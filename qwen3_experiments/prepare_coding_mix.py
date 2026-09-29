"""Build an audited, source-preserving coding RL mix from pinned local inputs."""

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from qwen3_experiments.coding_dataset_cleaning import (
    balanced_fill,
    canonical_statement,
    clean_stdio_tests,
    duplicate_groups,
    json_text,
    leetcode_test_issues,
    statement_digest,
    statement_issues,
    stratified_sample,
)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            value.update(chunk)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def codeforces_ids(record):
    return {c["id"] for c in record["candidates"]} | {
        o["id"] for c in record["candidates"] for o in c["official"]
    }


class Builder:
    def __init__(self, config):
        self.config = config
        self.root = Path(config["output_root"])
        self.scratch = Path(config["scratch"])
        self.cache = self.scratch / "candidates"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.excluded = []
        self.records = []
        self.statistics = Counter()
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True)
        self.blocked_ids = set()
        self.holdout_prompts = []
        self.holdout_counts = {}
        self.additional_holdout_exclusions = defaultdict(list)
        if config.get("additional_holdout_exclusions"):
            for record in read(config["additional_holdout_exclusions"]):
                self.additional_holdout_exclusions[(record["origin"], record["source_row_index"])].append(record)

    def progress(self, state, **values):
        result = {"state": state, "updated_at": datetime.now(timezone.utc).isoformat(), **values}
        write(self.root / "build_status.json", result)
        write(self.scratch / "build_status.json", result)
        print(json.dumps(result), flush=True)

    def reject(self, metadata, reasons):
        self.excluded.append({"origin": metadata["origin"], "source_row_index": metadata["source_row_index"],
                              "platform": metadata["platform"], "problem_id": metadata.get("problem_id"),
                              "name": metadata.get("name"), "reasons": sorted(set(reasons))})

    def prepare_holdouts(self):
        for path in sorted(Path(self.config["codecontests_columns"]).glob("*.json")):
            if not path.name.startswith(("test-", "valid-")):
                continue
            values = read(path)
            self.holdout_counts["codecontests_" + path.name.split("-", 1)[0]] = len(values)
            for row in values:
                if row.get("cf_contest_id") and row.get("cf_index"):
                    self.blocked_ids.add(f"codeforces:{row['cf_contest_id']}/{row['cf_index']}")
                self.blocked_ids.add("codecontests:" + row["name"].split()[0])
                self.holdout_prompts.append(("codecontests_holdout", canonical_statement(row["description"])))
        for label, filename in self.config.get("holdout_question_files", {}).items():
            rows = read(filename)
            self.holdout_counts[label] = len(rows)
            for row in rows:
                if row.get("messages"):
                    text = "\n".join(m["content"] for m in row["messages"])
                elif row.get("prompt"):
                    text = row["prompt"]
                else:
                    text = self.tokenizer.decode(row["prompt_token_ids"], skip_special_tokens=False)
                self.holdout_prompts.append((label, canonical_statement(text)))
                if row.get("platform") == "atcoder":
                    self.blocked_ids.add("atcoder:" + str(row["id"]))
                if row.get("platform") == "codeforces":
                    problem_id = str(row["id"]).replace("_", "/")
                    if re.fullmatch(r"\d+/[A-Z]\d*", problem_id):
                        self.blocked_ids.add("codeforces:" + problem_id)
                if row.get("platform") == "leetcode":
                    self.blocked_ids.add("leetcode_number:" + str(row["id"]))
        self.progress("holdouts_indexed", heldout_counts=self.holdout_counts)

    def holdout_matches(self, metadata, body):
        extra = self.additional_holdout_exclusions.get((metadata["origin"], metadata["source_row_index"]))
        if extra:
            return [f"heldout_exact_problem_overlap:{r['heldout_dataset']}:{r['heldout_id']}:{r['rule']}" for r in extra]
        if set(metadata["identity_keys"]) & self.blocked_ids:
            return ["heldout_problem_id_overlap"]
        if self.config.get("holdout_policy") == "exact_problem_only":
            return []
        normalized = canonical_statement(body)
        for label, text in self.holdout_prompts:
            if len(normalized) >= 100 and normalized in text:
                return ["heldout_statement_overlap:" + label]
        return []

    def add_candidate(self, metadata, messages, body, ground_truth, num_tests, duplicates=0):
        issues = self.holdout_matches(metadata, body)
        if issues:
            self.reject(metadata, issues)
            return
        token_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=True
        )
        if len(token_ids) > self.config["max_prompt_tokens"]:
            self.reject(metadata, ["prompt_exceeds_token_limit"])
            return
        identifier = f"{metadata['origin']}_{metadata['source_row_index']:05d}"
        record = {**metadata, "id": identifier, "statement_sha256": statement_digest(body),
                  "prompt_sha256": hashlib.sha256(json_text(messages).encode()).hexdigest(),
                  "prompt_tokens": len(token_ids), "num_tests": num_tests,
                  "duplicate_tests_removed": duplicates,
                  "payload_file": str(self.cache / f"{identifier}.json")}
        payload = {"id": identifier, "data_source": "nemotron_code_gen" if metadata["origin"] == "nemotron" else "leetcode_dataset",
                   "prompt": messages, "ability": "coding", "reward_model": {
                       "style": "rule", "ground_truth": json_text(ground_truth)}}
        write(record["payload_file"], payload)
        self.records.append(record)

    def nemotron_candidates(self):
        c = self.config
        population = read(c["nemotron_population"])
        matches = {r["source_index"]: r for r in read(c["nemotron_matches"])}
        ratings = {r["source_index"]: r for r in read(c["codeforces_ratings"])}
        atcoder = {r["name"]: r for r in read(c["atcoder_matches"])}
        candidates = {}
        for index, row in enumerate(population):
            self.statistics["nemotron_source_rows"] += 1
            platform = row["source"]
            match = matches[index]
            metadata = {"origin": "nemotron", "source_row_index": index, "platform": platform,
                        "source_repo": c["nemotron_repo"], "source_revision": c["nemotron_revision"],
                        "source_file": row["source_file"], "source_file_row": row["source_file_row"],
                        "source_hash_id": row["hash_id"], "source_dataset": row["dataset"],
                        "source_split": "train", "cf_rating": None, "atcoder_difficulty": None,
                        "atcoder_difficulty_raw": None, "leetcode_difficulty": None,
                        "reference_passed": None, "identity_keys": []}
            if platform == "codeforces":
                rating = ratings[index]
                values = rating["ratings"]
                if not values or not all(c["cf_min"] <= x <= c["cf_max"] for x in values):
                    self.statistics["codeforces_outside_requested_band_or_unrated"] += 1
                    continue
                if len(values) != 1 or rating["rating_provenance"] != "official_api":
                    self.reject(metadata, ["ambiguous_or_unverified_codeforces_rating"])
                    continue
                ids = sorted(codeforces_ids(rating))
                metadata.update(cf_rating=values[0], problem_id=ids[0], name=rating["candidates"][0]["title"],
                                identity_keys=["codeforces:" + pid for pid in ids], priority="primary")
            elif platform == "atcoder":
                choices = {atcoder[m["name"]]["atcoder_problem_id"]: atcoder[m["name"]]
                           for m in match["matches"] if m.get("name") in atcoder}
                if len(choices) != 1:
                    self.reject(metadata, ["unresolved_atcoder_problem_identity"])
                    continue
                item = next(iter(choices.values()))
                difficulty = item["difficulty"]
                if difficulty is None or not c["atcoder_min"] <= difficulty <= c["atcoder_max"]:
                    self.statistics["atcoder_outside_requested_band_or_unrated"] += 1
                    continue
                metadata.update(atcoder_difficulty=difficulty, atcoder_difficulty_raw=item["difficulty_raw"],
                                problem_id=item["atcoder_problem_id"], name=item["source_title"],
                                identity_keys=["atcoder:" + item["atcoder_problem_id"]], priority="primary")
            elif platform in c["filler_platforms"]:
                if not match["matches"]:
                    self.reject(metadata, ["unresolved_source_problem_identity"])
                    continue
                ids = sorted({m["name"].split()[0] for m in match["matches"]})
                metadata.update(problem_id=ids[0], name=match["matches"][0]["name"],
                                identity_keys=["codecontests:" + pid for pid in ids], priority="filler")
            else:
                continue
            self.statistics[platform + "_in_scope_raw"] += 1
            issues = statement_issues(match["body"], standard_input=True)
            if any(Path(m["upstream_file"]).name.startswith(("test-", "valid-")) for m in match["matches"]):
                issues.append("upstream_validation_or_test_split")
            if issues:
                self.reject(metadata, issues)
            else:
                candidates[index] = (metadata, row, match["body"])
        self.progress("checking_nemotron_tests", candidate_rows=len(candidates))
        import pyarrow.parquet as pq

        offset = 0
        source_files = []
        for path in sorted(Path(c["nemotron_parquet_root"]).glob("train-*.parquet")):
            source_files.append({"file": "data/" + path.name, "sha256": digest(path), "bytes": path.stat().st_size})
            source = pq.ParquetFile(path)
            file_offset = 0
            for batch in source.iter_batches(batch_size=16):
                for raw in batch.to_pylist():
                    index = offset + file_offset
                    file_offset += 1
                    if index not in candidates:
                        continue
                    metadata, original, body = candidates[index]
                    assert raw["hash_id"] == original["hash_id"]
                    assert raw["responses_create_params"] == original["responses_create_params"]
                    assert raw["source"] == metadata["platform"]
                    verifier = raw.get("verifier_metadata") or {}
                    tests, issues, duplicates = clean_stdio_tests(verifier.get("unit_tests"))
                    if issues:
                        self.reject(metadata, issues)
                        continue
                    ground_truth = {"grader": "nemo_gym_code_gen", "unit_tests": tests,
                                    "unit_test_timeout_seconds": 10, "check_eos": False,
                                    "score_after_thinking": True}
                    self.add_candidate(metadata, raw["responses_create_params"]["input"], body,
                                       ground_truth, len(tests["inputs"]), duplicates)
            assert file_offset == source.metadata.num_rows
            offset += file_offset
            self.progress("checking_nemotron_tests", rows_scanned=offset, clean_candidates=len(self.records),
                          excluded=len(self.excluded), last_file=path.name)
        assert offset == len(population) == 16083
        write(self.root / "nemotron_source_file_hashes.json", source_files)

    def leetcode_candidates(self):
        c = self.config
        reference = read(c["leetcode_reference_summary"])
        assert reference["checked"] == 606
        rows = [json.loads(line) for line in Path(c["leetcode_source"]).open() if line.strip()]
        for index, row in enumerate(rows):
            if row["difficulty"] != "Hard":
                continue
            self.statistics["leetcode_in_scope_raw"] += 1
            metadata = {"origin": "leetcode", "source_row_index": index, "platform": "leetcode",
                        "source_repo": c["leetcode_repo"], "source_revision": c["leetcode_revision"],
                        "source_file": "LeetCodeDataset-train.jsonl", "source_file_row": index,
                        "source_hash_id": row["task_id"], "source_dataset": c["leetcode_repo"],
                        "source_split": "train", "cf_rating": None, "atcoder_difficulty": None,
                        "atcoder_difficulty_raw": None, "leetcode_difficulty": "Hard",
                        "reference_passed": reference["results"][row["task_id"]]["passed"],
                        "identity_keys": ["leetcode:" + row["task_id"], "leetcode_number:" + str(row["question_id"])],
                        "problem_id": row["task_id"], "name": row["task_id"], "priority": "primary"}
            issues = statement_issues(row["problem_description"], standard_input=False) + leetcode_test_issues(row)
            expected_hash = reference["results"][row["task_id"]]["source_hash"]
            assert hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == expected_hash
            if not metadata["reference_passed"]:
                reason = reference["results"][row["task_id"]]["attempts"][-1]["result"]
                if "No module named" in reason:
                    raise RuntimeError("Repair missing checker dependencies before filtering data")
                issues.append("reference_does_not_pass_checker:" + reason)
            if issues:
                self.reject(metadata, issues)
                continue
            ground_truth = {"grader": "leetcode_dataset", "problem": {k: row[k] for k in
                             ("task_id", "prompt", "entry_point", "test")},
                            "suite_timeout_seconds": 10, "check_eos": False, "score_after_thinking": True}
            count = sum(isinstance(node, ast.Assert) for node in ast.walk(ast.parse(row["test"])))
            self.add_candidate(metadata, [{"role": "user", "content": row["query"]}],
                               row["problem_description"], ground_truth, count)

    def export(self):
        groups = duplicate_groups(self.records)
        representatives = []
        duplicate_audit = []
        for group in groups:
            ordered = sorted(group, key=lambda r: (r["priority"] != "primary", -r["num_tests"],
                                                   hashlib.sha256(f"{self.config['seed']}:{r['id']}".encode()).hexdigest()))
            representative = ordered[0]
            representatives.append(representative)
            if len(ordered) > 1:
                duplicate_audit.append({"kept": representative["id"], "removed": [r["id"] for r in ordered[1:]],
                                        "identities": sorted({k for r in ordered for k in r["identity_keys"]})})
                for row in ordered[1:]:
                    self.reject(row, ["duplicate_of:" + representative["id"]])
        primary = [r for r in representatives if r["priority"] == "primary"]
        pools = defaultdict(list)
        for row in representatives:
            if row["priority"] == "filler":
                pools[row["platform"]].append(row)
        selected_primary = primary
        if len(primary) > self.config["target_rows"]:
            retained = [r for r in primary if r["platform"] != "codeforces"]
            cf_pools = defaultdict(list)
            for row in primary:
                if row["platform"] == "codeforces":
                    cf_pools[row["cf_rating"]].append(row)
            selected_primary = retained + stratified_sample(
                cf_pools, self.config["target_rows"] - len(retained), self.config["seed"]
            )
        filler = balanced_fill(pools, self.config["target_rows"] - len(selected_primary), self.config["seed"])
        selected = sorted(selected_primary + filler, key=lambda r: hashlib.sha256(f"{self.config['seed']}:{r['id']}".encode()).hexdigest())
        assert len(selected) == self.config["target_rows"] == 3200
        assert len(duplicate_groups(selected)) == 3200
        upload = self.root / "upload"
        (upload / "data").mkdir(parents=True, exist_ok=True)
        exported = []
        public_manifest = []
        for row in selected:
            payload = read(row["payload_file"])
            extra = {k: v for k, v in row.items() if k not in ("payload_file", "identity_keys", "priority", "id", "origin")}
            extra["canonical_ids"] = row["identity_keys"]
            extra["selection_group"] = row["priority"]
            extra["test_count_kind"] = "stdin_pairs" if row["origin"] == "nemotron" else "assert_nodes"
            payload["extra_info"] = extra
            exported.append(payload)
            public_manifest.append({"id": row["id"], **extra})
        import pyarrow as pa
        import pyarrow.parquet as pq

        destination = upload / "data/train-00000-of-00001.parquet"
        pq.write_table(pa.Table.from_pylist(exported), destination, compression="zstd", row_group_size=100)
        recovered = pq.read_table(destination).to_pylist()
        assert recovered == exported
        checks = Counter()
        for row in recovered:
            truth = json.loads(row["reward_model"]["ground_truth"])
            assert truth["check_eos"] is False and truth["score_after_thinking"] is True
            if truth["grader"] == "nemo_gym_code_gen":
                cleaned, issues, _ = clean_stdio_tests(truth["unit_tests"])
                assert not issues and cleaned == truth["unit_tests"]
            else:
                assert set(truth["problem"]) == {"task_id", "prompt", "entry_point", "test"}
                assert row["extra_info"]["reference_passed"] is True
            assert "completion" not in truth and "response" not in truth and "solution" not in truth
            checks[truth["grader"]] += 1
        summary = {"rows": len(selected), "seed": self.config["seed"], "split": "train",
                   "source_counts": dict(Counter(r["platform"] for r in selected)),
                   "selection_groups": dict(Counter(r["priority"] for r in selected)),
                   "clean_primary_counts": dict(Counter(r["platform"] for r in primary)),
                   "eligible_primary_not_selected": len(primary) - len(selected_primary),
                   "selection_policy": "keep_atcoder_and_leetcode; proportional_cf_rating_strata_if_needed; balanced_other_sources_if_short",
                   "clean_filler_pool_counts": {k: len(v) for k, v in pools.items()},
                   "cf_rating_counts": dict(Counter(r["cf_rating"] for r in selected if r["cf_rating"] is not None)),
                   "cf_range": [self.config["cf_min"], self.config["cf_max"]],
                   "atcoder_range": [self.config["atcoder_min"], self.config["atcoder_max"]],
                   "input_statistics": dict(self.statistics),
                   "all_exclusion_reasons": dict(Counter(reason.split(":", 1)[0] for r in self.excluded for reason in r["reasons"])),
                   "duplicate_groups_merged": len(duplicate_audit),
                   "duplicate_tests_removed_in_selected": sum(r["duplicate_tests_removed"] for r in selected),
                   "maximum_prompt_tokens": max(r["prompt_tokens"] for r in selected),
                   "holdout_question_counts": self.holdout_counts,
                   "grader_counts": dict(checks), "parquet_sha256": digest(destination),
                   "parquet_bytes": destination.stat().st_size,
                   "created_at": datetime.now(timezone.utc).isoformat()}
        write(self.root / "summary.json", summary)
        write(upload / "audit/summary.json", summary)
        write(upload / "audit/selection_manifest.json", public_manifest)
        write(upload / "audit/exclusions.json", self.excluded)
        write(upload / "audit/duplicate_groups.json", duplicate_audit)
        selected_ids = {r["id"] for r in selected}
        write(upload / "audit/eligible_not_selected.json", [
            {k: r[k] for k in ("id", "platform", "problem_id", "cf_rating", "priority")}
            for r in representatives if r["id"] not in selected_ids
        ])
        write(self.root / "selected_internal.json", selected)
        self.progress("complete", summary=summary)

    def run(self):
        self.progress("starting")
        self.prepare_holdouts()
        self.nemotron_candidates()
        self.leetcode_candidates()
        self.export()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    builder = Builder(read(args.config))
    try:
        builder.run()
    except Exception as exc:
        builder.progress("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
