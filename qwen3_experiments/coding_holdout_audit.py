"""Audit a coding training pool against the exact frozen evaluation questions."""

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import parse_qs, urlparse

from qwen3_experiments.coding_dataset_cleaning import canonical_statement


def read(path):
    return json.loads(Path(path).read_text())


def words(text):
    text = html.unescape(unicodedata.normalize("NFKC", text)).casefold()
    return re.findall(r"[a-z0-9]+", text)


def title_key(text):
    return " ".join(words(text or ""))


def shingles(text):
    tokens = words(text)
    return {tuple(tokens[i:i + 5]) for i in range(len(tokens) - 4)}


def url_identity(url):
    parsed = urlparse(url or "")
    host, path = parsed.netloc.casefold(), parsed.path
    if host.endswith("codeforces.com"):
        match = re.search(r"/(?:problemset/problem|contest)/(\d+)/(?:problem/)?([a-zA-Z]\d*)", path)
        if match:
            return f"codeforces:{match[1]}/{match[2].upper()}"
    if host.endswith("codechef.com") and "/problems/" in path:
        return "codechef:" + path.split("/problems/", 1)[1].strip("/").split("/")[0].casefold()
    if host.endswith("hackerearth.com"):
        for marker in ("/algorithm/", "/approximate/"):
            if marker in path:
                return "hackerearth:" + path.split(marker, 1)[1].strip("/").split("/")[0].casefold()
    if host.endswith("usaco.org") and parse_qs(parsed.query).get("cpid"):
        return "usaco:" + parse_qs(parsed.query)["cpid"][0]
    return None


def training_pool(config):
    matches = read(config["nemotron_matches"])
    ratings = {r["source_index"]: r for r in read(config["codeforces_ratings"])}
    atcoder = {r["name"]: r for r in read(config["atcoder_matches"])}
    for row in matches:
        platform, index = row["platform"], row["source_index"]
        keys, titles = set(), set()
        if platform == "codeforces":
            rating = ratings[index]
            for candidate in rating["candidates"]:
                keys.add("codeforces:" + candidate["id"])
                keys.update("codeforces:" + item["id"] for item in candidate["official"])
                titles.add(candidate["title"])
        elif platform == "atcoder":
            for candidate in row["matches"]:
                if candidate.get("name") in atcoder:
                    item = atcoder[candidate["name"]]
                    keys.add("atcoder:" + item["atcoder_problem_id"])
                    titles.add(item["source_title"])
        else:
            for candidate in row["matches"]:
                titles.add(candidate["name"])
                keys.add(platform + ":" + candidate["name"].split()[0].casefold())
        yield {"id": f"nemotron_{index:05d}", "origin": "nemotron", "source_row_index": index,
               "platform": platform, "identity_keys": sorted(keys), "titles": sorted(titles), "body": row["body"]}
    leetcode = [json.loads(line) for line in Path(config["leetcode_source"]).open() if line.strip()]
    for index, row in enumerate(leetcode):
        if row["difficulty"] == "Hard":
            yield {"id": f"leetcode_{index:05d}", "origin": "leetcode", "source_row_index": index,
                   "platform": "leetcode", "titles": [row["task_id"]],
                   "identity_keys": ["leetcode:" + row["task_id"], "leetcode_number:" + str(row["question_id"])],
                   "body": row["problem_description"]}


def evaluation_pool(config):
    import pyarrow.parquet as pq

    cfg = config["overlap_sources"]
    taco_eval = read(config["holdout_question_files"]["taco_test_non_spj"])
    taco = pq.read_table(cfg["taco_raw"], columns=["question", "name", "source", "url"]).to_pylist()
    assert len(taco) == 1000 and len(taco_eval) == 782
    for question in taco_eval:
        row = taco[question["source_index"]]
        identity = url_identity(row["url"])
        yield {"dataset": "taco_test_non_spj", "id": question["id"], "source_index": question["source_index"],
               "platform": row["source"], "titles": [row["name"] or ""], "url": row["url"],
               "identity_keys": [identity] if identity else [], "body": row["question"]}
    for path in Path(config["codecontests_columns"]).glob("test-*.json"):
        rows = read(path)
        assert len(rows) == 165
        for index, row in enumerate(rows):
            pid = f"{row['cf_contest_id']}/{row['cf_index']}"
            yield {"dataset": "codecontests_test", "id": f"code_contests_test_{index:03d}",
                   "source_index": index, "platform": "codeforces", "identity_keys": ["codeforces:" + pid],
                   "titles": [re.sub(r"^\d+_[A-Z]\d*\.\s*", "", row["name"])],
                   "url": "https://codeforces.com/problemset/problem/" + pid, "body": row["description"]}
    original = read(cfg["usaco_raw"])
    usaco_eval = read(config["holdout_question_files"]["usaco307"])
    assert len(usaco_eval) == 307
    for question in usaco_eval:
        row = original[question["id"]]
        assert row["problem_link"] == question["url"]
        yield {"dataset": "usaco307", "id": question["id"], "source_index": question["source_index"],
               "platform": "usaco", "identity_keys": ["usaco:" + str(row["cp_id"])], "titles": [row["name"]],
               "url": row["problem_link"], "body": row["description"]}
    original = [json.loads(line) for line in Path(cfg["lcb_raw"]).open() if line.strip()]
    lcb_eval = read(config["holdout_question_files"]["lcb_v6"])
    assert len(original) == len(lcb_eval) == 175
    for question in lcb_eval:
        row = original[question["source_index"]]
        assert row["question_id"] == question["id"]
        if row["platform"] == "leetcode":
            keys = ["leetcode_number:" + str(row["question_id"]), "leetcode:" + row["question_title"]]
            url = "https://leetcode.com/problems/" + row["question_title"] + "/"
        else:
            assert row["platform"] == "atcoder"
            keys = ["atcoder:" + row["question_id"]]
            url = f"https://atcoder.jp/contests/{row['contest_id']}/tasks/{row['question_id']}"
        yield {"dataset": "lcb_v6", "id": question["id"], "source_index": question["source_index"],
               "platform": row["platform"], "identity_keys": keys, "titles": [row["question_title"]],
               "url": url, "body": row["question_content"]}


def audit(config):
    root = Path(config["output_root"])
    holdouts = list(evaluation_pool(config))
    counts = dict(Counter(r["dataset"] for r in holdouts))
    assert counts == {"taco_test_non_spj": 782, "codecontests_test": 165, "usaco307": 307, "lcb_v6": 175}
    identities, titles, postings = defaultdict(set), defaultdict(set), defaultdict(set)
    for index, row in enumerate(holdouts):
        row["shingles"] = shingles(row["body"])
        row["compact"] = "".join(words(row["body"]))
        row["canonical"] = canonical_statement(row["body"])
        for key in row["identity_keys"]:
            identities[key].add(index)
        for title in row["titles"]:
            if title_key(title):
                titles[title_key(title)].add(index)
        for shingle in row["shingles"]:
            postings[shingle].add(index)
    # Discard common boilerplate grams for candidate retrieval, not for scoring.
    postings = {key: indexes for key, indexes in postings.items() if len(indexes) <= 10}
    confirmed, review = [], []
    for number, row in enumerate(training_pool(config)):
        exact = set().union(*(identities.get(key, set()) for key in row["identity_keys"]))
        same_title = set().union(*(titles.get(title_key(title), set()) for title in row["titles"]))
        grams = shingles(row["body"])
        hits = Counter(index for gram in grams for index in postings.get(gram, ()))
        candidates = exact | same_title | {index for index, count in hits.items() if count >= 3}
        compact = "".join(words(row["body"]))
        canonical = canonical_statement(row["body"])
        for index in candidates:
            heldout = holdouts[index]
            common = len(grams & heldout["shingles"])
            coverage = common / max(1, min(len(grams), len(heldout["shingles"])))
            normalized_full_match = min(len(compact), len(heldout["compact"])) >= 150 and (
                compact in heldout["compact"] or heldout["compact"] in compact
            )
            if index in exact:
                rule = "exact_problem_identity"
            elif len(canonical) >= 150 and canonical == heldout["canonical"]:
                rule = "normalized_statement_equality"
            elif normalized_full_match:
                rule = "statement_containment_review"
            elif coverage >= 0.8 and common >= 40:
                rule = "similar_statement_review"
            elif (coverage >= 0.45 and common >= 25) or index in same_title:
                rule = "manual_review"
            else:
                continue
            record = {"train_id": row["id"], "origin": row["origin"], "source_row_index": row["source_row_index"],
                      "platform": row["platform"], "titles": row["titles"], "heldout_dataset": heldout["dataset"],
                      "heldout_id": heldout["id"], "heldout_titles": heldout["titles"], "heldout_url": heldout["url"],
                      "rule": rule, "shared_word_5grams": common, "shorter_statement_coverage": round(coverage, 6),
                      "same_normalized_title": index in same_title}
            (confirmed if rule in ("exact_problem_identity", "normalized_statement_equality") else review).append(record)
        if number % 2000 == 0:
            print(json.dumps({"scanned": number, "confirmed_pairs": len(confirmed), "review_pairs": len(review)}), flush=True)
    selected = {r["id"] for r in read(root / "selected_internal.json")}
    result = {"evaluation_counts": counts, "training_rows_scanned": number + 1,
              "confirmed_pairs": confirmed, "manual_review_pairs": review,
              "current_selected_matches": [r for r in confirmed if r["train_id"] in selected],
              "current_selected_review": [r for r in review if r["train_id"] in selected],
              "method": "Exclude identical original tasks by platform identity, full normalized statement equality, or a reviewed identity alias. Similarity and titles trigger review only; distinct variants are retained.",
              "question_index": [{k: v for k, v in r.items() if k not in ('body', 'shingles', 'compact', 'canonical')} |
                                 {"statement_sha256": hashlib.sha256(r['body'].encode()).hexdigest()} for r in holdouts]}
    (root / "holdout_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"complete": True, "confirmed_pairs": len(confirmed), "review_pairs": len(review),
                      "selected_matches": len(result['current_selected_matches']),
                      "selected_review": len(result['current_selected_review'])}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    audit(read(args.config))


if __name__ == "__main__":
    main()
