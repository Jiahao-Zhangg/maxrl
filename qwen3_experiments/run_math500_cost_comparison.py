#!/usr/bin/env python3
"""Run a new checkpoint on four GPUs and compare all four historical cost variants."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from run_math500_eval1_eval2 import audit_point
from run_math500_offset256_comparison import BASELINE_LABEL, BASELINE_REPO, CHECKPOINT, LABEL, PROTOCOLS, REPO_ROOT, result_paths, sha256, write_json
from run_math500_two_eval3 import audit_result
from summarize_math500_offset256_comparison import TITLES, audit


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def normalize(result: dict) -> dict:
    return {**result, "accuracy": result.get("accuracy", result.get("fraction_solved"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("python", "artifact-root", "output-root", "prepared-manifest", "offset256-root", "f-cov-root", "ipc-root"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=4, required=True, metavar=("EVAL1", "EVAL2", "SWEEP", "IID"))
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--report-only", action="store_true", help="Audit and render available results without launching GPU work")
    args = parser.parse_args()
    if len(set(args.gpus)) != 4:
        raise ValueError("Each protocol must have its own GPU")
    for key in ("python", "artifact_root", "output_root", "prepared_manifest", "offset256_root", "f_cov_root", "ipc_root"):
        setattr(args, key, getattr(args, key).resolve())
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    prepared = read(args.prepared_manifest)
    fcov_source = read(args.f_cov_root / "manifest.json")
    fcov_label = "f_cov_offset256_step150"
    fcov_repo = fcov_source["checkpoint_repo"]
    model_repo, model_revision = prepared["checkpoint_repo"], prepared["checkpoint_revision"]
    fcov_args = SimpleNamespace(output_root=args.f_cov_root, model_label=fcov_label, checkpoint_repo=fcov_repo, checkpoint_revision=fcov_source["checkpoint_revision"])
    model_args = SimpleNamespace(output_root=root, model_label=args.model_label, checkpoint_repo=model_repo, checkpoint_revision=model_revision)
    display = {"cap8": "Cap8", "offset256": "L+256", "f_cov_offset256": "F-cov (L+256)", "new_model": args.display_name}
    checkpoints = {"cap8": BASELINE_REPO, "offset256": CHECKPOINT, "f_cov_offset256": fcov_repo, "new_model": model_repo}
    comparisons = {protocol: {key: {} for key in display} for protocol in PROTOCOLS}
    source_hashes = {}
    children = {}
    state = {"state": "preparing", "supervisor_pid": os.getpid(), "gpus": dict(zip(PROTOCOLS, args.gpus, strict=True)), "failed": []}

    def collect() -> int:
        completed = 0
        for protocol, (_, budgets, _) in PROTOCOLS.items():
            for budget in budgets:
                path = result_paths(root / protocol / "results", protocol, budget, args.model_label)[-1]
                if str(budget) not in comparisons[protocol]["new_model"] and path.is_file():
                    value = audit_point(root / protocol / "results", protocol, budget, args.model_label, model_repo) if protocol in ("eval1", "eval2") else audit_result(model_args, protocol, budget)
                    comparisons[protocol]["new_model"][str(budget)] = normalize(value)
            completed += len(comparisons[protocol]["new_model"])
        return completed

    def render() -> None:
        count = collect()
        state.update(completed_points=count, total_points=19, updated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        state["protocols"] = {p: read(root / p / "status.json") for p in PROTOCOLS if (root / p / "status.json").is_file()}
        write_json(root / "status.json", state)
        write_json(root / "comparison.json", comparisons)
        lines = ["# MATH-500: four cost variants (step 150)", ""]
        lines += [f"- [{display[key]}](https://huggingface.co/{checkpoints[key]})" for key in display]
        lines += [
            "",
            f"New checkpoint revision: `{model_revision}`.",
            "",
            "All 500 questions, temperature 0.6, top-p 0.95, top-k -1, seed 0, and the historical MathVerify scorer with a one-second timeout. "
            "The three completed models are reused unchanged; only the new model is evaluated. "
            "Only generated tokens are charged. These are single-seed results, not significance tests.",
            "",
        ]
        for protocol, (_, budgets, _) in PROTOCOLS.items():
            explanation = (
                "Columns are per-response output caps; mean@4 averages four binary scores per question."
                if protocol == "eval1"
                else (
                    "Columns are independent per-question total budgets; accuracy means at least one answer in the realized list is correct. Every question uses its full budget, including after success; each response is capped at 4096 or the remaining budget."
                    if protocol == "eval2"
                    else "Columns are b; all questions share a total budget of 500 × b. Each response is capped at 4096 or the remaining budget, and all 500 questions remain in the denominator."
                )
            )
            lines += [f"## {TITLES[protocol]}", "", explanation, "", "| Model | " + " | ".join(map(str, budgets)) + " |", "|---|" + "---:|" * len(budgets)]
            for key, name in display.items():
                values = comparisons[protocol][key]
                cells = [f"{values[str(b)]['accuracy']:.2%}" if str(b) in values else "pending" for b in budgets]
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
            lines.append("")
        lines += ["## Validation", "", f"New-model budget points audited: {count}/19. All 57 baseline result sets were checked against raw responses. All generated responses and per-question results are retained.", ""]
        temporary = root / "comparison.md.tmp"
        temporary.write_text("\n".join(lines))
        temporary.replace(root / "comparison.md")

    def interrupted(signum, frame) -> None:
        raise InterruptedError(f"Comparison supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for protocol, (baseline_dir, budgets, _) in PROTOCOLS.items():
                for budget in budgets:
                    for key, label, directory in (
                        ("cap8", BASELINE_LABEL, REPO_ROOT / "outputs" / baseline_dir / "results"),
                        ("offset256", LABEL, args.offset256_root / protocol / "results"),
                        ("f_cov_offset256", fcov_label, args.f_cov_root / protocol / "results"),
                    ):
                        if key != "f_cov_offset256":
                            value = audit(directory, protocol, budget, label)
                        elif protocol in ("eval1", "eval2"):
                            value = audit_point(directory, protocol, budget, label, fcov_repo)
                        else:
                            value = audit_result(fcov_args, protocol, budget)
                        comparisons[protocol][key][str(budget)] = normalize(value)
                        source_hashes.update({str(path): sha256(path) for path in result_paths(directory, protocol, budget, label)})
            print("All 57 baseline result sets passed raw-rollout checks", flush=True)
            manifest = {"checkpoint_repo": model_repo, "checkpoint_revision": model_revision, "prepared_manifest": str(args.prepared_manifest), "prepared_manifest_sha256": sha256(args.prepared_manifest), "baseline_sha256": source_hashes, "gpus": state["gpus"], "commands": {}}
            if (root / "manifest.json").is_file():
                previous = read(root / "manifest.json")
                for key, value in manifest.items():
                    if key != "commands" and previous.get(key) != value:
                        raise ValueError(f"Cannot resume: manifest field {key} changed")
                manifest["commands"] = previous.get("commands", {})
            write_json(root / "manifest.json", manifest)
            render()
            if args.report_only:
                state["state"] = "complete" if state["completed_points"] == 19 else "partial_report"
                render()
                return
            for protocol, gpu in zip(PROTOCOLS, args.gpus, strict=True):
                phase = root / protocol
                phase.mkdir(parents=True, exist_ok=True)
                if len(comparisons[protocol]["new_model"]) == len(PROTOCOLS[protocol][1]):
                    continue
                if (phase / "status.json").is_file():
                    prior_pid = read(phase / "status.json").get("supervisor_pid")
                    if prior_pid and Path(f"/proc/{prior_pid}").exists():
                        raise RuntimeError(f"An existing supervisor may still own {protocol}: PID {prior_pid}; inspect before resuming")
                runner = "run_math500_eval1_eval2.py" if protocol in ("eval1", "eval2") else "run_math500_two_eval3.py"
                command = [
                    sys.executable,
                    "-u",
                    str(REPO_ROOT / "qwen3_experiments" / runner),
                    "--python",
                    str(args.python),
                    "--protocol",
                    protocol,
                    "--gpu",
                    str(gpu),
                    "--model-path",
                    prepared["model_path"],
                    "--model-label",
                    args.model_label,
                    "--checkpoint-repo",
                    model_repo,
                    "--checkpoint-revision",
                    model_revision,
                    "--artifact-root",
                    str(args.artifact_root),
                    "--ipc-root",
                    str(args.ipc_root / f"g{gpu}"),
                    "--output-root",
                    str(root),
                    "--source-manifest",
                    str(args.prepared_manifest),
                ]
                manifest["commands"][protocol] = command
                write_json(root / "manifest.json", manifest)
                with (phase / "supervisor.log").open("a") as log:
                    child = subprocess.Popen(command, cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
                children[protocol] = child
                print(f"Started {protocol} supervisor on GPU {gpu}: PID={child.pid}", flush=True)
            state["state"] = "running"
            while any(child.poll() is None for child in children.values()):
                render()
                time.sleep(10)
            failed = [f"{p}: supervisor exit code {child.returncode}" for p, child in children.items() if child.returncode]
            if failed:
                raise RuntimeError("; ".join(failed))
            if collect() != 19:
                raise ValueError("The workers exited without all 19 budget results")
            if any(sha256(Path(path)) != digest for path, digest in source_hashes.items()):
                raise ValueError("Baseline artifacts changed during the run")
            state["state"] = "complete"
            render()
            print(f"All 19 new results audited; four-model comparison complete: {root / 'comparison.md'}", flush=True)
        except BaseException as exc:
            state.update(state="failed", failed=[str(exc)])
            write_json(root / "status.json", state)
            raise
        finally:
            for child in children.values():
                if child.poll() is None:
                    try:
                        child.terminate()
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    main()
