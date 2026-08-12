#!/usr/bin/env python3
"""No-GPU regression tests for the full-context receipt comparator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable


ARMS = ("off-a", "on", "off-b")
KINDS = ("prime", "ship", "quality")


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def read_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{path}: expected JSON object")
    return value


def replace_bytes(path: pathlib.Path, data: bytes) -> None:
    # Break the temporary hardlink before writing so the source receipt is immutable.
    path.unlink()
    path.write_bytes(data)


def write_json(path: pathlib.Path, value: object) -> None:
    replace_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )


def receipt_files() -> list[str]:
    files = ["manifest.txt"]
    for arm in ARMS:
        files.extend((f"{arm}-server.log", f"{arm}-summary.json"))
        for kind in KINDS:
            files.extend((f"{arm}-{kind}.request.json", f"{arm}-{kind}.json"))
            if kind != "prime":
                files.extend((f"{arm}-{kind}.tokens.json", f"{arm}-{kind}.content.txt"))
    return files


def materialize(source: pathlib.Path, target: pathlib.Path) -> None:
    target.mkdir()
    for relative in receipt_files():
        src = source / relative
        dst = target / relative
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def manifest_values(path: pathlib.Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines())


def mutate_manifest(path: pathlib.Path, key: str, value: str) -> None:
    values = manifest_values(path / "manifest.txt")
    if key not in values:
        raise AssertionError(f"manifest lacks {key!r}")
    values[key] = value
    replace_bytes(
        path / "manifest.txt",
        "".join(f"{name}={item}\n" for name, item in values.items()).encode(),
    )


def set_ship_speed(path: pathlib.Path, arm: str, tps: float) -> None:
    result_path = path / f"{arm}-ship.json"
    result = read_json(result_path)
    result["timings"]["predicted_per_second"] = tps
    write_json(result_path, result)
    summary_path = path / f"{arm}-summary.json"
    summary = read_json(summary_path)
    summary["ship_predicted_per_second"] = tps
    write_json(summary_path, summary)


def mutate_off_instability(path: pathlib.Path) -> None:
    off_a = read_json(path / "off-a-summary.json")["ship_predicted_per_second"]
    set_ship_speed(path, "off-b", float(off_a) * 1.10)


def mutate_insufficient_gain(path: pathlib.Path) -> None:
    off_a = float(read_json(path / "off-a-summary.json")["ship_predicted_per_second"])
    off_b = float(read_json(path / "off-b-summary.json")["ship_predicted_per_second"])
    set_ship_speed(path, "on", (off_a + off_b) / 2.0)


def mutate_greedy_token(path: pathlib.Path) -> None:
    result_path = path / "on-quality.json"
    result = read_json(result_path)
    replacement = int(result["tokens"][0]) + 1
    result["tokens"][0] = replacement
    result["completion_probabilities"][0]["id"] = replacement
    write_json(result_path, result)
    replace_bytes(path / "on-quality.tokens.json", canonical_json(result["tokens"]))
    summary_path = path / "on-summary.json"
    summary = read_json(summary_path)
    summary["quality_tokens_sha256"] = hashlib.sha256(canonical_json(result["tokens"])).hexdigest()
    write_json(summary_path, summary)


def mutate_probability_drift(path: pathlib.Path) -> None:
    result_path = path / "on-quality.json"
    result = read_json(result_path)
    row = result["completion_probabilities"][0]
    selected_id = row["id"]
    row["logprob"] = float(row["logprob"]) - 1.0
    for entry in row["top_logprobs"]:
        if entry["id"] == selected_id:
            entry["logprob"] = row["logprob"]
            break
    else:
        raise AssertionError("selected token absent from canonical top_logprobs")
    write_json(result_path, result)


def run_compare(validator: pathlib.Path, out_dir: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(validator),
            "compare",
            "--out-dir",
            str(out_dir),
            "--arithmetic-expected",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def run_case(
    source: pathlib.Path,
    validator: pathlib.Path,
    name: str,
    mutate: Callable[[pathlib.Path], None] | None,
    expected_error: str | None,
) -> None:
    with tempfile.TemporaryDirectory(prefix=".fullctx-compare-test-", dir=source.parent) as raw:
        case_dir = pathlib.Path(raw) / "receipt"
        materialize(source, case_dir)
        if mutate is not None:
            mutate(case_dir)
        result = run_compare(validator, case_dir)
        output = result.stdout + result.stderr
        if expected_error is None:
            if result.returncode != 0:
                raise AssertionError(f"{name}: expected PASS\n{output[-4000:]}")
            comparison = read_json(case_dir / "comparison.json")
            if comparison.get("gate_passed") is not True:
                raise AssertionError(f"{name}: command succeeded without gate_passed=true")
        else:
            if result.returncode == 0:
                raise AssertionError(f"{name}: expected failure, got PASS")
            if expected_error not in output:
                raise AssertionError(
                    f"{name}: missing expected diagnostic {expected_error!r}\n{output[-4000:]}"
                )
        print(f"PASS {name}")


def file_hashes(source: pathlib.Path) -> dict[str, str]:
    paths = [source / relative for relative in receipt_files()]
    fixture = pathlib.Path(manifest_values(source / "manifest.txt")["fixture"])
    paths.append(fixture)
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--receipt",
        type=pathlib.Path,
        default=root / "results/fullctx-server-final-20260812T1605Z",
    )
    parser.add_argument(
        "--validator",
        type=pathlib.Path,
        default=root / "scripts/fullctx-server-bench.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.receipt.resolve()
    validator = args.validator.resolve()
    before = file_hashes(source)
    cases: tuple[tuple[str, Callable[[pathlib.Path], None] | None, str | None], ...] = (
        ("canonical receipt", None, None),
        ("wrong context provenance", lambda path: mutate_manifest(path, "ctx", "65536"), "manifest provenance mismatch"),
        ("wrong model provenance", lambda path: mutate_manifest(path, "model", "/invalid/model.gguf"), "fixture provenance mismatch"),
        ("unstable OFF controls", mutate_off_instability, "OFF speed spread"),
        ("insufficient ON gain", mutate_insufficient_gain, "candidate speed improvement"),
        ("greedy token mismatch", mutate_greedy_token, "greedy quality mismatch"),
        ("excessive probability drift", mutate_probability_drift, "candidate probability max_selected_logprob_delta"),
    )
    for name, mutate, expected_error in cases:
        run_case(source, validator, name, mutate, expected_error)
    after = file_hashes(source)
    if before != after:
        changed = sorted(path for path in before if before[path] != after.get(path))
        raise AssertionError(f"source receipt changed: {changed}")
    print(f"PASS source receipt preserved ({len(before)} files)")


if __name__ == "__main__":
    main()
