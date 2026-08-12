#!/usr/bin/env python3
"""Client and validator for the Muse one-seat full-context server gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import time
import urllib.error
import urllib.request


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def write_pretty(path: pathlib.Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def post_json(base_url: str, endpoint: str, payload: object, timeout_s: float) -> tuple[bytes, object]:
    request = urllib.request.Request(
        base_url + endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
            if response.status != 200:
                raise RuntimeError(f"{endpoint}: HTTP {response.status}: {raw[:1000]!r}")
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise RuntimeError(f"{endpoint}: HTTP {exc.code}: {body[:4000]!r}") from exc
    try:
        return raw, json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{endpoint}: non-JSON response: {raw[:4000]!r}") from exc


def wait_ready(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                body = response.read()
                if response.status == 200:
                    health = json.loads(body)
                    if health.get("status") == "ok":
                        return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy within {timeout_s:g}s: {last_error}")


def require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label}: expected integer, got {value!r}")
    return value


def require_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError(f"{label}: expected finite number, got {value!r}")
    return float(value)


def require_result_shape(
    result: object,
    *,
    label: str,
    n_predict: int,
    expected_cache_n: int,
    expected_prompt_n: int,
    n_ctx: int,
    require_probs: bool,
) -> dict:
    if not isinstance(result, dict):
        raise RuntimeError(f"{label}: expected JSON object, got {type(result).__name__}")
    if result.get("id_slot") != 0:
        raise RuntimeError(f"{label}: expected id_slot=0, got {result.get('id_slot')!r}")
    if result.get("truncated") is not False:
        raise RuntimeError(f"{label}: expected truncated=false, got {result.get('truncated')!r}")
    if result.get("stop_type") != "limit":
        raise RuntimeError(f"{label}: expected stop_type='limit', got {result.get('stop_type')!r}")

    tokens = result.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != n_predict:
        got = len(tokens) if isinstance(tokens, list) else tokens
        raise RuntimeError(f"{label}: expected {n_predict} returned tokens, got {got!r}")
    for index, token in enumerate(tokens):
        require_int(token, f"{label}.tokens[{index}]")
    if require_int(result.get("tokens_predicted"), f"{label}.tokens_predicted") != n_predict:
        raise RuntimeError(f"{label}: tokens_predicted does not equal {n_predict}")
    if require_int(result.get("tokens_evaluated"), f"{label}.tokens_evaluated") != expected_cache_n + expected_prompt_n:
        raise RuntimeError(
            f"{label}: tokens_evaluated does not equal {expected_cache_n + expected_prompt_n}"
        )

    timings = result.get("timings")
    if not isinstance(timings, dict):
        raise RuntimeError(f"{label}: missing timings object")
    cache_n = require_int(timings.get("cache_n"), f"{label}.timings.cache_n")
    prompt_n = require_int(timings.get("prompt_n"), f"{label}.timings.prompt_n")
    predicted_n = require_int(timings.get("predicted_n"), f"{label}.timings.predicted_n")
    if cache_n != expected_cache_n or prompt_n != expected_prompt_n:
        raise RuntimeError(
            f"{label}: cache proof failed: cache_n={cache_n}, prompt_n={prompt_n}, "
            f"expected {expected_cache_n}/{expected_prompt_n}"
        )
    if predicted_n != n_predict:
        raise RuntimeError(f"{label}: timings.predicted_n={predicted_n}, expected {n_predict}")
    require_number(timings.get("predicted_ms"), f"{label}.timings.predicted_ms")
    require_number(timings.get("predicted_per_second"), f"{label}.timings.predicted_per_second")

    generation_settings = result.get("generation_settings")
    if not isinstance(generation_settings, dict):
        raise RuntimeError(f"{label}: missing generation_settings")
    # The native completion response does not carry n_ctx in generation_settings.
    # The shell launcher proves n_ctx_slot from the server's own load log instead.
    if require_int(generation_settings.get("n_predict"), f"{label}.generation_settings.n_predict") != n_predict:
        raise RuntimeError(f"{label}: generation_settings.n_predict does not equal {n_predict}")

    if require_probs:
        probabilities = result.get("completion_probabilities")
        if not isinstance(probabilities, list) or len(probabilities) != n_predict:
            got = len(probabilities) if isinstance(probabilities, list) else probabilities
            raise RuntimeError(f"{label}: expected {n_predict} probability rows, got {got!r}")
        for index, row in enumerate(probabilities):
            if not isinstance(row, dict) or not isinstance(row.get("top_logprobs"), list):
                raise RuntimeError(f"{label}: malformed probability row {index}")
            if not row["top_logprobs"]:
                raise RuntimeError(f"{label}: empty top_logprobs at row {index}")
            if require_int(row.get("id"), f"{label}.completion_probabilities[{index}].id") != tokens[index]:
                raise RuntimeError(f"{label}: probability row {index} does not match returned token ID")

    return result


def fixture_command(args: argparse.Namespace) -> None:
    source = args.source.read_text(encoding="utf-8")
    source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if args.fixture.exists() and not args.force:
        fixture, tokens = load_fixture(args.fixture, args.token_count)
        expected = {
            "schema": 2,
            "model": str(args.model),
            "model_sha256": args.model_sha256,
            "source": str(args.source),
            "source_sha256": source_sha,
            "tokenize_add_special": True,
            "tokenize_parse_special": True,
        }
        wrong = {key: (fixture.get(key), value) for key, value in expected.items() if fixture.get(key) != value}
        if wrong:
            raise RuntimeError(
                f"existing fixture {args.fixture} has incompatible provenance {wrong!r}; "
                "pass --force to rebuild it"
            )
        print(json.dumps({"fixture": str(args.fixture), "tokens": len(tokens), "reused": True}))
        return

    raw, result = post_json(
        args.base_url,
        "/tokenize",
        {"content": source, "add_special": True, "parse_special": True},
        args.timeout,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "fixture-tokenize.raw.json").write_bytes(raw)
    write_pretty(args.out_dir / "fixture-tokenize.json", result)
    all_tokens = result.get("tokens") if isinstance(result, dict) else None
    if not isinstance(all_tokens, list) or len(all_tokens) < args.token_count:
        got = len(all_tokens) if isinstance(all_tokens, list) else all_tokens
        raise RuntimeError(f"wiki source tokenized to {got!r}; need at least {args.token_count}")
    tokens = [require_int(token, f"tokens[{index}]") for index, token in enumerate(all_tokens[: args.token_count])]
    token_sha = hashlib.sha256(canonical_json(tokens)).hexdigest()
    fixture = {
        "schema": 2,
        "model": str(args.model),
        "model_sha256": args.model_sha256,
        "source": str(args.source),
        "source_sha256": source_sha,
        "tokenize_add_special": True,
        "tokenize_parse_special": True,
        "token_count": len(tokens),
        "first_token_id": tokens[0],
        "last_token_id": tokens[-1],
        "tokens_sha256": token_sha,
        "tokens": tokens,
    }
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    write_pretty(args.fixture, fixture)
    print(json.dumps({"fixture": str(args.fixture), "tokens": len(tokens), "sha256": token_sha}))


def load_fixture(path: pathlib.Path, token_count: int) -> tuple[dict, list[int]]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(fixture, dict):
        raise RuntimeError(f"{path}: fixture root is not an object")
    tokens = fixture.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != token_count:
        got = len(tokens) if isinstance(tokens, list) else tokens
        raise RuntimeError(f"{path}: expected {token_count} tokens, got {got!r}")
    clean = [require_int(token, f"fixture.tokens[{index}]") for index, token in enumerate(tokens)]
    actual_sha = hashlib.sha256(canonical_json(clean)).hexdigest()
    if fixture.get("tokens_sha256") != actual_sha:
        raise RuntimeError(
            f"{path}: token hash mismatch: manifest={fixture.get('tokens_sha256')!r}, actual={actual_sha}"
        )
    return fixture, clean


def request_command(args: argparse.Namespace) -> None:
    fixture, tokens = load_fixture(args.fixture, args.token_count)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    prime_payload = {
        "prompt": tokens,
        "id_slot": 0,
        "n_predict": 0,
        "cache_prompt": False,
        "stream": False,
    }
    write_pretty(args.out_dir / f"{args.arm}-prime.request.json", prime_payload)
    started = time.monotonic()
    prime_raw, prime = post_json(args.base_url, "/completion", prime_payload, args.timeout)
    prime_elapsed = time.monotonic() - started
    (args.out_dir / f"{args.arm}-prime.raw.json").write_bytes(prime_raw)
    write_pretty(args.out_dir / f"{args.arm}-prime.json", prime)

    if not isinstance(prime, dict):
        raise RuntimeError(f"{args.arm} prime: response is not an object")
    if prime.get("id_slot") != 0 or prime.get("truncated") is not False:
        raise RuntimeError(
            f"{args.arm} prime: expected id_slot=0 and truncated=false, got "
            f"{prime.get('id_slot')!r}/{prime.get('truncated')!r}"
        )
    if require_int(prime.get("tokens_evaluated"), f"{args.arm}.prime.tokens_evaluated") != args.token_count:
        raise RuntimeError(f"{args.arm} prime: did not evaluate exactly {args.token_count} tokens")
    prime_timings = prime.get("timings")
    if not isinstance(prime_timings, dict):
        raise RuntimeError(f"{args.arm} prime: missing timings")
    if require_int(prime_timings.get("cache_n"), f"{args.arm}.prime.cache_n") != 0:
        raise RuntimeError(f"{args.arm} prime: fresh process unexpectedly reused cache")
    if require_int(prime_timings.get("prompt_n"), f"{args.arm}.prime.prompt_n") != args.token_count:
        raise RuntimeError(f"{args.arm} prime: did not process exactly {args.token_count} tokens")

    common = {
        "prompt": tokens,
        "id_slot": 0,
        "cache_prompt": True,
        "return_tokens": True,
        "stream": False,
        "ignore_eos": True,
    }
    ship_payload = {
        **common,
        "n_predict": args.n_predict,
        "seed": args.seed,
        "temperature": 1.0,
        "top_k": 64,
        "top_p": 0.95,
        "min_p": 0.0,
    }
    write_pretty(args.out_dir / f"{args.arm}-ship.request.json", ship_payload)
    started = time.monotonic()
    ship_raw, ship_obj = post_json(args.base_url, "/completion", ship_payload, args.timeout)
    ship_elapsed = time.monotonic() - started
    ship = require_result_shape(
        ship_obj,
        label=f"{args.arm} ship",
        n_predict=args.n_predict,
        expected_cache_n=args.token_count - 1,
        expected_prompt_n=1,
        n_ctx=args.n_ctx,
        require_probs=False,
    )
    (args.out_dir / f"{args.arm}-ship.raw.json").write_bytes(ship_raw)
    write_pretty(args.out_dir / f"{args.arm}-ship.json", ship)
    (args.out_dir / f"{args.arm}-ship.tokens.json").write_bytes(canonical_json(ship["tokens"]))
    (args.out_dir / f"{args.arm}-ship.content.txt").write_text(ship.get("content", ""), encoding="utf-8")

    # The ship completion extended the cached prompt. Supplying the original frozen
    # prompt again makes the server remove that suffix and re-evaluate its final token.
    quality_payload = {
        **common,
        "n_predict": args.quality_n_predict,
        "seed": args.seed,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "n_probs": args.n_probs,
        "post_sampling_probs": False,
    }
    write_pretty(args.out_dir / f"{args.arm}-quality.request.json", quality_payload)
    started = time.monotonic()
    quality_raw, quality_obj = post_json(args.base_url, "/completion", quality_payload, args.timeout)
    quality_elapsed = time.monotonic() - started
    quality = require_result_shape(
        quality_obj,
        label=f"{args.arm} quality",
        n_predict=args.quality_n_predict,
        expected_cache_n=args.token_count - 1,
        expected_prompt_n=1,
        n_ctx=args.n_ctx,
        require_probs=True,
    )
    (args.out_dir / f"{args.arm}-quality.raw.json").write_bytes(quality_raw)
    write_pretty(args.out_dir / f"{args.arm}-quality.json", quality)
    (args.out_dir / f"{args.arm}-quality.tokens.json").write_bytes(canonical_json(quality["tokens"]))
    (args.out_dir / f"{args.arm}-quality.content.txt").write_text(
        quality.get("content", ""), encoding="utf-8"
    )
    write_pretty(
        args.out_dir / f"{args.arm}-summary.json",
        {
            "arm": args.arm,
            "fixture": str(args.fixture),
            "fixture_tokens_sha256": fixture["tokens_sha256"],
            "token_count": args.token_count,
            "prime_elapsed_s": prime_elapsed,
            "ship_elapsed_s": ship_elapsed,
            "ship_predicted_per_second": ship["timings"]["predicted_per_second"],
            "ship_tokens_sha256": hashlib.sha256(canonical_json(ship["tokens"])).hexdigest(),
            "quality_elapsed_s": quality_elapsed,
            "quality_predicted_per_second": quality["timings"]["predicted_per_second"],
            "quality_tokens_sha256": hashlib.sha256(canonical_json(quality["tokens"])).hexdigest(),
        },
    )
    print(
        json.dumps(
            {
                "arm": args.arm,
                "depth": args.token_count,
                "ship_tps": ship["timings"]["predicted_per_second"],
                "quality_tps": quality["timings"]["predicted_per_second"],
            },
            sort_keys=True,
        )
    )


def read_json_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return value


def parse_manifest(path: pathlib.Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line or "=" not in raw_line:
            raise RuntimeError(f"{path}:{line_number}: expected key=value")
        key, value = raw_line.split("=", 1)
        if not key or key in result:
            raise RuntimeError(f"{path}:{line_number}: empty or duplicate key {key!r}")
        result[key] = value
    return result


def audit_completion_artifact(
    *,
    arm: str,
    kind: str,
    out_dir: pathlib.Path,
    fixture_token_sha: str,
    token_count: int,
) -> dict:
    request = read_json_object(out_dir / f"{arm}-{kind}.request.json")
    result = read_json_object(out_dir / f"{arm}-{kind}.json")
    prompt = request.get("prompt")
    if not isinstance(prompt, list) or len(prompt) != token_count:
        got = len(prompt) if isinstance(prompt, list) else prompt
        raise RuntimeError(f"{arm} {kind}: request has {got!r} prompt tokens, expected {token_count}")
    prompt_sha = hashlib.sha256(canonical_json(prompt)).hexdigest()
    if prompt_sha != fixture_token_sha:
        raise RuntimeError(f"{arm} {kind}: request prompt does not match the frozen fixture")
    if request.get("id_slot") != 0 or request.get("stream") is not False:
        raise RuntimeError(f"{arm} {kind}: request is not pinned to non-streaming slot 0")
    if result.get("id_slot") != 0 or result.get("truncated") is not False:
        raise RuntimeError(f"{arm} {kind}: result does not prove slot 0 / truncated=false")
    if require_int(result.get("tokens_evaluated"), f"{arm}.{kind}.tokens_evaluated") != token_count:
        raise RuntimeError(f"{arm} {kind}: tokens_evaluated does not equal {token_count}")
    timings = result.get("timings")
    if not isinstance(timings, dict):
        raise RuntimeError(f"{arm} {kind}: missing timings")

    if kind == "prime":
        if request.get("cache_prompt") is not False or request.get("n_predict") != 0:
            raise RuntimeError(f"{arm} prime: request is not cache-off / n_predict=0")
        if require_int(timings.get("cache_n"), f"{arm}.prime.cache_n") != 0:
            raise RuntimeError(f"{arm} prime: unexpectedly reused cache")
        if require_int(timings.get("prompt_n"), f"{arm}.prime.prompt_n") != token_count:
            raise RuntimeError(f"{arm} prime: did not process the full frozen prompt")
        # Current llama-server reports one internal sampled/logit token even for
        # n_predict=0, while returning no token IDs. The request setting, exact
        # prompt count, and the following cache_n proof are the authoritative
        # evidence that this was a cache-prime operation.
        if result.get("tokens") != []:
            raise RuntimeError(f"{arm} prime: n_predict=0 nevertheless returned token IDs")
        if require_int(result.get("tokens_cached"), f"{arm}.prime.tokens_cached") != token_count:
            raise RuntimeError(f"{arm} prime: did not leave the full prompt cached")
        return result

    if request.get("cache_prompt") is not True or request.get("return_tokens") is not True:
        raise RuntimeError(f"{arm} {kind}: request does not enable prompt cache and raw tokens")
    if request.get("ignore_eos") is not True:
        raise RuntimeError(f"{arm} {kind}: request does not force the complete token count")
    if kind == "ship":
        expected_sampling = {
            "temperature": 1.0,
            "top_k": 64,
            "top_p": 0.95,
            "min_p": 0.0,
        }
    else:
        expected_sampling = {
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "post_sampling_probs": False,
        }
        n_probs = require_int(request.get("n_probs"), f"{arm}.{kind}.n_probs")
        if n_probs < 2:
            raise RuntimeError(f"{arm} {kind}: n_probs={n_probs} cannot characterize distribution drift")
    wrong_sampling = {
        key: (request.get(key), value)
        for key, value in expected_sampling.items()
        if request.get(key) != value
    }
    if wrong_sampling:
        raise RuntimeError(f"{arm} {kind}: sampling provenance mismatch: {wrong_sampling!r}")
    tokens = result.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise RuntimeError(f"{arm} {kind}: missing returned token IDs")
    n_predict = len(tokens)
    if request.get("n_predict") != n_predict:
        raise RuntimeError(f"{arm} {kind}: request/result token count mismatch")
    require_result_shape(
        result,
        label=f"{arm} {kind}",
        n_predict=n_predict,
        expected_cache_n=token_count - 1,
        expected_prompt_n=1,
        n_ctx=131072,
        require_probs=kind == "quality",
    )
    token_file = json.loads((out_dir / f"{arm}-{kind}.tokens.json").read_text(encoding="utf-8"))
    if token_file != tokens:
        raise RuntimeError(f"{arm} {kind}: token receipt differs from response")
    content = result.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"{arm} {kind}: content is not a string")
    if (out_dir / f"{arm}-{kind}.content.txt").read_text(encoding="utf-8") != content:
        raise RuntimeError(f"{arm} {kind}: content receipt differs from response")
    return result


def probability_row(row: object, label: str) -> tuple[int, float, dict[int, float], float]:
    if not isinstance(row, dict):
        raise RuntimeError(f"{label}: probability row is not an object")
    selected_id = require_int(row.get("id"), f"{label}.id")
    selected_logprob = require_number(row.get("logprob"), f"{label}.logprob")
    entries = row.get("top_logprobs")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"{label}: top_logprobs is empty")
    probabilities: dict[int, float] = {}
    selected_entry_logprob: float | None = None
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RuntimeError(f"{label}.top_logprobs[{index}]: expected object")
        token_id = require_int(entry.get("id"), f"{label}.top_logprobs[{index}].id")
        logprob = require_number(entry.get("logprob"), f"{label}.top_logprobs[{index}].logprob")
        if token_id in probabilities:
            raise RuntimeError(f"{label}: duplicate top-logprob token ID {token_id}")
        probabilities[token_id] = math.exp(logprob) if logprob > -745.0 else 0.0
        if token_id == selected_id:
            selected_entry_logprob = logprob
    if selected_entry_logprob is None:
        raise RuntimeError(f"{label}: selected token {selected_id} is absent from top_logprobs")
    if abs(selected_entry_logprob - selected_logprob) > 1e-7:
        raise RuntimeError(f"{label}: selected-token logprob disagrees with its top_logprobs entry")
    mass = sum(probabilities.values())
    if mass > 1.0001:
        raise RuntimeError(f"{label}: top-logprob probability mass is invalid ({mass})")
    return selected_id, selected_logprob, probabilities, max(0.0, 1.0 - mass)


def probability_distance(left: object, right: object, label: str) -> dict[str, float | int]:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right) or not left:
        raise RuntimeError(f"{label}: probability payload lengths are invalid")
    total_variations: list[float] = []
    js_divergences: list[float] = []
    selected_logprob_deltas: list[float] = []
    top_id_symmetric_differences: list[int] = []
    for index, (left_row, right_row) in enumerate(zip(left, right)):
        lid, llp, lprobs, ltail = probability_row(left_row, f"{label}.left[{index}]")
        rid, rlp, rprobs, rtail = probability_row(right_row, f"{label}.right[{index}]")
        if lid != rid:
            raise RuntimeError(f"{label}: selected token differs at row {index}: {lid} != {rid}")
        token_ids = set(lprobs) | set(rprobs)
        left_values = [lprobs.get(token_id, 0.0) for token_id in token_ids] + [ltail]
        right_values = [rprobs.get(token_id, 0.0) for token_id in token_ids] + [rtail]
        total_variations.append(0.5 * sum(abs(a - b) for a, b in zip(left_values, right_values)))
        js = 0.0
        for a, b in zip(left_values, right_values):
            midpoint = 0.5 * (a + b)
            if a > 0.0:
                js += 0.5 * a * math.log(a / midpoint)
            if b > 0.0:
                js += 0.5 * b * math.log(b / midpoint)
        js_divergences.append(js)
        selected_logprob_deltas.append(abs(llp - rlp))
        top_id_symmetric_differences.append(len(set(lprobs) ^ set(rprobs)))
    return {
        "rows": len(left),
        # Each endpoint returns only top-N entries. Missing probability mass is
        # grouped into one tail bucket, so this is explicitly a top-N drift
        # proxy, not a claim of exact full-vocabulary total variation.
        "mean_top_n_tv_proxy": sum(total_variations) / len(total_variations),
        "max_top_n_tv_proxy": max(total_variations),
        "mean_jensen_shannon": sum(js_divergences) / len(js_divergences),
        "max_jensen_shannon": max(js_divergences),
        "mean_selected_logprob_delta": sum(selected_logprob_deltas) / len(selected_logprob_deltas),
        "max_selected_logprob_delta": max(selected_logprob_deltas),
        "max_top_id_symmetric_difference": max(top_id_symmetric_differences),
    }


def compare_command(args: argparse.Namespace) -> None:
    arm_names = ("off-a", "on", "off-b")
    manifest = parse_manifest(args.out_dir / "manifest.txt")
    expected_manifest = {
        "ctx": str(args.n_ctx),
        "slots": "1",
        "token_count": str(args.token_count),
        "kv": "f16/f16",
        "swa_full": "1",
        "tokenize_add_special": "true",
        "tokenize_parse_special": "true",
        "arithmetic_expected": "1" if args.arithmetic_expected else "0",
    }
    wrong_manifest = {
        key: (manifest.get(key), value)
        for key, value in expected_manifest.items()
        if manifest.get(key) != value
    }
    if wrong_manifest:
        raise RuntimeError(f"manifest provenance mismatch: {wrong_manifest!r}")
    for hash_key in ("server_sha256", "sycl_sha256", "model_sha256", "fixture_tokens_sha256"):
        value = manifest.get(hash_key, "")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise RuntimeError(f"manifest has invalid {hash_key}: {value!r}")

    fixture_path = pathlib.Path(manifest.get("fixture", ""))
    fixture, fixture_tokens = load_fixture(fixture_path, args.token_count)
    fixture_token_sha = fixture["tokens_sha256"]
    fixture_file_sha = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    if fixture_file_sha != manifest.get("fixture_sha256"):
        raise RuntimeError("manifest and frozen fixture file hashes differ")
    if fixture_token_sha != manifest["fixture_tokens_sha256"]:
        raise RuntimeError("manifest and frozen fixture token hashes differ")
    fixture_checks = {
        "schema": 2,
        "model": manifest.get("model"),
        "model_sha256": manifest["model_sha256"],
        "tokenize_add_special": True,
        "tokenize_parse_special": True,
        "first_token_id": fixture_tokens[0],
        "last_token_id": fixture_tokens[-1],
    }
    wrong_fixture = {
        key: (fixture.get(key), value)
        for key, value in fixture_checks.items()
        if fixture.get(key) != value
    }
    if wrong_fixture:
        raise RuntimeError(f"fixture provenance mismatch: {wrong_fixture!r}")
    if manifest.get("fixture_first_token_id") != str(fixture_tokens[0]):
        raise RuntimeError("manifest and fixture first token IDs differ")
    if manifest.get("fixture_last_token_id") != str(fixture_tokens[-1]):
        raise RuntimeError("manifest and fixture last token IDs differ")

    summaries: dict[str, dict] = {}
    receipts: dict[str, dict[str, dict]] = {}
    for arm in arm_names:
        summary = read_json_object(args.out_dir / f"{arm}-summary.json")
        summaries[arm] = summary
        if summary.get("arm") != arm:
            raise RuntimeError(f"{arm}: summary arm label mismatch")
        if summary.get("fixture") != str(fixture_path):
            raise RuntimeError(f"{arm}: summary fixture path mismatch")
        if summary.get("fixture_tokens_sha256") != fixture_token_sha:
            raise RuntimeError(f"{arm}: summary fixture hash mismatch")
        if summary.get("token_count") != args.token_count:
            raise RuntimeError(f"{arm}: summary depth mismatch")
        log = (args.out_dir / f"{arm}-server.log").read_text(encoding="utf-8")
        if f"n_slots = 1, n_ctx_slot = {args.n_ctx}" not in log:
            raise RuntimeError(f"{arm}: server log does not prove one {args.n_ctx}-token slot")
        if "using full-size SWA cache" not in log:
            raise RuntimeError(f"{arm}: server log does not prove --swa-full")
        receipts[arm] = {
            kind: audit_completion_artifact(
                arm=arm,
                kind=kind,
                out_dir=args.out_dir,
                fixture_token_sha=fixture_token_sha,
                token_count=args.token_count,
            )
            for kind in ("prime", "ship", "quality")
        }
        for kind in ("ship", "quality"):
            tokens = receipts[arm][kind]["tokens"]
            token_sha = hashlib.sha256(canonical_json(tokens)).hexdigest()
            if summary.get(f"{kind}_tokens_sha256") != token_sha:
                raise RuntimeError(f"{arm}: {kind} summary token hash mismatch")
            reported_tps = require_number(
                summary.get(f"{kind}_predicted_per_second"), f"{arm}.{kind}_summary_tps"
            )
            actual_tps = require_number(
                receipts[arm][kind]["timings"].get("predicted_per_second"),
                f"{arm}.{kind}_result_tps",
            )
            if reported_tps != actual_tps:
                raise RuntimeError(f"{arm}: {kind} summary/result speed mismatch")

    for kind in ("prime", "ship", "quality"):
        reference_request = (args.out_dir / f"off-a-{kind}.request.json").read_bytes()
        for arm in ("on", "off-b"):
            if (args.out_dir / f"{arm}-{kind}.request.json").read_bytes() != reference_request:
                raise RuntimeError(f"{kind}: {arm} request differs from off-a")

    ship_diagnostics = {
        "off_a_vs_off_b_tokens_equal": receipts["off-a"]["ship"]["tokens"] == receipts["off-b"]["ship"]["tokens"],
        "off_a_vs_off_b_content_equal": receipts["off-a"]["ship"]["content"] == receipts["off-b"]["ship"]["content"],
        "on_vs_off_a_tokens_equal": receipts["on"]["ship"]["tokens"] == receipts["off-a"]["ship"]["tokens"],
        "on_vs_off_a_content_equal": receipts["on"]["ship"]["content"] == receipts["off-a"]["ship"]["content"],
        "on_vs_off_b_tokens_equal": receipts["on"]["ship"]["tokens"] == receipts["off-b"]["ship"]["tokens"],
        "on_vs_off_b_content_equal": receipts["on"]["ship"]["content"] == receipts["off-b"]["ship"]["content"],
        "hard_gate": False,
        "note": "official sampling is seeded but process-level numerical noise can change sampled paths",
    }

    greedy_checks = {
        "off_a_vs_off_b_tokens_equal": receipts["off-a"]["quality"]["tokens"] == receipts["off-b"]["quality"]["tokens"],
        "off_a_vs_off_b_content_equal": receipts["off-a"]["quality"]["content"] == receipts["off-b"]["quality"]["content"],
        "on_vs_off_a_tokens_equal": receipts["on"]["quality"]["tokens"] == receipts["off-a"]["quality"]["tokens"],
        "on_vs_off_a_content_equal": receipts["on"]["quality"]["content"] == receipts["off-a"]["quality"]["content"],
        "on_vs_off_b_tokens_equal": receipts["on"]["quality"]["tokens"] == receipts["off-b"]["quality"]["tokens"],
        "on_vs_off_b_content_equal": receipts["on"]["quality"]["content"] == receipts["off-b"]["quality"]["content"],
        "hard_gate": True,
    }

    probability_payloads = {
        arm: receipts[arm]["quality"]["completion_probabilities"] for arm in arm_names
    }
    probability_exact = {
        "off_a_vs_off_b": probability_payloads["off-a"] == probability_payloads["off-b"],
        "on_vs_off_a": probability_payloads["on"] == probability_payloads["off-a"],
        "on_vs_off_b": probability_payloads["on"] == probability_payloads["off-b"],
    }
    greedy_outputs_passed = all(
        value for key, value in greedy_checks.items() if key.endswith("_equal")
    )
    greedy_checks["passed"] = greedy_outputs_passed

    off_a_tps = require_number(summaries["off-a"].get("ship_predicted_per_second"), "off-a tps")
    off_b_tps = require_number(summaries["off-b"].get("ship_predicted_per_second"), "off-b tps")
    on_tps = require_number(summaries["on"].get("ship_predicted_per_second"), "on tps")
    if min(off_a_tps, off_b_tps, on_tps) <= 0.0:
        raise RuntimeError("all measured decode rates must be positive")
    off_mean = (off_a_tps + off_b_tps) / 2.0
    off_spread_pct = abs(off_a_tps - off_b_tps) / off_mean * 100.0
    on_delta_pct = (on_tps / off_mean - 1.0) * 100.0
    required_on_improvement_pct = max(
        args.min_on_improvement_pct,
        args.speed_noise_multiplier * off_spread_pct,
    )

    probability_gate_specs = {
        "mean_top_n_tv_proxy": (args.prob_mean_tv_floor, args.max_prob_mean_tv),
        "max_top_n_tv_proxy": (args.prob_max_tv_floor, args.max_prob_max_tv),
        "max_selected_logprob_delta": (
            args.prob_selected_logprob_floor,
            args.max_prob_selected_logprob_delta,
        ),
    }
    probability_distances: dict[str, object]
    probability_gate: dict[str, object] = {}
    probability_failures: list[str] = []
    if greedy_outputs_passed:
        probability_distances = {
            "comparable": True,
            "off_a_vs_off_b": probability_distance(
                probability_payloads["off-a"], probability_payloads["off-b"], "off-a vs off-b"
            ),
            "on_vs_off_a": probability_distance(
                probability_payloads["on"], probability_payloads["off-a"], "on vs off-a"
            ),
            "on_vs_off_b": probability_distance(
                probability_payloads["on"], probability_payloads["off-b"], "on vs off-b"
            ),
        }
        baseline_probability = probability_distances["off_a_vs_off_b"]
        candidate_a_probability = probability_distances["on_vs_off_a"]
        candidate_b_probability = probability_distances["on_vs_off_b"]
        if not isinstance(baseline_probability, dict) or not isinstance(candidate_a_probability, dict) or not isinstance(candidate_b_probability, dict):
            raise RuntimeError("internal probability comparison shape error")
        for metric, (floor, absolute_cap) in probability_gate_specs.items():
            baseline_value = float(baseline_probability[metric])
            candidate_worst = max(
                float(candidate_a_probability[metric]),
                float(candidate_b_probability[metric]),
            )
            noise_limit = max(floor, args.prob_noise_multiplier * baseline_value)
            effective_limit = min(absolute_cap, noise_limit)
            baseline_passed = baseline_value <= absolute_cap
            absolute_candidate_passed = candidate_worst <= absolute_cap
            relative_candidate_passed = candidate_worst <= effective_limit
            enforced_candidate_passed = absolute_candidate_passed and (
                relative_candidate_passed or not args.arithmetic_expected
            )
            probability_gate[metric] = {
                "baseline_off_a_vs_off_b": baseline_value,
                "candidate_worst_vs_either_off": candidate_worst,
                "noise_multiplier": args.prob_noise_multiplier,
                "noise_relative_limit": noise_limit,
                "absolute_limit": absolute_cap,
                "effective_arithmetic_limit": effective_limit,
                "baseline_passed": baseline_passed,
                "candidate_absolute_passed": absolute_candidate_passed,
                "candidate_relative_passed": relative_candidate_passed,
                "candidate_gate_enforced_relative_to_noise": args.arithmetic_expected,
                "passed": baseline_passed and enforced_candidate_passed,
            }
            if not baseline_passed:
                probability_failures.append(
                    f"OFF probability {metric}={baseline_value:.9g} exceeds absolute limit {absolute_cap:.9g}"
                )
            if not absolute_candidate_passed:
                probability_failures.append(
                    f"candidate probability {metric}={candidate_worst:.9g} exceeds absolute limit {absolute_cap:.9g}"
                )
            elif args.arithmetic_expected and not relative_candidate_passed:
                probability_failures.append(
                    f"candidate probability {metric}={candidate_worst:.9g} exceeds noise-derived limit {effective_limit:.9g}"
                )
        probability_gate["comparable"] = True
        probability_gate["passed"] = not probability_failures
    else:
        probability_distances = {
            "comparable": False,
            "reason": "greedy token/content paths differ; aligned per-position probability drift is invalid",
        }
        probability_gate = {
            "comparable": False,
            "passed": False,
            "reason": "greedy quality gate failed before probability comparison",
        }

    gate_failures: list[str] = []
    if off_spread_pct > args.max_off_spread_pct:
        gate_failures.append(
            f"OFF speed spread {off_spread_pct:.4f}% exceeds {args.max_off_spread_pct:.4f}%"
        )
    if on_delta_pct < required_on_improvement_pct:
        gate_failures.append(
            f"candidate speed improvement {on_delta_pct:.4f}% does not exceed the required "
            f"{required_on_improvement_pct:.4f}%"
        )
    for key, passed in greedy_checks.items():
        if key.endswith("_equal") and not passed:
            gate_failures.append(f"greedy quality mismatch: {key}")
    gate_failures.extend(probability_failures)

    comparison = {
        "gate_passed": not gate_failures,
        "gate_failures": gate_failures,
        "provenance_and_depth_gate": {
            "passed": True,
            "fixture_tokens_sha256": fixture_token_sha,
            "token_count": args.token_count,
            "n_ctx": args.n_ctx,
            "n_slots": 1,
            "kv": "f16/f16",
            "swa_full": True,
            "requests_identical_across_arms": True,
            "cache_n": args.token_count - 1,
            "prompt_n": 1,
            "truncated": False,
        },
        "arithmetic_expected": args.arithmetic_expected,
        "speed_gate": {
            "passed": (
                off_spread_pct <= args.max_off_spread_pct
                and on_delta_pct >= required_on_improvement_pct
            ),
            "off_a_ship_tps": off_a_tps,
            "off_b_ship_tps": off_b_tps,
            "off_mean_ship_tps": off_mean,
            "off_spread_pct": off_spread_pct,
            "max_off_spread_pct": args.max_off_spread_pct,
            "on_ship_tps": on_tps,
            "on_vs_off_mean_pct": on_delta_pct,
            "min_on_improvement_pct": args.min_on_improvement_pct,
            "speed_noise_multiplier": args.speed_noise_multiplier,
            "required_on_improvement_pct": required_on_improvement_pct,
        },
        "sampled_ship_diagnostics": ship_diagnostics,
        "greedy_quality_gate": greedy_checks,
        "quality_probability_payload_exact_diagnostic": probability_exact,
        "quality_probability_distances": probability_distances,
        "quality_probability_gate": probability_gate,
    }
    write_pretty(args.out_dir / "comparison.json", comparison)
    print(json.dumps(comparison, sort_keys=True))
    if gate_failures:
        raise RuntimeError("full-context comparison gate failed: " + "; ".join(gate_failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    fixture = subparsers.add_parser("fixture")
    fixture.add_argument("--base-url", required=True)
    fixture.add_argument("--source", type=pathlib.Path, required=True)
    fixture.add_argument("--model", type=pathlib.Path, required=True)
    fixture.add_argument("--model-sha256", required=True)
    fixture.add_argument("--fixture", type=pathlib.Path, required=True)
    fixture.add_argument("--out-dir", type=pathlib.Path, required=True)
    fixture.add_argument("--token-count", type=int, default=129024)
    fixture.add_argument("--timeout", type=float, default=600.0)
    fixture.add_argument("--force", action="store_true")
    fixture.set_defaults(function=fixture_command)

    request = subparsers.add_parser("request")
    request.add_argument("--base-url", required=True)
    request.add_argument("--arm", required=True)
    request.add_argument("--fixture", type=pathlib.Path, required=True)
    request.add_argument("--out-dir", type=pathlib.Path, required=True)
    request.add_argument("--token-count", type=int, default=129024)
    request.add_argument("--n-ctx", type=int, default=131072)
    request.add_argument("--n-predict", type=int, default=256)
    request.add_argument("--quality-n-predict", type=int, default=32)
    request.add_argument("--n-probs", type=int, default=128)
    request.add_argument("--seed", type=int, default=424242)
    request.add_argument("--timeout", type=float, default=7200.0)
    request.set_defaults(function=request_command)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--out-dir", type=pathlib.Path, required=True)
    compare.add_argument("--arithmetic-expected", action="store_true")
    compare.add_argument("--token-count", type=int, default=129024)
    compare.add_argument("--n-ctx", type=int, default=131072)
    compare.add_argument("--max-off-spread-pct", type=float, default=2.0)
    compare.add_argument("--min-on-improvement-pct", type=float, default=1.0)
    compare.add_argument("--speed-noise-multiplier", type=float, default=2.0)
    compare.add_argument("--prob-noise-multiplier", type=float, default=2.0)
    compare.add_argument("--prob-mean-tv-floor", type=float, default=0.002)
    compare.add_argument("--prob-max-tv-floor", type=float, default=0.01)
    compare.add_argument("--prob-selected-logprob-floor", type=float, default=0.02)
    compare.add_argument("--max-prob-mean-tv", type=float, default=0.02)
    compare.add_argument("--max-prob-max-tv", type=float, default=0.05)
    compare.add_argument("--max-prob-selected-logprob-delta", type=float, default=0.10)
    compare.set_defaults(function=compare_command)

    args = parser.parse_args()
    if hasattr(args, "token_count") and args.token_count <= 1:
        parser.error("--token-count must be greater than 1")
    for name in (
        "max_off_spread_pct",
        "min_on_improvement_pct",
        "speed_noise_multiplier",
        "prob_noise_multiplier",
        "prob_mean_tv_floor",
        "prob_max_tv_floor",
        "prob_selected_logprob_floor",
        "max_prob_mean_tv",
        "max_prob_max_tv",
        "max_prob_selected_logprob_delta",
    ):
        if hasattr(args, name) and (not math.isfinite(getattr(args, name)) or getattr(args, name) < 0.0):
            parser.error(f"--{name.replace('_', '-')} must be a finite non-negative number")
    return args


def main() -> None:
    args = parse_args()
    if args.command in {"fixture", "request"}:
        wait_ready(args.base_url, 300.0)
    args.function(args)


if __name__ == "__main__":
    main()
