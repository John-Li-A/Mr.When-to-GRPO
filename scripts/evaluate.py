from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer

from when_to_grpo.core import normalize_prompt, sha256_file


def protocol(config: dict, n: int) -> dict:
    evaluation = config["evaluation"]
    max_tokens = int(evaluation["max_response_length"])
    if max_tokens != int(config["rollout"]["max_response_length"]):
        raise ValueError("evaluation and training response caps must match")
    return {
        "n": int(n),
        "temperature": float(evaluation["temperature"]),
        "top_p": float(evaluation["top_p"]),
        "top_k": int(evaluation["top_k"]),
        "repetition_penalty": float(evaluation["repetition_penalty"]),
        "max_tokens": max_tokens,
        "max_prompt_length": int(config["rollout"]["max_prompt_length"]),
        "max_model_len": int(config["rollout"]["max_prompt_length"]) + max_tokens,
        "seed": int(evaluation["seed"]),
        "enable_thinking": bool(config["rollout"]["enable_thinking"]),
        "verifier": str(config.get("verifier", {}).get("label", "configured-verifier")),
    }


def load_panel(path: Path, config: dict) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"prompt", "reward_model"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"evaluation panel is missing columns: {missing}")
    for index, prompt in enumerate(frame["prompt"]):
        messages = normalize_prompt(prompt)
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError(f"eval row {index} is not one raw user message")
        content = messages[0]["content"]
        for token in config.get("evaluation", {}).get("required_prompt_substrings", []):
            if content.count(str(token)) != 1:
                raise ValueError(f"eval row {index} must contain exactly one {token!r}")
    return frame


def render_prompts(frame: pd.DataFrame, tokenizer, config: dict) -> tuple[list[str], list[int]]:
    rendered = [
        tokenizer.apply_chat_template(
            normalize_prompt(prompt),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config["rollout"]["enable_thinking"],
        )
        for prompt in frame["prompt"]
    ]
    lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in rendered]
    cap = int(config["rollout"]["max_prompt_length"])
    overlong = [index for index, length in enumerate(lengths) if length > cap]
    if overlong:
        raise ValueError(f"evaluation prompts exceed max_prompt_length at rows: {overlong[:20]}")
    return rendered, lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frame = load_panel(args.dataset, config)
    tokenizer_path = args.model_path if args.model_path.is_dir() else Path(config["paths"]["student_model"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    rendered, prompt_lengths = render_prompts(frame, tokenizer, config)
    locked = protocol(config, args.n)
    manifest = {
        **locked,
        "panel": args.panel,
        "dataset": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "rows": len(frame),
        "model_path": str(args.model_path),
        "max_rendered_prompt_tokens": max(prompt_lengths),
        "metric": f"avg_at_{args.n}",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.execute:
        print(json.dumps(manifest, indent=2))
        return
    if os.environ.get("HANDOFF_GPU_AUTHORIZED") != "1":
        raise SystemExit("GPU execution requires HANDOFF_GPU_AUTHORIZED=1")
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)

    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    verifier = config.get("verifier", {})
    verifier_module = importlib.import_module(str(verifier.get("eval_module", "verl.utils.reward_score.ttrl_math")))
    compute_score = getattr(verifier_module, str(verifier.get("eval_function", "compute_score")))

    stop_ids = []
    for token in ("<|im_end|>", "<|endoftext|>"):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) == 1:
            stop_ids.append(encoded[0])
    engine = LLM(
        model=str(args.model_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        trust_remote_code=False,
        enforce_eager=bool(config["evaluation"]["enforce_eager"]),
        gpu_memory_utilization=float(config["evaluation"]["gpu_memory_utilization"]),
        max_model_len=locked["max_model_len"],
        max_num_batched_tokens=int(config["evaluation"]["max_num_batched_tokens"]),
        seed=locked["seed"],
    )
    sampling = SamplingParams(
        n=locked["n"],
        temperature=locked["temperature"],
        top_p=locked["top_p"],
        top_k=locked["top_k"],
        repetition_penalty=locked["repetition_penalty"],
        max_tokens=locked["max_tokens"],
        stop_token_ids=stop_ids or None,
        detokenize=True,
        skip_special_tokens=True,
    )
    start = time.perf_counter()
    outputs = engine.generate(rendered, sampling, use_tqdm=True)
    generation_seconds = time.perf_counter() - start
    records = []
    prompt_scores = []
    for request, row in zip(outputs, frame.to_dict(orient="records"), strict=True):
        truth = str(row["reward_model"]["ground_truth"])
        scores = []
        if len(request.outputs) != locked["n"]:
            raise ValueError("vLLM returned the wrong number of samples")
        for sample_index, sample in enumerate(request.outputs):
            raw_score = compute_score(sample.text, truth)
            if isinstance(raw_score, dict):
                score = raw_score
                task_score = float(score[verifier.get("score_field", "score")])
            else:
                task_score = float(raw_score)
                score = {"score": task_score}
            scores.append(task_score)
            records.append(
                {
                    "id": row.get("id"),
                    "sample_index": sample_index,
                    "ground_truth": truth,
                    "token_count": len(sample.token_ids),
                    "finish_reason": sample.finish_reason,
                    "score": score,
                    "output": sample.text,
                }
            )
        prompt_scores.append(scores)
    temp_path = args.output_dir / "trajectories.jsonl.tmp"
    temp_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n", encoding="utf-8")
    temp_path.replace(args.output_dir / "trajectories.jsonl")
    flat_scores = [score for group in prompt_scores for score in group]
    token_counts = [item["token_count"] for item in records]
    summary = {
        **manifest,
        "trajectory_count": len(records),
        "generation_seconds": generation_seconds,
        "avg_at_n": sum(flat_scores) / len(flat_scores),
        "pass_at_n": sum(any(group) for group in prompt_scores) / len(prompt_scores),
        "all_fail_rate": sum(not any(group) for group in prompt_scores) / len(prompt_scores),
        "all_correct_rate": sum(all(group) for group in prompt_scores) / len(prompt_scores),
        "nondegenerate_group_rate": sum(any(group) and not all(group) for group in prompt_scores) / len(prompt_scores),
        "mean_tokens": sum(token_counts) / len(token_counts),
        "cap_hit_rate": sum(count >= locked["max_tokens"] for count in token_counts) / len(token_counts),
    }
    format_field = verifier.get("format_field", "format_score")
    if all(format_field in item["score"] for item in records):
        summary["format_rate"] = sum(item["score"][format_field] > 0 for item in records) / len(records)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    del engine
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
