from __future__ import annotations

import argparse
import json
import os
import shlex
from dataclasses import asdict
from pathlib import Path

import yaml

from .core import (
    EMPTY_SHA256,
    CheckpointIdentity,
    canonical_json,
    checkpoint_identity,
    compare_branch_identities,
    guard_estimator,
    sha256_bytes,
    tree_identity,
)


def boolean(value: bool) -> str:
    return "True" if value else "False"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record_pre_intervention_identity(config: dict, spec: dict) -> dict:
    output = Path(config["project"]["output_dir"])
    manifest = json.loads((output / "canonical_manifest.json").read_text(encoding="utf-8"))
    config_hash = str(spec["config_sha256"])
    source_hash = str(spec["source_manifest_sha256"])
    if config_hash != str(manifest["config_sha256"]):
        raise ValueError("run spec config identity does not match canonical_manifest.json")
    actual_source_hash = sha256_bytes(canonical_json(manifest["source"]).encode("utf-8"))
    if source_hash != actual_source_hash:
        raise ValueError("run spec source identity does not match canonical_manifest.json")
    prompt_window_hash = str(spec["planned_dose"]["prompt_batches_sha256"])
    global_step = int(spec.get("pre_intervention_step", spec.get("start_step", 0)))

    resume_from = spec.get("resume_from_path")
    if resume_from:
        identity, evidence = checkpoint_identity(
            Path(resume_from),
            global_step=global_step,
            rollout_seed=int(spec["rollout_seed"]),
            config_hash=config_hash,
            prompt_window_hash=prompt_window_hash,
            source_manifest_hash=source_hash,
        )
        kind = "resumed_checkpoint"
    else:
        actor = tree_identity(Path(config["paths"]["student_model"]))
        identity = CheckpointIdentity(
            global_step=0,
            actor_hash=actor["sha256"],
            optimizer_hash=EMPTY_SHA256,
            trainer_state_hash=EMPTY_SHA256,
            dataloader_hash=EMPTY_SHA256,
            rollout_seed=int(spec["rollout_seed"]),
            config_hash=config_hash,
            prompt_window_hash=prompt_window_hash,
            source_manifest_hash=source_hash,
        )
        identity.validate()
        evidence = {"student_model": actor}
        kind = "fresh_model"

    payload = {
        "schema_version": 1,
        "execution_id": spec.get("execution_id", spec["run_id"]),
        "identity_group": spec.get("identity_group"),
        "kind": kind,
        "identity": asdict(identity),
        "identity_sha256": identity.sha256,
        "evidence": evidence,
    }
    identities = output / "identities"
    group = spec.get("identity_group")
    if group:
        group_path = identities / "groups" / f"{group}.json"
        if group_path.is_file():
            locked = json.loads(group_path.read_text(encoding="utf-8"))
            compare_branch_identities(
                CheckpointIdentity(**locked["identity"]),
                identity,
            )
        else:
            _write_json(group_path, payload)
    path = identities / "runs" / f"{payload['execution_id']}.json"
    _write_json(path, payload)
    return {"path": str(path), "sha256": identity.sha256, "kind": kind}


def build_argv(config: dict, spec: dict) -> list[str]:
    training = config["training"]
    estimator = spec["estimator"]
    guard_estimator(estimator, training.get("forbidden_estimators", []))
    if estimator not in {training["opd_estimator"], training["rl_estimator"]}:
        raise ValueError(
            f"unsupported estimator {estimator!r}; the reference launcher accepts only "
            f"{training['opd_estimator']!r} and {training['rl_estimator']!r}"
        )
    is_opd = estimator == training["opd_estimator"]
    norm_adv_by_std = True if is_opd else bool(training.get("rl_norm_adv_by_std_in_grpo", True))
    loss_agg_mode = (
        training.get("opd_loss_agg_mode", "token-mean")
        if is_opd
        else training.get("rl_loss_agg_mode", "token-mean")
    )
    use_dynamic_bsz = bool(
        training.get("opd_use_dynamic_bsz", True)
        if is_opd
        else training.get("rl_use_dynamic_bsz", True)
    )
    rollout = config["rollout"]
    runtime = config["runtime"]
    verifier = config.get("verifier", {})
    output = Path(config["project"]["output_dir"])
    required_model_len = rollout["max_prompt_length"] + rollout["max_response_length"]
    max_model_len = int(runtime.get("max_model_len", required_model_len))
    if max_model_len < required_model_len:
        raise ValueError(
            f"runtime max_model_len={max_model_len} is below required context "
            f"{required_model_len}"
        )
    actor_max_token_len = int(
        runtime.get("actor_max_token_len_per_gpu", max_model_len)
    )
    teacher_max_token_len = int(
        runtime.get("teacher_forward_max_token_len_per_gpu", actor_max_token_len)
    )
    rollout_max_num_batched_tokens = int(
        runtime.get("rollout_max_num_batched_tokens", max_model_len)
    )
    topk_logprobs_with_chunking = bool(
        is_opd and runtime.get("topk_logprobs_with_chunking", False)
    )
    topk_logprobs_chunk_size = int(runtime.get("topk_logprobs_chunk_size", 4096))
    entropy_from_logits_with_chunking = bool(
        is_opd and runtime.get("entropy_from_logits_with_chunking", False)
    )
    save_freq = int(training.get("checkpoint_frequency", 10)) if spec["phase"].startswith("discovery") else 999999
    if spec["phase"].startswith("discovery") and spec["target_global_step"] % save_freq:
        raise ValueError("discovery endpoint must align with checkpoint_frequency")
    # All scientific evaluation runs through the independent JustRL pipeline.
    # The bundled verl validation path is known to underestimate this model pair.
    test_freq = -1
    val_file = output / ("branch_eval.parquet" if spec["phase"].startswith("discovery") else "final_test.parquet")
    overrides = [
        f"algorithm.adv_estimator={estimator}",
        f"algorithm.norm_adv_by_std_in_grpo={boolean(norm_adv_by_std)}",
        "data.shuffle=False",
        f"data.seed={spec['rollout_seed']}",
        f"data.train_files={spec['train_file']}",
        f"data.val_files={val_file}",
        f"data.train_batch_size={config['data']['prompt_batch_size']}",
        f"data.dataloader_num_workers={config['data']['dataloader_num_workers']}",
        f"data.max_prompt_length={rollout['max_prompt_length']}",
        f"data.max_response_length={rollout['max_response_length']}",
        "data.filter_overlong_prompts=False",
        "data.truncation=error",
        "data.return_raw_chat=True",
        f"+data.apply_chat_template_kwargs.enable_thinking={boolean(rollout['enable_thinking'])}",
        f"actor_rollout_ref.model.path={config['paths']['student_model']}",
        "+actor_rollout_ref.model.override_config.attn_implementation="
        f"{runtime['attention_implementation']}",
        f"actor_rollout_ref.model.use_fused_kernels={boolean(runtime['use_fused_kernels'])}",
        "actor_rollout_ref.model.fused_kernel_options.impl_backend="
        f"{runtime['fused_kernel_backend']}",
        f"actor_rollout_ref.model.use_remove_padding={boolean(runtime['use_remove_padding'])}",
        f"actor_rollout_ref.model.enable_activation_offload={boolean(runtime['enable_activation_offload'])}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={boolean(runtime['enable_gradient_checkpointing'])}",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size="
        f"{training.get('ppo_mini_batch_size', config['data']['prompt_batch_size'])}",
        f"actor_rollout_ref.actor.ppo_epochs={training.get('ppo_epochs', 1)}",
        f"actor_rollout_ref.actor.use_dynamic_bsz={boolean(use_dynamic_bsz)}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={runtime['actor_micro_batch_size_per_gpu']}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={actor_max_token_len}",
        f"actor_rollout_ref.actor.strategy={runtime.get('actor_strategy', 'fsdp')}",
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1",
        f"actor_rollout_ref.actor.topk_logprobs_with_chunking={boolean(topk_logprobs_with_chunking)}",
        f"actor_rollout_ref.actor.topk_logprobs_chunk_size={topk_logprobs_chunk_size}",
        "actor_rollout_ref.actor.entropy_from_logits_with_chunking="
        f"{boolean(entropy_from_logits_with_chunking)}",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.optim.weight_decay=0.01",
        "actor_rollout_ref.actor.optim.betas=[0.9,0.999]",
        "actor_rollout_ref.actor.grad_clip=1.0",
        "actor_rollout_ref.actor.clip_ratio=0.2",
        "actor_rollout_ref.actor.clip_ratio_low=0.2",
        "actor_rollout_ref.actor.clip_ratio_high=0.2",
        f"actor_rollout_ref.actor.loss_agg_mode={loss_agg_mode}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={boolean(runtime['actor_param_offload'])}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={boolean(runtime['optimizer_offload'])}",
        "actor_rollout_ref.actor.fsdp_config.offload_policy="
        f"{boolean(runtime.get('actor_offload_policy', False))}",
        f"actor_rollout_ref.actor.fsdp_config.model_dtype={runtime['model_dtype']}",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        f"actor_rollout_ref.ref.fsdp_config.model_dtype={runtime['model_dtype']}",
        "actor_rollout_ref.rollout.name=vllm",
        f"+actor_rollout_ref.rollout.seed={spec['rollout_seed']}",
        f"actor_rollout_ref.rollout.temperature={rollout['temperature']}",
        f"actor_rollout_ref.rollout.top_p={rollout['top_p']}",
        f"actor_rollout_ref.rollout.repetition_penalty={rollout['repetition_penalty']}",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="
        f"{boolean(use_dynamic_bsz)}",
        f"+actor_rollout_ref.rollout.log_prob_top_k={training['top_k'] if is_opd else 0}",
        f"+actor_rollout_ref.rollout.top_k_strategy={training['top_k_strategy']}",
        f"+actor_rollout_ref.rollout.reward_weight_mode={training['reward_weight_mode']}",
        "+actor_rollout_ref.rollout.teacher_temperature=1.0",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.enforce_eager={boolean(runtime['rollout_enforce_eager'])}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={runtime['rollout_gpu_memory_utilization']}",
        f"+actor_rollout_ref.rollout.engine_kwargs.vllm.cpu_offload_gb={runtime['rollout_cpu_offload_gb']}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={rollout_max_num_batched_tokens}",
        f"actor_rollout_ref.rollout.max_model_len={max_model_len}",
        f"actor_rollout_ref.rollout.n={rollout['n']}",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        f"+actor_rollout_ref.rollout.val_kwargs.max_tokens={rollout['eval_max_response_length']}",
        f"actor_rollout_ref.rollout.val_kwargs.n={rollout['n']}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={rollout['temperature']}",
        f"actor_rollout_ref.rollout.val_kwargs.top_p={rollout['top_p']}",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        f"reward_model.enable={boolean(is_opd)}",
        f"reward_model.model.path={config['paths']['teacher_model']}",
        "+reward_model.model.attn_implementation="
        f"{runtime['attention_implementation']}",
        f"reward_model.model.use_fused_kernels={boolean(runtime['use_fused_kernels'])}",
        "reward_model.model.fused_kernel_options.impl_backend="
        f"{runtime['fused_kernel_backend']}",
        "reward_model.model.input_tokenizer=null",
        f"reward_model.model.use_remove_padding={boolean(runtime['use_remove_padding'])}",
        f"reward_model.model.fsdp_config.param_offload={boolean(runtime['teacher_param_offload'])}",
        f"+reward_model.model.dtype={runtime['model_dtype']}",
        f"reward_model.micro_batch_size_per_gpu={runtime['teacher_micro_batch_size_per_gpu']}",
        f"reward_model.forward_max_token_len_per_gpu={teacher_max_token_len}",
        f"reward_model.compute_entropy={boolean(runtime['teacher_compute_entropy'])}",
        "custom_reward_function.path="
        f"{Path(config['paths']['verl_root']) / verifier.get('train_relative_path', 'verl/utils/reward_score/ttrl_math/__init__.py')}",
        f"custom_reward_function.name={verifier.get('train_function', 'reward_func')}",
        "trainer.val_before_train=False",
        "trainer.balance_batch=False",
        "trainer.logger=[\"console\"]",
        f"trainer.project_name={config['project']['name']}",
        f"trainer.experiment_name={spec['run_id']}",
        f"trainer.n_gpus_per_node={runtime['n_gpus_per_node']}",
        "trainer.nnodes=1",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={test_freq}",
        "trainer.total_epochs=1000",
        f"trainer.total_training_steps={spec['target_global_step']}",
        f"trainer.resume_mode={spec['resume_mode']}",
        f"trainer.default_local_dir={spec['default_local_dir']}",
        f"trainer.validation_data_dir={output.parent / 'validation' / spec['run_id']}",
        f"trainer.rollout_data_dir={output.parent / 'rollouts' / spec['run_id']}",
    ]
    if spec.get("resume_from_path"):
        overrides.append(f"trainer.resume_from_path={spec['resume_from_path']}")
    if not use_dynamic_bsz:
        overrides.append(
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="
            f"{runtime['actor_micro_batch_size_per_gpu']}"
        )
    return ["python", "-m", "verl.trainer.main_ppo", *overrides]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(config["project"]["output_dir"])
    execution_specs = output / "execution_specs.json"
    specs = json.loads(
        (execution_specs if execution_specs.is_file() else output / "run_specs.json").read_text(
            encoding="utf-8"
        )
    )
    benchmark_specs = output / "benchmark_specs.json"
    if benchmark_specs.is_file():
        specs.extend(json.loads(benchmark_specs.read_text(encoding="utf-8")))
    spec = next(
        (
            item
            for item in specs
            if item.get("execution_id", item["run_id"]) == args.run_id
        ),
        None,
    )
    if spec is None:
        raise SystemExit(f"unknown run id: {args.run_id}")
    pre_intervention = record_pre_intervention_identity(config, spec)
    argv = build_argv(config, spec)
    command_dir = Path(config["project"]["output_dir"]) / "commands"
    command_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "execution_id": args.run_id,
        "scientific_run_id": spec.get("scientific_run_id", spec["run_id"]),
        "pre_intervention_identity": pre_intervention,
        "argv": argv,
        "shell": shlex.join(argv),
    }
    (command_dir / f"{args.run_id}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(payload["shell"])
    if not args.dry_run:
        if os.environ.get("HANDOFF_GPU_AUTHORIZED") != "1":
            raise SystemExit("GPU execution requires HANDOFF_GPU_AUTHORIZED=1")
        os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
