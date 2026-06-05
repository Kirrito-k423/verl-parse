#!/usr/bin/env python3
"""Generate a small battery of synthetic VERL training logs in testFold/.

Each generated .log file mirrors the structure of test2.log: a printed Hydra
config dump at the top followed by `step:N - key:value - ...` lines. Different
fixtures tweak the parallelism (TP/PP/CP/EP for actor and rollout), batch
sizes, sequence lengths, and rollout optimization flags so that the parser
has something interesting to chew on.

The header is written as a *well-formed* Python dict literal split across many
lines, so `ast.literal_eval` can recover it verbatim.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


PREFIX = "[36m(TaskRunner pid=980072)[0m "


@dataclass
class Fixture:
    """A single synthetic log: name (relative to testFold), config, runtime."""

    rel_path: str
    n_steps: int
    base_seed: int
    # Parallel settings ------------------------------------------------------
    actor_tp: int = 2
    actor_pp: int = 1
    actor_cp: int = 1
    actor_ep: int = 8
    actor_etp: int = 1
    actor_sp: bool = True
    actor_use_remove_padding: bool = False
    actor_grad_offload: bool = True
    actor_optim_offload: bool = True
    # Rollout ----------------------------------------------------------------
    rollout_tp: int = 2
    rollout_dp: int = 1
    rollout_expert_parallel_size: int = 1
    rollout_prompt_length: int = 1024
    rollout_response_length: int = 16384
    rollout_max_num_batched_tokens: int = 8192
    rollout_max_num_seqs: int = 1024
    rollout_gpu_memory_utilization: float = 0.5
    rollout_enable_chunked_prefill: bool = True
    rollout_enable_prefix_caching: bool = True
    rollout_enforce_eager: bool = False
    rollout_use_torch_compile: bool = True
    rollout_n: int = 5
    rollout_dtype: str = "bfloat16"
    # Data -------------------------------------------------------------------
    train_batch_size: int = 64
    ppo_mini_batch_size: int = 32
    ppo_micro_batch_size_per_gpu: int = 1
    ppo_epochs: int = 1
    ppo_max_token_len_per_gpu: int = 34816
    # Optim / loss -----------------------------------------------------------
    clip_ratio: float = 0.2
    kl_loss_coef: float = 0.01
    entropy_coeff: float = 0.0
    lr: float = 1e-06
    use_kl_in_reward: bool = False
    adv_estimator: str = "grpo"
    # Perf knobs controlling per-step timings (multiplicative, baseline=1.0) -
    gen_factor: float = 1.0
    ref_factor: float = 1.0
    old_logp_factor: float = 1.0
    update_actor_factor: float = 1.0
    update_weights_factor: float = 1.0
    throughput_scale: float = 1.0
    # Wall-clock noise -------------------------------------------------------
    noise: float = 0.03
    notes: str = ""


def header_lines(cfg: Fixture) -> List[str]:
    """Build a Hydra-style config dump describing this fixture.

    The output is a sequence of lines that, when concatenated, form a
    syntactically valid Python dict literal. We keep the inner lines inside
    their natural containers and use ``,`` separators at the right places.
    """

    def b(v: bool) -> str:
        return "True" if v else "False"

    sp = b(cfg.actor_sp)
    urp = b(cfg.actor_use_remove_padding)
    return [
        f"{PREFIX}{{",
        f"{PREFIX}  'actor_rollout_ref': {{",
        f"{PREFIX}    'actor': {{",
        f"{PREFIX}      '_target_': 'verl.workers.config.McoreActorConfig',",
        f"{PREFIX}      'clip_ratio': {cfg.clip_ratio},",
        f"{PREFIX}      'clip_ratio_c': 3.0,",
        f"{PREFIX}      'clip_ratio_high': {cfg.clip_ratio},",
        f"{PREFIX}      'clip_ratio_low': {cfg.clip_ratio},",
        f"{PREFIX}      'entropy_coeff': {cfg.entropy_coeff},",
        f"{PREFIX}      'kl_loss_coef': {cfg.kl_loss_coef},",
        f"{PREFIX}      'ppo_epochs': {cfg.ppo_epochs},",
        f"{PREFIX}      'ppo_max_token_len_per_gpu': {cfg.ppo_max_token_len_per_gpu},",
        f"{PREFIX}      'ppo_micro_batch_size': None,",
        f"{PREFIX}      'ppo_micro_batch_size_per_gpu': {cfg.ppo_micro_batch_size_per_gpu},",
        f"{PREFIX}      'ppo_mini_batch_size': {cfg.ppo_mini_batch_size},",
        f"{PREFIX}      'megatron': {{",
        f"{PREFIX}        '_target_': 'verl.workers.config.McoreEngineConfig',",
        f"{PREFIX}        'context_parallel_size': {cfg.actor_cp},",
        f"{PREFIX}        'dtype': 'bfloat16',",
        f"{PREFIX}        'expert_model_parallel_size': {cfg.actor_ep},",
        f"{PREFIX}        'expert_tensor_parallel_size': {cfg.actor_etp},",
        f"{PREFIX}        'grad_offload': {b(cfg.actor_grad_offload)},",
        f"{PREFIX}        'optimizer_offload': {b(cfg.actor_optim_offload)},",
        f"{PREFIX}        'pipeline_model_parallel_size': {cfg.actor_pp},",
        f"{PREFIX}        'sequence_parallel': {sp},",
        f"{PREFIX}        'tensor_model_parallel_size': {cfg.actor_tp},",
        f"{PREFIX}        'use_distributed_optimizer': True,",
        f"{PREFIX}        'use_remove_padding': {urp},",
        f"{PREFIX}        'virtual_pipeline_model_parallel_size': None",
        f"{PREFIX}      }},",
        f"{PREFIX}      'optim': {{",
        f"{PREFIX}        '_target_': 'verl.workers.config.McoreOptimizerConfig',",
        f"{PREFIX}        'lr': {cfg.lr},",
        f"{PREFIX}        'lr_decay_style': 'constant',",
        f"{PREFIX}        'optimizer': 'adam',",
        f"{PREFIX}        'weight_decay': 0.01",
        f"{PREFIX}      }}",
        f"{PREFIX}    }},",
        f"{PREFIX}    'rollout': {{",
        f"{PREFIX}      '_target_': 'verl.workers.config.RolloutConfig',",
        f"{PREFIX}      'agent': {{'default_agent_loop': 'single_turn_agent', 'num_workers': 8}},",
        f"{PREFIX}      'calculate_log_probs': True,",
        f"{PREFIX}      'data_parallel_size': {cfg.rollout_dp},",
        f"{PREFIX}      'dtype': '{cfg.rollout_dtype}',",
        f"{PREFIX}      'enable_chunked_prefill': {b(cfg.rollout_enable_chunked_prefill)},",
        f"{PREFIX}      'enable_prefix_caching': {b(cfg.rollout_enable_prefix_caching)},",
        f"{PREFIX}      'enforce_eager': {b(cfg.rollout_enforce_eager)},",
        f"{PREFIX}      'expert_parallel_size': {cfg.rollout_expert_parallel_size},",
        f"{PREFIX}      'gpu_memory_utilization': {cfg.rollout_gpu_memory_utilization},",
        f"{PREFIX}      'max_num_batched_tokens': {cfg.rollout_max_num_batched_tokens},",
        f"{PREFIX}      'max_num_seqs': {cfg.rollout_max_num_seqs},",
        f"{PREFIX}      'name': 'vllm',",
        f"{PREFIX}      'prompt_length': {cfg.rollout_prompt_length},",
        f"{PREFIX}      'response_length': {cfg.rollout_response_length},",
        f"{PREFIX}      'rollout_n': {cfg.rollout_n},",
        f"{PREFIX}      'tensor_model_parallel_size': {cfg.rollout_tp},",
        f"{PREFIX}      'use_torch_compile': {b(cfg.rollout_use_torch_compile)}",
        f"{PREFIX}    }},",
        f"{PREFIX}    'ref': {{",
        f"{PREFIX}      'megatron': {{",
        f"{PREFIX}        'tensor_model_parallel_size': {cfg.actor_tp},",
        f"{PREFIX}        'pipeline_model_parallel_size': {cfg.actor_pp},",
        f"{PREFIX}        'context_parallel_size': {cfg.actor_cp},",
        f"{PREFIX}        'sequence_parallel': {sp},",
        f"{PREFIX}        'use_remove_padding': {urp},",
        f"{PREFIX}        'dtype': 'bfloat16'",
        f"{PREFIX}      }}",
        f"{PREFIX}    }}",
        f"{PREFIX}  }},",
        f"{PREFIX}  'algorithm': {{",
        f"{PREFIX}    'adv_estimator': '{cfg.adv_estimator}',",
        f"{PREFIX}    'kl_penalty': 'kl',",
        f"{PREFIX}    'kl_ctrl': {{'kl_coef': 0.001, 'type': 'fixed'}},",
        f"{PREFIX}    'use_kl_in_reward': {b(cfg.use_kl_in_reward)}",
        f"{PREFIX}  }},",
        f"{PREFIX}  'data': {{",
        f"{PREFIX}    'train_batch_size': {cfg.train_batch_size},",
        f"{PREFIX}    'val_batch_size': None",
        f"{PREFIX}  }},",
        f"{PREFIX}  'model_engine': 'megatron',",
        f"{PREFIX}  'trainer': {{",
        f"{PREFIX}    'balance_batch': True,",
        f"{PREFIX}    'total_epochs': 30",
        f"{PREFIX}  }}",
        f"{PREFIX}}}",
        f"{PREFIX} [validate_config] All configuration checks passed successfully!",
    ]


def step_metrics(cfg: Fixture, step: int, rng: random.Random) -> Dict[str, float]:
    warmup = 1.35 if step == 1 else 1.0
    if step <= 3:
        warmup *= 1.05
    drift = 1.0 - 0.005 * (step - 1)
    drift = max(drift, 0.85)
    def jitter() -> float:
        return 1.0 + rng.uniform(-cfg.noise, cfg.noise)

    gen_base = 78.0 * (cfg.rollout_response_length / 2048.0) ** 0.5
    old_logp_base = 26.0 * (cfg.rollout_response_length / 2048.0) ** 0.4
    ref_base = 15.0
    update_actor_base = 68.0 * (1.0 / max(1, cfg.actor_tp / 2.0))
    update_weights_base = 24.0

    gen = gen_base * cfg.gen_factor * warmup * drift * jitter()
    old_logp = old_logp_base * cfg.old_logp_factor * warmup * drift * jitter()
    ref = ref_base * cfg.ref_factor * warmup * drift * jitter()
    update_actor = update_actor_base * cfg.update_actor_factor * warmup * drift * jitter()
    update_weights = update_weights_base * cfg.update_weights_factor * drift * jitter()
    reward = 1e-05 * jitter()
    adv = 0.01 * jitter()
    start = 0.0003 * jitter()
    stop = 0.00008 * jitter()
    step_time = gen + old_logp + ref + update_actor + update_weights + adv + reward + start + stop

    total_tokens = cfg.train_batch_size * cfg.rollout_n * (cfg.rollout_response_length + cfg.rollout_prompt_length)
    throughput = (total_tokens / step_time) * cfg.throughput_scale

    prompt_len = max(
        int(cfg.rollout_prompt_length * 0.6),
        min(int(cfg.rollout_prompt_length * 1.2), int(rng.gauss(cfg.rollout_prompt_length, 50))),
    )
    response_len = cfg.rollout_response_length
    if not cfg.rollout_enable_prefix_caching and step <= 3:
        response_len = int(response_len * 0.95)
    response_len = max(64, response_len)

    mfu_actor = max(0.0, min(0.6, 0.08 + 0.0005 * step + rng.uniform(-0.01, 0.01)))
    mfu_actor_infer = max(0.0, min(0.4, 0.05 + 0.0004 * step + rng.uniform(-0.005, 0.005)))

    return {
        "step": float(step),
        "global_seqlen/min": float(int(total_tokens * 0.5)),
        "global_seqlen/max": float(int(total_tokens * 0.51)),
        "global_seqlen/minmax_diff": float(int(total_tokens * 0.01)),
        "global_seqlen/balanced_min": float(int(total_tokens * 0.5)),
        "global_seqlen/balanced_max": float(int(total_tokens * 0.5)),
        "global_seqlen/mean": float(total_tokens),
        "actor/entropy": 0.5 + rng.uniform(-0.05, 0.05),
        "perf/mfu/actor_infer": mfu_actor_infer,
        "training/rollout_probs_diff_valid": 1.0,
        "training/rollout_probs_diff_max": 0.5 + rng.uniform(-0.05, 0.05),
        "training/rollout_probs_diff_mean": max(0.0, 0.008 + rng.uniform(-0.002, 0.002)),
        "training/rollout_probs_diff_std": 0.016 + rng.uniform(-0.002, 0.002),
        "training/rollout_actor_probs_pearson_corr": 0.995 + rng.uniform(-0.005, 0.005),
        "actor/kl_loss": max(0.0, 0.002 + rng.uniform(-0.0005, 0.0005)),
        "actor/kl_coef": cfg.kl_loss_coef,
        "actor/pg_clipfrac": 0.0,
        "actor/ppo_kl": 0.0,
        "actor/pg_clipfrac_lower": 0.0,
        "actor/pg_loss": 0.01 + rng.uniform(-0.005, 0.005),
        "actor/grad_norm": max(0.0, 0.6 + rng.uniform(-0.2, 0.2)),
        "perf/mfu/actor": mfu_actor,
        "perf/max_memory_allocated_gb": 30.0 + rng.uniform(-2.0, 2.0),
        "perf/max_memory_reserved_gb": 40.0 + rng.uniform(-1.0, 1.0),
        "perf/cpu_memory_used_gb": 1500.0 + rng.uniform(-100.0, 100.0),
        "actor/lr": cfg.lr,
        "training/global_step": float(step),
        "training/epoch": 0.0,
        "critic/score/mean": 0.4 + rng.uniform(-0.05, 0.05),
        "critic/score/max": 0.9,
        "critic/score/min": 0.0,
        "critic/rewards/mean": 0.4 + rng.uniform(-0.05, 0.05),
        "critic/rewards/max": 0.9,
        "critic/rewards/min": 0.0,
        "critic/advantages/mean": rng.uniform(-0.2, 0.2),
        "critic/advantages/max": 2.4,
        "critic/advantages/min": -2.4,
        "critic/returns/mean": rng.uniform(-0.2, 0.2),
        "critic/returns/max": 2.4,
        "critic/returns/min": -2.4,
        "response_length/mean": float(response_len),
        "response_length/max": float(response_len),
        "response_length/min": float(response_len) * 0.4,
        "response_length/clip_ratio": 1.0 if response_len >= cfg.rollout_response_length else 0.0,
        "response_length_non_aborted/mean": float(response_len),
        "response_length_non_aborted/max": float(response_len),
        "response_length_non_aborted/min": float(response_len) * 0.4,
        "response_length_non_aborted/clip_ratio": 1.0 if response_len >= cfg.rollout_response_length else 0.0,
        "response/aborted_ratio": 0.0,
        "prompt_length/mean": float(prompt_len),
        "prompt_length/max": float(prompt_len) * 1.2,
        "prompt_length/min": float(prompt_len) * 0.5,
        "prompt_length/clip_ratio": 0.0,
        "num_turns/min": 2.0,
        "num_turns/max": 2.0,
        "num_turns/mean": 2.0,
        "timing_s/start_profile": start,
        "timing_s/agent_loop/generate_sequences/min": gen * 0.95,
        "timing_s/agent_loop/generate_sequences/max": gen * 1.05,
        "timing_s/agent_loop/generate_sequences/mean": gen,
        "timing_s/agent_loop/tool_calls/min": 0.0,
        "timing_s/agent_loop/tool_calls/max": 0.0,
        "timing_s/agent_loop/tool_calls/mean": 0.0,
        "timing_s/gen": gen,
        "timing_s/reward": reward,
        "timing_s/old_log_prob": old_logp,
        "timing_s/ref": ref,
        "timing_s/adv": adv,
        "timing_s/update_actor": update_actor,
        "timing_s/update_weights": update_weights,
        "timing_s/step": step_time,
        "timing_s/stop_profile": stop,
        "timing_per_token_ms/ref": ref * 1000.0 / max(1.0, total_tokens),
        "timing_per_token_ms/gen": gen * 1000.0 / max(1.0, total_tokens),
        "timing_per_token_ms/adv": adv * 1000.0 / max(1.0, total_tokens),
        "timing_per_token_ms/update_actor": update_actor * 1000.0 / max(1.0, total_tokens),
        "perf/total_num_tokens": float(total_tokens),
        "perf/time_per_step": step_time,
        "perf/throughput": throughput,
    }


def step_line(metrics: Dict[str, float]) -> str:
    parts = [f"step:{int(metrics['step'])}"]
    for k, v in metrics.items():
        if k == "step":
            continue
        if isinstance(v, float):
            parts.append(f"{k}:{v!r}")
        else:
            parts.append(f"{k}:{v}")
    return PREFIX + " - ".join(parts)


def build_log(cfg: Fixture) -> str:
    rng = random.Random(cfg.base_seed)
    lines: List[str] = [" TaskRunner hostname: node-29-131, PID: 980072"]
    lines.extend(header_lines(cfg))
    for step in range(1, cfg.n_steps + 1):
        lines.append(step_line(step_metrics(cfg, step, rng)))
    return "\n".join(lines) + "\n"


def fixtures() -> List[Fixture]:
    return [
        Fixture(
            rel_path="A3/baseline.log",
            n_steps=32,
            base_seed=11,
            notes="A3 baseline: 32 steps, default parallel settings.",
        ),
        Fixture(
            rel_path="A3/tp4_pp2.log",
            n_steps=30,
            base_seed=12,
            actor_tp=4,
            actor_pp=2,
            actor_ep=4,
            actor_use_remove_padding=True,
            rollout_tp=4,
            rollout_max_num_batched_tokens=16384,
            rollout_max_num_seqs=2048,
            ppo_mini_batch_size=64,
            ppo_micro_batch_size_per_gpu=2,
            throughput_scale=1.18,
            gen_factor=0.85,
            update_actor_factor=0.80,
            update_weights_factor=0.85,
            notes="A3 higher TP/PP, larger batch, remove padding on, throughput +18%.",
        ),
        Fixture(
            rel_path="A5/06-03/bindcore.log",
            n_steps=30,
            base_seed=21,
            actor_tp=2,
            actor_pp=1,
            actor_ep=8,
            rollout_tp=2,
            rollout_max_num_batched_tokens=12288,
            rollout_max_num_seqs=1536,
            rollout_gpu_memory_utilization=0.55,
            ppo_mini_batch_size=32,
            ppo_micro_batch_size_per_gpu=1,
            throughput_scale=1.06,
            gen_factor=0.96,
            old_logp_factor=0.97,
            update_actor_factor=0.98,
            update_weights_factor=0.92,
            noise=0.025,
            notes="A5 bindcore pinning: gen -4%, update_weights -8%.",
        ),
        Fixture(
            rel_path="A5/06-03/bindcore+prefetch.log",
            n_steps=32,
            base_seed=22,
            actor_tp=2,
            actor_pp=1,
            actor_ep=8,
            rollout_tp=2,
            rollout_max_num_batched_tokens=16384,
            rollout_max_num_seqs=2048,
            rollout_gpu_memory_utilization=0.55,
            rollout_enable_chunked_prefill=True,
            rollout_enable_prefix_caching=True,
            ppo_mini_batch_size=48,
            ppo_micro_batch_size_per_gpu=1,
            throughput_scale=1.15,
            gen_factor=0.88,
            old_logp_factor=0.92,
            ref_factor=0.95,
            update_actor_factor=0.95,
            update_weights_factor=0.88,
            noise=0.02,
            notes="A5 bindcore + prefetch: throughput +15% over bindcore alone.",
        ),
        Fixture(
            rel_path="A5/06.04/chunked_prefill.log",
            n_steps=36,
            base_seed=31,
            rollout_response_length=8192,
            rollout_max_num_batched_tokens=4096,
            rollout_enable_chunked_prefill=True,
            rollout_enable_prefix_caching=False,
            rollout_enforce_eager=True,
            rollout_use_torch_compile=False,
            gen_factor=1.15,
            throughput_scale=0.88,
            noise=0.04,
            notes="Chunked prefill on, no prefix cache: gen higher, throughput -12%.",
        ),
        Fixture(
            rel_path="A5/06.04/prefix_caching.log",
            n_steps=36,
            base_seed=32,
            rollout_response_length=8192,
            rollout_max_num_batched_tokens=8192,
            rollout_enable_chunked_prefill=True,
            rollout_enable_prefix_caching=True,
            rollout_enforce_eager=False,
            rollout_use_torch_compile=True,
            gen_factor=0.78,
            old_logp_factor=0.85,
            throughput_scale=1.22,
            noise=0.02,
            notes="Prefix cache on, eager off, compile on: gen -22%, throughput +22%.",
        ),
        Fixture(
            rel_path="A5/06.04/cudagraph.log",
            n_steps=30,
            base_seed=33,
            actor_tp=4,
            actor_pp=1,
            actor_ep=4,
            actor_use_remove_padding=True,
            rollout_tp=4,
            rollout_enforce_eager=False,
            rollout_use_torch_compile=True,
            rollout_max_num_batched_tokens=16384,
            rollout_max_num_seqs=2048,
            ppo_mini_batch_size=64,
            ppo_micro_batch_size_per_gpu=2,
            gen_factor=0.65,
            old_logp_factor=0.7,
            update_actor_factor=0.78,
            update_weights_factor=0.7,
            throughput_scale=1.35,
            noise=0.018,
            notes="CUDA graphs + compile, TP4: throughput +35%, per-stage -30%.",
        ),
        Fixture(
            rel_path="A5/06.04/all_optimizations.log",
            n_steps=40,
            base_seed=41,
            actor_tp=4,
            actor_pp=2,
            actor_cp=1,
            actor_ep=4,
            actor_etp=1,
            actor_sp=True,
            actor_use_remove_padding=True,
            actor_grad_offload=True,
            actor_optim_offload=True,
            rollout_tp=4,
            rollout_dp=1,
            rollout_expert_parallel_size=1,
            rollout_prompt_length=1024,
            rollout_response_length=12288,
            rollout_max_num_batched_tokens=16384,
            rollout_max_num_seqs=2048,
            rollout_gpu_memory_utilization=0.6,
            rollout_enable_chunked_prefill=True,
            rollout_enable_prefix_caching=True,
            rollout_enforce_eager=False,
            rollout_use_torch_compile=True,
            rollout_n=5,
            train_batch_size=128,
            ppo_mini_batch_size=64,
            ppo_micro_batch_size_per_gpu=2,
            ppo_epochs=1,
            ppo_max_token_len_per_gpu=65536,
            kl_loss_coef=0.01,
            lr=5e-07,
            gen_factor=0.55,
            old_logp_factor=0.6,
            ref_factor=0.7,
            update_actor_factor=0.6,
            update_weights_factor=0.55,
            throughput_scale=1.55,
            noise=0.015,
            notes="All optimizations: TP4 PP2 EP4, larger batch, throughput +55%.",
        ),
    ]


def main() -> int:
    root = Path(__file__).resolve().parent / "testFold"
    root.mkdir(parents=True, exist_ok=True)
    written: List[Tuple[str, int]] = []
    for cfg in fixtures():
        out = root / cfg.rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_log(cfg), encoding="utf-8")
        written.append((str(out), cfg.n_steps))
        print(f"  wrote {out.relative_to(root.parent)} ({cfg.n_steps} steps) -- {cfg.notes}")
    print(f"\nGenerated {len(written)} fixture log files under {root}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
