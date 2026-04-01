from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.config import load_config
from svh_dp.data.factory import build_data_bundle
from svh_dp.models.system import SVHDPModel
from svh_dp.training.engine import fit_model
from svh_dp.utils.common import ensure_dir, set_seed, write_json


def build_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_single_variant(config: dict, variant: str, search_override: dict | None) -> dict:
    model_cfg = config["model"]
    training_cfg = deepcopy(config["training"])
    if search_override:
        training_cfg.update(search_override)

    set_seed(config["seed"])
    device = build_device(config["device"])
    data_bundle = build_data_bundle(config, training_cfg, seed=config["seed"])
    dataset_cfg = data_bundle["dataset_cfg"]
    loaders = data_bundle["loaders"]
    model = SVHDPModel(dataset_cfg, model_cfg, variant=variant).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    result = fit_model(model, loaders, optimizer, training_cfg, device)
    model.load_state_dict(result.best_state)
    return {
        "variant": variant,
        "data_source": data_bundle["source"],
        "dataset_summary": data_bundle["summary"],
        "search_config": search_override or {},
        "metrics": result.metrics,
        "state_dict": result.best_state,
    }


def result_selection_key(result: dict) -> tuple[float, float]:
    metrics = result["metrics"]
    return (
        metrics.get("best_weighted_completion", float("-inf")),
        metrics["best_weighted_success"],
        -metrics.get("final_nominal_terminal_step_mean_mse", float("inf")),
        -metrics.get("best_weighted_mse", float("inf")),
    )


def apply_runtime_overrides(
    config: dict,
    dataset_source: str | None,
    rollout_path: str | None,
    seed: int | None,
    enable_task_structured_decoder: bool,
    enable_prefix_decoder: bool,
    enable_turn_on_led_transition_decoder: bool,
) -> dict:
    updated = deepcopy(config)
    if seed is not None:
        updated["seed"] = seed
    if enable_task_structured_decoder:
        updated.setdefault("model", {}).setdefault("task_structured_decoder", {})
        updated["model"]["task_structured_decoder"]["enabled"] = True
    if enable_prefix_decoder:
        updated.setdefault("model", {}).setdefault("prefix_decoder", {})
        updated["model"]["prefix_decoder"]["enabled"] = True
    if enable_turn_on_led_transition_decoder:
        updated.setdefault("model", {}).setdefault("turn_on_led_transition_decoder", {})
        updated["model"]["turn_on_led_transition_decoder"]["enabled"] = True
    if dataset_source is not None:
        updated["dataset"]["source"] = dataset_source
    if rollout_path is not None:
        if updated["dataset"]["source"] == "calvin_export":
            updated["dataset"].setdefault("calvin_export", {})
            updated["dataset"]["calvin_export"]["rollout_path"] = rollout_path
        else:
            updated["dataset"].setdefault("rlbench_export", {})
            updated["dataset"]["rlbench_export"]["rollout_path"] = rollout_path
    return updated


def build_artifact_names(
    args_variant: str,
    dataset_source: str,
    artifact_tag: str | None,
) -> tuple[str, str]:
    suffix = ""
    if artifact_tag:
        suffix = f"_{artifact_tag}"
    if dataset_source == "synthetic":
        return (
            f"model_final_{args_variant}{suffix}.pth",
            f"{args_variant}_metrics{suffix}.json",
        )
    return (
        f"model_final_{args_variant}_{dataset_source}{suffix}.pth",
        f"{args_variant}_{dataset_source}_metrics{suffix}.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "default.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--variant",
        default="full",
        choices=["diffusion_only", "vla_only", "modular", "full"],
    )
    parser.add_argument(
        "--dataset-source",
        choices=["synthetic", "rlbench_export", "calvin_export"],
        help="Optional dataset backend override.",
    )
    parser.add_argument(
        "--rollout-path",
        help="Optional rollout export path when using rlbench_export or calvin_export.",
    )
    parser.add_argument(
        "--artifact-tag",
        help="Optional suffix to keep checkpoint/metric artifacts from different runs separate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed override.",
    )
    parser.add_argument(
        "--enable-task-structured-decoder",
        action="store_true",
        help="Enable the task-structured residual joint decoder on top of the shared executor.",
    )
    parser.add_argument(
        "--enable-prefix-decoder",
        action="store_true",
        help="Enable the prefix-conditioned early-action residual decoder on top of the shared executor.",
    )
    parser.add_argument(
        "--enable-turn-on-led-transition-decoder",
        action="store_true",
        help="Enable a dedicated step-conditioned gripper transition decoder for turn_on_led early rollout steps.",
    )
    args = parser.parse_args()

    config = apply_runtime_overrides(
        load_config(args.config).data,
        dataset_source=args.dataset_source,
        rollout_path=args.rollout_path,
        seed=args.seed,
        enable_task_structured_decoder=args.enable_task_structured_decoder,
        enable_prefix_decoder=args.enable_prefix_decoder,
        enable_turn_on_led_transition_decoder=args.enable_turn_on_led_transition_decoder,
    )
    checkpoint_dir = ensure_dir(config["paths"]["checkpoint_dir"])
    log_dir = ensure_dir(config["paths"]["log_dir"])
    dataset_source = config["dataset"].get("source", "synthetic")

    if args.variant != "full":
        run_result = run_single_variant(config, args.variant, search_override=None)
    else:
        candidates = config["search"]["candidates"] if config["search"]["enabled"] else [{}]
        candidate_results = [
            run_single_variant(config, "modular", search_override=candidate) for candidate in candidates
        ]
        run_result = max(
            candidate_results,
            key=result_selection_key,
        )
        run_result["variant"] = "full"
        run_result["candidate_results"] = [
            {
                "search_config": item["search_config"],
                "best_weighted_completion": item["metrics"].get("best_weighted_completion"),
                "best_weighted_success": item["metrics"]["best_weighted_success"],
                "best_weighted_mse": item["metrics"].get("best_weighted_mse"),
                "final_nominal_task_completion_rate": item["metrics"].get("final_nominal_task_completion_rate"),
                "final_perturbed_task_completion_rate": item["metrics"].get("final_perturbed_task_completion_rate"),
                "final_nominal_terminal_step_mean_mse": item["metrics"].get("final_nominal_terminal_step_mean_mse"),
                "final_perturbed_terminal_step_mean_mse": item["metrics"].get("final_perturbed_terminal_step_mean_mse"),
                "final_nominal_success_rate": item["metrics"]["final_nominal_success_rate"],
                "final_perturbed_success_rate": item["metrics"]["final_perturbed_success_rate"],
                "final_nominal_mse": item["metrics"]["final_nominal_mse"],
                "final_perturbed_mse": item["metrics"]["final_perturbed_mse"],
            }
            for item in candidate_results
        ]

    checkpoint_name, metrics_name = build_artifact_names(
        args_variant=args.variant,
        dataset_source=dataset_source,
        artifact_tag=args.artifact_tag,
    )
    checkpoint_path = checkpoint_dir / checkpoint_name
    metrics_path = log_dir / metrics_name
    torch.save(
        {
            "variant": run_result["variant"],
            "data_source": run_result["data_source"],
            "state_dict": run_result["state_dict"],
            "search_config": run_result.get("search_config", {}),
            "dataset_cfg": run_result["dataset_summary"],
            "model_cfg": config["model"],
            "task_name_to_id": run_result["dataset_summary"].get("task_name_to_id"),
        },
        checkpoint_path,
    )
    serializable = {
        key: value
        for key, value in run_result.items()
        if key != "state_dict"
    }
    serializable["seed"] = config["seed"]
    write_json(metrics_path, serializable)
    print(f"saved_checkpoint={checkpoint_path}")
    print(f"saved_metrics={metrics_path}")
    print(
        "summary",
        {
            "variant": serializable["variant"],
            "data_source": serializable["data_source"],
            "seed": serializable["seed"],
            "best_weighted_success": serializable["metrics"]["best_weighted_success"],
            "final_nominal_success_rate": serializable["metrics"]["final_nominal_success_rate"],
            "final_perturbed_success_rate": serializable["metrics"]["final_perturbed_success_rate"],
        },
    )


if __name__ == "__main__":
    main()
