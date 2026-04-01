from __future__ import annotations

from copy import deepcopy

from svh_dp.data.calvin_rollout import build_data_bundle as build_calvin_rollout_bundle
from svh_dp.data.rlbench_rollout import build_data_bundle as build_rlbench_rollout_bundle
from svh_dp.data.synthetic import build_loaders as build_synthetic_loaders


def build_data_bundle(config: dict, training_cfg: dict, seed: int) -> dict:
    dataset_cfg = deepcopy(config["dataset"])
    source = dataset_cfg.get("source", "synthetic")
    if source == "synthetic":
        return {
            "source": source,
            "dataset_cfg": dataset_cfg,
            "loaders": build_synthetic_loaders(dataset_cfg, training_cfg, seed=seed),
            "summary": {
                "train_samples": dataset_cfg["train_size"],
                "val_samples": dataset_cfg["val_size"],
                "obs_dim": dataset_cfg["obs_dim"],
                "proprio_dim": dataset_cfg["proprio_dim"],
                "action_dim": dataset_cfg["action_dim"],
                "chunk_len": dataset_cfg["chunk_len"],
            },
        }
    if source == "rlbench_export":
        return build_rlbench_rollout_bundle(
            dataset_cfg=dataset_cfg,
            training_cfg=training_cfg,
            benchmark_cfg=config["benchmarks"]["rlbench"],
            seed=seed,
        )
    if source == "calvin_export":
        return build_calvin_rollout_bundle(
            dataset_cfg=dataset_cfg,
            training_cfg=training_cfg,
            benchmark_cfg=config["benchmarks"]["calvin"],
            seed=seed,
        )
    raise ValueError(f"unsupported dataset source: {source}")
