"""Open-loop evaluation of a trained openpi policy on a LeRobot dataset.

Feeds dataset observations into the model (no environment interaction), compares
predicted action chunks against ground-truth actions, writes action-dimension
curve plots, and aggregates metrics.

Works with any TrainConfig that uses a LeRobot dataset (via `data.repo_id` and
`data.action_sequence_keys`). Robot-specific input/output mapping is taken from
the config's data transforms.

Example:
    python scripts/eval_openloop.py --config-name <config> --exp-name <exp>
    python scripts/eval_openloop.py --config-name <config> --checkpoint-dir <ckpt>
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import math
import pathlib
from typing import Any

import numpy as np
import rich
import rich.table
import torch
import tqdm
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

logger = logging.getLogger("openpi")


@dataclasses.dataclass
class Args:
    """Open-loop evaluation arguments."""

    config_name: str
    """Training config name (must exist in openpi.training.config)."""

    exp_name: str | None = None
    """Experiment name. Used with --config-name to resolve checkpoints/<config>/<exp> when --checkpoint-dir is omitted."""

    checkpoint_dir: pathlib.Path | None = None
    """Checkpoint directory. Can be a step dir (params/ or model.safetensors) or an experiment dir (latest step is used)."""

    output_dir: pathlib.Path | None = None
    """Where to write plots and metric files. Defaults to eval_openloop/<config>/<exp_or_step>."""

    repo_id: str | None = None
    """Override the dataset repo_id from the training config."""

    dataset_root: pathlib.Path | None = None
    """Optional local LeRobot dataset root. If omitted, uses the HF cache for repo_id."""

    default_prompt: str | None = None
    """Fallback language prompt when a sample has no prompt/task."""

    num_episodes: int | None = None
    """Max number of episodes to evaluate. None = all episodes."""

    episode_indices: tuple[int, ...] | None = None
    """Explicit episode indices to evaluate. Overrides --num-episodes if set."""

    stride: int | None = None
    """Query the policy every N frames and stitch the predicted chunk. Default: action_horizon (non-overlapping)."""

    max_plots: int = 20
    """Max number of per-episode action-curve figures to save."""


@dataclasses.dataclass
class EpisodeResult:
    episode_index: int
    prompt: str
    fps: float
    gt: np.ndarray  # (T, D)
    pred: np.ndarray  # (T, D), NaN where not covered
    first_abs: np.ndarray  # (Q, D)
    chunk_abs: np.ndarray  # (valid_steps_total, D)
    horizon_abs: dict[int, list[float]]
    infer_ms: list[float]


def _tree_to_numpy(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _tree_to_numpy(v) for k, v in tree.items()}
    if isinstance(tree, torch.Tensor):
        return tree.detach().cpu().numpy()
    if hasattr(tree, "mode") and hasattr(tree, "convert"):
        # Pillow Image
        return np.asarray(tree)
    return tree


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def resolve_checkpoint_dir(path: pathlib.Path | str) -> pathlib.Path:
    path_str = str(path)
    if path_str.startswith("gs://"):
        return pathlib.Path(path_str)

    path = pathlib.Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if (path / "params").exists() or (path / "model.safetensors").exists():
        return path

    step_dirs = [p for p in path.iterdir() if p.is_dir() and p.name.isdigit()]
    if not step_dirs:
        raise FileNotFoundError(
            f"No checkpoint steps found under {path}. Expected a directory containing `params/` "
            "or `model.safetensors`, or numeric step folders."
        )
    latest = max(step_dirs, key=lambda p: int(p.name))
    logger.info("Using latest checkpoint step: %s", latest)
    return latest


def _episode_bounds(dataset: Any, episode_index: int) -> tuple[int, int]:
    start = int(dataset.episode_data_index["from"][episode_index])
    end = int(dataset.episode_data_index["to"][episode_index])
    return start, end


def _action_dim_names(dataset: Any, action_key: str, action_dim: int) -> list[str]:
    """Resolve action-dimension names from LeRobot metadata when available."""
    features = getattr(getattr(dataset, "meta", None), "features", None) or {}
    for key in (action_key, "action", "actions"):
        action_feat = features.get(key)
        if not isinstance(action_feat, dict):
            continue
        names = action_feat.get("names")
        if not names:
            continue
        if isinstance(names[0], (list, tuple)):
            names = list(names[0])
        names = [str(n) for n in names]
        if len(names) >= action_dim:
            return names[:action_dim]
    return [f"dim_{i}" for i in range(action_dim)]


def _resolve_action_key(dataset: Any, action_sequence_keys: tuple[str, ...] | list[str]) -> str:
    """Pick the dataset column used as ground-truth actions."""
    features = getattr(getattr(dataset, "meta", None), "features", None) or {}
    candidates = list(action_sequence_keys) + ["action", "actions"]
    seen: set[str] = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        if key in features:
            return key
        # Some LeRobot versions expose columns only via hf_dataset.
        if hasattr(dataset, "hf_dataset") and key in dataset.hf_dataset.column_names:
            return key
    raise KeyError(
        f"Could not find an action column. Tried {candidates}. "
        f"Available features: {sorted(features) if features else 'unknown'}"
    )


def _load_gt_actions(dataset: Any, start: int, end: int, action_key: str) -> np.ndarray:
    column = dataset.hf_dataset.select(range(start, end))[action_key]
    return np.stack([np.asarray(row, dtype=np.float32) for row in column])


def _get_prompt(sample: dict[str, Any], tasks: dict[int, str], default_prompt: str | None) -> str:
    for key in ("prompt", "task"):
        if key not in sample:
            continue
        value = sample[key]
        if isinstance(value, bytes):
            return value.decode()
        if isinstance(value, str) and value:
            return value
        if isinstance(value, np.ndarray) and value.dtype.kind in {"U", "S", "O"}:
            item = value.item()
            if item:
                return str(item)

    if "task_index" in sample and tasks:
        task_index = int(np.asarray(sample["task_index"]).item())
        if task_index in tasks:
            return str(tasks[task_index])

    if default_prompt:
        return default_prompt
    raise KeyError(
        "No language prompt found on sample. Provide --default-prompt, or ensure the dataset "
        "has prompt/task fields (or task_index with meta.tasks)."
    )


def _mae_rmse(abs_err: np.ndarray) -> tuple[float, float, float]:
    if abs_err.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(abs_err)), float(np.sqrt(np.mean(np.square(abs_err)))), float(np.max(abs_err))


def _pearson(gt: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(pred).all(axis=1)
    if mask.sum() < 2:
        return float("nan")
    rs = []
    for dim in range(gt.shape[1]):
        g = gt[mask, dim]
        p = pred[mask, dim]
        if np.std(g) < 1e-8 or np.std(p) < 1e-8:
            continue
        rs.append(float(np.corrcoef(g, p)[0, 1]))
    return float(np.mean(rs)) if rs else float("nan")


def eval_episode(
    *,
    policy: Any,
    dataset: Any,
    episode_index: int,
    stride: int,
    action_horizon: int,
    action_key: str,
    tasks: dict[int, str],
    default_prompt: str | None,
) -> EpisodeResult:
    start, end = _episode_bounds(dataset, episode_index)
    gt = _load_gt_actions(dataset, start, end, action_key)
    action_dim = gt.shape[1]
    pred = np.full_like(gt, np.nan)
    first_abs: list[np.ndarray] = []
    chunk_abs: list[np.ndarray] = []
    horizon_abs: dict[int, list[float]] = {h: [] for h in range(action_horizon)}
    infer_ms: list[float] = []
    prompt = default_prompt or ""

    query_indices = list(range(start, end, stride))
    for frame_idx in query_indices:
        sample = _tree_to_numpy(dataset[frame_idx])
        prompt = _get_prompt(sample, tasks, default_prompt)
        sample["prompt"] = prompt

        local = frame_idx - start
        remaining = end - frame_idx
        valid = min(action_horizon, remaining)

        result = policy.infer(sample)
        pred_chunk = np.asarray(result["actions"], dtype=np.float32)
        if pred_chunk.ndim != 2:
            raise ValueError(f"Expected actions of shape (horizon, dim), got {pred_chunk.shape}")
        pred_chunk = pred_chunk[:valid, :action_dim]
        gt_chunk = gt[local : local + valid]

        abs_err = np.abs(pred_chunk - gt_chunk)
        first_abs.append(abs_err[0])
        chunk_abs.append(abs_err)
        for h in range(valid):
            horizon_abs[h].extend(abs_err[h].tolist())

        pred[local : local + valid] = pred_chunk
        infer_ms.append(float(result.get("policy_timing", {}).get("infer_ms", 0.0)))

    fps = float(getattr(dataset.meta, "fps", 1.0) or 1.0)
    return EpisodeResult(
        episode_index=episode_index,
        prompt=prompt,
        fps=fps,
        gt=gt,
        pred=pred,
        first_abs=np.stack(first_abs) if first_abs else np.zeros((0, action_dim), dtype=np.float32),
        chunk_abs=np.concatenate(chunk_abs, axis=0) if chunk_abs else np.zeros((0, action_dim), dtype=np.float32),
        horizon_abs=horizon_abs,
        infer_ms=infer_ms,
    )


def _stitched_abs(result: EpisodeResult) -> np.ndarray:
    mask = np.isfinite(result.pred).all(axis=1)
    return np.abs(result.pred[mask] - result.gt[mask])


def plot_episode_actions(result: EpisodeResult, dim_names: list[str], path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    action_dim = result.gt.shape[1]
    t = np.arange(result.gt.shape[0]) / max(result.fps, 1e-6)
    ncols = 2 if action_dim > 1 else 1
    nrows = int(math.ceil(action_dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 2.1 * nrows), sharex=True, squeeze=False)
    for dim in range(action_dim):
        ax = axes[dim // ncols][dim % ncols]
        ax.plot(t, result.gt[:, dim], color="C0", lw=1.4, label="GT")
        ax.plot(t, result.pred[:, dim], color="C1", lw=1.2, ls="--", label="Pred")
        ax.set_ylabel(dim_names[dim], fontsize=8)
        ax.grid(True, alpha=0.3)
        if dim == 0:
            ax.legend(loc="upper right", fontsize=8)
    for dim in range(action_dim, nrows * ncols):
        axes[dim // ncols][dim % ncols].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"episode {result.episode_index}: {result.prompt}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_per_dim_mae(per_dim_mae: np.ndarray, dim_names: list[str], path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(dim_names))))
    y = np.arange(len(dim_names))
    ax.barh(y, per_dim_mae, color="C0")
    ax.set_yticks(y, dim_names)
    ax.invert_yaxis()
    ax.set_xlabel("MAE")
    ax.set_title("Per-dimension MAE (stitched open-loop trajectory)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_horizon_mae(horizon_mae: list[float], path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(np.arange(len(horizon_mae)), horizon_mae, marker="o", lw=1.5)
    ax.set_xlabel("step in action chunk")
    ax.set_ylabel("MAE")
    ax.set_title("Open-loop error vs. action-chunk horizon")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_episode_heatmap(
    episode_maes: np.ndarray,
    dim_names: list[str],
    episode_ids: list[int],
    path: pathlib.Path,
) -> None:
    import matplotlib.pyplot as plt

    fig_w = max(8, 0.45 * len(dim_names))
    fig_h = max(3.5, 0.35 * len(episode_ids))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(episode_maes, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_xticks(np.arange(len(dim_names)), dim_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(episode_ids)), [str(i) for i in episode_ids], fontsize=8)
    ax.set_xlabel("action dim")
    ax.set_ylabel("episode")
    ax.set_title("Per-episode per-dimension MAE")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="MAE")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_tables(
    *,
    overall: dict[str, Any],
    per_dim: list[dict[str, Any]],
    per_episode: list[dict[str, Any]],
) -> None:
    overall_table = rich.table.Table(title="Open-loop evaluation", show_header=True, header_style="bold")
    overall_table.add_column("Metric")
    overall_table.add_column("First step", justify="right")
    overall_table.add_column("Full chunk", justify="right")
    overall_table.add_column("Stitched traj", justify="right")
    for key, label in (("mae", "MAE"), ("rmse", "RMSE"), ("max", "MaxAE")):
        overall_table.add_row(
            label,
            f"{overall['first'][key]:.5f}",
            f"{overall['chunk'][key]:.5f}",
            f"{overall['stitched'][key]:.5f}",
        )
    rich.print(overall_table)

    dim_table = rich.table.Table(title="Per-dimension MAE (stitched)", show_header=True, header_style="bold")
    dim_table.add_column("Dim")
    dim_table.add_column("MAE", justify="right")
    dim_table.add_column("RMSE", justify="right")
    for row in per_dim:
        dim_table.add_row(row["dim"], f"{row['mae']:.5f}", f"{row['rmse']:.5f}")
    rich.print(dim_table)

    ep_table = rich.table.Table(title="Per-episode MAE (stitched)", show_header=True, header_style="bold")
    ep_table.add_column("Episode")
    ep_table.add_column("T")
    ep_table.add_column("MAE", justify="right")
    ep_table.add_column("RMSE", justify="right")
    ep_table.add_column("Pearson r", justify="right")
    for row in per_episode:
        ep_table.add_row(
            str(row["episode_index"]),
            str(row["num_frames"]),
            f"{row['mae']:.5f}",
            f"{row['rmse']:.5f}",
            f"{row['pearson_r']:.4f}" if row["pearson_r"] is not None else "n/a",
        )
    rich.print(ep_table)


def _write_summary_md(
    path: pathlib.Path,
    *,
    args: Args,
    checkpoint_dir: pathlib.Path,
    repo_id: str,
    overall: dict[str, Any],
    per_dim: list[dict[str, Any]],
    per_episode: list[dict[str, Any]],
) -> None:
    lines = [
        "# Open-loop evaluation",
        "",
        f"- config: `{args.config_name}`",
        f"- checkpoint: `{checkpoint_dir}`",
        f"- dataset: `{repo_id}`",
        f"- episodes: {len(per_episode)}",
        f"- stride: {args.stride}",
        f"- mean infer: {overall['mean_infer_ms']:.1f} ms",
        "",
        "## Overall",
        "",
        "| Metric | First step | Full chunk | Stitched traj |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (("mae", "MAE"), ("rmse", "RMSE"), ("max", "MaxAE")):
        lines.append(
            f"| {label} | {overall['first'][key]:.5f} | {overall['chunk'][key]:.5f} | {overall['stitched'][key]:.5f} |"
        )
    lines.extend(
        [
            "",
            f"- stitched Pearson r (mean over dims): {overall['stitched']['pearson_r']:.4f}",
            "",
            "## Per-dimension (stitched)",
            "",
            "| Dim | MAE | RMSE |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in per_dim:
        lines.append(f"| {row['dim']} | {row['mae']:.5f} | {row['rmse']:.5f} |")
    lines.extend(
        [
            "",
            "## Per-episode (stitched)",
            "",
            "| Episode | Frames | MAE | RMSE | Pearson r | Prompt |",
            "| ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in per_episode:
        pearson = f"{row['pearson_r']:.4f}" if row["pearson_r"] is not None else "n/a"
        prompt = str(row["prompt"]).replace("|", "/")
        lines.append(
            f"| {row['episode_index']} | {row['num_frames']} | {row['mae']:.5f} | {row['rmse']:.5f} | {pearson} | {prompt} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _load_task_map(dataset: Any) -> dict[int, str]:
    raw_tasks = getattr(getattr(dataset, "meta", None), "tasks", None)
    if raw_tasks is None:
        return {}
    if isinstance(raw_tasks, dict):
        return {int(k): str(v) for k, v in raw_tasks.items()}
    return {i: str(v) for i, v in enumerate(raw_tasks)}


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    train_config = _config.get_config(args.config_name)
    if args.checkpoint_dir is not None:
        checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir)
    else:
        if not args.exp_name:
            raise ValueError("Provide --checkpoint-dir, or both --config-name and --exp-name")
        checkpoint_dir = resolve_checkpoint_dir(pathlib.Path("checkpoints") / args.config_name / args.exp_name)

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        run_name = args.exp_name or checkpoint_dir.name
        output_dir = pathlib.Path("eval_openloop") / args.config_name / run_name
    plots_dir = output_dir / "plots"
    traj_dir = output_dir / "trajectories"
    plots_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    repo_id = args.repo_id or data_config.repo_id
    if not repo_id or repo_id == "fake":
        raise ValueError(
            f"Config '{args.config_name}' has no usable LeRobot repo_id. "
            "Pass --repo-id, or use a config that points at a real dataset."
        )

    action_horizon = int(train_config.model.action_horizon)
    stride = args.stride if args.stride is not None else action_horizon
    if stride <= 0:
        raise ValueError("--stride must be > 0")
    args.stride = stride

    logger.info("Loading dataset %s", repo_id)
    dataset_kwargs: dict[str, Any] = {}
    if args.dataset_root is not None:
        dataset_kwargs["root"] = str(args.dataset_root)

    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, **dataset_kwargs)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        **dataset_kwargs,
    )
    action_key = _resolve_action_key(dataset, data_config.action_sequence_keys)
    tasks = _load_task_map(dataset)
    default_prompt = args.default_prompt

    num_episodes = int(dataset.num_episodes)
    if args.episode_indices is not None:
        episode_ids = list(args.episode_indices)
    else:
        episode_ids = list(range(num_episodes))
        if args.num_episodes is not None:
            episode_ids = episode_ids[: args.num_episodes]
    for ep in episode_ids:
        if ep < 0 or ep >= num_episodes:
            raise IndexError(f"episode_index {ep} is out of range [0, {num_episodes})")

    logger.info(
        "Evaluating %d episodes from %s (action_key=%s, action_horizon=%d, stride=%d)",
        len(episode_ids),
        repo_id,
        action_key,
        action_horizon,
        stride,
    )
    logger.info("Loading policy from %s", checkpoint_dir)
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        # Same repack as training so LeRobot sample keys match the policy's data transforms.
        repack_transforms=data_config.repack_transforms,
        default_prompt=default_prompt,
    )

    results: list[EpisodeResult] = []
    for ep in tqdm.tqdm(episode_ids, desc="open-loop eval"):
        result = eval_episode(
            policy=policy,
            dataset=dataset,
            episode_index=ep,
            stride=stride,
            action_horizon=action_horizon,
            action_key=action_key,
            tasks=tasks,
            default_prompt=default_prompt,
        )
        results.append(result)
        np.savez_compressed(
            traj_dir / f"ep{ep:04d}.npz",
            gt=result.gt,
            pred=result.pred,
            prompt=np.asarray(result.prompt),
            fps=np.asarray(result.fps),
        )

    if not results:
        raise RuntimeError("No episodes were evaluated.")

    dim_names = _action_dim_names(dataset, action_key, results[0].gt.shape[1])
    first_abs = np.concatenate([r.first_abs for r in results], axis=0)
    chunk_abs = np.concatenate([r.chunk_abs for r in results], axis=0)
    stitched_abs = np.concatenate([_stitched_abs(r) for r in results], axis=0)

    first_mae, first_rmse, first_max = _mae_rmse(first_abs)
    chunk_mae, chunk_rmse, chunk_max = _mae_rmse(chunk_abs)
    stitched_mae, stitched_rmse, stitched_max = _mae_rmse(stitched_abs)
    stitched_pearson = float(np.nanmean([_pearson(r.gt, r.pred) for r in results]))

    per_dim_mae = np.mean(stitched_abs, axis=0)
    per_dim_rmse = np.sqrt(np.mean(np.square(stitched_abs), axis=0))
    per_dim_rows = [
        {"dim": name, "mae": float(per_dim_mae[i]), "rmse": float(per_dim_rmse[i])} for i, name in enumerate(dim_names)
    ]

    per_episode_rows = []
    episode_dim_maes = []
    for result in results:
        abs_err = _stitched_abs(result)
        mae, rmse, max_ae = _mae_rmse(abs_err)
        pearson = _pearson(result.gt, result.pred)
        per_episode_rows.append(
            {
                "episode_index": result.episode_index,
                "num_frames": int(result.gt.shape[0]),
                "prompt": result.prompt,
                "mae": mae,
                "rmse": rmse,
                "max": max_ae,
                "pearson_r": None if math.isnan(pearson) else pearson,
                "mean_infer_ms": float(np.mean(result.infer_ms)) if result.infer_ms else None,
            }
        )
        mask = np.isfinite(result.pred).all(axis=1)
        episode_dim_maes.append(np.mean(np.abs(result.pred[mask] - result.gt[mask]), axis=0))

    horizon_mae = []
    for h in range(action_horizon):
        vals = [v for r in results for v in r.horizon_abs[h]]
        horizon_mae.append(float(np.mean(vals)) if vals else float("nan"))

    overall = {
        "config_name": args.config_name,
        "checkpoint_dir": str(checkpoint_dir),
        "dataset": repo_id,
        "action_key": action_key,
        "num_episodes": len(results),
        "action_horizon": action_horizon,
        "stride": stride,
        "mean_infer_ms": float(np.mean([ms for r in results for ms in r.infer_ms])) if results else float("nan"),
        "first": {"mae": first_mae, "rmse": first_rmse, "max": first_max},
        "chunk": {"mae": chunk_mae, "rmse": chunk_rmse, "max": chunk_max},
        "stitched": {
            "mae": stitched_mae,
            "rmse": stitched_rmse,
            "max": stitched_max,
            "pearson_r": stitched_pearson,
        },
        "horizon_mae": horizon_mae,
    }

    (output_dir / "metrics.json").write_text(json.dumps(_jsonable(overall), indent=2) + "\n")
    _write_csv(output_dir / "per_episode.csv", per_episode_rows)
    _write_csv(output_dir / "per_dim.csv", per_dim_rows)
    _write_summary_md(
        output_dir / "summary.md",
        args=args,
        checkpoint_dir=checkpoint_dir,
        repo_id=str(repo_id),
        overall=overall,
        per_dim=per_dim_rows,
        per_episode=per_episode_rows,
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        for result in results[: args.max_plots]:
            plot_episode_actions(result, dim_names, plots_dir / f"actions_ep{result.episode_index:04d}.png")
        plot_per_dim_mae(per_dim_mae, dim_names, plots_dir / "per_dim_mae.png")
        plot_horizon_mae(horizon_mae, plots_dir / "horizon_mae.png")
        plot_episode_heatmap(
            np.stack(episode_dim_maes, axis=0),
            dim_names,
            [r.episode_index for r in results],
            plots_dir / "episode_mae_heatmap.png",
        )
    except ImportError:
        logger.warning("matplotlib is not installed; metrics were saved but plots were skipped.")

    _print_tables(overall=overall, per_dim=per_dim_rows, per_episode=per_episode_rows)
    logger.info("Wrote evaluation artifacts to %s", output_dir.resolve())


if __name__ == "__main__":
    main(tyro.cli(Args))
