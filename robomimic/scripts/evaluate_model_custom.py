"""
Evaluate a diffusion_policy checkpoint on a Robomimic/Robosuite environment.

Usage:
    python robomimic/scripts/evaluate_model_custom.py \
        --checkpoint /path/to/diffusion_policy.ckpt \
        --horizon 450 \
        --n-rollouts 10 \
        --video-path /path/to/output.mp4

    # Write per-trial results to a specific directory and resume later:
    python robomimic/scripts/evaluate_model_custom.py \
        --checkpoint /path/to/diffusion_policy.ckpt \
        --horizon 450 --n-rollouts 10 \
        --output-dir outputs/my_run --resume
"""

import argparse
import collections
import csv
import datetime
import torch.multiprocessing as mp
import os
import pickle
import time
from copy import deepcopy
from pathlib import Path

import dill
import hydra
import imageio
import numpy as np
import torch
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper

SEED = 42
DATASET_PATH = os.environ.get(
    "ROBOMIMIC_DATASET_PATH",
    "/home/michzeng/diffusion-policy/data/diffusion_experiments/robomimic/tool_hang/ph/image_v15.hdf5",
)

# Required so hydra configs that embed "${eval:...}" expressions can be loaded.
OmegaConf.register_new_resolver("eval", eval, replace=True)


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------


def load_policy(checkpoint_path: str, device: torch.device):
    """Load a diffusion_policy workspace + policy from a .ckpt file.

    - Reconstructs the workspace from the baked-in cfg.
    - Loads state dicts, skipping optimizer/lr_scheduler (not needed for inference).
    - Resolves or generates the paired normalizer.pt expected at
      <run_dir>/normalizer.pt (two levels above the .ckpt file).

    Returns:
        policy: EMA (or base) model on ``device``, in eval mode.
        cfg:    OmegaConf config from the checkpoint.
    """
    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(
        payload,
        exclude_keys=["optimizer", "lr_scheduler"],
        include_keys=None,
    )

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model

    ckpt_path = Path(checkpoint_path)
    normalizer_path = ckpt_path.parent.parent / "normalizer.pt"
    if normalizer_path.exists():
        print(f"Loading normalizer from {normalizer_path}")
        normalizer = torch.load(normalizer_path, weights_only=False)
    else:
        print(f"Normalizer not found at {normalizer_path}, generating from dataset...")
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        normalizer = dataset.get_normalizer()
        torch.save(normalizer, normalizer_path)
        print(f"Saved normalizer to {normalizer_path}")

    policy.set_normalizer(normalizer)
    policy.to(device).eval()
    # Belt-and-suspenders: obs_encoder may contain BatchNorm (e.g. ResNet18 in
    # R3M) that behaves incorrectly in train mode with the tiny batch sizes
    # (n_obs_steps rows) typical at inference time.
    if hasattr(policy, "obs_encoder"):
        policy.obs_encoder.eval()
    return policy, cfg


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def preprocess_obs(obs: dict, device: torch.device, obs_keys: set) -> dict:
    """Convert a robomimic env obs dict to float32 tensors on ``device``.

    Only the keys the policy was trained on (``obs_keys``) are kept.

    Image obs are cast to float32 but NOT scaled — the dataset normalizer is a
    passthrough for images (stores [0, 255]), and RobomimicObsEncoder handles
    the CHW / [-1, 1] conversion internally.
    """
    result = {}
    for key in obs_keys:
        if key not in obs:
            continue
        val = obs[key]
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val.copy())
        result[key] = val.float().to(device)
    return result


def build_obs_dict(obs_deque: collections.deque, device: torch.device) -> dict:
    """Stack the obs deque into the format expected by policy.predict_action.

    Each deque element is a dict {key: (...) tensor} without a batch dim
    (robomimic envs are not vectorized).  Returns:
        {"obs": {key: (1, T_obs, ...) tensor}}
    """
    keys = obs_deque[0].keys()
    obs_stacked = {k: torch.stack([o[k] for o in obs_deque], dim=0).unsqueeze(0) for k in keys}
    return {"obs": obs_stacked}


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_rollout(
    env,
    policy,
    n_obs_steps: int,
    horizon: int,
    device: torch.device,
    obs_keys: set,
    n_action_steps: int = None,
    render: bool = False,
    record_video: bool = False,
    camera_names: list = (),
) -> dict:
    """Run one rollout episode and return per-episode statistics.

    Returns:
        dict with keys:
            success      (bool)
            total_reward (float)
            result       (str) — "success" or "failure"
            steps        (int) — number of env steps executed
            frames       list[np.ndarray] — only present if record_video=True;
                         each element is a (H, W*n_cams, 3) uint8 frame
    """
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)

    # Reset and stabilise initial state (mirrors run_trained_agent.py).
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    preprocessed = preprocess_obs(obs, device, obs_keys)
    obs_deque = collections.deque([preprocessed] * n_obs_steps, maxlen=n_obs_steps)
    action_queue: collections.deque = collections.deque()

    total_reward = 0.0
    success = False
    step_i = 0
    frames = [] if record_video else None

    try:
        for step_i in range(horizon):
            # Refill action queue by running the diffusion model.
            if len(action_queue) == 0:
                obs_dict = build_obs_dict(obs_deque, device)
                pred = policy.predict_action(obs_dict, use_DDIM=True)
                # pred["action"]: (1, Ta, Da) — already sliced to n_action_steps.
                # For an override, pull from action_pred at the canonical offset.
                if n_action_steps is not None:
                    start = n_obs_steps - 1
                    actions = pred["action_pred"][:, start : start + n_action_steps]
                else:
                    actions = pred["action"]  # (1, Ta, Da)
                for t in range(actions.shape[1]):
                    action_queue.append(actions[0, t].cpu().numpy())

            action = action_queue.popleft()
            next_obs, r, done, _ = env.step(action)
            total_reward += float(r)
            success = env.is_success()["task"]

            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if record_video:
                imgs = [env.render(mode="rgb_array", height=512, width=512, camera_name=cam) for cam in camera_names]
                frames.append(np.concatenate(imgs, axis=1))

            if done or success:
                break

            obs = deepcopy(next_obs)
            obs_deque.append(preprocess_obs(obs, device, obs_keys))

    except env.rollout_exceptions as e:
        print(f"WARNING: rollout exception: {e}")

    out = {
        "success": bool(success),
        "total_reward": total_reward,
        "result": "success" if success else "failure",
        "steps": step_i + 1,
    }
    if record_video:
        out["frames"] = frames
    return out


# ---------------------------------------------------------------------------
# Parallel worker (must be module-level for multiprocessing spawn pickling)
# ---------------------------------------------------------------------------

# Per-worker state populated by _init_worker in each subprocess.
_worker_state: dict = {}


def _init_worker(
    checkpoint_path,
    video_worker_claimed,
    dataset_path,
    env_name,
    is_image_policy,
    horizon,
    n_obs_steps,
    policy_obs_keys,
    n_action_steps,
    camera_names,
    device_str,
    save_video,
):
    """Pool initializer: load policy + env once per worker process.

    Only the first worker to initialise claims the video-recording role.
    All other workers skip offscreen rendering, saving OpenGL framebuffer memory.
    """
    global _worker_state
    worker_seed = SEED + os.getpid()
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    device = torch.device(device_str)
    policy, _ = load_policy(checkpoint_path, device)

    # Exactly one worker gets video-recording capability.
    with video_worker_claimed.get_lock():
        can_record = save_video and not video_worker_claimed.value
        if can_record:
            video_worker_claimed.value = True

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_name,
        render=False,
        render_offscreen=can_record,  # only the designated worker allocates framebuffers
        use_image_obs=is_image_policy,
    )

    _worker_state.update(
        {
            "env": env,
            "policy": policy,
            "device": device,
            "horizon": horizon,
            "n_obs_steps": n_obs_steps,
            "obs_keys": policy_obs_keys,
            "n_action_steps": n_action_steps,
            "camera_names": camera_names,
            "can_record": can_record,
        }
    )


def _worker_run_rollout(record_video: bool) -> dict:
    """Called in each worker process to run one rollout."""
    s = _worker_state
    # Only the designated video worker actually records frames.
    actual_record = record_video and s["can_record"]
    result = run_rollout(
        env=s["env"],
        policy=s["policy"],
        n_obs_steps=s["n_obs_steps"],
        horizon=s["horizon"],
        device=s["device"],
        obs_keys=s["obs_keys"],
        n_action_steps=s["n_action_steps"],
        render=False,
        record_video=actual_record,
        camera_names=s["camera_names"],
    )
    result["_record_flag"] = actual_record
    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _write_mp4(frames: list, path: Path, fps: int = 20) -> None:
    """Write a list of (H, W, 3) uint8 numpy frames to an MP4 file."""
    with imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p") as writer:
        for frame in frames:
            writer.append_data(frame)


def _write_summary(n_success: int, n_total: int, n_requested: int, trial_records: list, path: Path) -> None:
    n_failure = sum(1 for r in trial_records if r["result"] == "failure")
    rate = n_success / n_total if n_total > 0 else 0.0
    avg_steps = sum(r["steps"] for r in trial_records) / len(trial_records) if trial_records else 0.0
    with open(path, "w") as f:
        f.write(f"Trials completed : {n_total} / {n_requested}\n")
        f.write(f"Successes        : {n_success}\n")
        f.write(f"Failures         : {n_failure}\n")
        f.write(f"Success rate     : {rate:.1%}\n")
        f.write(f"Avg trial steps  : {avg_steps:.1f}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a diffusion_policy checkpoint on a Robomimic env.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to diffusion_policy .ckpt file",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=450,
        help="Maximum number of env steps per rollout",
    )
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Override the env name embedded in the training dataset metadata",
    )
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel worker processes. Each loads its own policy+env copy. "
        "Requires --headless (on-screen rendering is incompatible with workers).",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Override action horizon (default: use value from checkpoint config)",
    )
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument("--no-save-video", dest="save_video", action="store_false")
    parser.add_argument(
        "--n-video-trials",
        type=int,
        default=20,
        help="Save videos for only the first N trials (default: 20). Set to -1 to save all.",
    )
    parser.add_argument(
        "--record-failures",
        action="store_true",
        default=False,
        help="If set, only save videos of failed trials, and save all of them (overrides --n-video-trials).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for results.csv / results.pkl / summary.txt (default: outputs/<date>/<time>)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from an existing results.pkl in --output-dir",
    )
    args = parser.parse_args()

    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir to be specified")
    if args.n_envs > 1 and not args.headless:
        print("WARNING: --n-envs > 1 is incompatible with on-screen rendering; forcing headless.")
        args.headless = True

    device = torch.device(args.device)

    # --- policy ---
    print(f"Loading policy from {args.checkpoint}")
    policy, cfg = load_policy(args.checkpoint, device)
    n_obs_steps: int = int(cfg.n_obs_steps)
    n_action_steps = args.n_action_steps
    policy_obs_keys = set(cfg.shape_meta.obs.keys())
    # Derive video camera names from the rgb obs keys: "sideview_image" → "sideview".
    camera_names = [
        k[: -len("_image")] if k.endswith("_image") else k
        for k, v in cfg.shape_meta.obs.items()
        if v.get("type") == "rgb"
    ]
    print(
        f"n_obs_steps={n_obs_steps}, "
        f"n_action_steps={'from_cfg' if n_action_steps is None else n_action_steps}, "
        f"obs_keys={sorted(policy_obs_keys)}, "
        f"camera_names={camera_names}"
    )

    # --- environment ---
    # The training dataset embeds env_args (env name + kwargs) written by
    # robomimic at data-collection time; use those to reconstruct the env.
    dataset_path = DATASET_PATH
    print(f"Loading env metadata from dataset: {dataset_path}")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    is_image_policy = any(v.get("type") == "rgb" for v in cfg.shape_meta.obs.values())
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=args.env,
        render=not args.headless,
        render_offscreen=args.save_video,
        use_image_obs=is_image_policy,
    )
    print(f"Created env: {env_meta['env_name']}, horizon={args.horizon}")

    # --- seeds ---
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --- output directory ---
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        now = datetime.datetime.now()
        out_dir = Path("outputs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = out_dir / "videos"
    if args.save_video:
        videos_dir.mkdir(parents=True, exist_ok=True)

    # --- state: fresh or resumed ---
    n_success = 0
    n_total = 0
    all_trial_records = []

    if args.resume:
        pkl_path = out_dir / "results.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                saved = pickle.load(f)
            if saved.get("n_total", 0) > 0:
                n_success = saved["n_success"]
                n_total = saved["n_total"]
                all_trial_records = saved["trials"]
                print(
                    f"Resuming: {n_total}/{args.n_rollouts} trials done "
                    f"({n_success} successes, {n_success / n_total:.1%}); "
                    f"starting from trial {n_total + 1}"
                )
            else:
                print("Found results.pkl but no completed trials; starting fresh.")
        else:
            print(f"--resume set but no results.pkl found in {out_dir}; starting fresh.")

    csv_path = out_dir / "results.csv"
    csv_fields = ["trial", "result", "reward", "steps"]
    summary_path = out_dir / "summary.txt"

    csv_mode = "a" if (args.resume and n_total > 0) else "w"
    csv_file = open(csv_path, csv_mode, newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if csv_mode == "w":
        csv_writer.writeheader()
        csv_file.flush()

    # --- rollout loop ---
    video_budget = args.n_video_trials if args.n_video_trials >= 0 else args.n_rollouts

    def _record_this(trial_idx_0based: int) -> bool:
        if args.record_failures:
            return args.save_video
        return args.save_video and (trial_idx_0based < video_budget)

    def _process_result(rollout_result: dict, record_flag: bool, n_success: int, n_total: int):
        n_success += int(rollout_result["success"])
        n_total += 1
        result_str = rollout_result["result"]

        record = {
            "trial": n_total,
            "result": result_str,
            "reward": rollout_result["total_reward"],
            "steps": rollout_result["steps"],
        }
        all_trial_records.append(record)
        csv_writer.writerow(record)

        if record_flag:
            save_this = result_str != "success" if args.record_failures else n_total <= video_budget
            if save_this:
                video_path = videos_dir / f"trial_{n_total:04d}_{result_str}.mp4"
                _write_mp4(rollout_result["frames"], video_path)
                print(f"  Saved video: {video_path.name}")

        print(
            f"Trial {n_total}/{args.n_rollouts}: "
            f"result={result_str}, "
            f"steps={rollout_result['steps']}, "
            f"reward={rollout_result['total_reward']:.3f}  "
            f"running {n_success}/{n_total} ({n_success / n_total:.1%})"
        )
        return n_success, n_total

    def _flush_state() -> None:
        csv_file.flush()
        _write_summary(n_success, n_total, args.n_rollouts, all_trial_records, summary_path)
        with open(out_dir / "results.pkl", "wb") as f:
            pickle.dump(
                {
                    "trials": all_trial_records,
                    "n_success": n_success,
                    "n_total": n_total,
                    "success_rate": n_success / n_total,
                    "checkpoint": args.checkpoint,
                    "horizon": args.horizon,
                    "n_obs_steps": n_obs_steps,
                },
                f,
            )

    if args.n_envs == 1:
        # ---- serial path ------------------------------------------------
        for trial_idx in range(n_total, args.n_rollouts):
            record_flag = _record_this(trial_idx)
            t_start = time.time()
            rollout_result = run_rollout(
                env=env,
                policy=policy,
                n_obs_steps=n_obs_steps,
                horizon=args.horizon,
                device=device,
                obs_keys=policy_obs_keys,
                n_action_steps=n_action_steps,
                render=not args.headless,
                record_video=record_flag,
                camera_names=camera_names,
            )
            n_success, n_total = _process_result(rollout_result, record_flag, n_success, n_total)
            _flush_state()
            print(f"  wall time: {time.time() - t_start:.1f}s")

    else:
        # ---- parallel path ----------------------------------------------
        # imap_unordered keeps all workers busy with no batch-level waiting.
        # Only 1 worker is designated for video recording; the rest skip
        # offscreen rendering entirely to save OpenGL framebuffer memory.
        ctx = mp.get_context("spawn")
        video_worker_claimed = ctx.Value("b", False)
        print(f"Spawning {args.n_envs} worker processes...")
        pool = ctx.Pool(
            processes=args.n_envs,
            initializer=_init_worker,
            initargs=(
                args.checkpoint,
                video_worker_claimed,
                dataset_path,
                args.env,
                is_image_policy,
                args.horizon,
                n_obs_steps,
                policy_obs_keys,
                n_action_steps,
                camera_names,
                args.device,
                args.save_video,
            ),
        )

        remaining = args.n_rollouts - n_total
        record_flags = [_record_this(n_total + j) for j in range(remaining)]

        for rollout_result in pool.imap_unordered(_worker_run_rollout, record_flags):
            record_flag = rollout_result.pop("_record_flag")
            n_success, n_total = _process_result(rollout_result, record_flag, n_success, n_total)
            _flush_state()

        pool.close()
        pool.join()

    csv_file.close()

    final_rate = n_success / n_total
    print(f"\nFinal success rate: {n_success}/{n_total} ({final_rate:.1%})")
    print(f"Results written to {out_dir}/")
