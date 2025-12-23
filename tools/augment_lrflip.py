#!/usr/bin/env python3
"""Left-right (y-axis) flip augmentation for *_prcd.json scenario files.

What it does
- Finds all '*_prcd.json' files inside a given scenario folder.
- For each file, mirrors spatial quantities left-right:
  * position: y -> -y
  * lin_vel: vy -> -vy
  * lin_acc: ay -> -ay
  * ang_vel: w -> -w
  * scan: angle -> -angle (range unchanged), then re-sorted by angle ascending
- Writes a new JSON next to the source file (or to --out_dir) with a suffix.

Input schema expected (from data_processor.py):
{
  "frames": [
    {
      "timestamp": float,
      "lin_vel": [vx, vy] or null,
      "lin_acc": [ax, ay] or null,
      "ang_vel": float or null,
      "scan": [[angle0, range0], [angle1, range1], ...] or null,
      "position": [x, y] (or [x, y, z]) or null
    },
    ...
  ]
}

This script is intentionally conservative: it only flips the known fields above and
leaves all other keys untouched.

Usage
  python tools/augment_lrflip.py --scenario_dir /path/to/scenario_folder

"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class Stats:
    files_processed: int = 0
    frames_processed: int = 0


def _flip_vec_y(vec: Any) -> Any:
    """Flip the second component (y) of a 2D/3D vector-like list."""
    if vec is None:
        return None
    if not isinstance(vec, list):
        return vec
    if len(vec) < 2:
        return vec
    out = vec.copy()
    try:
        out[1] = -float(out[1])
    except Exception:
        # If it's not numeric, keep as-is.
        return vec
    return out


def _flip_ang(ang: Any) -> Any:
    if ang is None:
        return None
    try:
        return -float(ang)
    except Exception:
        return ang


def _flip_scan(scan: Any) -> Any:
    """Flip scan angles: angle -> -angle; keep range; re-sort by angle asc."""
    if scan is None:
        return None
    if not isinstance(scan, list):
        return scan

    flipped: List[List[float]] = []
    for pair in scan:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        try:
            ang = -float(pair[0])
            rng = float(pair[1])
        except Exception:
            continue
        flipped.append([ang, rng])

    flipped.sort(key=lambda x: x[0])
    return flipped

def _flip_obj_pos(obj_pos: Any) -> Any:
    return _flip_vec_y(obj_pos)

def flip_prcd_data(data: Dict[str, Any], *, keep_nonstandard: bool = True) -> Dict[str, Any]:
    """Return a left-right flipped copy of a *_prcd.json dict."""
    out = deepcopy(data) if keep_nonstandard else {}

    frames = data.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Input JSON does not contain a list at key 'frames'.")

    new_frames: List[Dict[str, Any]] = []
    for fr in frames:
        if not isinstance(fr, dict):
            continue
        fr_new = deepcopy(fr) if keep_nonstandard else {}

        # Known fields
        if "position" in fr:
            fr_new["position"] = _flip_vec_y(fr.get("position"))
        if "lin_vel" in fr:
            fr_new["lin_vel"] = _flip_vec_y(fr.get("lin_vel"))
        if "lin_acc" in fr:
            fr_new["lin_acc"] = _flip_vec_y(fr.get("lin_acc"))
        if "ang_vel" in fr:
            fr_new["ang_vel"] = _flip_ang(fr.get("ang_vel"))
        if "scan" in fr:
            fr_new["scan"] = _flip_scan(fr.get("scan"))
        if "obj_pos" in fr:
            fr_new["obj_pos"] = _flip_obj_pos(fr.get("obj_pos"))

        new_frames.append(fr_new)

    out["frames"] = new_frames
    return out


def augment_folder(
    scenario_dir: Path,
    *,
    out_dir: Optional[Path] = None,
    suffix: str = "_aug_lrflip",
    overwrite: bool = False,
    skip_existing: bool = True,
) -> Stats:
    scenario_dir = scenario_dir.resolve()
    if out_dir is None:
        out_dir = scenario_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prcd_files = sorted(
        list(scenario_dir.glob("*_prcd.json")) +
        list(scenario_dir.glob("*_prcd_linear.json"))
    )
    if not prcd_files:
        raise FileNotFoundError(f"No '*_prcd.json' or '*_prcd_linear.json' found in: {scenario_dir}")


    stats = Stats()

    for src in prcd_files:
        with src.open("r", encoding="utf-8") as f:
            data = json.load(f)

        flipped = flip_prcd_data(data)

        # output name: keep *_prcd.json ending; insert suffix before it
        name = src.name
        if name.endswith("_prcd_linear.json"):
            out_name = name[: -len("_prcd_linear.json")] + f"{suffix}_prcd_linear.json"
        elif name.endswith("_prcd.json"):
            out_name = name[: -len("_prcd.json")] + f"{suffix}_prcd.json"
        else:
            out_name = src.stem + suffix + ".json"


        dst = out_dir / out_name
        if dst.exists() and not overwrite:
            if skip_existing:
                print(f"[SKIP] exists: {dst.name}")
                continue
            raise FileExistsError(
                f"Output already exists: {dst}\n"
                f"Use --overwrite to replace it, or change --suffix/--out_dir."
            )


        with dst.open("w", encoding="utf-8") as f:
            json.dump(flipped, f, indent=2, ensure_ascii=False)

        stats.files_processed += 1
        frames = flipped.get("frames")
        stats.frames_processed += len(frames) if isinstance(frames, list) else 0

        print(f"[OK] {src.name} -> {dst.name}")

    print(
        f"[DONE] files={stats.files_processed}, total_frames={stats.frames_processed}, out_dir={out_dir}"
    )
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Left-right flip augmentation for *_prcd.json")
    p.add_argument("--scenario_dir", type=str, required=True, help="Scenario folder containing *_prcd.json")
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Where to write augmented jsons (default: same as scenario_dir)",
    )
    p.add_argument(
        "--suffix",
        type=str,
        default="_aug_lrflip",
        help="Suffix inserted before '_prcd.json' (default: _aug_lrflip)",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite if output exists")

    args = p.parse_args()

    augment_folder(
        Path(args.scenario_dir),
        out_dir=Path(args.out_dir) if args.out_dir else None,
        suffix=args.suffix,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
