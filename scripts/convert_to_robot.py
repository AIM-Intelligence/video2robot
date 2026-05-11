#!/usr/bin/env python3
"""
Convert SMPL-X motion to robot motion using GMR

Usage:
    # From project folder
    python scripts/convert_to_robot.py --project data/video_001

    # Skip TWIST compatibility (default: generate both)
    python scripts/convert_to_robot.py --project data/video_001 --no-twist

    # From SMPL-X file (creates new project)
    python scripts/convert_to_robot.py --smplx poses.npz
"""

import argparse
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from video2robot.robot import RobotRetargeter
from video2robot.config import DATA_DIR
from video2robot.pose.tracks import TrackInfo, get_smplx_tracks, get_track_by_index
from video2robot.utils import get_next_project_dir, emit_progress


def prepare_smplx_betas_for_gmr(smplx_path: Path, tmp_dir: Path, target_num_betas: int = 16) -> Path:
    """Create a temporary SMPL-X file with betas padded/truncated for GMR."""
    with np.load(smplx_path, allow_pickle=True) as smplx_data:
        data = {key: smplx_data[key] for key in smplx_data.files}

    betas = np.asarray(data["betas"], dtype=np.float32).reshape(-1)
    if betas.shape[0] == target_num_betas:
        return smplx_path

    if betas.shape[0] < target_num_betas:
        adjusted_betas = np.pad(betas, (0, target_num_betas - betas.shape[0]))
    else:
        adjusted_betas = betas[:target_num_betas]

    data["betas"] = adjusted_betas.astype(np.float32)

    tmp = tempfile.NamedTemporaryFile(
        prefix=f"{smplx_path.stem}_",
        suffix=".npz",
        dir=tmp_dir,
        delete=False,
    )
    tmp_path = Path(tmp.name)
    tmp.close()
    np.savez(tmp_path, **data)
    print(f"[SMPL-X] Adjusted betas for GMR: {betas.shape[0]} → {target_num_betas}")
    return tmp_path


def main():
    parser = argparse.ArgumentParser(description="Convert SMPL-X to robot motion")
    
    # Input options
    parser.add_argument("--project", "-p", help="Project folder path (e.g., data/video_001)")
    parser.add_argument("--smplx", "-s", help="SMPL-X .npz file path (creates new project)")
    
    # Robot options
    parser.add_argument("--robot", "-r", default="unitree_g1", help="Target robot type")
    parser.add_argument("--fps", type=int, default=0, help="Target FPS (0 = keep original, recommended)")
    parser.add_argument("--visualize", action="store_true", help="Show MuJoCo visualization")
    parser.add_argument("--track-index", type=int, help="Retarget only this track index (default: first track unless --all-tracks)")
    parser.add_argument("--all-tracks", action="store_true", help="Retarget every available SMPL-X track in the project")
    
    # TWIST options
    parser.add_argument("--no-twist", action="store_true",
                        help="Skip TWIST-compatible motion generation (default: generate TWIST)")
    
    # Other options
    parser.add_argument("--name", "-n", help="Project name (when using --smplx)")
    parser.add_argument("--list-robots", action="store_true", help="List supported robots")

    args = parser.parse_args()

    if args.all_tracks and args.track_index is not None:
        parser.error("--track-index cannot be used together with --all-tracks")
    if args.track_index is not None and args.track_index < 1:
        parser.error("--track-index must be >= 1")

    if args.list_robots:
        print("Supported robots:")
        for robot in RobotRetargeter.get_supported_robots():
            print(f"  - {robot}")
        return

    if not args.project and not args.smplx:
        parser.error("Provide --project or --smplx")

    tracks: list[TrackInfo] = []

    # Determine project directory
    if args.project:
        project_dir = Path(args.project)
        if not project_dir.exists():
            parser.error(f"Project not found: {project_dir}")
        tracks = get_smplx_tracks(project_dir)
        if not tracks:
            parser.error(f"No SMPL-X tracks found in {project_dir}. Run extract_pose.py first.")
    else:
        # Create new project from SMPL-X file
        smplx_path = Path(args.smplx)
        if not smplx_path.exists():
            parser.error(f"SMPL-X not found: {smplx_path}")
        
        if args.name:
            project_dir = DATA_DIR / args.name
        else:
            project_dir = get_next_project_dir()
        
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy SMPL-X to project
        dst_smplx = project_dir / "smplx.npz"
        shutil.copy(smplx_path, dst_smplx)
        smplx_path = dst_smplx
        print(f"[Project] Created: {project_dir}")
        tracks = [TrackInfo(index=1, smplx_path=smplx_path)]

    if not tracks:
        parser.error("No SMPL-X tracks available for retargeting.")

    if args.all_tracks:
        selected_tracks = tracks
    elif args.track_index is not None:
        track = get_track_by_index(tracks, args.track_index)
        if track is None:
            parser.error(f"Track index {args.track_index} not found. Available tracks: {[t.index for t in tracks]}")
        selected_tracks = [track]
    else:
        selected_tracks = [tracks[0]]

    print(f"[Project] {project_dir}")
    print(f"[Robot]   {args.robot}")
    print(f"[Tracks]  {', '.join(str(track.index) for track in selected_tracks)}")
    emit_progress("init", 0.02, "Initializing")

    retargeter = RobotRetargeter(robot_type=args.robot)
    motion_paths: dict[int, Path] = {}
    twist_paths: dict[int, Path] = {}
    temp_smplx_dir = tempfile.TemporaryDirectory(prefix="video2robot_gmr_")
    temp_smplx_path = Path(temp_smplx_dir.name)

    for track in selected_tracks:
        smplx_path = track.smplx_path
        if not smplx_path.exists():
            print(f"[Warning] SMPL-X not found for track {track.index}: {smplx_path}")
            continue

        output_path = project_dir / f"robot_motion_track_{track.index}.pkl"
        print(f"\n[Robot] Track #{track.index}: {smplx_path.name} → {output_path.name}")
        if track.track_id:
            print(f"[Robot] Track ID: {track.track_id}")

        gmr_smplx_path = prepare_smplx_betas_for_gmr(smplx_path, temp_smplx_path)
        retargeter.retarget(
            smplx_path=gmr_smplx_path,
            output_path=output_path,
            target_fps=args.fps,
            visualize=args.visualize,
        )
        motion_paths[track.index] = output_path

        if not args.no_twist and args.robot == "unitree_g1":
            twist_output_path = project_dir / f"robot_motion_track_{track.index}_twist.pkl"
            emit_progress("twist", 0.90, f"TWIST conversion (track {track.index})")
            print(f"\n[TWIST] Track #{track.index}: Converting 29 DOF → 23 DOF...")

            with open(output_path, "rb") as f:
                robot_motion = pickle.load(f)

            keep_indices = list(range(19)) + list(range(22, 26))
            twist_motion = {
                "fps": float(robot_motion["fps"]),
                "robot_type": robot_motion["robot_type"],
                "num_frames": int(robot_motion["num_frames"]),
                "human_height": float(robot_motion["human_height"]),
                "root_pos": robot_motion["root_pos"].tolist(),
                "root_rot": robot_motion["root_rot"].tolist(),
                "dof_pos": robot_motion["dof_pos"][:, keep_indices].tolist(),
                "local_body_pos": robot_motion["local_body_pos"].tolist(),
                "link_body_list": robot_motion["link_body_list"],
            }

            dof_shape = robot_motion["dof_pos"][:, keep_indices].shape
            with open(twist_output_path, "wb") as f:
                pickle.dump(twist_motion, f, protocol=2)

            print(f"[TWIST] Track #{track.index} dof_pos: {dof_shape}")
            print(f"[TWIST] Saved: {twist_output_path}")
            twist_paths[track.index] = twist_output_path

    if not motion_paths:
        raise SystemExit("No robot motions were generated.")

    # Maintain legacy aliases for track 1 if available.
    track1_motion = motion_paths.get(1)
    if track1_motion and track1_motion.exists():
        default_robot_motion = project_dir / "robot_motion.pkl"
        shutil.copy(track1_motion, default_robot_motion)
        print(f"\n[Robot] Default alias updated -> {default_robot_motion.name}")

    if not args.no_twist and args.robot == "unitree_g1":
        track1_twist = twist_paths.get(1)
        if track1_twist and track1_twist.exists():
            default_twist = project_dir / "robot_motion_twist.pkl"
            shutil.copy(track1_twist, default_twist)
            print(f"[TWIST] Default alias updated -> {default_twist.name}")

    emit_progress("done", 1.0, "Done")
    print(f"\nDone!")
    print(f"  Project: {project_dir}")
    for idx in sorted(motion_paths.keys()):
        print(f"  Track #{idx}: {motion_paths[idx].name}")
        if idx in twist_paths:
            print(f"             {twist_paths[idx].name}")


if __name__ == "__main__":
    main()
