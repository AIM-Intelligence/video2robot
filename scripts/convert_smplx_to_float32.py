#!/usr/bin/env python3
"""
Convert SMPL-X models from float64 to float32 format.

This script converts SMPL-X .npz model files from float64 (double precision) to 
float32 (single precision), which:
- Reduces file size by approximately 50%
- Improves compatibility with PyTorch and other frameworks
- Maintains sufficient precision for most use cases

Usage:
    python scripts/convert_smplx_to_float32.py \
        --input_dir third_party/PromptHMR/data/body_models/smplx \
        --output_dir third_party/PromptHMR/data/body_models/smplx

    # Convert in-place (overwrites original files)
    python scripts/convert_smplx_to_float32.py \
        --input_dir third_party/PromptHMR/data/body_models/smplx \
        --in_place

    # Backup originals before converting
    python scripts/convert_smplx_to_float32.py \
        --input_dir third_party/PromptHMR/data/body_models/smplx \
        --output_dir third_party/PromptHMR/data/body_models/smplx \
        --backup
"""
import os
import os.path as osp
import argparse
import numpy as np
import shutil
from pathlib import Path


def convert_npz_to_float32(in_path, out_path):
    """
    Convert an NPZ file from float64 to float32 format.
    
    Args:
        in_path: Input NPZ file path
        out_path: Output NPZ file path
    """
    if not osp.exists(in_path):
        print(f"Warning: Input file {in_path} does not exist, skipping")
        return False
    
    print(f"Processing {osp.basename(in_path)} -> {osp.basename(out_path)}")
    in_npz = np.load(in_path, allow_pickle=True)
    out_dict = {}
    converted_count = 0
    
    for key, val in in_npz.items():
        # Handle special cases for UV and face texture indices
        if key in ["vt", "ft"]:
            if isinstance(val, np.ndarray):
                if val.dtype == object:
                    # Try to convert object arrays
                    try:
                        val = np.array(val.tolist(), dtype=np.uint32)
                        converted_count += 1
                    except Exception as e:
                        print(f"  Warning: Could not convert {key} from object: {e}")
                        continue
                elif val.dtype == np.float64:
                    val = np.float32(val)
                    converted_count += 1
                elif val.dtype == np.int64:
                    val = np.int32(val)
                    converted_count += 1
            out_dict[key] = val
            continue
        
        # Convert numpy arrays
        if isinstance(val, np.ndarray):
            original_dtype = val.dtype
            if val.dtype == np.float64:
                val = np.float32(val)
                converted_count += 1
            elif val.dtype == np.int64:
                val = np.int32(val)
                converted_count += 1
            elif val.dtype == np.uint64:
                val = np.uint32(val)
                converted_count += 1
            elif val.dtype == object:
                # Try to handle object arrays
                try:
                    # For most object arrays in SMPLX, they're usually lists/arrays
                    val = np.array(val.tolist())
                    if val.dtype == np.float64:
                        val = np.float32(val)
                        converted_count += 1
                    elif val.dtype == np.int64:
                        val = np.int32(val)
                        converted_count += 1
                except Exception as e:
                    print(f"  Warning: Could not convert {key} from object: {e}")
                    # Skip problematic keys
                    continue
            out_dict[key] = val
        else:
            # Handle non-array values (scalars, strings, etc.)
            if isinstance(val, (float, np.floating)) and val.dtype == np.float64:
                out_dict[key] = np.float32(val)
                converted_count += 1
            else:
                out_dict[key] = val
    
    # Save as compressed NPZ
    np.savez_compressed(out_path, **out_dict)
    
    # Get file sizes
    in_size = osp.getsize(in_path) / (1024 * 1024)  # MB
    out_size = osp.getsize(out_path) / (1024 * 1024)  # MB
    reduction = ((in_size - out_size) / in_size * 100) if in_size > 0 else 0
    
    print(f"  ✓ Converted {converted_count} arrays/values")
    print(f"  ✓ Size: {in_size:.2f} MB -> {out_size:.2f} MB ({reduction:.1f}% reduction)")
    print(f"  ✓ Saved to {out_path}\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert SMPL-X models from float64 to float32 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing SMPL-X .npz files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as input_dir if --in_place not set)"
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Convert files in-place (overwrites original files)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of original files before converting (only with --in_place)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.npz",
        help="File pattern to match (default: *.npz)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1
    
    # Determine output directory
    if args.in_place:
        output_dir = input_dir
        if args.backup:
            backup_dir = input_dir / "backup_float64"
            backup_dir.mkdir(exist_ok=True)
            print(f"Backup directory: {backup_dir}")
    else:
        output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    npz_files = list(input_dir.glob(args.pattern))
    if not npz_files:
        print(f"No {args.pattern} files found in {input_dir}")
        return 1
    
    print("=" * 60)
    print("SMPL-X Float32 Conversion")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Files to convert: {len(npz_files)}")
    if args.in_place and args.backup:
        print(f"Backup directory: {backup_dir}")
    print("=" * 60)
    print()
    
    converted = 0
    for npz_file in npz_files:
        if args.in_place:
            # Create backup if requested
            if args.backup:
                backup_path = backup_dir / npz_file.name
                shutil.copy2(npz_file, backup_path)
                print(f"Backed up: {npz_file.name} -> backup_float64/{npz_file.name}")
            
            # Convert in-place (use temporary file to avoid corruption)
            temp_path = npz_file.with_suffix('.npz.tmp')
            if convert_npz_to_float32(npz_file, temp_path):
                shutil.move(temp_path, npz_file)
                converted += 1
        else:
            # Convert to output directory
            output_path = output_dir / npz_file.name
            if convert_npz_to_float32(npz_file, output_path):
                converted += 1
    
    print("=" * 60)
    print(f"Conversion complete! Converted {converted}/{len(npz_files)} files")
    print("=" * 60)
    
    if args.in_place and args.backup:
        print(f"\nOriginal files backed up to: {backup_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())

