# video2robot

End-to-end pipeline: Video (or Prompt) → Human Pose Extraction → Robot Motion Conversion

## Demo

<p align="center">
<video src="https://github.com/user-attachments/assets/a0f1bfb1-7e06-4672-8f6a-320ab60b0bfe" width="800" controls></video>
</p>
<p align="center"><b>Demo Video</b></p>

<table>
<tr>
<td align="center" width="50%">
<video src="https://github.com/user-attachments/assets/1d58bac8-173c-499d-b245-65013371d50f" width="400" controls></video>
<br><b>Backflip</b>
</td>
<td align="center" width="50%">
<video src="https://github.com/user-attachments/assets/94e6d12d-afae-4300-8c5c-c244ad208bdb" width="400" controls></video>
<br><b>Dance Motion</b>
</td>
</tr>
</table>

## Pipeline

```
[Prompt] → Veo → [Video] → PromptHMR → [SMPL-X] → GMR → [Robot Motion]
```

## Project Structure

```
video2robot/
├── video2robot/            # Main package
│   ├── config.py           # Configuration management
│   ├── pipeline.py         # (Optional) Python API pipeline
│   ├── cli.py              # Console entrypoint for installation
│   ├── video/              # Video generation/processing
│   │   └── veo_client.py   # Google Veo API
│   ├── pose/               # Pose extraction (PromptHMR wrapper)
│   │   └── extractor.py
│   └── robot/              # Robot conversion (GMR wrapper)
│       └── retargeter.py
│
├── scripts/                # CLI scripts
│   ├── run_pipeline.py     # Full pipeline
│   ├── generate_video.py   # Veo video generation
│   ├── extract_pose.py     # Pose extraction
│   └── convert_to_robot.py # Robot conversion
│   └── visualize.py        # Result visualization
│
├── configs/                # Configuration files
├── data/                   # Data (gitignored)
│
└── third_party/            # External dependencies (submodules)
    ├── PromptHMR/          # Pose extraction model
    └── GMR/                # Motion retargeting
```

## Installation

This project requires **two conda environments**: `gmr` and `phmr`.

```bash
# Clone repo (with submodules)
git clone --recursive https://github.com/AIM-Intelligence/video2robot.git
cd video2robot

# Or initialize submodules after cloning
git submodule update --init --recursive
```

### 1. GMR Environment (Robot Retargeting)

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e .
```

For details, see [GMR README](third_party/GMR/README.md).

### 2. PromptHMR Environment (Pose Extraction)

**For Blackwell GPU (sm_120) users:**
```bash
conda create -n phmr python=3.11 -y
conda activate phmr
cd third_party/PromptHMR
bash scripts/install_blackwell.sh
```

**For other GPUs (Ampere, Hopper, etc.):**
```bash
conda create -n phmr python=3.10 -y
conda activate phmr
cd third_party/PromptHMR
pip install -e .
```

For details, see [PromptHMR README](third_party/PromptHMR/README.md).

## Data Download and Setup

After installing the environments, you need to download the required model files and data.

### Download PromptHMR Data

From the `third_party/PromptHMR` directory:

```bash
cd third_party/PromptHMR
bash scripts/fetch_data.sh
```

This script downloads:
- PromptHMR checkpoints
- Third-party checkpoints (camcalib, droidcalib, vitpose, SAM, SAM2, YOLO, Detectron2)
- Dataset annotations (for evaluation)
- Example images

### Download SMPL-X and SMPL Body Models

From the `third_party/PromptHMR` directory:

```bash
cd third_party/PromptHMR
bash scripts/fetch_smplx.sh
```

This script requires registration at:
- [SMPL-X](https://smpl-x.is.tue.mpg.de) - for SMPL-X models
- [SMPL](https://smpl.is.tue.mpg.de) - for SMPL models

The script automatically downloads, extracts, and reorganizes the models into the correct directory structure:
- SMPL-X models → `data/body_models/smplx/`
- SMPL models → `data/body_models/smpl/`
- Accessory files → `data/body_models/accessory/`

### Convert SMPL-X Models to Float32 (Required)

**Important:** The downloaded SMPL-X models use `float64` (double precision) which can cause compatibility issues with PyTorch and other frameworks. **You must convert them to `float32` (single precision) after downloading.** This also reduces file size by ~50%.

Convert downloaded models to float32:

```bash
# From the project root directory
python scripts/convert_smplx_to_float32.py \
    --input_dir third_party/PromptHMR/data/body_models/smplx \
    --output_dir third_party/PromptHMR/data/body_models/smplx
```

Or convert in-place (overwrites originals - backup recommended):

```bash
# Backup before converting
cp -r third_party/PromptHMR/data/body_models/smplx third_party/PromptHMR/data/body_models/smplx_backup

# Convert in-place
python scripts/convert_smplx_to_float32.py \
    --input_dir third_party/PromptHMR/data/body_models/smplx \
    --in_place
```

The conversion script:
- Converts all `.npz` files from `float64` to `float32`
- Converts `int64` to `int32` and `uint64` to `uint32` for consistency
- Handles object arrays and nested structures
- Creates compressed output files
- Preserves original files (unless using `--in_place`)

## Usage

> **Note**: Scripts automatically switch to the appropriate conda environment (`gmr` or `phmr`) as needed. Just ensure both environments are installed - no need to manually activate them.

```bash
# Full pipeline (action → robot motion) - BASE_PROMPT auto-applied
python scripts/run_pipeline.py --action "Action sequence:
The subject walks forward with four steps."

# Use Sora
python scripts/run_pipeline.py --action "..." --provider sora

# Start from existing video (video.mp4 → robot motion)
python scripts/run_pipeline.py --video /path/to/video.mp4

# Resume from existing project
python scripts/run_pipeline.py --project data/video_001

# Run individual steps
python scripts/generate_video.py --action "Action sequence: The subject walks forward."
python scripts/extract_pose.py --project data/video_001
python scripts/convert_to_robot.py --project data/video_001

# Visualization (auto env switching)
python scripts/visualize.py --project data/video_001
python scripts/visualize.py --project data/video_001 --pose
python scripts/visualize.py --project data/video_001 --robot
```

## Web UI

```bash
# Run server (from video2robot root)
uvicorn web.app:app --host 0.0.0.0 --port 8000

# Access in browser
# http://localhost:8000
```

Features:
- Prompt input → Video generation → Pose extraction → Robot conversion automatic pipeline
- Video upload support
- Veo/Sora model selection
- 3D visualization (viser)
- Video-3D synchronized playback

## Environment Setup

```bash
# Create .env file
cp .env.example .env

# Set API key
echo "GOOGLE_API_KEY=your-api-key" >> .env
```

## Supported Robots

| Robot | ID | DOF |
|-------|-----|-----|
| Unitree G1 | `unitree_g1` | 29 |
| Unitree H1 | `unitree_h1` | 19 |
| Booster T1 | `booster_t1` | 23 |

See [GMR README](third_party/GMR/README.md) for full list

## Output Format

```python
# robot_motion.pkl
{
    "fps": 30.0,
    "robot_type": "unitree_g1",
    "num_frames": 240,
    "root_pos": np.ndarray,    # (N, 3)
    "root_rot": np.ndarray,    # (N, 4) quaternion xyzw
    "dof_pos": np.ndarray,     # (N, DOF)
}
```


## TODO

- [ ] **`lastFrame` (Start/End Frame Interpolation)** - Veo 3.1 only
  - Start image + End image → Generate video smoothly connecting the two
  - Useful for "Pose A → Pose B" robot motion videos

- [ ] **`referenceImages` (Reference Images)** - Veo 3.1 only
  - Up to 3 reference images to maintain character/style
  - Generate videos with specific character performing actions

## Acknowledgements

This project builds upon the following excellent open source projects:

- [PromptHMR](https://github.com/yufu-wang/PromptHMR): 3D human mesh recovery from video
- [GMR](https://github.com/YanjieZe/GMR): general motion retargeting framework

## License

This project depends on third-party libraries with their own licenses:

- **[GMR](third_party/GMR/LICENSE)**: MIT License
- **[PromptHMR](third_party/PromptHMR/LICENSE)**: Non-Commercial Scientific Research Use Only

Please review both licenses before use.

> The core video2robot code is MIT-licensed, but using this
> repository end-to-end (including PromptHMR) inherits PromptHMR's
> **Non-Commercial Scientific Research Only** restriction. Commercial use
> requires obtaining appropriate permission from the PromptHMR authors.
