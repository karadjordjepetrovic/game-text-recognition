# Game Text Recognition

Detect on-screen text in game frames and separate **HUD text** (UI overlays such as objectives, subtitles, compass, ammo counters) from **in-world text** (signs, posters, labels that are part of the 3D scene).

The end goal is a per-frame **HUD mask**: a pixel mask of everything that is drawn on top of the scene rather than being part of it. That mask feeds into monocular depth estimation work (see [realtime-depth-optimization](https://github.com/karadjordjepetrovic/realtime-depth-optimization)), where HUD overlays produce wrong depth values and distort error metrics.

## Status

> **This is the first, deliberately simple baseline. It runs on CPU and is slow: about 15 seconds per 1080p frame.**
> The following commits will make it fast (GPU inference, smaller models, reduced detection resolution, TensorRT) and add the HUD vs in-world classification.

What works today:

- Extracting frames from a gameplay video
- Running OCR (text detection + recognition) on every frame with [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- Saving per-frame results: boxed preview images, raw JSON, and one CSV with every detected text region
- A timing breakdown per stage (model loading, inference, saving)

### Why the baseline is slow

| Factor | Effect |
|---|---|
| CPU inference instead of GPU | The largest cost by far |
| oneDNN (Intel CPU acceleration) disabled | Required to avoid a crash in recent PaddlePaddle CPU builds on Windows |
| "Medium" accuracy models on full 1080p frames | Accurate but heavy |

Writing the output files is a small fraction of the time. Almost all of it is model inference.

## Project layout

```
game-text-recognition/
├── examples/                ← a few sample results shown in this README
├── data/
│   ├── gameplay.mp4         ← your recording (not in git)
│   └── frames/              ← extracted frames (not in git)
├── outputs/ocr/             ← results (not in git)
│   ├── images/              ← frames with detected text boxed
│   ├── json/                ← raw OCR output per frame
│   └── summary.csv          ← one row per detected text region
├── scripts/
│   ├── extract_frames.py    ← video → PNG frames (OpenCV)
│   └── run_ocr.py           ← OCR over all frames + timing report
└── requirements.txt
```

The full gameplay footage and outputs are not included in the repository (size and copyright). `examples/` holds a few sample frames and the CSV from one run. Use your own recordings to reproduce.

## Setup (Windows)

1. Install [Miniconda](https://docs.anaconda.com/miniconda/) and open **Anaconda Prompt**.
2. Use the free conda-forge channel:
   ```
   conda config --add channels conda-forge
   conda config --remove channels defaults
   conda config --set channel_priority strict
   ```
3. Create and activate an isolated environment:
   ```
   conda create -n hudmask python=3.11 -y
   conda activate hudmask
   ```
4. Install the packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

From the project root, with the `hudmask` environment active:

1. Put a gameplay recording at `data/gameplay.mp4`, then extract frames (1 per second, max 50):
   ```
   python scripts/extract_frames.py
   ```
   Options: `--fps 2`, `--max-frames 100`, `--width 1920` (downscale), `--video path/to/other.mp4`.

2. Run OCR:
   ```
   python scripts/run_ocr.py
   ```
   The first run downloads the OCR models (about 100 MB). At the end the script prints a timing breakdown.

   Speed options (being evaluated in the next commits):
   - `--device gpu`: GPU inference (requires `paddlepaddle-gpu`)
   - `--fast`: small mobile models
   - `--det-size 960`: run detection on a downscaled frame
   - `--mkldnn`: re-enable oneDNN on PaddlePaddle versions without the crash
   - `--no-images --no-json`: only write the CSV

## Example output

Detected text boxed on sample frames. HUD text (objective, subtitles, compass, counter) and in-world text (signs, notices) are both found; telling them apart is the next step.

![HUD objective, subtitles and in-world signs](examples/example_1.jpg)

![In-world notices while the HUD stays fixed](examples/example_2.jpg)

All detected text regions from the run: [`examples/summary.csv`](examples/summary.csv)

## Early observations

From a first run over 50 frames of a first-person shooter:

- HUD text is consistently detected and read correctly: objectives, subtitles, compass letters and counters.
- HUD elements appear at the **same screen position across many frames**, while in-world text (signs, notices) moves as the camera moves. This is the main signal for the HUD vs in-world classification.
- Low-confidence detections are mostly textures misread as text, and a higher confidence threshold removes most of them.

## Roadmap

- [x] CPU OCR baseline with timing breakdown
- [ ] GPU inference and a speed comparison table (CPU vs GPU, model size, detection resolution)
- [ ] Track text boxes across frames
- [ ] Classify HUD vs in-world text (screen-space position stability, optical flow vs camera motion, geometry)
- [ ] Static-pixel mask for non-text HUD (minimap, crosshair, icons)
- [ ] Combined per-frame HUD mask with temporal smoothing
- [ ] Ground-truth evaluation with Unreal Engine 5 captures (frames rendered with and without UI)
- [ ] Use the mask in depth estimation: exclude from metrics, inpaint before inference
- [ ] TensorRT optimization of the slowest stage
