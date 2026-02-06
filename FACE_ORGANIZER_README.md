# Face Organizer — Offline Face-Recognition Photo Organizer

Automatically group a large personal photo collection by detected faces — similar to Google Photos face grouping — but **fully offline**, with no cloud APIs or paid services.

---

## Features

- **Offline & Private** — All processing runs locally on your machine; no data ever leaves your device.
- **Automatic Face Grouping** — Detects faces, computes 128-d embeddings, and clusters them by identity using DBSCAN.
- **Interactive Labeling** — Prompts you to name unknown face clusters; saves labels for future runs.
- **Multi-face Handling** — When a photo contains multiple faces, the largest face is used as the primary identity.
- **Copy or Move** — Keeps originals intact (`--action copy`) or relocates them (`--action move`).
- **Scalable** — Handles thousands of photos on a student laptop.

---

## Architecture

```
photos/                     ← Input: mixed photos from your phone
  ├── IMG_001.jpg
  ├── IMG_002.jpg
  └── …

        ┌──────────┐
        │  Detect   │  face_recognition (HOG / CNN)
        └────┬─────┘
             ▼
        ┌──────────┐
        │  Encode   │  128-d face embeddings (dlib ResNet)
        └────┬─────┘
             ▼
        ┌──────────┐
        │ Cluster   │  DBSCAN (scikit-learn)
        └────┬─────┘
             ▼
        ┌──────────┐
        │  Label    │  Interactive or automatic naming
        └────┬─────┘
             ▼
        ┌──────────┐
        │ Organize  │  Copy / move into per-person folders
        └──────────┘

organized/                  ← Output
  ├── Alice/
  │   ├── IMG_001.jpg
  │   └── IMG_042.jpg
  ├── Bob/
  │   └── IMG_007.jpg
  └── unknown/
      └── IMG_099.jpg
```

---

## Quick Start

### 1. Install Dependencies

```bash
# Requires Python 3.10+
# Install dlib first (needs CMake):
pip install cmake
pip install dlib

# Then install the project:
pip install -r requirements.txt
```

### 2. Run

```bash
# Basic usage — organize photos from a folder:
python main.py /path/to/photos

# With options:
python main.py /path/to/photos \
    -o organized_output \
    --detection-model hog \
    --threshold 0.5 \
    --action copy \
    --no-interactive \
    -v
```

### 3. CLI Options

| Option | Default | Description |
|---|---|---|
| `input_dir` | *(required)* | Folder containing photos |
| `-o, --output-dir` | `organized/` | Output folder for grouped photos |
| `--detection-model` | `hog` | `hog` (fast, CPU) or `cnn` (accurate, GPU) |
| `--threshold` | `0.6` | Clustering distance (0–1). Lower = stricter |
| `--action` | `copy` | `copy` or `move` files |
| `--no-interactive` | `False` | Skip interactive labeling |
| `--num-jitters` | `1` | Encoding re-samples (higher = slower, more accurate) |
| `--batch-size` | `32` | Processing batch size |
| `-v, --verbose` | `False` | Enable debug logging |

---

## Python Library Stack

| Library | Purpose | Why |
|---|---|---|
| **face_recognition** | Detection + encoding | High-level API over dlib, 99.38% accuracy on LFW |
| **dlib** | Underlying ML models | Proven CNN face detector + ResNet encoder |
| **Pillow** | Image loading/conversion | Fast, supports all common formats |
| **scikit-learn** | DBSCAN clustering | Battle-tested, no training data needed |
| **NumPy** | Numeric operations | Core array/vector math |

---

## Accuracy & Limitations

| Metric | Expected Range |
|---|---|
| Face detection recall | 95–99% (frontal), 80–90% (profile/occluded) |
| Encoding accuracy (LFW) | ~99.38% on Labeled Faces in the Wild |
| Clustering quality | Depends on threshold; 0.5–0.6 works well for most |

**Known limitations:**
- Side profiles, sunglasses, and heavy occlusion reduce detection.
- Children's faces may cluster less reliably (training data bias).
- Very low-resolution faces (< 80px) are harder to encode.
- DBSCAN may create single-person fragments if lighting varies a lot.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | Any modern dual-core | 4+ cores (for HOG parallelism) |
| RAM | 4 GB | 8+ GB for large collections |
| GPU | Not required (HOG) | NVIDIA + CUDA (for CNN model) |
| Disk | Enough for photos + copies | SSD recommended |

**Performance tips:**
- Use `--detection-model hog` for CPU-only machines (default).
- Increase `--num-jitters` for better accuracy at the cost of speed.
- Pre-sort very large collections into batches of ~1,000.

---

## Project Structure

```
face_organizer/
├── __init__.py          # Package metadata
├── config.py            # Centralized configuration dataclass
├── utils.py             # Image I/O, validation, helpers
├── detector.py          # Face detection (HOG/CNN)
├── encoder.py           # 128-d face encoding
├── cluster.py           # DBSCAN clustering
├── labeler.py           # Interactive face labeling
├── organizer.py         # File copy/move into person folders
├── pipeline.py          # End-to-end orchestration
└── cli.py               # Command-line interface
tests/
├── test_config.py
├── test_utils.py
├── test_cluster.py
├── test_organizer.py
├── test_labeler.py
├── test_encoder.py
└── test_cli.py
main.py                  # Top-level entry point
requirements.txt
pyproject.toml
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Evaluation & Improvement

1. **Manual spot-check:** Review 50–100 photos per cluster for purity.
2. **Adjust threshold:** Lower `--threshold` (e.g., 0.45) for stricter matching if clusters merge different people.
3. **Increase jitters:** Use `--num-jitters 10` for more accurate encodings.
4. **Use CNN model:** Switch to `--detection-model cnn` if you have a GPU.
5. **Re-label:** Re-run with interactive mode to correct mis-clustered faces.

---

## Future Upgrades

- **Video support** — Extract frames from video files and process them.
- **Duplicate detection** — Perceptual hashing to find and remove duplicate photos.
- **Web GUI** — Flask/Streamlit dashboard for browsing and labeling faces visually.
- **Incremental processing** — Cache encodings to avoid reprocessing already-scanned photos.
- **Age/gender estimation** — Add demographic metadata per cluster.
- **EXIF-based sorting** — Combine face identity with date/location metadata.

---

## License

MIT
