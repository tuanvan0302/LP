"""
This script is used to create the real dataset for evaluation.
Input: video car/truck.mp4 + labels from data/real_dataset/real_data.txt
Output: 15 images extracted from each video and their corresponding labels.
Saved in data/real_dataset/images/ and data/real_dataset/labels/

Flow: Video -> process each frame -> detect plate with YOLO tracking -> list plates by track ID
-> split top 3 tracks, each track selects 5 frames -> 15 images + label per video

Label file format (real_data.txt):
    0. 51C 337.84   -> video truck_0.mp4, label "51C 337.84"
Skip entries where label is "na" or has multiple values (comma-separated).
"""

import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# ─── Configuration ────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT_DIR / "data" / "truck_video"
LABEL_FILE = ROOT_DIR / "data" / "real_dataset" / "real_data.txt"
WEIGHTS_PATH = ROOT_DIR / "weights_detect_plate_v2.pt"
OUTPUT_DIR = ROOT_DIR / "data" / "real_dataset"

IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"

NUM_TRACKS = 3           # number of unique tracks to select per video
NUM_SAMPLES_PER_TRACK = 5 # number of frames to sample per track
FRAME_STEP = 1            # process every Nth frame for speed
CONF_THRESHOLD = 0.5      # confidence threshold for detections


def parse_label_file(label_file_path: str) -> list[tuple[int, str]]:
    """Parse real_data.txt, return list of (video_index, label) for valid entries.

    Skip rules:
    - Empty lines
    - Label is "na" (case-insensitive)
    - Label contains comma (multiple values)
    """
    valid_entries = []

    with open(label_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse: "0. 51C 337.84"
            parts = line.split(". ", 1)
            if len(parts) != 2:
                continue

            try:
                idx = int(parts[0])
            except ValueError:
                continue

            label = parts[1].strip()

            # Skip "na"
            if label.lower() == "na":
                print(f"  [SKIP] Entry {idx}: label is 'na'")
                continue

            # Skip multiple values (comma-separated)
            if "," in label:
                print(f"  [SKIP] Entry {idx}: multiple values detected -> '{label}'")
                continue

            valid_entries.append((idx, label))

    return valid_entries


def process_video(video_path: Path, model: YOLO) -> list[dict]:
    """Process a video with YOLO tracking, return list of plate detections."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    print(f"  Duration: {duration:.1f}s, {total_frames} frames @ {fps:.1f} fps")

    results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every FRAME_STEP frames for efficiency
        if frame_idx % FRAME_STEP == 0:
            # Run YOLO with tracking (BoT-SORT tracker)
            detections = model.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                verbose=False,
                conf=CONF_THRESHOLD,
            )

            if detections[0].boxes is not None and detections[0].boxes.id is not None:
                boxes = detections[0].boxes.xyxy.cpu().numpy()
                track_ids = detections[0].boxes.id.cpu().numpy().astype(int)
                confs = detections[0].boxes.conf.cpu().numpy()

                for box, tid, conf in zip(boxes, track_ids, confs):
                    x1, y1, x2, y2 = box.astype(int)
                    # Clamp to frame boundaries
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                    plate_crop = frame[y1:y2, x1:x2]
                    if plate_crop.size > 0 and plate_crop.shape[0] > 10 and plate_crop.shape[1] > 10:
                        results.append({
                            "track_id": tid,
                            "frame_idx": frame_idx,
                            "image": plate_crop.copy(),
                            "conf": float(conf),
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        })

        frame_idx += 1

        # Progress
        if frame_idx % 1000 == 0:
            pct = frame_idx / total_frames * 100
            print(f"    Processed {frame_idx}/{total_frames} ({pct:.0f}%)...")

    cap.release()
    print(f"  Total detections: {len(results)}")
    return results


def select_samples(results: list[dict]) -> list[dict]:
    """Split the sorted detection list into NUM_TRACKS segments, sample NUM_SAMPLES_PER_TRACK from each."""
    if not results:
        return []

    # Sort all detections by frame index (chronological order)
    results.sort(key=lambda x: x["frame_idx"])
    total = len(results)
    print(f"  Total detections: {total}")

    # Split into NUM_TRACKS equal segments
    segments = np.array_split(range(total), NUM_TRACKS)

    selected = []
    for seg_idx, seg_indices in enumerate(segments):
        seg_detections = [results[i] for i in seg_indices]
        n = len(seg_detections)

        # Evenly sample NUM_SAMPLES_PER_TRACK frames from this segment
        if n <= NUM_SAMPLES_PER_TRACK:
            samples = seg_detections
        else:
            indices = np.linspace(0, n - 1, NUM_SAMPLES_PER_TRACK, dtype=int)
            samples = [seg_detections[i] for i in indices]

        print(f"    Segment {seg_idx + 1}: {n} detections -> {len(samples)} samples "
              f"(frames {seg_detections[0]['frame_idx']}-{seg_detections[-1]['frame_idx']})")
        selected.extend(samples)

    return selected


def save_samples(selected: list[dict], video_idx: int, label: str) -> None:
    """Save selected plate images and write label file."""
    images_dir = IMAGES_DIR
    labels_dir = LABELS_DIR
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    if not selected:
        print(f"  [WARN] No samples to save for video {video_idx}")
        return

    # Save images
    image_paths = []
    for i, sample in enumerate(selected):
        filename = f"real_{video_idx}_image_{i + 1}.png"
        filepath = images_dir / filename
        cv2.imwrite(str(filepath), sample["image"])
        image_paths.append(filename)

    # Save label file (one label line, same label for all samples from this video)
    label_filename = f"real_{video_idx}_label.txt"
    label_filepath = labels_dir / label_filename
    with open(label_filepath, "w") as f:
        for img_name in image_paths:
            f.write(f"{img_name} {label}\n")

    print(f"  Saved {len(selected)} images + label file '{label_filename}' for '{label}'")
    print(f"    Images: {', '.join(image_paths[:3])}..." if len(image_paths) > 3 else f"    Images: {', '.join(image_paths)}")


def main():
    print("=" * 60)
    print("CREATE REAL DATASET FROM VIDEOS")
    print("=" * 60)

    # ─── Step 1: Parse labels ──────────────────────────────────
    print("\n[1] Parsing label file...")
    if not LABEL_FILE.exists():
        print(f"  [ERROR] Label file not found: {LABEL_FILE}")
        return

    entries = parse_label_file(str(LABEL_FILE))
    print(f"  Valid entries: {len(entries)}")
    for idx, label in entries:
        print(f"    Video truck_{idx}.mp4 -> '{label}'")

    # ─── Step 2: Load YOLO model ───────────────────────────────
    print("\n[2] Loading YOLO model...")
    if not WEIGHTS_PATH.exists():
        print(f"  [ERROR] Weights file not found: {WEIGHTS_PATH}")
        return

    model = YOLO(str(WEIGHTS_PATH))
    print(f"  Model loaded: {WEIGHTS_PATH.name}")

    # ─── Step 3: Process each video ────────────────────────────
    print(f"\n[3] Processing {len(entries)} videos...")
    total_saved = 0
    videos_with_issues = []

    for video_idx, label in entries:
        video_path = VIDEO_DIR / f"truck_{video_idx}.mp4"
        print(f"\n  ─── Video truck_{video_idx}.mp4 | Label: '{label}' ───")

        if not video_path.exists():
            print(f"  [WARN] Video not found: {video_path}")
            videos_with_issues.append(video_idx)
            continue

        # Process video → detect plates with tracking
        all_detections = process_video(video_path, model)

        # Select representative samples
        selected = select_samples(all_detections)

        # Save results
        save_samples(selected, video_idx, label)
        total_saved += len(selected)

    # ─── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Videos processed: {len(entries)}")
    print(f"  Total images saved: {total_saved}")
    print(f"  Images directory: {IMAGES_DIR}")
    print(f"  Labels directory: {LABELS_DIR}")
    if videos_with_issues:
        print(f"  Videos with issues (not found): {videos_with_issues}")
    print("Done!")


if __name__ == "__main__":
    main() 



