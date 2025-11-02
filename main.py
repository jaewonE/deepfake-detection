"""Run LAA-Net inference on folders of images and videos.

Images and video frames are first processed through a YuNet-based face
alignment pipeline that expands the detected face, applies aspect-ratio
preserving letterbox padding, and optionally tracks faces across frames when
detections drop out. The resulting square crops reuse the normalization metadata
from ``scripts/test.py`` before being forwarded through the network.

Helper routines:

``prepare_model`` loads the YAML configuration, builds the network, restores
weights, and prepares the normalization transforms.

``preprocess_image`` detects and standardizes a single RGB image ahead of the
torch preprocessing chain.

``preprocess_frame`` performs the same detection pipeline on BGR video frames
while smoothing boxes over time and falling back to tracking when possible.

``run_inference`` enumerates the ``data`` directory, runs preprocessing, saves
optional face crops, and reports logits alongside auxiliary face metadata.
"""

import argparse
import csv
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from configs.get_config import load_config
from losses.losses import _sigmoid
from models import MODELS, build_model, load_pretrained
from package_utils.face_processing import FacePreprocessor, FacePreprocessingConfig
from package_utils.image_utils import load_image
from package_utils.transform import final_transform


# The SBI checkpoint pairs with the SBI configuration, whose metadata keeps the
# preprocessing codepath aligned with ``scripts/test.py``.
DEFAULT_CONFIG_PATH = "configs/efn4_fpn_sbi_adv.yaml"
DEFAULT_WEIGHT_PATH = os.path.join(
    "model",
    "laa_net",
    "PoseEfficientNet_EFN_hm10_EFPN_NoBasedCLS_Focal_C3_256Cst100_8SBI_"
    "SAM(Adam)_ADV_Era1_OutSigmoid_1e7_boost500_UnFZ_model_best.pth",
)
DEFAULT_DATA_DIR = "data"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for configuring the inference script."""

    parser = argparse.ArgumentParser(
        description=(
            "Run SBI single-image inference using the exact preprocessing"
            " pipeline from scripts/test.py."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config that defines the model architecture.",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHT_PATH,
        help=(
            "Checkpoint path to load. The default matches the SBI config;"
            " use the BI config if you swap in the BI weights."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory containing RGB images to evaluate.",
    )
    parser.add_argument(
        "--skip-fixed-crop",
        action="store_true",
        help=(
            "Disable the fixed 257x257 crop after resizing to 317x317. "
            "This can be useful if the face is heavily off-center."
        ),
    )
    parser.add_argument(
        "--video-frame-step",
        type=int,
        default=1,
        help=(
            "Decode one frame out of every N when processing videos. "
            "Keeping the default of 1 mirrors the evaluation pipeline's "
            "use of all decoded frames before averaging logits."
        ),
    )
    parser.add_argument(
        "--yunet-path",
        default="face_detection_yunet_2023mar.onnx",
        help="Path to the YuNet ONNX model used for face detection.",
    )
    parser.add_argument(
        "--save-preprocessed-images-path",
        default=None,
        help="Optional directory to store square face crops and metadata.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file prediction details.",
    )
    return parser.parse_args()


def _validate_inference_fields(cfg) -> None:
    """Ensure the loaded config exposes the metadata required by this demo."""

    missing_paths = []
    dataset_cfg = getattr(cfg, "DATASET", None)
    if dataset_cfg is None:
        missing_paths.append("DATASET")
    else:
        if not hasattr(dataset_cfg, "IMAGE_SIZE"):
            missing_paths.append("DATASET.IMAGE_SIZE")
        if not hasattr(dataset_cfg, "TRANSFORM") or not hasattr(
            dataset_cfg.TRANSFORM, "normalize"
        ):
            missing_paths.append("DATASET.TRANSFORM.normalize")

    if not hasattr(cfg, "TEST") or not hasattr(cfg.TEST, "threshold"):
        missing_paths.append("TEST.threshold")

    if missing_paths:
        raise ValueError(
            "The supplied config is missing fields required for preprocessing "
            f"or scoring: {', '.join(missing_paths)}"
        )

    # ``DATASET.DATA`` documents how training/evaluation loaders enumerate
    # samples. The ad-hoc ``data/`` directory this demo scans can differ, but
    # surfacing unsupported types early helps users diagnose mismatches.
    data_cfg = getattr(dataset_cfg, "DATA", None)
    if data_cfg is not None and hasattr(data_cfg, "TYPE"):
        supported = {"frames", "images"}
        if data_cfg.TYPE not in supported:
            raise ValueError(
                "This demo only supports configs whose DATASET.DATA.TYPE is in "
                f"{sorted(supported)}. Got '{data_cfg.TYPE}'."
            )


def prepare_model(
    cfg_path: str, weight_path: str
) -> Tuple[object, torch.nn.Module, object, torch.device, List[int]]:
    """Assemble the model and preprocessing primitives required for inference."""

    cfg = load_config(cfg_path)
    _validate_inference_fields(cfg)
    model = build_model(cfg.MODEL, MODELS).to(torch.float64)
    model = load_pretrained(model, weight_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    transforms = final_transform(cfg.DATASET)
    image_size = cfg.DATASET.IMAGE_SIZE

    return cfg, model, transforms, device, image_size


def _prepare_tensor(
    image_rgb: np.ndarray, transforms, device: torch.device
) -> torch.Tensor:
    """Normalize a preprocessed RGB image and move it onto the target device."""

    tensor = transforms(image_rgb.astype(np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).to(device=device, dtype=torch.float64)
    return tensor


def preprocess_image(
    image_path: str,
    face_processor: FacePreprocessor,
    transforms,
    device: torch.device,
    *,
    save_dir: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Detect, standardize, and normalize an RGB image located on disk."""

    img = load_image(image_path)
    face_rgb, metadata = face_processor.process_image(
        img,
        identifier=image_path,
        save_dir=save_dir,
    )
    img_tensor = _prepare_tensor(face_rgb, transforms, device)
    return img_tensor, metadata


def preprocess_frame(
    frame: np.ndarray,
    video_processor,
    transforms,
    device: torch.device,
    *,
    frame_index: int,
    save_dir: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Process a single BGR video frame through detection and normalization."""

    face_rgb, metadata = video_processor.process_frame(
        frame,
        frame_index,
        save_dir=save_dir,
    )
    img_tensor = _prepare_tensor(face_rgb, transforms, device)
    return img_tensor, metadata


def _forward_model(model: torch.nn.Module, inputs: torch.Tensor):
    outputs = model(inputs)
    if isinstance(outputs, list):
        outputs = outputs[0]

    hm_outputs = outputs["hm"]
    cls_outputs = outputs["cls"]
    hm_preds = _sigmoid(hm_outputs).cpu().numpy()
    cls_preds = cls_outputs.detach().cpu().numpy()
    return hm_preds, cls_preds


def _collect_media_paths(
    data_dir: str,
) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    media_entries: List[Tuple[str, str]] = []
    image_paths: List[str] = []
    video_paths: List[str] = []

    for root, _, files in os.walk(data_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            full_path = os.path.join(root, fname)
            if ext in IMAGE_EXTENSIONS:
                image_paths.append(full_path)
                media_entries.append(("image", full_path))
            elif ext in VIDEO_EXTENSIONS:
                video_paths.append(full_path)

                media_entries.append(("video", full_path))

    return media_entries, image_paths, video_paths


def run_inference():
    """Load checkpoints, preprocess images, and print prediction metadata."""

    args = parse_args()

    cfg, model, transforms, device, image_size = prepare_model(
        args.config, args.weights
    )

    target_size = int(max(image_size))
    face_config = FacePreprocessingConfig(
        target_size=target_size,
        align_size=target_size,
    )
    face_processor = FacePreprocessor(args.yunet_path, face_config)
    save_dir = args.save_preprocessed_images_path

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Could not find data directory: {data_dir}")

    media_entries, image_paths, video_paths = _collect_media_paths(data_dir)

    if not media_entries:
        raise FileNotFoundError(
            f"No compatible media found in '{data_dir}'. Supported image extensions: "
            f"{', '.join(IMAGE_EXTENSIONS)}. Supported video extensions: "
            f"{', '.join(VIDEO_EXTENSIONS)}."
        )

    if image_paths:
        print(f"Loaded {len(image_paths)} images from {data_dir}/")
    if video_paths:
        print(f"Loaded {len(video_paths)} videos from {data_dir}/")
    print(f"Using device: {device}")
    if args.skip_fixed_crop:
        print(
            "--skip-fixed-crop is ignored; adaptive face preprocessing is always applied.")

    predictions: List[Tuple[str, int]] = []

    for media_type, media_path in media_entries:
        if media_type == "image":
            image_path = media_path
            with torch.no_grad():
                start_time = time.time()
                inputs, face_metadata = preprocess_image(
                    image_path,
                    face_processor,
                    transforms,
                    device,
                    save_dir=save_dir,
                )
                hm_preds, cls_preds = _forward_model(model, inputs)
                score = float(cls_preds[0][-1])
                label_int = 1 if score > cfg.TEST.threshold else 0
                label_text = "Fake" if label_int else "Real"
                elapsed = time.time() - start_time

            predictions.append((os.path.basename(image_path), label_int))

            if args.verbose:
                print("----------------------------------------")
                print(f"Image: {os.path.basename(image_path)}")
                print(f"Prediction: {label_text}")
                print(f"Fake score: {score:.4f}")
                print(f"Heatmap max value: {hm_preds.max():.4f}")
                print(f"Inference time: {elapsed:.3f}s")
                print(f"Face detected: {face_metadata['face_detected']}")
                if face_metadata["face_detected"]:
                    det_score = face_metadata.get("detection_score")
                    if det_score is not None:
                        print(
                            f"Detector: {face_metadata['detector']} (score={det_score:.3f})"
                        )
                    else:
                        print(f"Detector: {face_metadata['detector']}")
                    fallback_stage = face_metadata.get("fallback")
                    if fallback_stage and fallback_stage != "full_frame_letterbox":
                        print(f"Fallback stage: {fallback_stage}")
                else:
                    print(
                        f"Fallback pipeline: {face_metadata.get('fallback')} (reason={face_metadata.get('detection_reason')})"
                    )
                if face_metadata.get("output_path"):
                    print(f"Saved preprocess: {face_metadata['output_path']}")

        elif media_type == "video":
            video_path = media_path
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("----------------------------------------")
                print(f"Video: {os.path.basename(video_path)}")
                print("Error: unable to open video file.")
                cap.release()
                continue

            video_id = os.path.splitext(os.path.basename(video_path))[0]
            video_processor = face_processor.create_video_processor(video_id)
            frame_scores: List[np.ndarray] = []
            heatmap_maxes: List[float] = []
            frame_count = 0
            processed_frames = 0
            start_time = time.time()
            frame_metadata: List[Dict[str, object]] = []

            with torch.no_grad():
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    if frame_count % args.video_frame_step:
                        continue

                    inputs, face_meta = preprocess_frame(
                        frame,
                        video_processor,
                        transforms,
                        device,
                        frame_index=frame_count - 1,
                        save_dir=save_dir,
                    )
                    hm_preds, cls_preds = _forward_model(model, inputs)
                    heatmap_maxes.append(float(hm_preds.max()))
                    frame_scores.append(cls_preds[0])
                    processed_frames += 1
                    frame_metadata.append(face_meta)

            cap.release()

            if not frame_scores:
                print("----------------------------------------")
                print(f"Video: {os.path.basename(video_path)}")
                print(
                    "Error: no frames were processed. Check --video-frame-step or file integrity.")
                continue

            mean_logits = np.mean(frame_scores, axis=0)
            score = float(mean_logits[-1])
            label_int = 1 if score > cfg.TEST.threshold else 0
            label_text = "Fake" if label_int else "Real"
            elapsed = time.time() - start_time

            predictions.append((os.path.basename(video_path), label_int))

            if args.verbose:
                print("----------------------------------------")
                print(f"Video: {os.path.basename(video_path)}")
                print(f"Prediction: {label_text}")
                print(f"Fake score (mean logits): {score:.4f}")
                print(f"Frames decoded: {frame_count}")
                print(f"Frames evaluated: {processed_frames}")
                print(
                    f"Heatmap max value (max over frames): {max(heatmap_maxes):.4f}")
                print(f"Inference time: {elapsed:.3f}s")
                if frame_metadata:
                    successes = sum(
                        1 for meta in frame_metadata if meta.get("face_detected"))
                    tracker_frames = sum(
                        1
                        for meta in frame_metadata
                        if meta.get("temporal_source") == "tracker"
                    )
                    fallback_frames = sum(
                        1
                        for meta in frame_metadata
                        if meta.get("fallback") == "full_frame_letterbox"
                        or meta.get("temporal_source") == "fallback"
                    )
                    print(
                        f"Face detection success frames: {successes}/{len(frame_metadata)}")
                    if tracker_frames:
                        print(f"Tracker-rescued frames: {tracker_frames}")
                    if fallback_frames:
                        print(f"Letterbox fallback frames: {fallback_frames}")

    submission_path = os.path.join(os.getcwd(), "submission.csv")
    with open(submission_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "label"])
        for filename, label in predictions:
            writer.writerow([filename, label])

    print(f"Saved {len(predictions)} predictions to {submission_path}")


if __name__ == "__main__":
    run_inference()
