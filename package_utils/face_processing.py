"""Face-centric preprocessing utilities built around OpenCV YuNet."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class FacePreprocessingConfig:
    target_size: int = 512
    margin: float = 1.30
    pad_mode: str = "reflect"
    pad_value: Tuple[int, int, int] = (114, 114, 114)
    min_score: float = 0.9
    min_face_size: float = 40.0
    use_alignment: bool = False
    align_size: Optional[int] = None
    retry_scales: Tuple[float, ...] = (1.25, 1.5)
    retry_equalize: bool = True
    ema_alpha: float = 0.7
    tracker_type: str = "CSRT"


@dataclass
class DetectionOutcome:
    success: bool
    bbox: Optional[np.ndarray] = None  # [x, y, w, h]
    landmarks: Optional[np.ndarray] = None  # shape (5, 2)
    score: Optional[float] = None
    detector: str = "yunet"
    fallback: Optional[str] = None
    reason: Optional[str] = None
    history: List[str] = field(default_factory=list)


def _create_yunet(model_path: str):
    create_fn = getattr(cv2, "FaceDetectorYN_create", None)
    if create_fn is None:
        face_detector_yn = getattr(cv2, "FaceDetectorYN", None)
        if face_detector_yn is None or not hasattr(face_detector_yn, "create"):
            raise RuntimeError(
                "Unable to locate cv2.FaceDetectorYN. Install opencv-contrib-python >= 4.8."
            )
        create_fn = face_detector_yn.create

    detector = create_fn(
        model=model_path,
        config="",
        input_size=(320, 320),
        score_threshold=0.0,
        nms_threshold=0.3,
        top_k=5000,
        backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
        target_id=cv2.dnn.DNN_TARGET_CPU,
    )
    return detector


def _create_tracker(tracker_type: str):
    tracker_type = tracker_type.upper()
    constructor_candidates: List = []
    if tracker_type == "CSRT":
        constructor_candidates.extend(
            [
                getattr(cv2, "TrackerCSRT_create", None),
                getattr(getattr(cv2, "legacy", object), "TrackerCSRT_create", None),
            ]
        )
    elif tracker_type == "KCF":
        constructor_candidates.extend(
            [
                getattr(cv2, "TrackerKCF_create", None),
                getattr(getattr(cv2, "legacy", object), "TrackerKCF_create", None),
            ]
        )

    if not constructor_candidates:
        return None

    for ctor in constructor_candidates:
        if callable(ctor):
            try:
                tracker = ctor()
                if tracker is not None:
                    return tracker
            except Exception:  # pragma: no cover - defensive guard
                continue
    return None


def _haar_cascade():
    cascade_path = getattr(cv2.data, "haarcascades", "")
    if not cascade_path:
        return None
    cascade = cv2.CascadeClassifier(os.path.join(cascade_path, "haarcascade_frontalface_default.xml"))
    if cascade.empty():
        return None
    return cascade


def _equalize_bgr(image: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel, cr, cb = cv2.split(ycrcb)
    y_channel = cv2.equalizeHist(y_channel)
    merged = cv2.merge((y_channel, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


class YuNetDetector:
    def __init__(self, model_path: str, config: FacePreprocessingConfig):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YuNet model not found: {model_path}. Supply the ONNX file before running."
            )
        self.config = config
        self.detector = _create_yunet(model_path)
        self.cascade = _haar_cascade()

    def _prepare_input(self, image: np.ndarray):
        h, w = image.shape[:2]
        if hasattr(self.detector, "setInputSize"):
            self.detector.setInputSize((w, h))

    def _run_yunet(self, image: np.ndarray) -> Optional[np.ndarray]:
        self._prepare_input(image)
        _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            return None
        return np.array(faces, dtype=np.float32)

    def _select_face(self, faces: np.ndarray, scale: float = 1.0) -> Optional[DetectionOutcome]:
        scores = faces[:, -1]
        best_idx = int(np.argmax(scores))
        best = faces[best_idx]
        score = float(best[-1])
        bbox = best[0:4] / scale
        landmarks = best[4:14].reshape(5, 2) / scale

        side = max(bbox[2], bbox[3])
        if score < self.config.min_score:
            return DetectionOutcome(
                success=False,
                reason="low_score",
                history=["yunet"],
            )
        if side < self.config.min_face_size:
            return DetectionOutcome(
                success=False,
                reason="small_face",
                history=["yunet"],
            )
        return DetectionOutcome(
            success=True,
            bbox=bbox,
            landmarks=landmarks,
            score=score,
            detector="yunet",
            history=["yunet"],
        )

    def detect(self, image: np.ndarray) -> DetectionOutcome:
        history: List[str] = []
        faces = self._run_yunet(image)
        if faces is not None:
            outcome = self._select_face(faces)
            if outcome.success:
                return outcome
            history.extend(outcome.history)
            last_reason = outcome.reason
        else:
            last_reason = "no_face"
            history.append("yunet")

        if self.config.retry_equalize:
            eq_image = _equalize_bgr(image)
            faces = self._run_yunet(eq_image)
            if faces is not None:
                outcome = self._select_face(faces)
                if outcome.success:
                    outcome.history = history + ["equalize"]
                    outcome.fallback = "hist_equalize"
                    return outcome
                last_reason = outcome.reason
                history.extend(outcome.history)
            else:
                history.append("equalize")

        for scale in self.config.retry_scales:
            scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            faces = self._run_yunet(scaled)
            if faces is None:
                history.append(f"scale_{scale}")
                continue
            outcome = self._select_face(faces, scale=scale)
            if outcome.success:
                outcome.history = history + [f"scale_{scale}"]
                outcome.fallback = f"scale_{scale}"
                return outcome
            last_reason = outcome.reason
            history.extend(outcome.history)

        if self.cascade is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detected = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(int(self.config.min_face_size), int(self.config.min_face_size)),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            if len(detected) > 0:
                x, y, w, h = max(detected, key=lambda b: b[2] * b[3])
                bbox = np.array([x, y, w, h], dtype=np.float32)
                return DetectionOutcome(
                    success=True,
                    bbox=bbox,
                    landmarks=None,
                    score=None,
                    detector="haar_cascade",
                    fallback="haar_cascade",
                    history=history + ["haar_cascade"],
                )
            history.append("haar_cascade")

        return DetectionOutcome(
            success=False,
            detector="yunet",
            fallback=None,
            reason=last_reason or "no_face",
            history=history,
        )


def _border_type(mode: str):
    mode = mode.lower()
    if mode == "reflect":
        return cv2.BORDER_REFLECT_101
    if mode == "replicate":
        return cv2.BORDER_REPLICATE
    if mode == "constant":
        return cv2.BORDER_CONSTANT
    raise ValueError(f"Unsupported pad_mode '{mode}'.")


def _expand_square(image: np.ndarray, bbox: np.ndarray, margin: float) -> Tuple[np.ndarray, Dict[str, float]]:
    h, w = image.shape[:2]
    x, y, bw, bh = bbox.astype(np.float32)
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    side = max(bw, bh) * margin

    half = side / 2.0
    left = math.floor(cx - half)
    top = math.floor(cy - half)
    right = math.ceil(cx + half)
    bottom = math.ceil(cy + half)

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - w)
    pad_bottom = max(0, bottom - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top

    crop = image[top:bottom, left:right]
    square_meta = {
        "square_left": float(left),
        "square_top": float(top),
        "square_right": float(right),
        "square_bottom": float(bottom),
        "square_side": float(side),
    }
    return crop, square_meta


def _letterbox(
    image: np.ndarray, target_size: int, pad_mode: str, pad_value: Tuple[int, int, int]
) -> Tuple[np.ndarray, Dict[str, float]]:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Cannot letterbox empty crop.")
    scale = target_size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC

    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    border_type = _border_type(pad_mode)
    if border_type == cv2.BORDER_CONSTANT:
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, border_type, value=pad_value)
    else:
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, border_type)

    letterbox_meta = {
        "letterbox_scale": float(scale),
        "letterbox_top": int(top),
        "letterbox_bottom": int(bottom),
        "letterbox_left": int(left),
        "letterbox_right": int(right),
        "interpolation": "INTER_AREA" if interpolation == cv2.INTER_AREA else "INTER_CUBIC",
        "pad_mode": pad_mode,
    }
    return padded, letterbox_meta


def _align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    align_size: int,
    pad_mode: str,
    pad_value: Tuple[int, int, int],
) -> Optional[np.ndarray]:
    if landmarks is None or landmarks.shape != (5, 2):
        return None

    dst = ARCFACE_TEMPLATE * (align_size / 112.0)
    ordered_landmarks = np.array(
        [
            landmarks[1],  # left eye
            landmarks[0],  # right eye
            landmarks[2],  # nose tip
            landmarks[4],  # left mouth
            landmarks[3],  # right mouth
        ],
        dtype=np.float32,
    )

    transform, inliers = cv2.estimateAffinePartial2D(
        ordered_landmarks,
        dst,
        method=cv2.LMEDS,
    )

    if transform is None or (inliers is not None and inliers.sum() < 3):
        return None

    border_type = _border_type(pad_mode)
    aligned = cv2.warpAffine(
        image,
        transform,
        (align_size, align_size),
        flags=cv2.INTER_LINEAR,
        borderMode=border_type,
        borderValue=pad_value if border_type == cv2.BORDER_CONSTANT else 0,
    )
    return aligned


class FacePreprocessor:
    def __init__(self, model_path: str, config: Optional[FacePreprocessingConfig] = None):
        self.config = config or FacePreprocessingConfig()
        if self.config.align_size is None:
            self.config.align_size = self.config.target_size
        self.detector = YuNetDetector(model_path, self.config)

    def _generate_metadata(
        self,
        outcome: DetectionOutcome,
        preprocessing_meta: Dict[str, float],
        align_used: bool,
    ) -> Dict[str, object]:
        metadata: Dict[str, object] = {
            "face_detected": outcome.success,
            "detector": outcome.detector,
            "detection_score": None if outcome.score is None else float(outcome.score),
            "detection_history": outcome.history,
            "fallback": outcome.fallback,
            "alignment_used": align_used,
        }
        if outcome.bbox is not None:
            metadata.update(
                {
                    "bbox_x": float(outcome.bbox[0]),
                    "bbox_y": float(outcome.bbox[1]),
                    "bbox_w": float(outcome.bbox[2]),
                    "bbox_h": float(outcome.bbox[3]),
                }
            )
        if not outcome.success:
            metadata["detection_reason"] = outcome.reason
        metadata.update(preprocessing_meta)
        return metadata

    def _save_outputs(
        self,
        image_rgb: np.ndarray,
        metadata: Dict[str, object],
        save_dir: str,
        identifier: str,
    ) -> None:
        if not identifier:
            return
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.basename(identifier)
        name, ext = os.path.splitext(base_name)
        if ext.lower() not in {".png", ".jpg", ".jpeg"}:
            ext = ".png"
        image_path = os.path.join(save_dir, f"{name}{ext}")
        metadata_path = os.path.join(save_dir, f"{name}.json")

        bgr_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, bgr_image)
        metadata = dict(metadata)
        metadata.update({"output_path": image_path})
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def process_image(
        self,
        image_rgb: np.ndarray,
        identifier: Optional[str] = None,
        save_dir: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        outcome = self.detector.detect(bgr)

        if outcome.success:
            expanded, square_meta = _expand_square(bgr, outcome.bbox, self.config.margin)
            target_img, letterbox_meta = _letterbox(
                expanded, self.config.target_size, self.config.pad_mode, self.config.pad_value
            )
            metadata = {**square_meta, **letterbox_meta}
            align_used = False
            if self.config.use_alignment:
                aligned = _align_face(
                    bgr,
                    outcome.landmarks,
                    self.config.align_size or self.config.target_size,
                    self.config.pad_mode,
                    self.config.pad_value,
                )
                if aligned is not None:
                    target_img = aligned
                    align_used = True
            metadata = self._generate_metadata(outcome, metadata, align_used)
        else:
            padded, letterbox_meta = _letterbox(
                bgr, self.config.target_size, self.config.pad_mode, self.config.pad_value
            )
            metadata = self._generate_metadata(
                outcome,
                {**letterbox_meta, "fallback": "full_frame_letterbox"},
                align_used=False,
            )
            target_img = padded

        rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        if save_dir is not None:
            self._save_outputs(rgb, metadata, save_dir, identifier or "output")
        return rgb, metadata

    def create_video_processor(self, video_id: str) -> "VideoFaceProcessor":
        return VideoFaceProcessor(self, video_id)


class VideoFaceProcessor:
    def __init__(self, processor: FacePreprocessor, video_id: str):
        self.processor = processor
        self.video_id = video_id
        self.ema_state: Optional[np.ndarray] = None
        self.tracker = None
        self.tracker_type = processor.config.tracker_type

    def _update_ema(self, cx: float, cy: float, side: float) -> Tuple[float, float, float]:
        current = np.array([cx, cy, side], dtype=np.float32)
        if self.ema_state is None:
            self.ema_state = current
        else:
            alpha = self.processor.config.ema_alpha
            self.ema_state = alpha * current + (1.0 - alpha) * self.ema_state
        return tuple(map(float, self.ema_state))

    def _init_tracker(self, frame: np.ndarray, bbox: np.ndarray):
        tracker = _create_tracker(self.tracker_type)
        if tracker is None:
            self.tracker = None
            return
        init_bbox = tuple(map(float, bbox))
        tracker.init(frame, init_bbox)
        self.tracker = tracker

    def _tracker_predict(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if self.tracker is None:
            return None
        ok, tracked = self.tracker.update(frame)
        if not ok:
            self.tracker = None
            return None
        return np.array(tracked, dtype=np.float32)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        save_dir: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        outcome = self.processor.detector.detect(frame_bgr)

        metadata: Dict[str, object]
        if outcome.success:
            cx = float(outcome.bbox[0] + outcome.bbox[2] / 2.0)
            cy = float(outcome.bbox[1] + outcome.bbox[3] / 2.0)
            side = float(max(outcome.bbox[2], outcome.bbox[3]))
            smoothed_cx, smoothed_cy, smoothed_side = self._update_ema(cx, cy, side)
            # smooth bounding box before cropping
            smoothed_bbox = np.array(
                [
                    smoothed_cx - smoothed_side / 2.0,
                    smoothed_cy - smoothed_side / 2.0,
                    smoothed_side,
                    smoothed_side,
                ],
                dtype=np.float32,
            )
            outcome.bbox = smoothed_bbox
            expanded, square_meta = _expand_square(frame_bgr, outcome.bbox, self.processor.config.margin)
            target_img, letterbox_meta = _letterbox(
                expanded,
                self.processor.config.target_size,
                self.processor.config.pad_mode,
                self.processor.config.pad_value,
            )
            align_used = False
            if self.processor.config.use_alignment:
                aligned = _align_face(
                    frame_bgr,
                    outcome.landmarks,
                    self.processor.config.align_size or self.processor.config.target_size,
                    self.processor.config.pad_mode,
                    self.processor.config.pad_value,
                )
                if aligned is not None:
                    target_img = aligned
                    align_used = True
            metadata = self.processor._generate_metadata(outcome, {**square_meta, **letterbox_meta}, align_used)
            metadata["temporal_source"] = "detection"
            if outcome.detector == "yunet":
                # reset tracker on fresh detection
                self._init_tracker(frame_bgr, outcome.bbox)
        else:
            tracker_bbox = self._tracker_predict(frame_bgr)
            if tracker_bbox is not None:
                pseudo_outcome = DetectionOutcome(
                    success=True,
                    bbox=tracker_bbox,
                    landmarks=None,
                    score=None,
                    detector=f"tracker_{self.tracker_type.lower()}",
                    fallback="tracker",
                )
                expanded, square_meta = _expand_square(
                    frame_bgr,
                    pseudo_outcome.bbox,
                    self.processor.config.margin,
                )
                target_img, letterbox_meta = _letterbox(
                    expanded,
                    self.processor.config.target_size,
                    self.processor.config.pad_mode,
                    self.processor.config.pad_value,
                )
                metadata = self.processor._generate_metadata(
                    pseudo_outcome, {**square_meta, **letterbox_meta}, align_used=False
                )
                metadata["temporal_source"] = "tracker"
                cx = float(pseudo_outcome.bbox[0] + pseudo_outcome.bbox[2] / 2.0)
                cy = float(pseudo_outcome.bbox[1] + pseudo_outcome.bbox[3] / 2.0)
                side = float(max(pseudo_outcome.bbox[2], pseudo_outcome.bbox[3]))
                self._update_ema(cx, cy, side)
            elif self.ema_state is not None:
                cx, cy, side = map(float, self.ema_state)
                fallback_bbox = np.array(
                    [cx - side / 2.0, cy - side / 2.0, side, side],
                    dtype=np.float32,
                )
                pseudo_outcome = DetectionOutcome(
                    success=True,
                    bbox=fallback_bbox,
                    landmarks=None,
                    score=None,
                    detector="ema_track",
                    fallback="ema_hold",
                )
                expanded, square_meta = _expand_square(
                    frame_bgr,
                    pseudo_outcome.bbox,
                    self.processor.config.margin,
                )
                target_img, letterbox_meta = _letterbox(
                    expanded,
                    self.processor.config.target_size,
                    self.processor.config.pad_mode,
                    self.processor.config.pad_value,
                )
                metadata = self.processor._generate_metadata(
                    pseudo_outcome, {**square_meta, **letterbox_meta}, align_used=False
                )
                metadata["temporal_source"] = "ema"
            else:
                padded, letterbox_meta = _letterbox(
                    frame_bgr,
                    self.processor.config.target_size,
                    self.processor.config.pad_mode,
                    self.processor.config.pad_value,
                )
                metadata = self.processor._generate_metadata(
                    outcome,
                    {**letterbox_meta, "fallback": "full_frame_letterbox"},
                    align_used=False,
                )
                metadata["temporal_source"] = "fallback"
                target_img = padded

        metadata["frame_index"] = frame_index
        identifier = f"{self.video_id}_frame{frame_index:06d}"
        rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        if save_dir is not None:
            self.processor._save_outputs(rgb, metadata, save_dir, identifier)
        return rgb, metadata
