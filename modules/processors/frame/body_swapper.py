"""Transfer source skin tone to exposed target skin without changing clothing."""

import os
import threading
from typing import Any

import cv2
import numpy as np

import modules.globals
from modules import imread_unicode, imwrite_unicode
from modules.paths import MODELS_DIR
from modules.model_downloader import _download

NAME = "DLC.BODY-SWAPPER"
MODEL_NAME = "pose_landmarker_full.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

_lock = threading.RLock()
_landmarker = None
_source_path = None
_source_image = None
_source_result = None
_source_skin = None


def pre_check() -> bool:
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        print(f"[{NAME}] Install mediapipe to use skin tone matching.")
        return False
    return True


def pre_start() -> bool:
    if modules.globals.map_faces:
        print(f"[{NAME}] Skin tone matching currently supports one person; turn off Map faces.")
        return False
    if modules.globals.many_faces:
        print(f"[{NAME}] Skin tone matching currently supports one person; turn off Many faces.")
        return False
    if not modules.globals.source_path:
        print(f"[{NAME}] Select a source face image first.")
        return False
    try:
        model_path = os.path.join(MODELS_DIR, MODEL_NAME)
        if not os.path.isfile(model_path) and not _download(
            MODEL_NAME, MODEL_URL, model_path, None
        ):
            raise OSError("Pose landmarker download failed")
        with _lock:
            return _load_source(modules.globals.source_path)
    except Exception as exc:
        print(f"[{NAME}] Could not prepare body model: {exc}")
        return False


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        import mediapipe as mp

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=os.path.join(MODELS_DIR, MODEL_NAME)
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            output_segmentation_masks=True,
        )
        _landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    return _landmarker


def _detect(frame: np.ndarray):
    import mediapipe as mp

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    return _get_landmarker().detect(image)


def _valid(result: Any) -> bool:
    return bool(result.pose_landmarks and result.segmentation_masks)


def _load_source(path: str) -> bool:
    global _source_path, _source_image, _source_result, _source_skin
    if path == _source_path and _source_image is not None:
        return True
    source = imread_unicode(path)
    if source is None:
        print(f"[{NAME}] Cannot read source image: {path}")
        return False
    result = _detect(source)
    skin = _skin_reference(source, result) if _valid(result) else None
    if skin is None:
        from modules.face_analyser import get_one_face

        face = get_one_face(source)
        skin = _skin_reference(source, None, face)
    if skin is None:
        print(f"[{NAME}] Source needs a clearly visible face or exposed skin: {path}")
        return False
    _source_path, _source_image, _source_result, _source_skin = path, source, result, skin
    return True


def _person_mask(result: Any, shape: tuple) -> np.ndarray | None:
    height, width = shape[:2]
    mask = np.asarray(result.segmentation_masks[0].numpy_view(), dtype=np.float32)
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    # Segmentation is a confidence score. Keep only the largest person-shaped
    # component so stray detections cannot bring source pixels into the scene.
    binary = (mask > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count < 2:
        return None
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[component, cv2.CC_STAT_AREA] < 100:
        return None
    return (labels == component).astype(np.uint8)


def _point(landmarks: list, index: int, shape: tuple) -> tuple[int, int] | None:
    landmark = landmarks[index]
    if (landmark.visibility or 0) < 0.55:
        return None
    height, width = shape[:2]
    x, y = landmark.x * width, landmark.y * height
    if not (0 <= x < width and 0 <= y < height):
        return None
    return int(x), int(y)


def _face_sample(lab: np.ndarray, result: Any, person_mask: np.ndarray,
                 face: Any = None):
    """Estimate skin chroma from two small cheek patches, away from eyes/lips."""
    if face is not None and getattr(face, 'bbox', None) is not None:
        x1, y1, x2, y2 = face.bbox
        face_width = x2 - x1
        centers = ((int(x1 + face_width * .32), int(y1 + (y2 - y1) * .55)),
                   (int(x1 + face_width * .68), int(y1 + (y2 - y1) * .55)))
    else:
        if result is None or not result.pose_landmarks:
            return None
        landmarks = result.pose_landmarks[0]
        nose = _point(landmarks, 0, lab.shape)
        if nose is None:
            return None
        left_ear = _point(landmarks, 7, lab.shape)
        right_ear = _point(landmarks, 8, lab.shape)
        left_eye = _point(landmarks, 2, lab.shape)
        right_eye = _point(landmarks, 5, lab.shape)
        if left_ear and right_ear:
            face_width = np.linalg.norm(np.subtract(left_ear, right_ear))
        elif left_eye and right_eye:
            face_width = 1.8 * np.linalg.norm(np.subtract(left_eye, right_eye))
        else:
            return None
        centers = tuple((int(nose[0] + direction * face_width * .22),
                         int(nose[1] + face_width * .04))
                        for direction in (-1, 1))
    if face_width < 8:
        return None
    radius = max(2, int(face_width * .09))
    patch = np.zeros(lab.shape[:2], np.uint8)
    for center in centers:
        cv2.circle(patch, center, radius, 1, -1)
    pixels = lab[(patch & person_mask) > 0]
    if len(pixels) < 8:
        return None
    return np.median(pixels, axis=0)


def _body_regions(result: Any, shape: tuple) -> np.ndarray:
    """Conservative limb/neck regions; torso garments are never candidates."""
    landmarks = result.pose_landmarks[0]
    height, width = shape[:2]
    regions = np.zeros((height, width), np.uint8)
    left = _point(landmarks, 11, shape)
    right = _point(landmarks, 12, shape)
    if left is None or right is None:
        return regions
    shoulder_width = np.linalg.norm(np.subtract(left, right))
    if shoulder_width < 8:
        return regions
    for a, b, fraction in (
        (11, 13, .23), (13, 15, .18), (12, 14, .23), (14, 16, .18),
        (23, 25, .30), (25, 27, .22), (24, 26, .30), (26, 28, .22),
        (15, 19, .16), (16, 20, .16), (27, 31, .19), (28, 32, .19),
    ):
        start = _point(landmarks, a, shape)
        end = _point(landmarks, b, shape)
        if start and end:
            cv2.line(regions, start, end, 1,
                     thickness=max(3, int(shoulder_width * fraction)))
    # Neck only: stop at the shoulder line so shirts are preserved.
    nose = _point(landmarks, 0, shape)
    if nose:
        midpoint = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2)
        start = (int(nose[0] * .4 + midpoint[0] * .6),
                 int(nose[1] * .4 + midpoint[1] * .6))
        cv2.line(regions, start, midpoint, 1,
                 thickness=max(3, int(shoulder_width * .18)))
    return regions


def _skin_candidates(lab: np.ndarray, result: Any, person_mask: np.ndarray,
                     face_color: np.ndarray) -> np.ndarray:
    regions = _body_regions(result, lab.shape)
    chroma = lab[:, :, 1:3].astype(np.float32)
    distance = np.linalg.norm(chroma - face_color[1:3], axis=2)
    return ((regions > 0) & (person_mask > 0) & (distance < 24)).astype(np.uint8)


def _skin_reference(frame: np.ndarray, result: Any,
                    face: Any = None) -> np.ndarray | None:
    person_mask = _person_mask(result, frame.shape) if result is not None else None
    if person_mask is None:
        if face is None:
            return None
        person_mask = np.ones(frame.shape[:2], np.uint8)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    face_color = _face_sample(lab, result, person_mask, face)
    if face_color is None:
        return None
    if result is not None and _valid(result):
        candidates = _skin_candidates(lab, result, person_mask, face_color)
        if np.count_nonzero(candidates) >= 40:
            return np.median(lab[candidates > 0], axis=0)
    return face_color


def _composite(source, source_result, target, target_result, source_skin=None):
    if source_skin is None:
        source_skin = _skin_reference(source, source_result)
    target_mask = _person_mask(target_result, target.shape)
    if source_skin is None or target_mask is None:
        return target
    lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    face = _face_sample(lab, target_result, target_mask)
    if face is None:
        return target
    candidates = _skin_candidates(lab, target_result, target_mask, face)
    if not np.any(candidates):
        return target
    target_skin = np.median(lab[candidates > 0], axis=0)
    delta = np.clip(source_skin - target_skin, [-100, -35, -35], [100, 35, 35])
    alpha = np.minimum(cv2.distanceTransform(candidates, cv2.DIST_L2, 3) / 3,
                       1.0)[:, :, None]
    changed = np.clip(lab.astype(np.float32) + delta * alpha, 0, 255).astype(np.uint8)
    recolored = cv2.cvtColor(changed, cv2.COLOR_LAB2BGR)
    output = target.copy()
    output[candidates > 0] = recolored[candidates > 0]
    return output


def process_frame(source_face, temp_frame, target_face=None):
    del source_face, target_face
    path = modules.globals.source_path
    if not path:
        return temp_frame
    with _lock:
        if not _load_source(path):
            return temp_frame
        result = _detect(temp_frame)
        if not _valid(result):
            return temp_frame
        return _composite(_source_image, _source_result, temp_frame, result,
                          _source_skin)


def process_frame_v2(temp_frame, temp_frame_path=""):
    del temp_frame_path
    return process_frame(None, temp_frame)


def process_image(source_path, target_path, output_path):
    target = imread_unicode(target_path)
    if target is None:
        raise ValueError(f"Cannot read target image: {target_path}")
    modules.globals.source_path = source_path
    if not imwrite_unicode(output_path, process_frame(None, target)):
        raise OSError(f"Cannot write body swap output: {output_path}")


def process_frames(source_path, temp_frame_paths, progress=None):
    modules.globals.source_path = source_path
    for path in temp_frame_paths:
        frame = imread_unicode(path)
        if frame is not None:
            imwrite_unicode(path, process_frame(None, frame))
        if progress:
            progress.update(1)


def process_video(source_path, temp_frame_paths):
    from modules.processors.frame.core import process_video as run_video
    run_video(source_path, temp_frame_paths, process_frames)
