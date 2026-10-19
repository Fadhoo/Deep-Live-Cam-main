"""Pose-guided, single-person appearance transfer from a full-body photo.

This is a 2-D compositor, not a generative model. It can only reuse pixels
visible in the reference image; large turns and occlusions cannot be invented.
"""

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


def pre_check() -> bool:
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        print(f"[{NAME}] Install mediapipe to use body swapping.")
        return False
    return True


def pre_start() -> bool:
    if modules.globals.map_faces:
        print(f"[{NAME}] Body swapping currently supports one person; turn off Map faces.")
        return False
    if modules.globals.many_faces:
        print(f"[{NAME}] Body swapping currently supports one person; turn off Many faces.")
        return False
    if not modules.globals.source_path:
        print(f"[{NAME}] Select a full-body source image first.")
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
    global _source_path, _source_image, _source_result
    if path == _source_path and _source_image is not None:
        return True
    source = imread_unicode(path)
    if source is None:
        print(f"[{NAME}] Cannot read source image: {path}")
        return False
    result = _detect(source)
    if not _valid(result):
        print(f"[{NAME}] No visible body found in source image: {path}")
        return False
    _source_path, _source_image, _source_result = path, source, result
    return True


def _mask_and_bounds(result: Any, shape: tuple) -> tuple[np.ndarray, tuple] | None:
    height, width = shape[:2]
    mask = np.asarray(result.segmentation_masks[0].numpy_view(), dtype=np.float32)
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    # MediaPipe returns a confidence map, not an opacity map. Using it as
    # opacity makes clothing and limbs translucent even for confident poses.
    mask = (mask > 0.45).astype(np.float32)
    ys, xs = np.where(mask > 0)
    if len(xs) < 100:
        return None
    return mask, (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def _control_points(source_result, target_result, source_shape, target_shape,
                    source_bounds, target_bounds):
    sh, sw = source_shape[:2]
    th, tw = target_shape[:2]
    src, dst = [], []
    # Joint positions anchor the articulated warp. A point is used only when
    # visible in both images so hidden limbs do not pull nearby triangles.
    for index in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32):
        a = source_result.pose_landmarks[0][index]
        b = target_result.pose_landmarks[0][index]
        if min(a.visibility or 0, b.visibility or 0) < 0.5:
            continue
        src.append((float(np.clip(a.x * sw, 0, sw - 1)),
                    float(np.clip(a.y * sh, 0, sh - 1))))
        dst.append((float(np.clip(b.x * tw, 0, tw - 1)),
                    float(np.clip(b.y * th, 0, th - 1))))

    # Silhouette anchors let the source mask extend beyond the skeleton.
    for u, v in ((0, 0), (.5, 0), (1, 0), (1, .5), (1, 1),
                 (.5, 1), (0, 1), (0, .5)):
        src.append((source_bounds[0] + u * (source_bounds[2] - source_bounds[0]),
                    source_bounds[1] + v * (source_bounds[3] - source_bounds[1])))
        dst.append((target_bounds[0] + u * (target_bounds[2] - target_bounds[0]),
                    target_bounds[1] + v * (target_bounds[3] - target_bounds[1])))

    unique_src, unique_dst = [], []
    for a, b in zip(src, dst):
        if all(np.linalg.norm(np.subtract(a, prior)) > 2 for prior in unique_src):
            unique_src.append(a)
            unique_dst.append(b)
    return np.asarray(unique_src, np.float32), np.asarray(unique_dst, np.float32)


def _triangles(points: np.ndarray, shape: tuple):
    height, width = shape[:2]
    subdiv = cv2.Subdiv2D((0, 0, width, height))
    for x, y in points:
        subdiv.insert((float(np.clip(x, 0, width - 1)),
                       float(np.clip(y, 0, height - 1))))
    for triangle in subdiv.getTriangleList().reshape(-1, 3, 2):
        indices = []
        for vertex in triangle:
            distances = np.linalg.norm(points - vertex, axis=1)
            index = int(np.argmin(distances))
            if distances[index] > 3:
                break
            indices.append(index)
        if len(indices) == 3 and len(set(indices)) == 3:
            yield indices


def _warp_triangle(source, source_mask, canvas, canvas_mask, src_tri, dst_tri):
    if abs(cv2.contourArea(src_tri)) < 1 or abs(cv2.contourArea(dst_tri)) < 1:
        return
    sh, sw = source.shape[:2]
    th, tw = canvas.shape[:2]
    sx, sy, sx2, sy2 = cv2.boundingRect(src_tri)
    dx, dy, dw, dh = cv2.boundingRect(dst_tri)
    sx0, sy0 = max(sx, 0), max(sy, 0)
    sx1, sy1 = min(sx + sx2, sw), min(sy + sy2, sh)
    dx0, dy0 = max(dx, 0), max(dy, 0)
    dx1, dy1 = min(dx + dw, tw), min(dy + dh, th)
    if sx1 <= sx0 or sy1 <= sy0 or dx1 <= dx0 or dy1 <= dy0:
        return
    transform = cv2.getAffineTransform(
        (src_tri - (sx0, sy0)).astype(np.float32),
        (dst_tri - (dx0, dy0)).astype(np.float32),
    )
    size = (dx1 - dx0, dy1 - dy0)
    warped = cv2.warpAffine(source[sy0:sy1, sx0:sx1], transform, size,
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    alpha = cv2.warpAffine(source_mask[sy0:sy1, sx0:sx1], transform, size,
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    triangle_mask = np.zeros((size[1], size[0]), dtype=np.float32)
    cv2.fillConvexPoly(triangle_mask, np.round(dst_tri - (dx0, dy0)).astype(np.int32), 1)
    alpha *= triangle_mask
    region = canvas[dy0:dy1, dx0:dx1]
    existing = canvas_mask[dy0:dy1, dx0:dx1]
    use = alpha > 0.01
    region[use] = warped[use]
    existing[use] = alpha[use]


def _composite(source, source_result, target, target_result):
    source_data = _mask_and_bounds(source_result, source.shape)
    target_data = _mask_and_bounds(target_result, target.shape)
    if source_data is None or target_data is None:
        return target
    source_mask, source_bounds = source_data
    target_mask, target_bounds = target_data
    src_points, dst_points = _control_points(
        source_result, target_result, source.shape, target.shape,
        source_bounds, target_bounds,
    )
    canvas = np.zeros_like(target)
    canvas_mask = np.zeros(target.shape[:2], dtype=np.float32)
    for indices in _triangles(src_points, source.shape):
        _warp_triangle(source, source_mask, canvas, canvas_mask,
                       src_points[indices], dst_points[indices])
    alpha = np.minimum(canvas_mask, target_mask)
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
    # Leave the head to the dedicated face swapper. A triangular body mesh
    # stretches the eyes and mouth when the head turns or mirrors.
    landmarks = target_result.pose_landmarks[0]
    nose = landmarks[0]
    left, right = landmarks[11], landmarks[12]
    if min(nose.visibility or 0, left.visibility or 0, right.visibility or 0) > 0.5:
        height, width = target.shape[:2]
        shoulder_width = abs(left.x - right.x) * width
        shoulder_y = (left.y + right.y) * height / 2
        nose_y = nose.y * height
        head_mask = np.zeros((height, width), np.float32)
        center = (int(nose.x * width), int(nose_y + .15 * (shoulder_y - nose_y)))
        radius = (max(1, int(shoulder_width * .38)),
                  max(1, int(abs(shoulder_y - nose_y) * 1.35)))
        cv2.ellipse(head_mask, center, radius, 0, 0, 360, 1, -1)
        head_mask = cv2.GaussianBlur(head_mask, (21, 21), 0)
        alpha *= 1 - head_mask
    alpha = alpha[:, :, None]
    return np.clip(canvas.astype(np.float32) * alpha +
                   target.astype(np.float32) * (1 - alpha), 0, 255).astype(np.uint8)


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
        return _composite(_source_image, _source_result, temp_frame, result)


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
