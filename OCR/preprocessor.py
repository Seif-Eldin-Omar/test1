"""
preprocessor.py

Image preprocessing pipeline to maximize OCR accuracy
on Arabic legal documents.

Surya handles many aspects internally, so preprocessing is lighter
than with older engines like EasyOCR. Each step is independently
toggleable via config. The pipeline works with PIL Images throughout,
converting to OpenCV format only for specific operations.
"""

import logging
import math
from typing import Optional

import cv2
import numpy as np
from PIL import Image

import config

logger = logging.getLogger(__name__)


def preprocess_image(
    image: Image.Image,
    enable_denoise: Optional[bool] = None,
    enable_deskew: Optional[bool] = None,
    enable_border_removal: Optional[bool] = None,
    enable_contrast_enhancement: Optional[bool] = None,
    enable_resolution_check: Optional[bool] = None,
) -> Image.Image:
    """
    Run the preprocessing pipeline on a single PIL Image.

    Surya works well with RGB input, so we avoid aggressive binarization.
    The pipeline focuses on resolution, deskewing, border removal, and
    contrast enhancement.

    Args:
        image: Input PIL Image in RGB mode.
        enable_denoise: Override config ENABLE_DENOISE.
        enable_deskew: Override config ENABLE_DESKEW.
        enable_border_removal: Override config ENABLE_BORDER_REMOVAL.
        enable_contrast_enhancement: Override config ENABLE_CONTRAST_ENHANCEMENT.
        enable_resolution_check: Override config ENABLE_RESOLUTION_CHECK.

    Returns:
        Preprocessed PIL Image in RGB mode, ready for Surya.
    """
    if enable_denoise is None:
        enable_denoise = config.ENABLE_DENOISE
    if enable_deskew is None:
        enable_deskew = config.ENABLE_DESKEW
    if enable_border_removal is None:
        enable_border_removal = config.ENABLE_BORDER_REMOVAL
    if enable_contrast_enhancement is None:
        enable_contrast_enhancement = config.ENABLE_CONTRAST_ENHANCEMENT
    if enable_resolution_check is None:
        enable_resolution_check = config.ENABLE_RESOLUTION_CHECK

    result = image.copy()

    # 1. Resolution check — upscale low-res images
    if enable_resolution_check:
        result = check_and_upscale_resolution(result)
        logger.debug("Resolution check done")

    # 2. Deskewing — correct rotation
    if enable_deskew:
        result = deskew(result)
        logger.debug("Deskewing done")

    # 3. Border removal — crop black scan borders
    if enable_border_removal:
        result = remove_borders(result)
        logger.debug("Border removal done")

    # 4. Contrast enhancement — CLAHE for faded documents
    if enable_contrast_enhancement:
        result = enhance_contrast(result)
        logger.debug("Contrast enhancement done")

    # 5. Noise removal — optional, off by default since Surya is robust
    if enable_denoise:
        result = denoise(result)
        logger.debug("Denoising done")

    return result


def check_and_upscale_resolution(image: Image.Image) -> Image.Image:
    """
    Ensure minimum resolution for OCR accuracy.

    If the image height is below what a standard A4 page would be
    at MIN_DPI, upscale using Lanczos interpolation.
    """
    min_dpi = config.MIN_DPI

    # A4 page at min DPI: 297mm * (dpi / 25.4)
    expected_height_at_min = int(min_dpi * 11.69)

    w, h = image.size

    if h >= expected_height_at_min:
        return image

    scale = expected_height_at_min / h
    # Cap upscaling at 3x to avoid excessive memory use
    scale = min(scale, 3.0)

    if scale <= 1.05:
        return image

    new_w = int(w * scale)
    new_h = int(h * scale)

    logger.info("Upscaling image from %dx%d to %dx%d (%.1fx)", w, h, new_w, new_h, scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


def deskew(image: Image.Image) -> Image.Image:
    """
    Detect and correct document rotation using multiple complementary methods.

    Photographed documents are often slightly tilted. This function detects
    the skew direction and magnitude using three independent approaches,
    then combines them for a robust angle estimate:

    1. **Hough Line Transform** -- detects dominant line angles from text
       baselines and page edges.
    2. **Projection Profile Analysis** -- finds the rotation angle that
       maximises horizontal projection variance (sharpest text lines).
    3. **minAreaRect fallback** -- uses the bounding rectangle of dark
       pixels when the other methods don't produce a clear result.

    Handles rotations up to ~15 degrees in either direction.

    Args:
        image: PIL Image to deskew.

    Returns:
        Rotated PIL Image with white fill for new border pixels.
    """
    gray = np.array(image.convert("L"))

    angle_hough = _detect_skew_hough(gray)
    angle_proj = _detect_skew_projection_profile(gray)
    angle_rect = _detect_skew_min_area_rect(gray)

    candidates = [a for a in (angle_hough, angle_proj, angle_rect) if a is not None]

    if not candidates:
        return image

    # When we have multiple estimates, take the median for robustness
    # against outliers from any single method.
    angle = float(np.median(candidates))

    if abs(angle) > 15 or abs(angle) < 0.1:
        return image

    logger.info(
        "Deskewing by %.2f° (hough=%.2f, projection=%.2f, rect=%.2f)",
        angle,
        angle_hough if angle_hough is not None else float("nan"),
        angle_proj if angle_proj is not None else float("nan"),
        angle_rect if angle_rect is not None else float("nan"),
    )
    return image.rotate(
        -angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255)
    )


# ------------------------------------------------------------------
# Skew detection helpers
# ------------------------------------------------------------------


def _detect_skew_hough(gray: np.ndarray, max_angle: float = 15.0):
    """
    Detect skew angle using Probabilistic Hough Line Transform.

    Finds near-horizontal lines (text baselines, ruled lines) and
    computes their median angle relative to horizontal.

    Returns:
        Estimated skew angle in degrees, or None if not enough lines.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Dilate horizontally to emphasise text baselines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    edges = cv2.dilate(edges, kernel, iterations=1)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=gray.shape[1] // 8,
        maxLineGap=20,
    )

    if lines is None or len(lines) < 3:
        return None

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1:
            continue
        angle_deg = math.degrees(math.atan2(dy, dx))
        if abs(angle_deg) <= max_angle:
            angles.append(angle_deg)

    if len(angles) < 3:
        return None

    return float(np.median(angles))


def _detect_skew_projection_profile(
    gray: np.ndarray,
    angle_range: float = 10.0,
    angle_step: float = 0.25,
):
    """
    Detect skew angle via projection profile analysis.

    For each candidate angle, rotate the image and compute the
    horizontal projection (row-wise sum of dark pixels). The correct
    angle produces the sharpest text lines, which shows up as the
    highest variance in the projection profile.

    To keep things fast, operates on a downscaled version of the image.

    Returns:
        Estimated skew angle in degrees, or None on failure.
    """
    # Downscale for speed
    scale = 600 / max(gray.shape)
    if scale < 1.0:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = gray

    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_angle = 0.0
    best_variance = 0.0
    h, w = binary.shape

    angles = np.arange(-angle_range, angle_range + angle_step, angle_step)
    for angle in angles:
        rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(
            binary, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
        )
        row_sums = np.sum(rotated, axis=1).astype(np.float64)
        variance = np.var(row_sums)
        if variance > best_variance:
            best_variance = variance
            best_angle = angle

    if best_variance == 0:
        return None

    # The rotation that maximises variance is the deskew correction,
    # so the detected skew is the negative of that.
    return float(-best_angle)


def _detect_skew_min_area_rect(gray: np.ndarray):
    """
    Detect skew angle using the minimum-area bounding rectangle of
    dark pixels. Simple fallback method.

    Returns:
        Estimated skew angle in degrees, or None if insufficient dark pixels.
    """
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 50:
        return None

    angle = cv2.minAreaRect(coords)[-1]

    # minAreaRect returns angles in [-90, 0)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) > 15:
        return None

    return float(angle)


def remove_borders(image: Image.Image) -> Image.Image:
    """
    Remove black scan borders by finding the largest contour
    that represents the document area.
    """
    gray = np.array(image.convert("L"))

    # Invert so document area is white
    inverted = cv2.bitwise_not(gray)

    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    img_area = gray.shape[0] * gray.shape[1]
    contour_area = w * h
    if contour_area < img_area * 0.5:
        return image

    # Add small padding
    pad = 5
    y_start = max(0, y - pad)
    y_end = min(gray.shape[0], y + h + pad)
    x_start = max(0, x - pad)
    x_end = min(gray.shape[1], x + w + pad)

    return image.crop((x_start, y_start, x_end, y_end))


def enhance_contrast(image: Image.Image) -> Image.Image:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to improve contrast on faded documents.

    Works on the L channel in LAB color space to preserve color information.
    """
    arr = np.array(image)

    # Convert RGB to LAB
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(enhanced)


def denoise(image: Image.Image) -> Image.Image:
    """
    Remove noise using Non-Local Means denoising.

    Optional step — off by default since Surya is robust to noise.
    """
    arr = np.array(image)

    # Use color denoising for RGB images (positional args for OpenCV compat)
    denoised = cv2.fastNlMeansDenoisingColored(arr, None, 10, 10, 7, 21)

    return Image.fromarray(denoised)
