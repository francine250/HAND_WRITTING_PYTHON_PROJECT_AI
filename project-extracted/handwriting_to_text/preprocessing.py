"""Step 3 - Data preparation: loading, cleaning and line segmentation."""
import cv2
import numpy as np


def load_pages(data: bytes, filename: str):
    """Return a list of BGR images (one per page) from image or PDF bytes."""
    if filename.lower().endswith(".pdf"):
        import fitz  # PyMuPDF
        pages = []
        with fitz.open(stream=data, filetype="pdf") as doc:
            for p in doc:
                pix = p.get_pixmap(dpi=200)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                else:
                    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                pages.append(arr)
        return pages
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unsupported or corrupted image file.")
    return [img]


def _deskew(gray, ink):
    coords = np.column_stack(np.where(ink > 0))
    if len(coords) < 50:
        return gray, ink
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.3 or abs(angle) > 15:      # ignore noise / unrealistic skew
        return gray, ink
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rot = lambda im, b: cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=b)
    return rot(gray, 255), rot(ink, 0)


def preprocess(img_bgr, cfg):
    """Return (clean_gray, ink_mask). ink_mask: handwriting = 255, background = 0."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # limit very large photos for speed
    h, w = gray.shape
    if w > 1800:
        gray = cv2.resize(gray, (1800, int(h * 1800 / w)), interpolation=cv2.INTER_AREA)
    if cfg.get("denoise", True):
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    # remove uneven lighting (phone photos)
    bg = cv2.medianBlur(gray, 51)
    gray = cv2.normalize(cv2.divide(gray, bg, scale=255), None, 0, 255, cv2.NORM_MINMAX)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 15)
    if cfg.get("deskew", True):
        gray, ink = _deskew(gray, ink)
    if cfg.get("binarize", False):
        gray = 255 - ink
    return gray, ink


def segment_lines(gray, ink, cfg):
    """Split a page into text-line images using a horizontal projection profile."""
    proj = ink.sum(axis=1).astype(np.float32) / 255.0
    proj = np.convolve(proj, np.ones(5) / 5, mode="same")
    if proj.max() == 0:
        return []
    active = proj > cfg.get("line_proj_ratio", 0.12) * proj.max()

    runs, start = [], None
    for y, a in enumerate(active):
        if a and start is None:
            start = y
        elif not a and start is not None:
            runs.append([start, y]); start = None
    if start is not None:
        runs.append([start, len(active)])

    merged = []                                  # merge runs separated by tiny gaps (i, j, accents)
    for r in runs:
        if merged and r[0] - merged[-1][1] <= cfg.get("merge_gap", 6):
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    lines, pad = [], cfg.get("line_padding", 8)
    for y0, y1 in merged:
        if y1 - y0 < cfg.get("min_line_height", 18):
            continue
        cols = np.where(ink[y0:y1].sum(axis=0) > 0)[0]
        x0, x1 = max(cols.min() - pad, 0), min(cols.max() + pad, gray.shape[1])
        crop = gray[max(y0 - pad, 0):min(y1 + pad, gray.shape[0]), x0:x1]
        lines.append(crop)
    return lines
