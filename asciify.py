import io
import os
import html
import json
import uuid
import base64
import threading
from collections import OrderedDict
from functools import lru_cache
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from flask import Flask, request, send_file, jsonify, Response, abort
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024   # 64 MB upload cap
Image.MAX_IMAGE_PIXELS = 200_000_000                  # decompression-bomb guard

RESAMPLE_BOX = getattr(Image, "Resampling", Image).BOX
RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

# ---------------------------------------------------------------------------
# Character sets (ordered dark -> light on a black background).
# Only single-width, non-emoji characters are used, so text output keeps its
# proportions in editors and browsers.
# ---------------------------------------------------------------------------

CHARSETS = {
    "ascii":     " .-_,^+r/?sJ7(Fi1tloxj2ShdbUHm0MQ%&@",
    "blocks":    " ░▒▓█",
    "blocks_lo": " ▁▂▃▄▅▆▇█",
    "dots":      " ⠁⠉⠋⠛⠟⠿⣿",
    "braille":   " ⠁⠂⠃⠄⠅⠆⠇⡀⡄⡆⡇⣀⣄⣆⣇⣿",
    "math":      " ·∙∘○●",
    "geometric": " ·▫□▪■",
    "stars":     " ·*✦✧★",
    "binary":    " 01",
    "hex":       " 0123456789abcdef",
    "minimal":   " .:+#@",
    "detailed":  " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
}
DEFAULT_CHARSET = "ascii"

DEFAULT_FONT_SIZE = 6
MIN_FONT_SIZE, MAX_FONT_SIZE = 4, 48
BAND_ROWS = 32                       # cell rows processed per band (memory)
MAX_CACHED_UPLOADS = 6

# ---------------------------------------------------------------------------
# Fonts. The first usable monospace font defines the character cell; other
# fonts are only used per glyph when the primary font lacks it.
# ---------------------------------------------------------------------------

MONO_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/lucon.ttf",
    "C:/Windows/Fonts/cour.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Courier.ttc",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
]
SYMBOL_FALLBACKS = [
    "C:/Windows/Fonts/seguisym.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Apple Symbols.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansSymbols-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
]
MONO_FONT_NAMES = ["DejaVuSansMono.ttf", "consola.ttf", "Menlo.ttc", "cour.ttf"]


@lru_cache(maxsize=1)
def _font_paths() -> tuple:
    """Usable font files, primary monospace font first."""
    paths = []
    for p in MONO_FONT_CANDIDATES + SYMBOL_FALLBACKS:
        if os.path.exists(p) and p not in paths:
            try:
                ImageFont.truetype(p, 12)
                paths.append(p)
            except OSError:
                continue
    if not paths:
        for name in MONO_FONT_NAMES:            # let Pillow search system dirs
            try:
                ImageFont.truetype(name, 12)
                paths.append(name)
                break
            except OSError:
                continue
    return tuple(paths)


def get_font(size: int, path: str | None = None):
    """New font object per call: FreeType faces are not thread-safe to share."""
    path = path or (_font_paths()[0] if _font_paths() else None)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size)      # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def font_advance(font) -> float:
    try:
        return float(font.getlength("M"))
    except AttributeError:
        return float(font.getsize("M")[0])


def cell_metrics(font) -> tuple[int, int]:
    """Integer pixel size of one character cell for the primary font."""
    cw = font_advance(font)
    try:
        asc, desc = font.getmetrics()
        ch = asc + desc
    except AttributeError:
        ch = font.getsize("M")[1]
    return max(1, int(round(cw))), max(2, int(ch))

# ---------------------------------------------------------------------------
# Glyph masks. Block elements and braille are drawn procedurally (they must
# fill the cell exactly, whatever font is installed); everything else comes
# from a font, with per-glyph fallback and automatic fit-to-cell.
# ---------------------------------------------------------------------------

def _block_mask(cp: int, cw: int, ch: int):
    m = np.zeros((ch, cw), np.uint8)
    h = lambda f: int(round(ch * f))
    w = lambda f: int(round(cw * f))
    if cp == 0x2580:                       # upper half
        m[:h(.5)] = 255
    elif 0x2581 <= cp <= 0x2588:           # lower 1/8 .. full
        m[ch - h((cp - 0x2580) / 8):] = 255
    elif 0x2589 <= cp <= 0x258F:           # left 7/8 .. 1/8
        m[:, :w((0x2590 - cp) / 8)] = 255
    elif cp == 0x2590:                     # right half
        m[:, cw - w(.5):] = 255
    elif cp in (0x2591, 0x2592, 0x2593):   # light / medium / dark shade
        m[:] = {0x2591: 64, 0x2592: 128, 0x2593: 191}[cp]
    elif cp == 0x2594:                     # upper 1/8
        m[:max(1, h(1 / 8))] = 255
    elif cp == 0x2595:                     # right 1/8
        m[:, cw - max(1, w(1 / 8)):] = 255
    elif 0x2596 <= cp <= 0x259F:           # quadrants
        UL, UR, LL, LR = 1, 2, 4, 8
        q = {0x2596: LL, 0x2597: LR, 0x2598: UL, 0x2599: UL | LL | LR,
             0x259A: UL | LR, 0x259B: UL | UR | LL, 0x259C: UL | UR | LR,
             0x259D: UR, 0x259E: UR | LL, 0x259F: UR | LL | LR}[cp]
        hh, hw = h(.5), w(.5)
        if q & UL: m[:hh, :hw] = 255
        if q & UR: m[:hh, hw:] = 255
        if q & LL: m[hh:, :hw] = 255
        if q & LR: m[hh:, hw:] = 255
    else:
        return None
    return m


_BRAILLE_DOTS = [(0, 0, 0x01), (0, 1, 0x02), (0, 2, 0x04), (1, 0, 0x08),
                 (1, 1, 0x10), (1, 2, 0x20), (0, 3, 0x40), (1, 3, 0x80)]


def _braille_mask(cp: int, cw: int, ch: int) -> np.ndarray:
    S = 6                                           # supersampling factor
    img = Image.new("L", (cw * S, ch * S), 0)
    d = ImageDraw.Draw(img)
    bits = cp - 0x2800
    r = 0.92 * min(cw / 4.0, ch / 8.0) * S
    for col, row, bit in _BRAILLE_DOTS:
        if bits & bit:
            x = (col * 0.5 + 0.25) * cw * S
            y = (row * 0.25 + 0.125) * ch * S
            d.ellipse((x - r, y - r, x + r, y + r), fill=255)
    return np.asarray(img.resize((cw, ch), RESAMPLE_BOX))


def _has_glyph(font, c: str) -> bool:
    try:
        m = font.getmask(c)
        if m.getbbox() is None:
            return False
        nd = font.getmask("\uffff")                 # .notdef box
        return not (m.size == nd.size and m.tobytes() == nd.tobytes())
    except Exception:
        return False


def _font_glyph_mask(c: str, font_size: int, cw: int, ch: int):
    paths = _font_paths()
    for n, path in enumerate(paths):
        size = font_size
        font = get_font(size, path)
        if not _has_glyph(font, c):
            continue
        for _ in range(4):                          # shrink until it fits the cell
            l, t, r, b = font.getbbox(c)
            gw, gh = r - l, b - t
            if gw <= cw + 1 and gh <= ch + 1:
                break
            scale = min((cw + 1) / max(gw, 1), (ch + 1) / max(gh, 1))
            size = max(2, int(size * scale))
            font = get_font(size, path)
        l, t, r, b = font.getbbox(c)
        if n == 0 and size == font_size:
            x = y = 0                               # natural monospace placement
        else:                                       # centre foreign / shrunk glyphs
            x = (cw - (r - l)) // 2 - l
            y = (ch - (b - t)) // 2 - t
        m = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(m).text((x, y), c, font=font, fill=255)
        return np.asarray(m), True
    # nobody has it: draw whatever the primary font gives (usually a box)
    m = Image.new("L", (cw, ch), 0)
    ImageDraw.Draw(m).text((0, 0), c, font=get_font(font_size), fill=255)
    return np.asarray(m), False


def _glyph_mask(c: str, font_size: int, cw: int, ch: int):
    cp = ord(c)
    if 0x2800 <= cp <= 0x28FF:
        return _braille_mask(cp, cw, ch), True
    if 0x2580 <= cp <= 0x259F:
        m = _block_mask(cp, cw, ch)
        if m is not None:
            return m, True
    return _font_glyph_mask(c, font_size, cw, ch)


@lru_cache(maxsize=16)
def build_glyph_masks(ramp: tuple, font_size: int, cw: int, ch: int):
    """(n, ch, cw) uint8 masks for the ramp + string of characters no font has."""
    masks = np.zeros((len(ramp), ch, cw), np.uint8)
    cache, missing = {}, []
    for k, c in enumerate(ramp):
        if c == " ":
            continue
        if c not in cache:
            m, ok = _glyph_mask(c, font_size, cw, ch)
            cache[c] = m
            if not ok:
                missing.append(c)
        masks[k] = cache[c]
    masks.setflags(write=False)
    return masks, "".join(missing)

# ---------------------------------------------------------------------------
# Upload cache (LRU). The FULL-RESOLUTION decoded image is kept so the output
# can match the original pixel size and sample the original pixels.
# ---------------------------------------------------------------------------

_upload_lock = threading.Lock()
_uploads: "OrderedDict[str, dict]" = OrderedDict()


def _store_upload(raw: bytes, filename: str):
    img = Image.open(io.BytesIO(raw))
    dpi = img.info.get("dpi")
    img = ImageOps.exif_transpose(img)              # respect camera rotation
    img.load()

    has_alpha = img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info)
    if has_alpha:                                   # transparent -> black
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        bg.alpha_composite(rgba)
        img = bg.convert("RGB")
    else:
        img = img.convert("RGB")

    rgb = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
    if isinstance(dpi, tuple) and len(dpi) == 2 and all(d and d > 0 for d in dpi):
        dpi = (float(dpi[0]), float(dpi[1]))
    else:
        dpi = None
    token = uuid.uuid4().hex
    with _upload_lock:
        _uploads[token] = {"name": filename, "rgb": rgb, "dpi": dpi}
        while len(_uploads) > MAX_CACHED_UPLOADS:
            _uploads.popitem(last=False)
    return token, img.size


def _get_upload(token: str) -> dict:
    with _upload_lock:
        entry = _uploads.get(token)
        if entry is not None:
            _uploads.move_to_end(token)
    if entry is None:
        abort(404, description="Upload expired or unknown - please select the image again.")
    return entry

# ---------------------------------------------------------------------------
# Conversion pipeline
# ---------------------------------------------------------------------------

def _adjust(arr, brightness, contrast, gamma):
    if brightness != 1.0:
        arr = arr * brightness
    if contrast != 1.0:
        arr = (arr - 128.0) * contrast + 128.0
    arr = np.clip(arr, 0.0, 255.0)
    if gamma != 1.0:
        arr = 255.0 * np.power(arr / 255.0, 1.0 / gamma)
    return arr


def make_grid(rgb: np.ndarray, p: dict, cw: int, ch: int):
    """One character per cw x ch block of ORIGINAL pixels.

    The grid is ceil(W/cw) x ceil(H/ch); the (<1 cell) overhang on the right and
    bottom is filled by edge replication, then cropped away again at render time.
    So: no borders, no cropped content, no stretching, exact original size.
    Returns idx (rows, cols) and colors (rows, cols, 3) uint8."""
    H, W, _ = rgb.shape
    cols, rows = -(-W // cw), -(-H // ch)
    PW, PH = cols * cw, rows * ch
    src = rgb
    if (PW, PH) != (W, H):
        src = np.pad(rgb, ((0, PH - H), (0, PW - W), (0, 0)), mode="edge")
    small = Image.fromarray(src).resize((cols, rows), RESAMPLE_BOX)  # exact block mean
    arr = _adjust(np.asarray(small, dtype=np.float32),
                  p["brightness"], p["contrast"], p["gamma"])

    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    if p["invert"]:
        lum = 255.0 - lum
    n = len(p["ramp"])
    idx = np.clip(np.rint(lum * (n - 1) / 255.0), 0, n - 1).astype(np.intp)
    return idx, arr.astype(np.uint8)


def render_image(rgb: np.ndarray, p: dict):
    """Lossless render at exactly the original pixel size. Returns (PIL image, info)."""
    H, W, _ = rgb.shape
    fs = p["font_size"]
    cw, ch = cell_metrics(get_font(fs))
    idx, colors = make_grid(rgb, p, cw, ch)
    rows, cols = idx.shape
    masks, missing = build_glyph_masks(tuple(p["ramp"]), fs, cw, ch)

    PW, PH = cols * cw, rows * ch
    out = np.zeros((PH, PW, 3) if p["color"] else (PH, PW), np.uint8)
    for y0 in range(0, rows, BAND_ROWS):
        y1 = min(rows, y0 + BAND_ROWS)
        mask = masks[idx[y0:y1]].transpose(0, 2, 1, 3).reshape((y1 - y0) * ch, PW)
        ys = slice(y0 * ch, y1 * ch)
        if p["color"]:
            cb = colors[y0:y1]
            for c in range(3):
                up = np.repeat(np.repeat(cb[..., c], ch, axis=0), cw, axis=1)
                prod = np.multiply(up, mask, dtype=np.uint16) + 127
                out[ys, :, c] = prod // 255
        else:
            out[ys] = mask

    img = Image.fromarray(out)
    if (PW, PH) != (W, H):
        img = img.crop((0, 0, W, H))
    info = {"cols": cols, "rows": rows, "cw": cw, "ch": ch,
            "w": W, "h": H, "font_size": fs, "missing": missing}
    return img, info

# ---------------------------------------------------------------------------
# Text / HTML outputs (same grid as the PNG)
# ---------------------------------------------------------------------------

def render_plain_text(rgb, p) -> str:
    cw, ch = cell_metrics(get_font(p["font_size"]))
    idx, _ = make_grid(rgb, p, cw, ch)
    ramp = np.array(p["ramp"], dtype="<U1")
    return "\n".join("".join(row) for row in ramp[idx]) + "\n"


def _rows_html(idx, colors, ramp, color: bool) -> str:
    """Neighbouring characters with the same (quantised) colour share a <span>;
    spaces never break a run."""
    esc = [html.escape(c, quote=False) for c in ramp]
    is_space = [c == " " for c in ramp]
    if not color:
        return "\n".join("".join(esc[i] for i in row) for row in idx.tolist())

    q = (colors >> 2) << 2
    packed = ((q[..., 0].astype(np.uint32) << 16)
              | (q[..., 1].astype(np.uint32) << 8)
              | q[..., 2].astype(np.uint32))

    def span(c, parts):
        return f'<span style="color:#{c:06x}">{"".join(parts)}</span>'

    lines = []
    for irow, crow in zip(idx.tolist(), packed.tolist()):
        out, buf, cur = [], [], None
        for i, c in zip(irow, crow):
            if is_space[i]:
                buf.append(" ")
                continue
            if c != cur:
                if buf:
                    out.append(span(cur, buf) if cur is not None else "".join(buf))
                    buf = []
                cur = c
            buf.append(esc[i])
        if buf:
            out.append(span(cur, buf) if cur is not None else "".join(buf))
        lines.append("".join(out))
    return "\n".join(lines)


def render_html_file(rgb, p) -> str:
    fs = p["font_size"]
    font = get_font(fs)
    cw, ch = cell_metrics(font)
    idx, colors = make_grid(rgb, p, cw, ch)
    body = _rows_html(idx, colors, p["ramp"], p["color"])

    face = ""
    paths = _font_paths()
    if paths and paths[0].lower().endswith((".ttf", ".otf")) and os.path.isfile(paths[0]):
        with open(paths[0], "rb") as f:             # embed font -> identical glyph widths
            b64 = base64.b64encode(f.read()).decode("ascii")
        face = ("@font-face{font-family:'AsciiMono';"
                f"src:url(data:font/ttf;base64,{b64}) format('truetype');}}")
    spacing = cw - font_advance(font)               # force the exact cell width
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ASCII art</title>
<style>
  {face}
  body {{ margin:0; background:#000; color:#ccc; }}
  pre {{ margin:0; white-space:pre; font-size:{fs}px; line-height:{ch}px;
        letter-spacing:{spacing:.4f}px; font-kerning:none; font-variant-ligatures:none;
        font-family:'AsciiMono','DejaVu Sans Mono',Consolas,Menlo,'Courier New',monospace; }}
</style></head>
<body><pre>{body}</pre></body></html>"""

# ---------------------------------------------------------------------------
# Steam profile export
#
# Steam renders profile text in a PROPORTIONAL font, so ordinary ASCII art falls
# apart (every character has a different width, spaces collapse). Braille
# patterns (U+2800..U+28FF) are the exception: they have one constant width even
# in proportional fonts, and U+2800 (the blank braille cell) is a "visible"
# character that is never trimmed or collapsed. So the Steam export is built
# ONLY from braille cells: 2x4 dots per character, blank = U+2800, never a
# normal space, tab or carriage return. Dot = bright pixel (Steam is dark).
# ---------------------------------------------------------------------------

STEAM_BLANK = "\u2800"
STEAM_NEWLINE_COST = 2          # a newline may be counted as CRLF -> be safe
STEAM_MIN_COLS, STEAM_MAX_COLS = 10, 120
STEAM_DEFAULT_COLS = 50
STEAM_DEFAULT_LIMIT = 2000      # profile summary budget; adjustable in the UI
STEAM_DEFAULT_ASPECT = 2.0      # rendered cell height / cell width (2.0 = square dots)


def _floyd_steinberg(g: np.ndarray) -> np.ndarray:
    """g: float 0..1 (h, w) -> bool (h, w), error-diffused."""
    h, w = g.shape
    g = g.astype(np.float64).copy()
    out = np.zeros((h, w), bool)
    for y in range(h):
        row = g[y]
        nxt = g[y + 1] if y + 1 < h else None
        for x in range(w):
            old = row[x]
            on = old >= 0.5
            out[y, x] = on
            err = old - (1.0 if on else 0.0)
            if x + 1 < w:
                row[x + 1] += err * 0.4375
            if nxt is not None:
                if x > 0:
                    nxt[x - 1] += err * 0.1875
                nxt[x] += err * 0.3125
                if x + 1 < w:
                    nxt[x + 1] += err * 0.0625
    return out


def _steam_lines(rgb: np.ndarray, p: dict, sp: dict, cols: int) -> list:
    H, W, _ = rgb.shape
    rows = max(1, int(round(cols * H / W / sp["aspect"])))
    dw, dh = cols * 2, rows * 4                       # dot grid
    method = RESAMPLE_BOX if (W >= dw and H >= dh) else RESAMPLE_LANCZOS
    small = Image.fromarray(rgb).resize((dw, dh), method)
    arr = _adjust(np.asarray(small, dtype=np.float32),
                  p["brightness"], p["contrast"], p["gamma"])
    lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]) / 255.0
    if p["invert"]:
        lum = 1.0 - lum
    lum = np.clip((lum - 0.12) / 0.76, 0.0, 1.0)      # dead zones: flat areas stay clean
    dots = _floyd_steinberg(lum) if sp["dither"] else (lum >= 0.5)

    code = np.zeros((rows, cols), np.int32)
    for dc, dr, bit in _BRAILLE_DOTS:
        code += dots[dr::4, dc::2].astype(np.int32) * bit
    lines = []
    for r in code.tolist():
        line = "".join(chr(0x2800 + c) for c in r).rstrip(STEAM_BLANK)
        lines.append(line or STEAM_BLANK)             # never emit an empty line
    return lines


def _steam_total(lines: list) -> int:
    return sum(map(len, lines)) + STEAM_NEWLINE_COST * (len(lines) - 1)


def render_steam_text(rgb: np.ndarray, p: dict, sp: dict):
    """Braille text for a Steam profile. Shrinks the width until it fits the
    character limit (the aspect ratio is kept). Returns (text, info)."""
    requested = cols = sp["cols"]
    lines = _steam_lines(rgb, p, sp, cols)
    total = _steam_total(lines)
    if total > sp["limit"]:
        cols = max(STEAM_MIN_COLS, min(cols - 1, int(cols * (sp["limit"] / total) ** 0.5)))
        lines = _steam_lines(rgb, p, sp, cols)
        total = _steam_total(lines)
        while total > sp["limit"] and cols > STEAM_MIN_COLS:
            cols -= 1
            lines = _steam_lines(rgb, p, sp, cols)
            total = _steam_total(lines)
    text = "\n".join(lines)
    info = {"cols": cols, "rows": len(lines), "requested_cols": requested,
            "chars": total, "limit": sp["limit"], "fits": total <= sp["limit"]}
    return text, info


def parse_steam_params(form) -> dict:
    def num(name, default, lo, hi):
        try:
            v = float(form.get(name, default))
            if v != v or v in (float("inf"), float("-inf")):
                v = default
        except (TypeError, ValueError):
            v = default
        return max(lo, min(hi, v))
    return {
        "cols": int(num("steamCols", STEAM_DEFAULT_COLS, STEAM_MIN_COLS, STEAM_MAX_COLS)),
        "limit": int(num("steamLimit", STEAM_DEFAULT_LIMIT, 100, 20000)),
        "aspect": num("steamAspect", STEAM_DEFAULT_ASPECT, 1.0, 3.0),
        "dither": form.get("steamDither", "1") != "0",
    }

# ---------------------------------------------------------------------------
# Param parsing
# ---------------------------------------------------------------------------

def parse_params(form) -> dict:
    def f(name, default, lo, hi):
        try:
            v = float(form.get(name, default))
            if v != v or v in (float("inf"), float("-inf")):
                v = default
        except (TypeError, ValueError):
            v = default
        return max(lo, min(hi, v))

    charset_key = form.get("charset", DEFAULT_CHARSET)
    custom = "".join(
        ch for ch in (form.get("custom_chars") or "")[:128]
        if ch == " " or ch.isprintable())

    charset = custom if (charset_key == "custom" and custom) \
        else CHARSETS.get(charset_key, CHARSETS[DEFAULT_CHARSET])
    if len(set(charset)) < 2:
        charset = CHARSETS[DEFAULT_CHARSET]

    return {
        "font_size": int(f("fontSize", DEFAULT_FONT_SIZE, MIN_FONT_SIZE, MAX_FONT_SIZE)),
        "ramp": list(charset),
        "brightness": f("brightness", 1.0, 0.2, 3.0),
        "contrast": f("contrast", 1.0, 0.2, 3.0),
        "gamma": f("gamma", 1.0, 0.2, 3.0),
        "invert": form.get("invert", "0") == "1",
        "color": form.get("color", "1") != "0",
    }

# ---------------------------------------------------------------------------
# Web interface
# ---------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image to ASCII</title>
<style>
  :root {
    color-scheme: dark;
    --bg:#0d0d0d; --panel:#161616; --border:#2a2a2a;
    --text:#e6e6e6; --muted:#888; --accent:#5b8def;
    --accent-hover:#4a7ade; --chip:#232323; --chip-on:#2f4a7a;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size:14px; }
  header { padding:1rem 1.5rem; border-bottom:1px solid var(--border);
    display:flex; align-items:center; justify-content:space-between; }
  header h1 { font-size:1rem; font-weight:600; margin:0; letter-spacing:.3px; }
  header .status { font-size:.8rem; color:var(--muted); }
  main { display:grid; grid-template-columns:340px 1fr; gap:1rem; padding:1rem;
    height:calc(100vh - 57px); }
  @media (max-width: 900px) { main { grid-template-columns:1fr; height:auto; }
    #stage { min-height:60vh; } }
  .panel { background:var(--panel); border:1px solid var(--border);
    border-radius:8px; padding:1rem; overflow:auto; }
  .panel h2 { font-size:.75rem; text-transform:uppercase; letter-spacing:.6px;
    color:var(--muted); margin:0 0 .75rem 0; font-weight:600; }
  label { display:block; font-size:.8rem; color:var(--muted); margin:.75rem 0 .25rem 0; }
  label:first-of-type { margin-top:0; }
  input[type=text], input[type=file] { width:100%; background:#1e1e1e; color:var(--text);
    border:1px solid var(--border); border-radius:5px; padding:.4rem .6rem;
    font-size:.85rem; font-family:inherit; }
  input[type=range] { width:100%; accent-color:var(--accent); }
  .range-row { display:flex; align-items:center; gap:.5rem; }
  .range-row output { min-width:2.5rem; text-align:right; font-family:monospace;
    font-size:.8rem; color:var(--muted); }
  .ends { display:flex; justify-content:space-between; font-size:.7rem; color:var(--muted); }
  .chips { display:flex; flex-wrap:wrap; gap:.35rem; margin-top:.25rem; }
  .chip { background:var(--chip); border:1px solid var(--border); color:var(--text);
    padding:.3rem .55rem; border-radius:5px; font-size:.8rem; cursor:pointer;
    user-select:none; font-family:monospace; transition:background .1s; }
  .chip:hover { background:#2c2c2c; }
  .chip.on { background:var(--chip-on); border-color:var(--accent); }
  .checkbox-row { display:flex; align-items:center; gap:.5rem; margin-top:.75rem; }
  .checkbox-row input { width:auto; }
  .checkbox-row label { margin:0; color:var(--text); font-size:.85rem; }
  .actions { display:flex; gap:.5rem; margin-top:1rem; }
  button { flex:1; background:var(--accent); color:#fff; border:0; border-radius:5px;
    padding:.55rem .75rem; font-weight:600; font-size:.85rem; cursor:pointer;
    font-family:inherit; }
  button:hover { background:var(--accent-hover); }
  button:disabled { opacity:.5; cursor:not-allowed; }
  button.alt { background:#2a2a2a; }
  button.alt:hover { background:#343434; }
  button.small { flex:none; padding:.25rem .6rem; font-size:.75rem; }
  .hint { font-size:.75rem; color:var(--muted); margin-top:.5rem; line-height:1.4; }

  .quality { border:1px solid var(--border); border-radius:8px; padding:.75rem;
    background:#111; margin-bottom:1.25rem; }
  .quality h2 { margin-bottom:.5rem; }
  .badge { display:inline-block; padding:.15rem .5rem; border-radius:999px;
    font-size:.75rem; font-weight:700; background:#2a2a2a; color:#ccc; }
  .badge.q4 { background:#1f5f3a; color:#c9ffe0; }
  .badge.q3 { background:#2b5a2b; color:#d6ffd6; }
  .badge.q2 { background:#6a5a1c; color:#fff2b8; }
  .badge.q1 { background:#6a3a1c; color:#ffd9b8; }
  .badge.q0 { background:#6a1c1c; color:#ffc9c9; }
  .stats { display:grid; grid-template-columns:auto 1fr; gap:.2rem .75rem;
    font-size:.78rem; margin-top:.6rem; }
  .stats dt { color:var(--muted); }
  .stats dd { margin:0; font-family:monospace; text-align:right; }
  #warn { display:none; margin-top:.6rem; font-size:.75rem; color:#ffcf8a; line-height:1.4; }

  .preview-panel { display:flex; flex-direction:column; padding:0; overflow:hidden; }
  .preview-head { padding:.5rem 1rem; border-bottom:1px solid var(--border);
    font-size:.75rem; color:var(--muted); display:flex; justify-content:space-between;
    align-items:center; gap:.5rem; }
  #stage { position:relative; flex:1; min-height:0; background:#000; overflow:hidden; }
  #stage.full { overflow:auto; }
  #stage img { display:block; }
  #stage.fit img { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; }
  #stage .ph { color:var(--muted); padding:2rem; text-align:center; }
  #busy { position:absolute; top:.6rem; right:.8rem; font-size:.75rem; color:var(--muted);
    background:#000a; padding:.15rem .5rem; border-radius:4px; display:none; z-index:2; }
</style></head>
<body>
<header>
  <h1>Image to ASCII</h1>
  <span class="status" id="status">No file loaded</span>
</header>
<main>
  <aside class="panel">
    <h2>Source</h2>
    <label for="file">Image file</label>
    <input id="file" type="file" accept="image/*">

    <div class="quality" style="margin-top:1.25rem;">
      <h2>Quality &amp; output size</h2>
      <label for="fontSize" style="margin-top:0;">Character size (px) &mdash; smaller = more detail</label>
      <div class="range-row">
        <input id="fontSize" type="range" min="__MINFS__" max="__MAXFS__" value="__DEFFS__">
        <output id="fontSizeVal">__DEFFS__</output>
      </div>
      <div class="ends"><span>&larr; finest (best quality)</span><span>coarsest &rarr;</span></div>
      <div style="margin-top:.6rem;">Quality: <span class="badge" id="qBadge">&ndash;</span></div>
      <dl class="stats">
        <dt>Original image</dt><dd id="stOrig">&ndash;</dd>
        <dt>PNG output</dt><dd id="stOut">&ndash;</dd>
        <dt>Characters</dt><dd id="stChars">&ndash;</dd>
        <dt>Character cell</dt><dd id="stCell">&ndash;</dd>
      </dl>
      <div class="hint">The PNG always has the exact pixel size of the original and is lossless.
        Smaller characters mean more of them, so the result follows the original more closely.
        The .txt and .html downloads use the same character grid.</div>
      <div id="warn"></div>
    </div>

    <h2>Character style</h2>
    <div class="chips" id="charsetChips"></div>
    <label for="customChars">Custom characters (dark &rarr; light)</label>
    <input id="customChars" type="text" placeholder="e.g.  .*#@" maxlength="64">

    <h2 style="margin-top:1.25rem;">Adjustments</h2>
    <label for="brightness">Brightness</label>
    <div class="range-row">
      <input id="brightness" type="range" min="0.2" max="3" step="0.05" value="1">
      <output id="brightnessVal">1.00</output>
    </div>
    <label for="contrast">Contrast</label>
    <div class="range-row">
      <input id="contrast" type="range" min="0.2" max="3" step="0.05" value="1">
      <output id="contrastVal">1.00</output>
    </div>
    <label for="gamma">Gamma</label>
    <div class="range-row">
      <input id="gamma" type="range" min="0.2" max="3" step="0.05" value="1">
      <output id="gammaVal">1.00</output>
    </div>
    <div class="checkbox-row">
      <input type="checkbox" id="color" checked>
      <label for="color">Colored characters</label>
    </div>
    <div class="checkbox-row">
      <input type="checkbox" id="invert">
      <label for="invert">Invert brightness</label>
    </div>

    <h2 style="margin-top:1.25rem;">Download</h2>
    <div class="actions">
      <button class="alt" id="dlTxt">.txt</button>
      <button class="alt" id="dlPng">.png</button>
      <button class="alt" id="dlHtml">.html</button>
    </div>
    <div class="hint">
      .png: lossless, exact original size (recommended).
      .txt: plain characters for copy-paste.
      .html: colored, selectable text with the font embedded.
    </div>

    <h2 style="margin-top:1.25rem;">Steam profile export</h2>
    <label for="steamCols">Width (braille characters per line)</label>
    <div class="range-row">
      <input id="steamCols" type="range" min="__STEAMMIN__" max="__STEAMMAX__" value="__STEAMCOLS__">
      <output id="steamColsVal">__STEAMCOLS__</output>
    </div>
    <label for="steamAspect">Line height / character width</label>
    <div class="range-row">
      <input id="steamAspect" type="range" min="1" max="3" step="0.05" value="__STEAMASPECT__">
      <output id="steamAspectVal">__STEAMASPECT__</output>
    </div>
    <label for="steamLimit">Character limit</label>
    <input id="steamLimit" type="text" inputmode="numeric" value="__STEAMLIMIT__">
    <div class="checkbox-row">
      <input type="checkbox" id="steamDither" checked>
      <label for="steamDither">Dithering (smoother shading)</label>
    </div>
    <div class="actions">
      <button class="alt" id="steamCopy">Copy for Steam</button>
      <button class="alt" id="steamTxt">.txt</button>
    </div>
    <div class="hint" id="steamInfo">Upload an image to generate the Steam text.</div>
    <pre id="steamPre" style="display:none; margin:.6rem 0 0; padding:.5rem; background:#000;
      border:1px solid var(--border); border-radius:5px; overflow:auto; max-height:260px;
      font-size:9px; line-height:1.1; font-family:'Segoe UI Symbol','DejaVu Sans','Noto Sans Symbols 2',sans-serif;"></pre>
    <div class="hint">Uses only braille characters (constant width even in Steam's proportional
      font) and the blank braille cell U+2800 instead of spaces, so nothing collapses or shifts.
      Bright areas become dots; use &ldquo;Invert brightness&rdquo; for light-on-dark vs dark-on-light.
      If the shape looks stretched or squashed on your profile, adjust the line-height ratio.
      The text is shrunk automatically to fit the character limit.</div>
  </aside>

  <section class="panel preview-panel">
    <div class="preview-head">
      <span id="previewMeta">Preview = exactly what the PNG will look like</span>
      <span>
        <button class="alt small" id="zoomBtn" title="Toggle between fit-to-window and actual pixels">Showing: fit &middot; click for 100%</button>
      </span>
    </div>
    <div id="stage" class="fit">
      <div id="busy">Rendering&hellip;</div>
      <div class="ph" id="ph">Upload an image to begin.</div>
      <img id="pimg" alt="Preview" hidden>
    </div>
  </section>
</main>

<script>
(() => {
  const CHARSETS = __CHARSETS_JSON__;
  CHARSETS.custom = "";

  const $ = id => document.getElementById(id);
  const status = $("status"), stage = $("stage"), pimg = $("pimg"), ph = $("ph");
  const busy = $("busy");

  let token = null, filename = "image", activeCharset = "ascii";
  let inflight = null, pendingTimer = null, zoomFull = false, objUrl = null;

  const showMessage = msg => { ph.textContent = msg; ph.hidden = false; pimg.hidden = true; };

  async function errorMessage(r, fallback) {
    try { const j = await r.json(); if (j && j.error) return j.error; } catch (_) {}
    return fallback;
  }

  // ---- charset chips ----
  Object.keys(CHARSETS).forEach(key => {
    const b = document.createElement("div");
    b.className = "chip" + (key === "ascii" ? " on" : "");
    b.textContent = key === "custom" ? "custom"
      : key + "  " + Array.from(CHARSETS[key]).slice(0, 6).join("");
    b.addEventListener("click", () => {
      activeCharset = key;
      document.querySelectorAll(".chip").forEach(c => c.classList.remove("on"));
      b.classList.add("on");
      if (key === "custom") $("customChars").focus();
      schedulePreview();
    });
    $("charsetChips").appendChild(b);
  });

  // ---- sliders ----
  const QUALITY = fs => fs <= 5 ? ["Maximum", 4] : fs <= 8 ? ["Very high", 3]
                      : fs <= 12 ? ["High", 2] : fs <= 18 ? ["Medium", 1] : ["Low (coarse)", 0];
  function updateBadge() {
    const [label, lvl] = QUALITY(parseInt($("fontSize").value, 10));
    $("qBadge").textContent = label;
    $("qBadge").className = "badge q" + lvl;
  }
  ["fontSize", "brightness", "contrast", "gamma"].forEach(id => {
    const el = $(id), out = $(id + "Val");
    const update = () => {
      out.textContent = id === "fontSize" ? el.value : parseFloat(el.value).toFixed(2);
      if (id === "fontSize") updateBadge();
    };
    el.addEventListener("input", () => { update(); schedulePreview(); });
    update();
  });
  $("invert").addEventListener("change", () => schedulePreview());
  $("color").addEventListener("change", () => schedulePreview());
  $("customChars").addEventListener("input", () => {
    if (activeCharset === "custom") schedulePreview();
  });

  // ---- zoom toggle ----
  $("zoomBtn").addEventListener("click", () => {
    zoomFull = !zoomFull;
    $("zoomBtn").textContent = zoomFull
      ? "Showing: 100% (actual pixels) \u00b7 click for fit"
      : "Showing: fit \u00b7 click for 100%";
    stage.className = zoomFull ? "full" : "fit";
    schedulePreview(0);
  });
  window.addEventListener("resize", () => { if (!zoomFull) schedulePreview(300); });

  // ---- upload ----
  $("file").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    filename = f.name.replace(/\.[^.]+$/, "");
    status.textContent = "Uploading...";
    const fd = new FormData();
    fd.append("image", f);
    try {
      const r = await fetch("/upload", { method: "POST", body: fd });
      if (!r.ok) throw new Error(await errorMessage(r, "Upload failed"));
      const data = await r.json();
      token = data.token;
      status.textContent = f.name;
      $("stOrig").textContent = `${data.width} \u00d7 ${data.height} px`;
      showMessage("Rendering...");
      schedulePreview(0);
    } catch (err) {
      token = null;
      status.textContent = "Upload failed";
      showMessage(err.message || "Upload failed.");
    }
  });

  // ---- preview ----
  function schedulePreview(delay = 200) {
    if (!token) return;
    clearTimeout(pendingTimer);
    pendingTimer = setTimeout(runPreview, delay);
    scheduleSteam(delay + 100);
  }

  function getParams(extra = {}) {
    const p = new FormData();
    p.append("token", token);
    p.append("fontSize", $("fontSize").value);
    p.append("charset", activeCharset);
    p.append("custom_chars", $("customChars").value);
    p.append("brightness", $("brightness").value);
    p.append("contrast", $("contrast").value);
    p.append("gamma", $("gamma").value);
    p.append("invert", $("invert").checked ? "1" : "0");
    p.append("color", $("color").checked ? "1" : "0");
    p.append("steamCols", $("steamCols").value);
    p.append("steamAspect", $("steamAspect").value);
    p.append("steamLimit", $("steamLimit").value);
    p.append("steamDither", $("steamDither").checked ? "1" : "0");
    for (const k in extra) p.append(k, extra[k]);
    return p;
  }

  function updateStats(h) {
    $("stOut").textContent = `${h.get("X-Out-W")} \u00d7 ${h.get("X-Out-H")} px (= original)`;
    const cols = +h.get("X-Cols"), rows = +h.get("X-Rows");
    $("stChars").textContent = `${cols} \u00d7 ${rows} = ${(cols * rows).toLocaleString()}`;
    $("stCell").textContent = `${h.get("X-Cell-W")} \u00d7 ${h.get("X-Cell-H")} px`;
    const miss = decodeURIComponent(h.get("X-Missing") || "");
    const warn = $("warn");
    if (miss) {
      warn.style.display = "block";
      warn.textContent = "No installed font has: " + Array.from(miss).join(" ") +
        " \u2014 install DejaVu Sans Mono for correct glyphs.";
    } else warn.style.display = "none";
  }

  async function runPreview() {
    if (!token) return;
    if (inflight) inflight.abort();
    const ctrl = new AbortController();
    inflight = ctrl;
    busy.style.display = "block";
    try {
      const dpr = window.devicePixelRatio || 1;
      const extra = zoomFull ? { zoom: "full" } : {
        zoom: "fit",
        fit_w: Math.round(stage.clientWidth * dpr),
        fit_h: Math.round(stage.clientHeight * dpr)
      };
      const r = await fetch("/preview", { method: "POST", body: getParams(extra), signal: ctrl.signal });
      if (!r.ok) throw new Error(await errorMessage(r, "Preview failed."));
      const blob = await r.blob();
      updateStats(r.headers);
      const url = URL.createObjectURL(blob);
      pimg.onload = () => {
        if (zoomFull) {
          pimg.style.width = (pimg.naturalWidth / dpr) + "px";
          pimg.style.height = "auto";
        } else { pimg.style.width = ""; pimg.style.height = ""; }
      };
      pimg.src = url;
      pimg.hidden = false; ph.hidden = true;
      if (objUrl) URL.revokeObjectURL(objUrl);
      objUrl = url;
    } catch (err) {
      if (err.name !== "AbortError") showMessage(err.message || "Preview failed.");
    } finally {
      if (inflight === ctrl) { inflight = null; busy.style.display = "none"; }
    }
  }

  // ---- downloads ----
  async function download(kind) {
    if (!token) { status.textContent = "Upload an image first"; return; }
    const buttons = document.querySelectorAll(".actions button");
    buttons.forEach(b => b.disabled = true);
    status.textContent = "Preparing " + kind.toUpperCase() + "...";
    try {
      const r = await fetch("/download/" + kind, { method: "POST", body: getParams() });
      if (!r.ok) throw new Error(await errorMessage(r, "Download failed"));
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = `${filename}_ascii.${kind}`;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
      status.textContent = "Downloaded " + a.download;
    } catch (err) {
      status.textContent = err.message || "Download failed";
    } finally {
      buttons.forEach(b => b.disabled = false);
    }
  }
  $("dlTxt").addEventListener("click", () => download("txt"));
  $("dlPng").addEventListener("click", () => download("png"));
  $("dlHtml").addEventListener("click", () => download("html"));

  // ---- Steam profile export ----
  let steamText = "", steamTimer = null, steamCtrl = null;
  function scheduleSteam(delay = 300) {
    if (!token) return;
    clearTimeout(steamTimer);
    steamTimer = setTimeout(runSteam, delay);
  }
  async function fetchSteam(signal) {
    const r = await fetch("/steam", { method: "POST", body: getParams(), signal });
    if (!r.ok) throw new Error(await errorMessage(r, "Steam export failed."));
    return r.json();
  }
  async function runSteam() {
    if (!token) return;
    if (steamCtrl) steamCtrl.abort();
    const ctrl = steamCtrl = new AbortController();
    try {
      const d = await fetchSteam(ctrl.signal);
      steamText = d.text;
      const pre = $("steamPre");
      pre.textContent = d.text; pre.style.display = "block";
      let msg = `${d.cols} \u00d7 ${d.rows} characters, ${d.chars.toLocaleString()} / ${d.limit.toLocaleString()} used`;
      if (d.cols < d.requested_cols) msg += ` (width reduced from ${d.requested_cols} to fit the limit)`;
      if (!d.fits) msg += " \u2014 still over the limit, lower the width.";
      $("steamInfo").textContent = msg;
    } catch (err) {
      if (err.name !== "AbortError") $("steamInfo").textContent = err.message || "Steam export failed.";
    } finally {
      if (steamCtrl === ctrl) steamCtrl = null;
    }
  }
  async function currentSteamText() {
    if (!token) { status.textContent = "Upload an image first"; return null; }
    clearTimeout(steamTimer);
    if (steamCtrl) steamCtrl.abort();
    const d = await fetchSteam();                    // always export the current settings
    steamText = d.text;
    return d.text;
  }
  ["steamCols", "steamAspect"].forEach(id => {
    const el = $(id), out = $(id + "Val");
    const update = () => { out.textContent = id === "steamCols" ? el.value : parseFloat(el.value).toFixed(2); };
    el.addEventListener("input", () => { update(); scheduleSteam(); });
    update();
  });
  $("steamLimit").addEventListener("input", () => scheduleSteam(500));
  $("steamDither").addEventListener("change", () => scheduleSteam(0));
  $("steamCopy").addEventListener("click", async () => {
    try {
      const t = await currentSteamText();
      if (t === null) return;
      try { await navigator.clipboard.writeText(t); }
      catch (_) {                                    // clipboard API unavailable (http, old browser)
        const ta = document.createElement("textarea");
        ta.value = t; document.body.appendChild(ta); ta.select();
        document.execCommand("copy"); ta.remove();
      }
      status.textContent = "Copied - paste it into your Steam profile";
    } catch (err) { status.textContent = err.message || "Copy failed"; }
  });
  $("steamTxt").addEventListener("click", async () => {
    try {
      const t = await currentSteamText();
      if (t === null) return;
      const url = URL.createObjectURL(new Blob([t], { type: "text/plain;charset=utf-8" }));
      const a = document.createElement("a");
      a.href = url; a.download = `${filename}_steam.txt`;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
      status.textContent = "Downloaded " + a.download;
    } catch (err) { status.textContent = err.message || "Download failed"; }
  });

  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); schedulePreview(0); }
  });
})();
</script>
</body></html>"""

PAGE = (PAGE.replace("__CHARSETS_JSON__", json.dumps(CHARSETS, ensure_ascii=False))
            .replace("__MINFS__", str(MIN_FONT_SIZE))
            .replace("__MAXFS__", str(MAX_FONT_SIZE))
            .replace("__DEFFS__", str(DEFAULT_FONT_SIZE))
            .replace("__STEAMMIN__", str(STEAM_MIN_COLS))
            .replace("__STEAMMAX__", str(STEAM_MAX_COLS))
            .replace("__STEAMCOLS__", str(STEAM_DEFAULT_COLS))
            .replace("__STEAMASPECT__", f"{STEAM_DEFAULT_ASPECT:.2f}")
            .replace("__STEAMLIMIT__", str(STEAM_DEFAULT_LIMIT)))

# ---------------------------------------------------------------------------
# Errors -> JSON so the UI can show a real message
# ---------------------------------------------------------------------------

@app.errorhandler(HTTPException)
def handle_http_error(e):
    return jsonify(error=e.description), e.code


@app.errorhandler(Exception)
def handle_error(e):
    app.logger.exception("Unhandled error")
    return jsonify(error=f"{type(e).__name__}: {e}"), 500

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("image")
    if not f:
        return jsonify(error="No file received."), 400
    try:
        token, (w, h) = _store_upload(f.read(), f.filename or "image")
    except Exception as e:
        return jsonify(error=f"Could not read image: {e}"), 400
    return jsonify(token=token, width=w, height=h)


def _png_bytes(img: Image.Image, level: int, dpi=None) -> io.BytesIO:
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=level, dpi=dpi)
    buf.seek(0)
    return buf


@app.route("/preview", methods=["POST"])
def preview():
    token = request.form.get("token")
    if not token:
        return jsonify(error="Missing upload token."), 400
    entry = _get_upload(token)
    p = parse_params(request.form)
    img, info = render_image(entry["rgb"], p)

    if request.form.get("zoom") != "full":          # preview-only downscale
        try:
            fw, fh = int(request.form.get("fit_w", 0)), int(request.form.get("fit_h", 0))
        except ValueError:
            fw = fh = 0
        if fw > 0 and fh > 0 and (img.width > fw or img.height > fh):
            img.thumbnail((fw, fh), RESAMPLE_LANCZOS)

    resp = Response(_png_bytes(img, 1).getvalue(), mimetype="image/png")
    resp.headers.update({
        "X-Out-W": str(info["w"]), "X-Out-H": str(info["h"]),
        "X-Cols": str(info["cols"]), "X-Rows": str(info["rows"]),
        "X-Cell-W": str(info["cw"]), "X-Cell-H": str(info["ch"]),
        "X-Missing": quote(info["missing"]),
        "Cache-Control": "no-store",
    })
    return resp


@app.route("/download/<kind>", methods=["POST"])
def download(kind):
    token = request.form.get("token")
    if not token:
        return jsonify(error="Missing upload token."), 400
    entry = _get_upload(token)
    rgb = entry["rgb"]
    p = parse_params(request.form)

    if kind == "png":
        img, info = render_image(rgb, p)
        assert img.size == (rgb.shape[1], rgb.shape[0]), "output size mismatch"
        return send_file(_png_bytes(img, 6, entry["dpi"]), mimetype="image/png",
                         as_attachment=True, download_name="ascii.png")
    if kind == "txt":
        text = render_plain_text(rgb, p)
        return send_file(io.BytesIO(text.encode("utf-8")), mimetype="text/plain",
                         as_attachment=True, download_name="ascii.txt")
    if kind == "html":
        page = render_html_file(rgb, p)
        return send_file(io.BytesIO(page.encode("utf-8")), mimetype="text/html",
                         as_attachment=True, download_name="ascii.html")
    return jsonify(error="Unknown format."), 400


@app.route("/steam", methods=["POST"])
def steam():
    token = request.form.get("token")
    if not token:
        return jsonify(error="Missing upload token."), 400
    entry = _get_upload(token)
    text, info = render_steam_text(entry["rgb"], parse_params(request.form),
                                   parse_steam_params(request.form))
    resp = jsonify(text=text, **info)
    resp.headers["Cache-Control"] = "no-store"
    return resp


if __name__ == "__main__":
    print("ASCII converter running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)