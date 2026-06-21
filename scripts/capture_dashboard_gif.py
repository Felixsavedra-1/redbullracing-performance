"""
Capture a cinematic widescreen GIF of the dashboard.
Usage: python scripts/capture_dashboard_gif.py
Output: docs/dashboard.gif

The dashboard scrolls through a 2.39:1 anamorphic letterbox slot (a short, wide
viewport) for a cinemascope feel. Frames are captured at 2x device scale for
crisp text, then encoded with a two-pass ffmpeg palettegen/paletteuse pipeline
(256-color per-scene palette + dithering) for vivid, band-free output.
Requires ffmpeg on PATH.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from logging_utils import setup_logging

_log = setup_logging()

REPO = Path(__file__).parent.parent
HTML_PATH = REPO / "data" / "exports" / "dashboard.html"
OUT_PATH = REPO / "docs" / "dashboard.gif"

# Short, wide viewport (~2.39:1) renders the desktop layout through an
# anamorphic letterbox slot; captured at 2x for crisp downscaling.
VIEWPORT_W = 1500
VIEWPORT_H = 628
DEVICE_SCALE = 2

TARGET_FRAMES = 84
RAMP_FRAC = 0.18
FRAME_MS = 60
TARGET_W = 1200
TARGET_H = 502
# Rational framerate so GIF centisecond rounding lands on the exact FRAME_MS
# delay (e.g. 1000/60 → 6cs = 60ms); a plain int would truncate to 50ms.
FPS_ARG = f"1000/{FRAME_MS}"


def build_scroll_sequence(frames=TARGET_FRAMES):
    ramp = max(int(frames * RAMP_FRAC), 1)

    vel = []
    for i in range(frames):
        if i < ramp:
            v = (i + 1) / ramp
        elif i >= frames - ramp:
            v = (frames - i) / ramp
        else:
            v = 1.0
        vel.append(v)

    cum, s = [], 0.0
    for v in vel:
        s += v
        cum.append(s)
    return [c / cum[-1] for c in cum]


def encode_gif(frame_dir: Path):
    """Two-pass ffmpeg encode: build an optimized palette, then apply it."""
    pattern = str(frame_dir / "f%04d.png")
    palette = frame_dir / "palette.png"
    scale = f"scale={TARGET_W}:{TARGET_H}:flags=lanczos"

    subprocess.run(
        ["ffmpeg", "-y", "-framerate", FPS_ARG, "-i", pattern,
         "-vf", f"{scale},palettegen=max_colors=256:stats_mode=diff",
         str(palette)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", FPS_ARG, "-i", pattern, "-i", str(palette),
         "-lavfi", f"{scale}[x];[x][1:v]paletteuse=dither=sierra2_4a:diff_mode=rectangle",
         "-loop", "0", str(OUT_PATH)],
        check=True, capture_output=True,
    )


def main():
    if not HTML_PATH.exists():
        _log.error("%s not found. Run: python scripts/run_analysis.py --export", HTML_PATH)
        sys.exit(1)

    if shutil.which("ffmpeg") is None:
        _log.error("ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg`).")
        sys.exit(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    url = HTML_PATH.as_uri()

    _log.info("Launching browser → %s", url)

    with tempfile.TemporaryDirectory(prefix="dashgif_") as tmp:
        frame_dir = Path(tmp)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                viewport={"width": VIEWPORT_W, "height": VIEWPORT_H},
                device_scale_factor=DEVICE_SCALE,
            )

            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(2500)

            page_h = page.evaluate("document.documentElement.scrollHeight")
            max_scroll = max(page_h - VIEWPORT_H, 1)
            _log.info("Page height: %dpx  max_scroll: %dpx", page_h, max_scroll)

            scroll_seq = build_scroll_sequence()
            total = len(scroll_seq)

            for idx, frac in enumerate(scroll_seq):
                scroll_y = int(frac * max_scroll)
                page.evaluate(f"window.scrollTo(0, {scroll_y})")
                page.wait_for_timeout(18)

                page.screenshot(path=str(frame_dir / f"f{idx:04d}.png"), type="png")

                if (idx + 1) % 10 == 0 or idx == total - 1:
                    _log.info("  Captured %d/%d frames (scroll=%dpx)", idx + 1, total, scroll_y)

            browser.close()

        _log.info("Encoding GIF (ffmpeg, %dms/frame, %dx%d, 256-color) → %s",
                  FRAME_MS, TARGET_W, TARGET_H, OUT_PATH)
        encode_gif(frame_dir)

    size_mb = OUT_PATH.stat().st_size / 1_048_576
    _log.info("Done. %d frames · %.1f MB → %s", total, size_mb, OUT_PATH)


if __name__ == "__main__":
    main()
