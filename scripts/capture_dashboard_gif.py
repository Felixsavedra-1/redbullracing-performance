"""Capture a looping GIF of data/exports/dashboard.html using Playwright + Pillow.

Usage:
    python scripts/capture_dashboard_gif.py

Output: docs/dashboard.gif
"""

import io
import os
import sys
import time

from PIL import Image
from playwright.sync_api import sync_playwright

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

HTML_PATH = os.path.join(REPO_ROOT, "data", "exports", "dashboard.html")
OUTPUT_PATH = os.path.join(REPO_ROOT, "docs", "dashboard.gif")

VIEWPORT_W = 1400
VIEWPORT_H = 900
INIT_WAIT_S = 2.5
FRAMES = 30
FRAME_INTERVAL_MS = 120
COLORS = 128


def _capture_frames(page) -> list[Image.Image]:
    page.wait_for_load_state("networkidle")
    time.sleep(INIT_WAIT_S)

    frames = []
    for _ in range(FRAMES):
        data = page.screenshot(type="png")
        img = Image.open(io.BytesIO(data)).convert("RGB")
        frames.append(img.quantize(colors=COLORS, method=Image.Quantize.MEDIANCUT))
        time.sleep(FRAME_INTERVAL_MS / 1000)

    return frames


def main() -> None:
    if not os.path.exists(HTML_PATH):
        print(f"error: dashboard not found at {HTML_PATH}")
        print("       run: python scripts/run_analysis.py --export")
        sys.exit(1)

    url = f"file://{HTML_PATH}"
    print(f"Capturing {FRAMES} frames from {HTML_PATH}...")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": VIEWPORT_W, "height": VIEWPORT_H})
        page.goto(url)
        frames = _capture_frames(page)
        browser.close()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    frames[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=FRAME_INTERVAL_MS,
        optimize=True,
    )

    size_mb = os.path.getsize(OUTPUT_PATH) / 1_000_000
    print(f"  → {OUTPUT_PATH}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
