"""
Capture a scrolling GIF of the dashboard.
Usage: python scripts/capture_dashboard_gif.py
Output: docs/dashboard.gif
"""
import sys
from pathlib import Path
from io import BytesIO

from PIL import Image
from playwright.sync_api import sync_playwright

REPO = Path(__file__).parent.parent
HTML_PATH = REPO / "data" / "exports" / "dashboard.html"
OUT_PATH = REPO / "docs" / "dashboard.gif"

VIEWPORT_W = 1440
VIEWPORT_H = 860

PX_PER_FRAME = 36
RAMP_FRAC = 0.18
MIN_FRAMES = 40
FRAME_MS = 50
TARGET_W = 700
GIF_COLORS = 64


def build_scroll_sequence(max_scroll):
    frames = max(round(max_scroll / PX_PER_FRAME), MIN_FRAMES)
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


def main():
    if not HTML_PATH.exists():
        print(f"ERROR: {HTML_PATH} not found. Run: python scripts/run_analysis.py --export")
        sys.exit(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    url = HTML_PATH.as_uri()

    print(f"Launching browser → {url}")
    frames_pil = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": VIEWPORT_W, "height": VIEWPORT_H})

        page.goto(url, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(2500)

        page_h = page.evaluate("document.documentElement.scrollHeight")
        max_scroll = max(page_h - VIEWPORT_H, 1)
        print(f"Page height: {page_h}px  max_scroll: {max_scroll}px")

        scroll_seq = build_scroll_sequence(max_scroll)
        total = len(scroll_seq)

        for idx, frac in enumerate(scroll_seq):
            scroll_y = int(frac * max_scroll)
            page.evaluate(f"window.scrollTo(0, {scroll_y})")
            page.wait_for_timeout(18)

            png_bytes = page.screenshot(type="png")
            img = Image.open(BytesIO(png_bytes)).convert("RGB")

            scale = TARGET_W / img.width
            new_h = int(img.height * scale)
            img = img.resize((TARGET_W, new_h), Image.LANCZOS)

            img_q = img.quantize(colors=GIF_COLORS, method=Image.Quantize.MEDIANCUT, dither=0)
            frames_pil.append(img_q)

            if (idx + 1) % 10 == 0 or idx == total - 1:
                print(f"  Captured {idx + 1}/{total} frames (scroll={scroll_y}px)")

        browser.close()

    print(f"Assembling GIF → {OUT_PATH}")
    frames_pil[0].save(
        OUT_PATH,
        format="GIF",
        save_all=True,
        append_images=frames_pil[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
    )
    size_mb = OUT_PATH.stat().st_size / 1_048_576
    print(f"Done. {len(frames_pil)} frames · {size_mb:.1f} MB → {OUT_PATH}")


if __name__ == "__main__":
    main()
