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

# (scroll_y_fraction, hold_frames) — hold longer at key visual sections
SCROLL_WAYPOINTS = [
    (0.000, 6),   # status bar + header
    (0.030, 4),   # stats row
    (0.100, 4),   # car viewer
    (0.200, 4),   # chart start
    (0.280, 6),   # championship trajectory
    (0.400, 4),   # positions + points gap row
    (0.540, 6),   # performance matrix heatmap
    (0.680, 6),   # grid vs finish scatter
    (0.840, 7),   # telemetry panels
    (1.000, 6),   # footer
]

INTERP_STEPS = 5   # frames between waypoints
FRAME_MS = 70      # ms per frame (~14 fps)


def eased(t: float) -> float:
    return t * t * (3 - 2 * t)


def build_scroll_sequence():
    positions = []
    for i in range(len(SCROLL_WAYPOINTS) - 1):
        y0, hold0 = SCROLL_WAYPOINTS[i]
        y1, _     = SCROLL_WAYPOINTS[i + 1]
        positions.extend([y0] * hold0)
        for step in range(1, INTERP_STEPS + 1):
            t = eased(step / INTERP_STEPS)
            positions.append(y0 + (y1 - y0) * t)
    y_last, hold_last = SCROLL_WAYPOINTS[-1]
    positions.extend([y_last] * hold_last)
    return positions


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
        # Wait for Three.js + Space Mono font
        page.wait_for_timeout(2500)

        page_h = page.evaluate("document.documentElement.scrollHeight")
        max_scroll = max(page_h - VIEWPORT_H, 1)
        print(f"Page height: {page_h}px  max_scroll: {max_scroll}px")

        scroll_seq = build_scroll_sequence()
        total = len(scroll_seq)

        for idx, frac in enumerate(scroll_seq):
            scroll_y = int(frac * max_scroll)
            page.evaluate(f"window.scrollTo(0, {scroll_y})")
            page.wait_for_timeout(18)

            png_bytes = page.screenshot(type="png")
            img = Image.open(BytesIO(png_bytes)).convert("RGB")

            # Downsample to 900px wide to balance quality and file size
            target_w = 900
            scale = target_w / img.width
            new_h = int(img.height * scale)
            img = img.resize((target_w, new_h), Image.LANCZOS)

            # Quantize to 72 colours — sufficient for dark UI palette
            img_q = img.quantize(colors=72, method=Image.Quantize.MEDIANCUT, dither=0)
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
