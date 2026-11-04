"""
render_code_screenshots.py
Generates beautiful, high-resolution code editor screenshots for prototype modules:
1. main.py (Cascaded Controller & Orchestrator)
2. cascaded_auth/ksd.py (Tier 1 Keystroke Dynamics & Dynamic Threshold)
3. cascaded_auth/cv_check.py (Tier 2 FaceMesh Iris & Mahalanobis Distance Check)
4. cv/matcher.py (Biometric Distance & Embedding Matcher)
5. cv/session.py (Burst Camera Acquisition & Landmark Tracking)
6. dashboard.py (Live Streamlit Security Telemetry & Control Panel)
"""

import os
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import ImageFormatter

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code_screenshots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SNIPPETS = {
    "01_main_orchestrator": {
        "title": "main.py — Two-Tier Cascaded System Controller",
        "badge": "Orchestrator / Main Thread",
        "file": "main.py",
        "lines": (24, 82),
    },
    "02_ksd_tier1_engine": {
        "title": "cascaded_auth/ksd.py — Tier 1 Keystroke Dynamics & Dynamic Threshold",
        "badge": "Tier 1: KSD",
        "file": "cascaded_auth/ksd.py",
        "lines": (38, 96),
    },
    "03_cv_tier2_verification": {
        "title": "cascaded_auth/cv_check.py — Tier 2 FaceMesh & Mahalanobis Verification",
        "badge": "Tier 2: Computer Vision",
        "file": "cascaded_auth/cv_check.py",
        "lines": (50, 108),
    },
    "04_cv_matcher": {
        "title": "cv/matcher.py — Biometric Embedding & Threshold Matcher",
        "badge": "Biometric Matcher",
        "file": "cv/matcher.py",
        "lines": (18, 76),
    },
    "05_cv_session": {
        "title": "cv/session.py — Camera Burst Session & Landmark Capture",
        "badge": "Vision Acquisition",
        "file": "cv/session.py",
        "lines": (38, 95),
    },
    "06_dashboard_telemetry": {
        "title": "dashboard.py — Real-time Security Dashboard & Telemetry",
        "badge": "Monitoring UI",
        "file": "dashboard.py",
        "lines": (20, 78),
    },
}

def render_editor_window(filename_key, meta):
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), meta["file"])
    with open(filepath, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    
    start, end = meta["lines"]
    snippet_lines = all_lines[start-1:end]
    code_text = "".join(snippet_lines)

    # Render syntax highlighted code
    formatter = ImageFormatter(
        font_name="Consolas",
        font_size=15,
        line_numbers=True,
        line_number_start=start,
        style="monokai",
        line_number_bg="#1e1e1e",
        line_number_fg="#6e7681",
    )
    code_bytes = highlight(code_text, PythonLexer(), formatter)
    code_img = Image.open(BytesIO(code_bytes))

    # Chrome / Window dimensions
    padding = 24
    header_height = 48
    win_w = max(code_img.width + padding * 2, 920)
    win_h = code_img.height + header_height + padding * 2

    # Canvas with modern dark background
    canvas_w = win_w + 60
    canvas_h = win_h + 60
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (13, 17, 23, 255))
    draw = ImageDraw.Draw(canvas)

    # Window card coordinates
    wx0 = 30
    wy0 = 30
    wx1 = wx0 + win_w
    wy1 = wy0 + win_h
    corner_radius = 12

    # Drop shadow / glow border
    draw.rounded_rectangle([wx0-2, wy0-2, wx1+2, wy1+2], corner_radius+2, fill=(48, 54, 61, 255))
    # Window body
    draw.rounded_rectangle([wx0, wy0, wx1, wy1], corner_radius, fill=(30, 30, 30, 255))

    # Window title bar
    draw.rounded_rectangle([wx0, wy0, wx1, wy0 + header_height], corner_radius, fill=(37, 37, 38, 255))
    draw.rectangle([wx0, wy0 + header_height - 10, wx1, wy0 + header_height], fill=(37, 37, 38, 255))

    # Window controls (macOS / modern IDE dots)
    dot_y = wy0 + 24
    draw.ellipse([wx0 + 18, dot_y - 6, wx0 + 30, dot_y + 6], fill=(255, 95, 87, 255))   # Close
    draw.ellipse([wx0 + 38, dot_y - 6, wx0 + 50, dot_y + 6], fill=(255, 189, 46, 255))  # Min
    draw.ellipse([wx0 + 58, dot_y - 6, wx0 + 70, dot_y + 6], fill=(40, 201, 64, 255))   # Max

    # Load font for title & badge
    try:
        title_font = ImageFont.truetype("segoeui.ttf", 15)
        badge_font = ImageFont.truetype("segoeuib.ttf", 12)
    except Exception:
        title_font = ImageFont.load_default()
        badge_font = ImageFont.load_default()

    # Title text
    title_text = meta["title"]
    draw.text((wx0 + 85, wy0 + 14), title_text, fill=(204, 204, 204, 255), font=title_font)

    # Badge on right
    badge_text = meta["badge"]
    badge_w = len(badge_text) * 8 + 18
    bx1 = wx1 - 20
    bx0 = bx1 - badge_w
    draw.rounded_rectangle([bx0, wy0 + 12, bx1, wy0 + 34], 5, fill=(59, 130, 246, 200))
    draw.text((bx0 + 9, wy0 + 14), badge_text, fill=(255, 255, 255, 255), font=badge_font)

    # Paste code image into window body
    canvas.paste(code_img, (wx0 + padding, wy0 + header_height + padding // 2))

    # Save output
    out_path = os.path.join(OUTPUT_DIR, f"{filename_key}.png")
    canvas.save(out_path, "PNG")
    print(f"Generated screenshot: {out_path} ({canvas_w}x{canvas_h})")
    return out_path

if __name__ == "__main__":
    for key, meta in SNIPPETS.items():
        render_editor_window(key, meta)
    print("\nAll module screenshots rendered successfully!")
