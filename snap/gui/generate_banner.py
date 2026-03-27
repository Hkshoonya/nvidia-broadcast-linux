#!/usr/bin/env python3
"""Generate Snap Store featured banner (1920x480) for NV Broadcast."""

from PIL import Image, ImageDraw, ImageFont
import math
import os

W, H = 1920, 480
img = Image.new("RGBA", (W, H))
draw = ImageDraw.Draw(img)

# --- Background: dark gradient with subtle grid ---
for y in range(H):
    r = int(14 + (y / H) * 12)
    g = int(14 + (y / H) * 14)
    b = int(18 + (y / H) * 16)
    for x in range(W):
        # Slight horizontal gradient (lighter center)
        cx = abs(x - W // 2) / (W // 2)
        fade = 1.0 - cx * 0.15
        img.putpixel((x, y), (int(r * fade), int(g * fade), int(b * fade), 255))

# Subtle grid lines
for x in range(0, W, 60):
    for y in range(H):
        px = img.getpixel((x, y))
        img.putpixel((x, y), (min(px[0] + 6, 255), min(px[1] + 8, 255), min(px[2] + 6, 255), 255))
for y in range(0, H, 60):
    for x in range(W):
        px = img.getpixel((x, y))
        img.putpixel((x, y), (min(px[0] + 6, 255), min(px[1] + 8, 255), min(px[2] + 6, 255), 255))

# --- NVIDIA green accent line at top ---
for x in range(W):
    for t in range(3):
        alpha = int(200 * (1 - t / 3))
        img.putpixel((x, t), (118, 185, 0, alpha))

# --- Broadcast wave arcs (right side decoration) ---
overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
odraw = ImageDraw.Draw(overlay)

# Signal waves emanating from right
cx_wave, cy_wave = 1580, 240
for i, radius in enumerate([120, 170, 220, 270, 320]):
    opacity = int(80 - i * 12)
    arc_w = 4 - i * 0.5
    odraw.arc(
        [cx_wave - radius, cy_wave - radius, cx_wave + radius, cy_wave + radius],
        start=-60, end=60,
        fill=(118, 185, 0, max(opacity, 20)),
        width=max(int(arc_w), 2),
    )

# Camera lens circles (right side)
for i, r in enumerate([55, 40, 22]):
    opacity = [50, 70, 100][i]
    odraw.ellipse(
        [cx_wave - r, cy_wave - r, cx_wave + r, cy_wave + r],
        outline=(118, 185, 0, opacity),
        width=2 + i,
    )
# Inner filled lens
odraw.ellipse(
    [cx_wave - 10, cy_wave - 10, cx_wave + 10, cy_wave + 10],
    fill=(118, 185, 0, 140),
)
# Recording dot
odraw.ellipse(
    [cx_wave + 35, cy_wave - 50, cx_wave + 47, cy_wave - 38],
    fill=(255, 51, 51, 180),
)

# Additional decorative elements - floating particles
import random
random.seed(42)
for _ in range(40):
    px = random.randint(1200, 1900)
    py = random.randint(30, 450)
    size = random.randint(1, 3)
    opacity = random.randint(20, 60)
    odraw.ellipse([px - size, py - size, px + size, py + size], fill=(118, 185, 0, opacity))

# Left side accent - vertical bar
odraw.rectangle([60, 80, 66, 400], fill=(118, 185, 0, 100))
odraw.rectangle([60, 80, 66, 200], fill=(118, 185, 0, 180))  # brighter top portion

img = Image.alpha_composite(img, overlay)
draw = ImageDraw.Draw(img)

# --- Typography ---
font_bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font_regular = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# App name - "NV" in green, "Broadcast" in white
title_font = ImageFont.truetype(font_bold, 86)
sub_font = ImageFont.truetype(font_regular, 32)
tag_font = ImageFont.truetype(font_bold, 28)
feature_font = ImageFont.truetype(font_regular, 22)

x_text = 100
y_base = 100

# "NV" part
nv_bbox = draw.textbbox((0, 0), "NV ", font=title_font)
nv_w = nv_bbox[2] - nv_bbox[0]
draw.text((x_text, y_base), "NV ", fill=(118, 185, 0, 255), font=title_font)
# "Broadcast" part
draw.text((x_text + nv_w, y_base), "Broadcast", fill=(240, 240, 240, 255), font=title_font)

# Tagline
y_tag = y_base + 105
draw.text(
    (x_text + 4, y_tag),
    "AI-Powered Virtual Camera for Linux",
    fill=(180, 180, 180, 255),
    font=sub_font,
)

# Divider line
y_div = y_tag + 50
draw.line([(x_text + 4, y_div), (x_text + 420, y_div)], fill=(118, 185, 0, 120), width=2)

# Feature pills
features = [
    "Background Blur",
    "Background Replace",
    "Eye Contact",
    "Face Relight",
    "Voice Studio",
    "AI Transcription",
]

y_feat = y_div + 20
x_feat = x_text + 4
pill_padding_x = 16
pill_padding_y = 8
pill_spacing = 14

for feat in features:
    bbox = draw.textbbox((0, 0), feat, font=feature_font)
    fw = bbox[2] - bbox[0]
    fh = bbox[3] - bbox[1]
    pill_w = fw + pill_padding_x * 2
    pill_h = fh + pill_padding_y * 2

    # Check if we need to wrap to next line
    if x_feat + pill_w > 1100:
        x_feat = x_text + 4
        y_feat += pill_h + 12

    # Pill background
    pill_overlay = Image.new("RGBA", (pill_w, pill_h), (0, 0, 0, 0))
    pill_draw = ImageDraw.Draw(pill_overlay)
    pill_draw.rounded_rectangle(
        [0, 0, pill_w - 1, pill_h - 1],
        radius=pill_h // 2,
        fill=(118, 185, 0, 30),
        outline=(118, 185, 0, 100),
        width=1,
    )
    img.paste(
        Image.alpha_composite(
            img.crop((x_feat, y_feat, x_feat + pill_w, y_feat + pill_h)),
            pill_overlay,
        ),
        (x_feat, y_feat),
    )
    # Pill text
    draw = ImageDraw.Draw(img)
    draw.text(
        (x_feat + pill_padding_x, y_feat + pill_padding_y - 2),
        feat,
        fill=(200, 220, 180, 255),
        font=feature_font,
    )
    x_feat += pill_w + pill_spacing

# "Works with" line
y_works = y_feat + 60
draw.text(
    (x_text + 4, y_works),
    "Works with Zoom  \u2022  Chrome  \u2022  OBS  \u2022  Discord  \u2022  Any v4l2 app",
    fill=(120, 120, 120, 255),
    font=ImageFont.truetype(font_regular, 18),
)

# --- Bottom green accent line ---
for x in range(W):
    for t in range(3):
        y = H - 1 - t
        alpha = int(200 * (1 - t / 3))
        img.putpixel((x, y), (118, 185, 0, alpha))

# --- Save ---
out_path = os.path.join(os.path.dirname(__file__), "banner.png")
img.convert("RGB").save(out_path, "PNG", quality=95)
print(f"Banner saved: {out_path} ({W}x{H})")

# Also save a 720x240 thumbnail version
thumb = img.resize((720, 240), Image.LANCZOS)
thumb_path = os.path.join(os.path.dirname(__file__), "banner_720.png")
thumb.convert("RGB").save(thumb_path, "PNG", quality=95)
print(f"Thumbnail saved: {thumb_path} (720x240)")
