"""High-resolution, unambiguous visual evidence for tracked objects.

Image operations run in the detector environment, not the Agent environment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def preprocess_image(image):
    """Match DGGT's RGB/BICUBIC/518-wide, multiple-of-14, center-crop geometry."""
    from PIL import Image

    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    image = image.convert("RGB")
    height = round(image.height * 518 / image.width / 14) * 14
    if height < 14:
        raise ValueError("image aspect ratio is unsupported")
    resized = image.resize((518, height), Image.Resampling.BICUBIC)
    top = max(0, (height - 518) // 2)
    return resized.crop((0, top, 518, min(height, top + 518)))


def original_box(box, original_size):
    """Map a DGGT box back into the original RGB image (including crop offset)."""
    width, height = original_size
    scaled_height = round(height * 518 / width / 14) * 14
    top = max(0, (scaled_height - 518) // 2)
    x1, y1, x2, y2 = box
    return [
        x1 * width / 518,
        (y1 + top) * height / scaled_height,
        x2 * width / 518,
        (y2 + top) * height / scaled_height,
    ]


def write_object_cards(instances, originals, directory):
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ]:
        if Path(font_path).is_file():
            font = ImageFont.truetype(font_path, 19)
            break
    cards = {}
    for instance in instances:
        track = instance["frames"]
        samples = sorted(
            {round((len(track) - 1) * fraction) for fraction in (0, 0.5, 1)}
        )
        canvas = Image.new("RGB", (1280, 70 + 300 * len(samples)), "#111827")
        draw = ImageDraw.Draw(canvas)
        ident = instance["instance_id"]
        draw.text(
            (15, 12),
            f"TARGET C{ident} | left: object only | right: context; cyan box = target",
            font=font,
            fill="white",
        )
        indices = []
        sizes = []
        for row, sample in enumerate(samples):
            record = track[sample]
            index = record["frame_index"]
            original = originals[index].convert("RGB")
            x1, y1, x2, y2 = original_box(record["bbox"], original.size)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(original.width, x2), min(original.height, y2)
            if x2 <= x1 or y2 <= y1:
                raise ValueError("object crop lies outside source image")
            tight = original.crop((int(x1), int(y1), int(x2 + 0.999), int(y2 + 0.999)))
            sizes.append(list(tight.size))
            indices.append(index)
            tight = ImageOps.contain(tight, (330, 240))
            offset = 70 + row * 300
            canvas.paste(tight, (15, offset + 32))
            draw.text(
                (15, offset), f"C{ident} frame {index}", font=font, fill="#66ddff"
            )
            # A wide context window keeps lanes and nearby reference objects visible.
            context = ImageOps.contain(original, (900, 260))
            sx, sy = context.width / original.width, context.height / original.height
            overlay = ImageDraw.Draw(context)
            overlay.rectangle(
                (x1 * sx, y1 * sy, x2 * sx, y2 * sy), outline="#00ffff", width=4
            )
            canvas.paste(context, (365, offset + 25))
        path = directory / f"object_C{ident}.jpg"
        canvas.save(path, quality=96)
        cards[str(ident)] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "frame_indices": indices,
            "native_crop_sizes": sizes,
        }
    return cards
