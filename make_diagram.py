"""Generate the graph diagram PNG for the assignment submission.

Pure Pillow — no graphviz/matplotlib dependency. Draws the node network with
boxes, arrows and route labels, including the revision/retry path and the
loop guard.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "diagrams" / "graph_diagram.png"

# Each node: (name, x, y, w, h, fill_color)
NODES = [
    ("triage", 280, 20, 120, 44, "#ffd966"),
    ("retrieve", 520, 140, 120, 44, "#a9cce3"),
    ("clarify", 60, 140, 120, 44, "#d5a6bd"),
    ("escalate", 60, 300, 120, 44, "#e6b8af"),
    ("oos", 520, 300, 120, 44, "#c9daf8"),
    ("safe_failure", 280, 300, 120, 44, "#f4cccc"),
    ("generate", 160, 460, 120, 44, "#a9cce3"),
    ("verify", 420, 460, 120, 44, "#d0e0e3"),
    ("revise", 300, 620, 120, 44, "#ffe599"),
    ("END", 600, 620, 120, 44, "#b6d7a8"),
]

# Edges: (source_name, target_name, route_label)
ARROWS = [
    ("triage", "retrieve", "answerable"),
    ("triage", "clarify", "requires_clarification"),
    ("triage", "escalate", "requires_escalation"),
    ("triage", "oos", "out_of_scope"),
    ("triage", "safe_failure", "safe_failure"),
    ("retrieve", "generate", ""),
    ("generate", "verify", ""),
    ("verify", "revise", "fail & revisions < 1"),
    ("revise", "generate", "retry"),
    ("verify", "safe_failure", "fail & revisions >= 1"),
    ("verify", "END", "pass"),
]

POS = {name: (x, y, w, h) for name, x, y, w, h, _ in NODES}


def center(name: str):
    x, y, w, h = POS[name]
    return (x + w // 2, y + h // 2)


def main() -> None:
    W, H = 760, 700
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
        font_small = ImageFont.truetype("arial.ttf", 10)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    # arrows first (under boxes)
    import math

    for src, dst, label in ARROWS:
        sx, sy = center(src)
        dx, dy = center(dst)
        d.line([(sx, sy), (dx, dy)], fill="#555555", width=2)
        ang = math.atan2(dy - sy, dx - sx)
        for off in (0.4, -0.4):
            ax = dx - 10 * math.cos(ang + off)
            ay = dy - 10 * math.sin(ang + off)
            d.line([(dx, dy), (ax, ay)], fill="#555555", width=2)
        if label:
            mx, my = (sx + dx) // 2, (sy + dy) // 2
            d.text((mx - 30, my - 14), label, fill="#333333", font=font_small)

    for name, x, y, w, h, color in NODES:
        d.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=color,
                            outline="#444444", width=2)
        d.text((x + 8, y + 14), name, fill="#111111", font=font)

    d.text((10, 10), "OrbitDesk Support Agent Network - graph workflow", fill="#111111", font=font)
    d.text((10, H - 40),
           "Loop guard: max_steps=12, max_visits=3/node  |  typed state  |  node logs  |  deterministic terminals",
           fill="#555555", font=font_small)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(f"diagram written to {OUT}")


if __name__ == "__main__":
    main()
