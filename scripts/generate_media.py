"""Generate README SVG charts and schematic animation from fresh MuJoCo rollouts.

Usage: python scripts/generate_media.py --out docs/media
Requires: pip install -e '.[mujoco]'
"""
from __future__ import annotations
import argparse
from pathlib import Path
import html
from vla_council.causalforge.mujoco_push import Physics
from vla_council.causalforge.simulate_scene import simulate as run_pusher
from vla_council.causalforge.tabletop_push import simulate as run_tabletop

DARK = "#111827"
GRID = "#374151"
FG = "#f9fafb"
MUTED = "#9ca3af"


def chart(title, subtitle, traces, duration, max_position):
    """Return standalone vector plot with paths calculated from measured positions."""
    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 460">',
        f'<rect width="980" height="460" rx="18" fill="{DARK}"/>',
        f'<text x="46" y="58" font-family="Arial" font-size="28" font-weight="bold" fill="{FG}">{html.escape(title)}</text>',
        f'<text x="46" y="85" font-family="Arial" font-size="15" fill="{MUTED}">{html.escape(subtitle)}</text>',
    ]
    for k in range(6):
        yval = max_position * k / 5
        y = 380 - 255 * k / 5
        pieces.append(f'<line x1="90" y1="{y:.1f}" x2="920" y2="{y:.1f}" stroke="{GRID}"/>')
        pieces.append(f'<text x="37" y="{y+4:.1f}" font-family="Arial" font-size="14" fill="{MUTED}">{yval:.2f}</text>')
    for k in range(6):
        x = 90 + 830 * k / 5
        pieces.append(f'<text x="{x:.1f}" y="412" text-anchor="middle" font-family="Arial" font-size="14" fill="{MUTED}">{duration*k/5:.1f}s</text>')
    for index, (name, rows, colour) in enumerate(traces):
        points = " ".join(
            f"{'M' if i == 0 else 'L'} {90 + 830 * row['time_s']/duration:.2f} {380 - 255 * row['payload_x_m']/max_position:.2f}"
            for i, row in enumerate(rows)
        )
        pieces.append(f'<path d="{points}" fill="none" stroke="{colour}" stroke-width="4" stroke-linejoin="round"/>')
        pieces.append(f'<circle cx="{90+index*270}" cy="443" r="5" fill="{colour}"/>')
        pieces.append(f'<text x="{108+index*270}" y="448" font-family="Arial" font-size="15" fill="#e5e7eb">{html.escape(name)}</text>')
    pieces.append('</svg>')
    return "\n".join(pieces) + "\n"


def replay(low_rows, high_rows):
    """Animation: geometry is schematic, positions are measured MuJoCo qpos samples."""
    p = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 405">',
        f'<rect width="980" height="405" rx="18" fill="{DARK}"/>',
        f'<text x="40" y="48" fill="{FG}" font-family="Arial" font-size="27" font-weight="bold">Tabletop push • trajectory replay</text>',
        f'<text x="40" y="74" fill="{MUTED}" font-family="Arial" font-size="15">Schematic animation generated from actual MuJoCo position samples</text>',
    ]
    for offset, rows, title in [(30, low_rows, "Low friction • μ=0.2"), (505, high_rows, "High friction • μ=1.2")]:
        p.append(f'<g transform="translate({offset} 100)"><rect x="0" y="0" width="445" height="258" rx="13" fill="#1f2937"/>')
        p.append(f'<text x="22" y="34" fill="white" font-size="19" font-family="Arial">{title}</text>')
        p.append('<path d="M15 192 H431" stroke="#64748b" stroke-width="7"/>')
        times = ";".join(f"{row['time_s']/rows[-1]['time_s']:.5f}" for row in rows)
        for key, colour, y, width, height in [
            ("pusher_x_m", "#60a5fa", 152, 30, 39),
            ("payload_x_m", "#fb923c", 145, 40, 45),
        ]:
            vals = ";".join(f"{25+310*row[key]:.2f}" for row in rows)
            p.append(f'<rect y="{y}" width="{width}" height="{height}" rx="4" fill="{colour}"><animate attributeName="x" dur="4s" repeatCount="indefinite" calcMode="linear" values="{vals}" keyTimes="{times}"/></rect>')
        p.append('<text x="20" y="237" fill="#d1d5db" font-family="Arial" font-size="14">blue = pusher    orange = payload</text></g>')
    p.append(f'<text x="40" y="389" fill="{MUTED}" font-family="Arial" font-size="12">For numerical results see the paired friction chart. This is not a MuJoCo camera render.</text>')
    p.append('</svg>')
    return "\n".join(p) + "\n"


def generate(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    traces = [
        ("nominal", run_pusher(Physics(1.0, 0.5), 7, 0.8, stride=50)[0], "#2563eb"),
        ("weak motor", run_pusher(Physics(0.42, 0.5), 7, 0.8, stride=50)[0], "#e27d39"),
        ("heavy payload", run_pusher(Physics(1.0, 3.0), 7, 0.8, stride=50)[0], "#16a34a"),
    ]
    low, _ = run_tabletop(force=7, friction=0.2, payload_mass=0.5, duration=1.0, stride=50)
    high, _ = run_tabletop(force=7, friction=1.2, payload_mass=0.5, duration=1.0, stride=50)
    artifacts = {
        "pusher-trajectories.svg": chart(
            "Pusher • payload position",
            "MuJoCo trajectories • 0.8 second rollout", traces, 0.8, 1.2),
        "tabletop-friction.svg": chart(
            "Tabletop • effect of floor friction",
            "Identical 1-second push with 0.5 kg payload",
            [("low friction μ=0.2", low, "#2563eb"), ("high friction μ=1.2", high, "#e27d39")],
            1.0, 1.0),
        "tabletop-replay.svg": replay(low, high),
    }
    for name, content in artifacts.items():
        (out / name).write_text(content, encoding="utf-8")
        print(out / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/media"))
    args = parser.parse_args()
    generate(args.out)


if __name__ == "__main__":
    main()
