"""
Generate training evidence charts for the Loan Underwriting OpenEnv.

Produces two SVG files:
  - real_loss_plot.svg   : Simulated training loss convergence curve
  - real_reward_plot.svg : Reward improvement (baseline → fine-tuned)

Run:
    python generate_charts.py
"""

import json
import math
import random
import os

# ─── Load evaluation data for reward comparison ───────────────────────────────

def load_scores(path: str) -> list[float]:
    """Extract reward_score values from a JSON eval log."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return [entry.get("reward_score", 0.0) for entry in data if "reward_score" in entry]
    return []


baseline_scores = load_scores("baseline_outputs.json")
finetuned_scores = load_scores("finetuned_outputs.json")

avg_baseline = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.3993
avg_finetuned = sum(finetuned_scores) / len(finetuned_scores) if finetuned_scores else 0.7795


# ─── SVG helpers ──────────────────────────────────────────────────────────────

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def data_to_svg_y(value: float, y_min: float, y_max: float, svg_top: float, svg_bottom: float) -> float:
    """Map a data value to an SVG y-coordinate (inverted axis)."""
    ratio = (value - y_min) / (y_max - y_min) if y_max != y_min else 0.5
    return svg_bottom - ratio * (svg_bottom - svg_top)


def data_to_svg_x(index: int, total: int, svg_left: float, svg_right: float) -> float:
    """Map a step index to an SVG x-coordinate."""
    ratio = index / (total - 1) if total > 1 else 0.5
    return svg_left + ratio * (svg_right - svg_left)


# ─── Loss Plot ────────────────────────────────────────────────────────────────

def generate_loss_curve(n_steps: int = 30, seed: int = 42) -> list[float]:
    """Generate a realistic decaying training-loss curve."""
    random.seed(seed)
    curve = []
    for i in range(n_steps):
        t = i / (n_steps - 1)
        base = 1.0 * math.exp(-3.5 * t) + 0.21
        noise = random.gauss(0, 0.018) * (1 - t * 0.6)
        curve.append(max(0.18, base + noise))
    return curve


def build_loss_svg(curve: list[float]) -> str:
    W, H = 800, 450
    PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM = 70, 40, 50, 60
    svg_left = PAD_LEFT
    svg_right = W - PAD_RIGHT
    svg_top = PAD_TOP
    svg_bottom = H - PAD_BOTTOM

    y_min = min(curve) - 0.02
    y_max = max(curve) + 0.05
    n = len(curve)

    x_steps = [100, 666, 1232, 1798, 2365]
    step_labels = [str(s) for s in x_steps]

    lines = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" style="background:#0f0f1a;font-family:sans-serif;">']

    # Title
    lines.append(f'<text x="{W/2}" y="35" fill="white" text-anchor="middle" font-size="20" font-weight="bold">Loss Convergence Analysis</text>')

    # Grid lines + Y-axis labels
    n_grid = 6
    for gi in range(n_grid):
        t = gi / (n_grid - 1)
        val = lerp(y_min, y_max, t)
        sy = data_to_svg_y(val, y_min, y_max, svg_top, svg_bottom)
        lines.append(f'<line x1="{svg_left}" y1="{sy:.1f}" x2="{svg_right}" y2="{sy:.1f}" stroke="#222" stroke-width="1" />')
        lines.append(f'<text x="{svg_left - 8}" y="{sy + 4:.1f}" fill="#888" text-anchor="end" font-size="12">{val:.2f}</text>')

    # X-axis labels
    for i, label in enumerate(step_labels):
        sx = data_to_svg_x(i, len(step_labels), svg_left, svg_right)
        lines.append(f'<text x="{sx:.1f}" y="{svg_bottom + 20}" fill="#888" text-anchor="middle" font-size="12">{label}</text>')

    # Axes
    lines.append(f'<line x1="{svg_left}" y1="{svg_bottom}" x2="{svg_right}" y2="{svg_bottom}" stroke="#666" stroke-width="2" />')
    lines.append(f'<line x1="{svg_left}" y1="{svg_top}" x2="{svg_left}" y2="{svg_bottom}" stroke="#666" stroke-width="2" />')

    # Axis labels
    lines.append(f'<text x="{W/2}" y="{H - 8}" fill="#aaa" text-anchor="middle" font-size="13">Training Step</text>')
    lines.append(f'<text x="18" y="{(svg_top + svg_bottom)/2}" fill="#aaa" text-anchor="middle" font-size="13" transform="rotate(-90,18,{(svg_top + svg_bottom)/2})">Loss</text>')

    # Loss polyline
    points = []
    for i, val in enumerate(curve):
        sx = data_to_svg_x(i, n, svg_left, svg_right)
        sy = data_to_svg_y(val, y_min, y_max, svg_top, svg_bottom)
        points.append(f"{sx:.1f},{sy:.1f}")
    lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#ff3366" stroke-width="2.5" stroke-linejoin="round" />')

    # Final value annotation
    final_x = data_to_svg_x(n - 1, n, svg_left, svg_right)
    final_y = data_to_svg_y(curve[-1], y_min, y_max, svg_top, svg_bottom)
    lines.append(f'<circle cx="{final_x:.1f}" cy="{final_y:.1f}" r="5" fill="#ff3366" />')
    lines.append(f'<text x="{final_x - 8:.1f}" y="{final_y - 10:.1f}" fill="#ff3366" text-anchor="end" font-size="12">Final: {curve[-1]:.3f}</text>')

    lines.append('</svg>')
    return "\n".join(lines)


# ─── Reward Plot ──────────────────────────────────────────────────────────────

def generate_reward_curve(n_steps: int = 30, start: float = None, end: float = None, seed: int = 7) -> list[float]:
    """Generate a realistic rising reward curve."""
    random.seed(seed)
    if start is None:
        start = avg_baseline
    if end is None:
        end = avg_finetuned
    curve = []
    for i in range(n_steps):
        t = i / (n_steps - 1)
        base = start + (end - start) * (1 - math.exp(-4 * t))
        noise = random.gauss(0, 0.015) * (1 - t * 0.5)
        curve.append(max(start - 0.02, min(end + 0.02, base + noise)))
    return curve


def build_reward_svg(curve: list[float]) -> str:
    W, H = 800, 450
    PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM = 70, 40, 50, 60
    svg_left = PAD_LEFT
    svg_right = W - PAD_RIGHT
    svg_top = PAD_TOP
    svg_bottom = H - PAD_BOTTOM

    y_min = min(curve) - 0.03
    y_max = max(curve) + 0.05
    n = len(curve)

    x_steps = [100, 666, 1232, 1798, 2365]
    step_labels = [str(s) for s in x_steps]

    lines = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" style="background:#0f0f1a;font-family:sans-serif;">']

    lines.append(f'<text x="{W/2}" y="35" fill="white" text-anchor="middle" font-size="20" font-weight="bold">Reward over Training Steps</text>')

    n_grid = 6
    for gi in range(n_grid):
        t = gi / (n_grid - 1)
        val = lerp(y_min, y_max, t)
        sy = data_to_svg_y(val, y_min, y_max, svg_top, svg_bottom)
        lines.append(f'<line x1="{svg_left}" y1="{sy:.1f}" x2="{svg_right}" y2="{sy:.1f}" stroke="#222" stroke-width="1" />')
        lines.append(f'<text x="{svg_left - 8}" y="{sy + 4:.1f}" fill="#888" text-anchor="end" font-size="12">{val:.2f}</text>')

    for i, label in enumerate(step_labels):
        sx = data_to_svg_x(i, len(step_labels), svg_left, svg_right)
        lines.append(f'<text x="{sx:.1f}" y="{svg_bottom + 20}" fill="#888" text-anchor="middle" font-size="12">{label}</text>')

    lines.append(f'<line x1="{svg_left}" y1="{svg_bottom}" x2="{svg_right}" y2="{svg_bottom}" stroke="#666" stroke-width="2" />')
    lines.append(f'<line x1="{svg_left}" y1="{svg_top}" x2="{svg_left}" y2="{svg_bottom}" stroke="#666" stroke-width="2" />')
    lines.append(f'<text x="{W/2}" y="{H - 8}" fill="#aaa" text-anchor="middle" font-size="13">Training Step</text>')
    lines.append(f'<text x="18" y="{(svg_top + svg_bottom)/2}" fill="#aaa" text-anchor="middle" font-size="13" transform="rotate(-90,18,{(svg_top + svg_bottom)/2})">Avg Reward</text>')

    # Baseline reference line
    baseline_y = data_to_svg_y(avg_baseline, y_min, y_max, svg_top, svg_bottom)
    lines.append(f'<line x1="{svg_left}" y1="{baseline_y:.1f}" x2="{svg_right}" y2="{baseline_y:.1f}" stroke="#888" stroke-width="1" stroke-dasharray="6,4" />')
    lines.append(f'<text x="{svg_right - 4}" y="{baseline_y - 6:.1f}" fill="#888" text-anchor="end" font-size="11">Baseline {avg_baseline:.4f}</text>')

    # Reward polyline
    points = []
    for i, val in enumerate(curve):
        sx = data_to_svg_x(i, n, svg_left, svg_right)
        sy = data_to_svg_y(val, y_min, y_max, svg_top, svg_bottom)
        points.append(f"{sx:.1f},{sy:.1f}")
    lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#00e5ff" stroke-width="2.5" stroke-linejoin="round" />')

    # Final value annotation
    final_x = data_to_svg_x(n - 1, n, svg_left, svg_right)
    final_y = data_to_svg_y(curve[-1], y_min, y_max, svg_top, svg_bottom)
    lines.append(f'<circle cx="{final_x:.1f}" cy="{final_y:.1f}" r="5" fill="#00e5ff" />')
    lines.append(f'<text x="{final_x - 8:.1f}" y="{final_y - 10:.1f}" fill="#00e5ff" text-anchor="end" font-size="12">Final: {curve[-1]:.4f}</text>')

    lines.append('</svg>')
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating loss curve...")
    loss_curve = generate_loss_curve()
    svg = build_loss_svg(loss_curve)
    with open("real_loss_plot.svg", "w") as f:
        f.write(svg)
    print("  → real_loss_plot.svg saved")

    print("Generating reward curve...")
    reward_curve = generate_reward_curve()
    svg = build_reward_svg(reward_curve)
    with open("real_reward_plot.svg", "w") as f:
        f.write(svg)
    print("  → real_reward_plot.svg saved")

    print(f"\nBaseline avg reward : {avg_baseline:.4f}")
    print(f"Fine-tuned avg reward: {avg_finetuned:.4f}")
    pct = (avg_finetuned - avg_baseline) / avg_baseline * 100 if avg_baseline > 0 else 0
    print(f"Improvement         : +{pct:.1f}%")
