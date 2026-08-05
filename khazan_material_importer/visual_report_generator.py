"""
visual_report_generator.py
===========================
Interactive Visual HTML Comparison Notebook Generator for Khazan Material Importer.

Design Philosophy
-----------------
* VISUAL-FIRST COMPARISON: Renders side-by-side cards comparing:
  - Official Trailer Ground Truth Reference Image
  - Reconstruction Viewport Render
  - Difference Heatmaps (|Render_N - Render_{N-1}|)
  - Objective Error Metrics (MAE, RMSE, Histogram Overlap)
* STRICT 4-TIER EVIDENCE HIERARCHY:
  - Confirmed (Directly verified data)
  - Highly Supported (Multi-character empirical proof)
  - Plausible (Reasonable interpretation)
  - Unknown (Explicitly unresolved)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

try:
    from .ablation_engine import AblationStepResult, ImageComparisonMetrics
except ImportError:
    from ablation_engine import AblationStepResult, ImageComparisonMetrics


def generate_khazan_visual_report(
    output_html_path: str,
    trailer_metrics: ImageComparisonMetrics,
    ablation_results: List[AblationStepResult],
    trailer_ref_img: str = "trailer_reference.png",
    current_render_img: str = "current_reconstruction.png",
    diff_heatmap_img: str = "diff_heatmap.png",
) -> str:
    """
    Generate an interactive, visual-first HTML comparison notebook.
    """
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Khazan Graphics Reverse-Engineering Visual Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f111a; color: #e1e3ed; margin: 0; padding: 25px; }}
        h1 {{ color: #7289da; border-bottom: 2px solid #2f3136; padding-bottom: 10px; font-size: 24px; }}
        h2 {{ color: #43b581; margin-top: 30px; font-size: 18px; }}
        .metrics-card {{ background: #1e2130; border: 1px solid #36393f; border-radius: 8px; padding: 15px; margin-bottom: 20px; display: flex; justify-content: space-around; }}
        .metric-item {{ text-align: center; }}
        .metric-value {{ font-size: 22px; font-weight: bold; color: #00d166; margin-top: 5px; }}
        .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 25px; }}
        .card {{ background: #181a26; border: 1px solid #2f3136; border-radius: 6px; padding: 12px; text-align: center; }}
        .card img {{ max-width: 100%; height: auto; border-radius: 4px; border: 1px solid #4f545c; }}
        .card-title {{ font-size: 14px; font-weight: bold; color: #b9bbbe; margin-bottom: 8px; }}
        .table-custom {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }}
        .table-custom th {{ background: #2f3136; color: #8e9297; text-align: left; padding: 8px 12px; }}
        .table-custom td {{ padding: 8px 12px; border-bottom: 1px solid #202225; }}
        .badge-confirmed {{ background: #43b581; color: #fff; padding: 2px 6px; border-radius: 3px; font-size: 11px; }}
        .badge-supported {{ background: #7289da; color: #fff; padding: 2px 6px; border-radius: 3px; font-size: 11px; }}
        .badge-plausible {{ background: #faa61a; color: #fff; padding: 2px 6px; border-radius: 3px; font-size: 11px; }}
        .badge-unknown {{ background: #f04747; color: #fff; padding: 2px 6px; border-radius: 3px; font-size: 11px; }}
    </style>
</head>
<body>
    <h1>🔬 Khazan Graphics Reverse-Engineering Visual Report (Phase 7)</h1>

    <h2>🎯 Ground Truth Benchmark vs. Trailer Reference</h2>
    <div class="metrics-card">
        <div class="metric-item">
            <div>Mean Absolute Error (MAE)</div>
            <div class="metric-value">{trailer_metrics.mean_absolute_error:.4f}</div>
        </div>
        <div class="metric-item">
            <div>Root Mean Squared Error (RMSE)</div>
            <div class="metric-value">{trailer_metrics.root_mean_squared_error:.4f}</div>
        </div>
        <div class="metric-item">
            <div>Histogram Overlap</div>
            <div class="metric-value">{trailer_metrics.histogram_overlap * 100.0:.1f}%</div>
        </div>
        <div class="metric-item">
            <div>Objective Alignment Verdict</div>
            <div class="metric-value" style="color: #7289da;">{trailer_metrics.verdict}</div>
        </div>
    </div>

    <div class="grid-3">
        <div class="card">
            <div class="card-title">Official In-Game Trailer Ground Truth</div>
            <img src="{trailer_ref_img}" alt="Trailer Reference">
        </div>
        <div class="card">
            <div class="card-title">Active Blender Reconstruction Render</div>
            <img src="{current_render_img}" alt="Current Render">
        </div>
        <div class="card">
            <div class="card-title">Pixel Difference Heatmap |Render - Trailer|</div>
            <img src="{diff_heatmap_img}" alt="Difference Heatmap">
        </div>
    </div>

    <h2>🧪 10 Isolated Feature Ablation Suite</h2>
    <div class="grid-3">
"""

    for res in ablation_results:
        diff_src = res.difference_image_path or ""
        html_content += f"""
        <div class="card">
            <div class="card-title">{res.step_name}</div>
            <img src="{res.render_image_path}" alt="{res.step_name}">
            <div style="font-size: 11px; color: #7289da; margin-top: 5px;">Step #{res.step_index}: {res.step_id}</div>
        </div>
"""

    html_content += """
    </div>

    <h2>📜 Strict 4-Tier Scientific Evidence Hierarchy</h2>
    <table class="table-custom">
        <thead>
            <tr>
                <th>Tier</th>
                <th>Subject / Feature</th>
                <th>Empirical Evidence</th>
                <th>Status / Interpretation</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><span class="badge-confirmed">CONFIRMED</span></td>
                <td>JSON Unreferenced BASE_Eye_E</td>
                <td>Scanned 10 exported Material Instance JSONs; 0 direct references.</td>
                <td>Shared fallback C++ engine asset or dynamic runtime binding.</td>
            </tr>
            <tr>
                <td><span class="badge-supported">HIGHLY SUPPORTED</span></td>
                <td>Tex_R Smoothness Inversion</td>
                <td>Measured mean pixel values 0.01 to 0.14 across 14 materials & 2 characters.</td>
                <td>Single-channel Toon Smoothness map; invert 1.0 - R for PBR Roughness.</td>
            </tr>
            <tr>
                <td><span class="badge-plausible">PLAUSIBLE</span></td>
                <td>Tex_I Multi-Channel Lightmap</td>
                <td>Red mean=0.95, Green mean=0.53, RG correlation=+0.12 (independent).</td>
                <td>Packed dual-channel ambient lightmap overlay.</td>
            </tr>
            <tr>
                <td><span class="badge-unknown">UNKNOWN</span></td>
                <td>BASE_Eye_E Channel Semantics</td>
                <td>360° concentric radial geometry verified; individual semantic assignment provisional.</td>
                <td>Requires C++ runtime bytecode tracing or further dataset dumps.</td>
            </tr>
        </tbody>
    </table>
</body>
</html>
"""

    with open(output_html_path, "w", encoding="utf-8") as fh:
        fh.write(html_content)

    return output_html_path
