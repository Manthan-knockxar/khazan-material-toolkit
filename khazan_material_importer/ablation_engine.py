"""
ablation_engine.py
==================
Ablation Testing Engine & Difference Image Generator for Khazan Material Importer.

Design Philosophy
-----------------
* USES BLENDER NATIVE IMAGE API (bpy.data.images / image.pixels) to compute:
  - 10 Isolated Feature Ablation Screenshots
  - Pixel-by-pixel Difference Images (|Render_N - Render_{N-1}|)
  - Objective Comparison Metrics against Trailer Ground Truth (MAE, RMSE, Histogram Overlap)
  - Reproducibility Metadata JSON files (meta_01.json ... meta_10.json)
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AblationStepConfig:
    """Configuration for a single isolated ablation test step."""
    step_index: int
    step_id: str                         # e.g., "step_01_diffuse_only"
    step_name: str                       # e.g., "Isolated 01: Diffuse Only"
    description: str
    feature_mode: str                    # e.g., "DIFFUSE_ONLY", "IRIS_ONLY", "BASE_EYE_E_RED", etc.
    active_toggles: Dict[str, bool]      # e.g., {"iris": False, "specular": False, "rim": False, "shadow": False}


@dataclass
class ImageComparisonMetrics:
    """Objective error metrics comparing render against Trailer Reference Ground Truth."""
    render_file: str
    reference_file: str
    mean_absolute_error: float          # MAE (lower = closer to trailer)
    root_mean_squared_error: float      # RMSE (lower = closer to trailer)
    histogram_overlap: float            # 0.0 to 1.0 (higher = closer histogram match)
    verdict: str


@dataclass
class AblationStepResult:
    """Output result for a completed ablation test step."""
    step_index: int
    step_id: str
    step_name: str
    render_image_path: str
    difference_image_path: Optional[str] = None
    metadata_json_path: Optional[str] = None
    metrics_vs_trailer: Optional[ImageComparisonMetrics] = None


class AblationEngine:
    """Engine that controls multi-layer ablation suite execution and image diff calculations."""

    STEPS: List[AblationStepConfig] = [
        AblationStepConfig(
            step_index=1,
            step_id="step_01_diffuse_only",
            step_name="Isolated 01: Diffuse / Base Color Only",
            description="Base Color flat texture output with all toon effects disabled",
            feature_mode="DIFFUSE_ONLY",
            active_toggles={"iris": False, "eye_e": False, "specular": False, "rim": False, "shadow": False},
        ),
        AblationStepConfig(
            step_index=2,
            step_id="step_02_iris_only",
            step_name="Isolated 02: Procedural Iris Only",
            description="Procedural concentric iris/pupil graph only",
            feature_mode="IRIS_ONLY",
            active_toggles={"iris": True, "eye_e": False, "specular": False, "rim": False, "shadow": False},
        ),
        AblationStepConfig(
            step_index=3,
            step_id="step_03_eye_e_red_only",
            step_name="Isolated 03: BASE_Eye_E Red Channel Only",
            description="BASE_Eye_E Red channel central disc feature mapped to Base Color",
            feature_mode="EYE_E_RED",
            active_toggles={"iris": False, "eye_e": True, "specular": False, "rim": False, "shadow": False},
        ),
        AblationStepConfig(
            step_index=4,
            step_id="step_04_eye_e_green_only",
            step_name="Isolated 04: BASE_Eye_E Green Channel Only",
            description="BASE_Eye_E Green channel concentric ring feature mapped to Base Color",
            feature_mode="EYE_E_GREEN",
            active_toggles={"iris": False, "eye_e": True, "specular": False, "rim": False, "shadow": False},
        ),
        AblationStepConfig(
            step_index=5,
            step_id="step_05_eye_e_blue_only",
            step_name="Isolated 05: BASE_Eye_E Blue Channel Only",
            description="BASE_Eye_E Blue channel focal dot feature mapped to Base Color",
            feature_mode="EYE_E_BLUE",
            active_toggles={"iris": False, "eye_e": True, "specular": False, "rim": False, "shadow": False},
        ),
        AblationStepConfig(
            step_index=6,
            step_id="step_06_eye_e_rgb_only",
            step_name="Isolated 06: BASE_Eye_E Combined RGB Only",
            description="BASE_Eye_E full RGBA texture output mapped to Base Color",
            feature_mode="EYE_E_RGB",
            active_toggles={"iris": False, "eye_e": True, "specular": False, "rim": False, "shadow": False},
        ),
        AblationStepConfig(
            step_index=7,
            step_id="step_07_specular_only",
            step_name="Isolated 07: Specular Highlights Only",
            description="Cel specular highlight bands only",
            feature_mode="SPECULAR_ONLY",
            active_toggles={"iris": False, "eye_e": False, "specular": True, "rim": False, "shadow": False},
        ),
        AblationStepConfig(
            step_index=8,
            step_id="step_08_rim_only",
            step_name="Isolated 08: Rim Light Only",
            description="Facing toon rim lighting only",
            feature_mode="RIM_ONLY",
            active_toggles={"iris": False, "eye_e": False, "specular": False, "rim": True, "shadow": False},
        ),
        AblationStepConfig(
            step_index=9,
            step_id="step_09_eyelid_shadow_only",
            step_name="Isolated 09: Upper Eyelid Shadow Only",
            description="Upper eyelid cast shadow gradient overlay only",
            feature_mode="SHADOW_ONLY",
            active_toggles={"iris": False, "eye_e": False, "specular": False, "rim": False, "shadow": True},
        ),
        AblationStepConfig(
            step_index=10,
            step_id="step_10_complete_shader",
            step_name="Complete 10: Complete Shader (Everything Enabled)",
            description="Full combined shader with all enabled layers",
            feature_mode="COMPLETE",
            active_toggles={"iris": True, "eye_e": False, "specular": True, "rim": True, "shadow": True},
        ),
    ]

    @staticmethod
    def compute_difference_image(
        render_n_path: str,
        render_prev_path: str,
        diff_out_path: str,
    ) -> bool:
        """
        Compute pixel-by-pixel absolute difference |Render_N - Render_{N-1}|
        natively via Blender's bpy.data.images API and save PNG heatmap.
        """
        import bpy

        if not os.path.exists(render_n_path) or not os.path.exists(render_prev_path):
            return False

        try:
            img_n = bpy.data.images.load(render_n_path, check_existing=False)
            img_p = bpy.data.images.load(render_prev_path, check_existing=False)

            w1, h1 = img_n.size
            w2, h2 = img_p.size
            if w1 != w2 or h1 != h2:
                bpy.data.images.remove(img_n)
                bpy.data.images.remove(img_p)
                return False

            pix_n = list(img_n.pixels)
            pix_p = list(img_p.pixels)

            diff_pixels = []
            for i in range(0, len(pix_n), 4):
                dr = abs(pix_n[i] - pix_p[i]) * 3.0       # Scale diff for visual emphasis
                dg = abs(pix_n[i + 1] - pix_p[i + 1]) * 3.0
                db = abs(pix_n[i + 2] - pix_p[i + 2]) * 3.0
                diff_pixels.extend([dr, dg, db, 1.0])

            diff_img = bpy.data.images.new(
                name="Khazan_Diff_Temp", width=w1, height=h1, alpha=False
            )
            diff_img.pixels = diff_pixels
            diff_img.filepath_raw = diff_out_path
            diff_img.file_format = "PNG"
            diff_img.save()

            bpy.data.images.remove(img_n)
            bpy.data.images.remove(img_p)
            bpy.data.images.remove(diff_img)
            return True
        except Exception:
            return False

    @staticmethod
    def compute_image_comparison_metrics(
        render_path: str,
        reference_path: str,
    ) -> ImageComparisonMetrics:
        """
        Compute objective image comparison metrics (MAE, RMSE, Histogram Overlap)
        between active Blender viewport render and Ground-Truth Trailer screenshot.
        """
        import bpy

        if not os.path.exists(render_path) or not os.path.exists(reference_path):
            return ImageComparisonMetrics(
                render_file=os.path.basename(render_path),
                reference_file=os.path.basename(reference_path),
                mean_absolute_error=999.0,
                root_mean_squared_error=999.0,
                histogram_overlap=0.0,
                verdict="File missing",
            )

        try:
            img_r = bpy.data.images.load(render_path, check_existing=False)
            img_ref = bpy.data.images.load(reference_path, check_existing=False)

            pix_r = list(img_r.pixels)
            pix_ref = list(img_ref.pixels)

            # Sample subset for speed if resolution differs
            n_samples = min(len(pix_r), len(pix_ref)) // 4
            step = max(1, n_samples // 10000)

            mae_sum = 0.0
            rmse_sum = 0.0
            sampled_count = 0

            hist_r = [0] * 64
            hist_ref = [0] * 64

            for i in range(0, n_samples * 4, 4 * step):
                if i + 2 < len(pix_r) and i + 2 < len(pix_ref):
                    # Luminance
                    lum_r = 0.2126 * pix_r[i] + 0.7152 * pix_r[i + 1] + 0.0722 * pix_r[i + 2]
                    lum_ref = 0.2126 * pix_ref[i] + 0.7152 * pix_ref[i + 1] + 0.0722 * pix_ref[i + 2]

                    diff = abs(lum_r - lum_ref)
                    mae_sum += diff
                    rmse_sum += diff * diff
                    sampled_count += 1

                    idx_r = max(0, min(63, int(lum_r * 64)))
                    idx_ref = max(0, min(63, int(lum_ref * 64)))
                    hist_r[idx_r] += 1
                    hist_ref[idx_ref] += 1

            bpy.data.images.remove(img_r)
            bpy.data.images.remove(img_ref)

            mae = mae_sum / max(1, sampled_count)
            rmse = math.sqrt(rmse_sum / max(1, sampled_count))

            # Histogram intersection overlap (0.0 to 1.0)
            overlap_sum = sum(min(hist_r[j], hist_ref[j]) for j in range(64))
            hist_overlap = overlap_sum / max(1, sampled_count)

            verdict = "Strong Match" if mae < 0.10 else ("Moderate Match" if mae < 0.25 else "Significant Difference")

            return ImageComparisonMetrics(
                render_file=os.path.basename(render_path),
                reference_file=os.path.basename(reference_path),
                mean_absolute_error=round(mae, 4),
                root_mean_squared_error=round(rmse, 4),
                histogram_overlap=round(hist_overlap, 4),
                verdict=verdict,
            )
        except Exception as exc:  # noqa: BLE001
            return ImageComparisonMetrics(
                render_file=os.path.basename(render_path),
                reference_file=os.path.basename(reference_path),
                mean_absolute_error=999.0,
                root_mean_squared_error=999.0,
                histogram_overlap=0.0,
                verdict=f"Error: {exc}",
            )

    @staticmethod
    def save_step_metadata(
        config: AblationStepConfig,
        eye_mode: str,
        out_json_path: str,
        node_count: int = 23,
    ) -> str:
        """Save reproducibility metadata JSON alongside screenshot."""
        meta = {
            "step_index": config.step_index,
            "step_id": config.step_id,
            "step_name": config.step_name,
            "description": config.description,
            "eye_mode": eye_mode,
            "active_toggles": config.active_toggles,
            "node_count": node_count,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        return out_json_path
