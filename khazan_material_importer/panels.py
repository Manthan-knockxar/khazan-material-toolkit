"""
panels.py
=========
Blender UI panels for the Khazan Material Importer.

Panel hierarchy:
  View3D > Sidebar > Khazan tab
    └── KHAZAN_PT_main         (main panel)
          ├── Actions sub-panel   (Import Materials, Rebuild Selected, etc.)
          └── Settings sub-panel  (folder paths, toggles)
"""

from __future__ import annotations

from typing import Set

import bpy
from bpy.types import Panel


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
class KHAZAN_PT_main(Panel):
    """Main Khazan Material Importer panel in the N-panel sidebar."""

    bl_label = "Khazan Material Importer"
    bl_idname = "KHAZAN_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Khazan"
    bl_order = 0

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        layout.label(
            text="The First Berserker: Khazan",
            icon="MATERIAL",
        )


# ---------------------------------------------------------------------------
# Actions sub-panel
# ---------------------------------------------------------------------------
class KHAZAN_PT_actions(Panel):
    """Action buttons: Import, Rebuild, Dry Run."""

    bl_label = "Actions"
    bl_idname = "KHAZAN_PT_actions"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Khazan"
    bl_parent_id = "KHAZAN_PT_main"
    bl_order = 1

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        col = layout.column(align=True)

        col.scale_y = 1.4  # slightly taller buttons for readability

        col.operator(
            "khazan.import_materials",
            text="Import Materials",
            icon="IMPORT",
        )
        col.separator(factor=0.5)
        col.operator(
            "khazan.rebuild_selected",
            text="Rebuild Selected",
            icon="OBJECT_DATA",
        )
        col.separator(factor=0.5)
        col.operator(
            "khazan.rebuild_all",
            text="Rebuild All",
            icon="LOOP_FORWARDS",
        )
        col.separator(factor=0.5)
        col.operator(
            "khazan.dry_run",
            text="Dry Run",
            icon="HIDE_OFF",
        )
        col.separator(factor=0.5)
        col.operator(
            "khazan.analyze_material",
            text="Analyze Material",
            icon="VIEWZOOM",
        )
        col.separator(factor=0.5)
        col.operator(
            "khazan.inspect_textures",
            text="Inspect Textures",
            icon="TEXTURE",
        )
        col.separator(factor=0.5)
        col.operator(
            "khazan.inspect_mesh",
            text="Inspect Mesh & UVs",
            icon="MESH_DATA",
        )
        col.separator(factor=0.5)
        col.operator(
            "khazan.toggle_debug_preview",
            text="Toggle Debug Preview",
            icon="RESTRICT_VIEW_OFF",
        )
        col.separator(factor=0.5)
        col.operator(
            "khazan.export_research_report",
            text="Export Research Report",
            icon="FILE_TEXT",
        )
        col.separator(factor=0.5)
        col.operator(
            "khazan.run_hypothesis_validation",
            text="Run Hypothesis Validation",
            icon="PHYSICS",
        )
        col.separator(factor=0.5)
        col.operator(
            "khazan.investigate_eye_base",
            text="Investigate BASE_Eye_E",
            icon="HIDE_OFF",
        )
        col.separator(factor=0.5)

        # Eye channel mapping experiment sub-row
        box = col.box()
        box.label(text="BASE_Eye_E Mesh Experiment:", icon="EXPERIMENTAL")
        row = box.row(align=True)
        op_all = row.operator("khazan.preview_base_eye_channel", text="Full")
        op_all.target_channel = "ALL"
        op_r = row.operator("khazan.preview_base_eye_channel", text="Red")
        op_r.target_channel = "R"
        op_g = row.operator("khazan.preview_base_eye_channel", text="Green")
        op_g.target_channel = "G"
        op_b = row.operator("khazan.preview_base_eye_channel", text="Blue")
        op_b.target_channel = "B"
        op_rst = box.operator("khazan.preview_base_eye_channel", text="Reset Procedural Eye", icon="LOOP_BACK")
        op_rst.target_channel = "RESET"


# ---------------------------------------------------------------------------
# Settings sub-panel
# ---------------------------------------------------------------------------
class KHAZAN_PT_settings(Panel):
    """Folder paths and toggle settings."""

    bl_label = "Settings"
    bl_idname = "KHAZAN_PT_settings"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Khazan"
    bl_parent_id = "KHAZAN_PT_main"
    bl_order = 2
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        settings = context.scene.khazan_settings
        col = layout.column(align=True)

        # --- Folder paths ---
        col.label(text="Paths:", icon="FILE_FOLDER")
        col.prop(settings, "texture_folder", text="Textures")
        col.prop(settings, "material_folder", text="Materials")

        col.separator()

        # --- Toggles ---
        col.label(text="Options:", icon="PREFERENCES")
        col.prop(settings, "verbose_logging")
        col.prop(settings, "clean_existing_nodes")
        col.prop(settings, "ignore_unknown_maps")
        col.prop(settings, "auto_match_materials")

        col.separator()

        # --- Status indicators ---
        self._draw_path_status(col, context)

    @staticmethod
    def _draw_path_status(
        col: bpy.types.UILayout,
        context: bpy.types.Context,
    ) -> None:
        """Show ✓/✗ icons next to folder paths to warn on missing dirs."""
        import os
        settings = context.scene.khazan_settings
        tex = bpy.path.abspath(settings.texture_folder)
        mat = bpy.path.abspath(settings.material_folder)

        row = col.row()
        if os.path.isdir(tex):
            row.label(text="Texture folder: OK", icon="CHECKMARK")
        else:
            row.label(text="Texture folder: NOT FOUND", icon="ERROR")

        row = col.row()
        if os.path.isdir(mat):
            row.label(text="Material folder: OK", icon="CHECKMARK")
        else:
            row.label(text="Material folder: NOT FOUND", icon="ERROR")


# ---------------------------------------------------------------------------
# Phase 7: Validation Framework & Trailer Inspector Sub-Panel
# ---------------------------------------------------------------------------
class KHAZAN_PT_validation_framework(Panel):
    """Validation framework, Eye Preset Modes, Feature Toggles, and Live Inspector."""

    bl_label = "Validation Framework & Trailer Inspector"
    bl_idname = "KHAZAN_PT_validation_framework"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Khazan"
    bl_parent_id = "KHAZAN_PT_main"
    bl_order = 3

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        settings = context.scene.khazan_settings
        col = layout.column(align=True)

        # --- 1. Eye Mode Selector ---
        col.label(text="Eye Shading Mode:", icon="HIDE_OFF")
        col.prop(settings, "eye_mode", text="")

        col.separator()

        # --- 2. Feature Toggles ---
        box_tog = col.box()
        box_tog.label(text="Shader Feature Toggles:", icon="CHECKBOX_HLT")
        box_tog.prop(settings, "enable_procedural_iris")
        box_tog.prop(settings, "enable_base_eye_e")
        box_tog.prop(settings, "enable_specular")
        box_tog.prop(settings, "enable_rim_light")
        box_tog.prop(settings, "enable_eyelid_shadow")

        col.separator()

        # --- 3. Live Parameter Sliders ---
        box_param = col.box()
        box_param.label(text="Live Parameter Inspector:", icon="PROPERTIES")
        box_param.prop(settings, "live_pupil_scale")
        box_param.prop(settings, "live_iris_color")
        box_param.prop(settings, "live_limbal_color")

        col.separator()

        # --- 4. Validation Suite Operators ---
        col.label(text="Validation Actions:", icon="PHYSICS")
        col.operator("khazan.run_ablation_suite", text="Run 10-Step Ablation Suite", icon="PLAY")
        col.separator(factor=0.5)
        col.operator("khazan.compute_trailer_metrics", text="Compute Trailer Metrics", icon="VIEWZOOM")
        col.separator(factor=0.5)
        col.operator("khazan.export_visual_report", text="Export Visual Report", icon="FILE_TEXT")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
_CLASSES = (
    KHAZAN_PT_main,
    KHAZAN_PT_actions,
    KHAZAN_PT_validation_framework,
    KHAZAN_PT_settings,
)


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
