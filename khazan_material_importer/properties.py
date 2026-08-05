"""
properties.py
=============
Blender PropertyGroup definitions for the Khazan Material Importer.
All user-facing settings live here; they are stored on Scene.khazan_settings.
"""

import bpy
from bpy.props import (
    BoolProperty,
    StringProperty,
)
from bpy.types import PropertyGroup


# ---------------------------------------------------------------------------
# Default paths – match the NekoPixil Daphrona release layout
# ---------------------------------------------------------------------------
_DEFAULT_TEX = (
    r"C:\Users\lemon\Downloads"
    r"\_fbx__daphrona__the_first_berserker_khazan____dl_by_nekopixil_djl42pw"
    r"\Daphrona (The First Berserker Khazan)\Textures"
)
_DEFAULT_MAT = (
    r"C:\Users\lemon\Downloads"
    r"\_fbx__daphrona__the_first_berserker_khazan____dl_by_nekopixil_djl42pw"
    r"\Daphrona (The First Berserker Khazan)\Material"
)


class KhazanSettings(PropertyGroup):
    """Persistent add-on settings stored on the active Scene."""

    texture_folder: StringProperty(
        name="Texture Folder",
        description="Folder containing .png texture files",
        subtype="DIR_PATH",
        default=_DEFAULT_TEX,
    )

    material_folder: StringProperty(
        name="Material Folder",
        description="Folder containing .json material files",
        subtype="DIR_PATH",
        default=_DEFAULT_MAT,
    )

    verbose_logging: BoolProperty(
        name="Verbose Logging",
        description="Print extra detail for every texture lookup to the console",
        default=False,
    )

    clean_existing_nodes: BoolProperty(
        name="Clean Existing Nodes",
        description=(
            "Remove previously auto-generated Image Texture / Normal Map nodes "
            "before rebuilding (manual nodes are preserved)"
        ),
        default=True,
    )

    ignore_unknown_maps: BoolProperty(
        name="Ignore Unknown Maps",
        description=(
            "Silently skip texture slots whose Blender mapping is uncertain "
            "(Tex_S, Tex_E, Tex_I, etc.) instead of logging them as warnings"
        ),
        default=False,
    )

    auto_match_materials: BoolProperty(
        name="Auto Match Materials",
        description=(
            "Fuzzy-match Blender material names to JSON filenames "
            "when exact names differ"
        ),
        default=True,
    )

    # --- Phase 7: Validation Framework Settings ---
    eye_mode: bpy.props.EnumProperty(
        name="Eye Shading Mode",
        description="Select eye shading preset mode",
        items=[
            ("MODE_A_RAW", "Mode A — Raw Export", "Exact exported JSON values (Slate Blue)"),
            ("MODE_B_CONCEPT", "Mode B — Concept Art", "Artistic Amber Gold preset override"),
            ("MODE_C_TRAILER", "Mode C — Trailer Match", "Muted pale gray-slate blue (Primary Research Target)"),
        ],
        default="MODE_C_TRAILER",
    )

    enable_procedural_iris: BoolProperty(
        name="Procedural Iris",
        description="Enable procedural concentric iris/pupil node graph",
        default=True,
    )

    enable_base_eye_e: BoolProperty(
        name="BASE_Eye_E Contribution",
        description="Enable BASE_Eye_E texture contribution",
        default=False,
    )

    enable_specular: BoolProperty(
        name="Specular Highlights",
        description="Enable cel specular highlight bands",
        default=True,
    )

    enable_rim_light: BoolProperty(
        name="Rim Light",
        description="Enable special facing rim lighting",
        default=True,
    )

    enable_eyelid_shadow: BoolProperty(
        name="Eyelid Shadow",
        description="Enable upper eyelid cast shadow overlay",
        default=True,
    )

    live_pupil_scale: bpy.props.FloatProperty(
        name="Pupil Scale",
        description="Live pupil radius scaling factor",
        min=0.1,
        max=1.0,
        default=0.9,
    )

    live_iris_color: bpy.props.FloatVectorProperty(
        name="Iris Color",
        description="Live iris color tint",
        subtype="COLOR",
        size=3,
        min=0.0,
        max=1.0,
        default=(0.48, 0.55, 0.65),
    )

    live_limbal_color: bpy.props.FloatVectorProperty(
        name="Limbal Color",
        description="Live limbal ring border color",
        subtype="COLOR",
        size=3,
        min=0.0,
        max=1.0,
        default=(0.20, 0.25, 0.35),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
_CLASSES = (KhazanSettings,)


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.khazan_settings = bpy.props.PointerProperty(
        type=KhazanSettings
    )


def unregister() -> None:
    del bpy.types.Scene.khazan_settings
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
