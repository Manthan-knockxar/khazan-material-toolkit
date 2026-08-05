"""
node_builder.py
===============
Blender node-tree manipulation for the Khazan Material Importer.

Responsibilities
----------------
* Reuse existing auto-generated nodes (preserves user-adjusted positions).
* Remove only leftover autogen nodes not claimed on this run.
* Apply per-material-type Principled BSDF defaults.
* Wire confirmed channels (D → Base Color, N → Normal, R → Roughness).
* For BLEND_Masked materials, also wire D-Alpha → PBSDF Alpha (hair cutout).
* Approximate SpecialRimLight via Layer Weight + ColorRamp + Add Shader.
* Handle EYE type with a procedural concentric-circle iris/pupil node graph.
* Handle EYESHADOW type with flat colour from JSON parameters.
* Leave uncertain channels (S, E, I, NTP, F2) disconnected with rich labels.

Changelog (v1.5 → v1.6)  Phase 3 - Procedural Shader Reconstruction
---------------------------------------------------------------------
* Procedural Eye: _apply_eye_setup() replaced by _build_procedural_eye().
  Generates concentric iris/pupil zones from EyeWhiteColor0, Pupil_Circle0,
  Pupil_Lens0, Pupil_Ring0, PupilScale, EyeUScale, EyeVScale using standard
  Blender nodes (TexCoord → Mapping → VectorMath.LENGTH → ColorRamp).
  Result: anime-style eye with visible iris and pupil instead of white sphere.
* Rim light improvements:
  - Swapped Fresnel → Facing output (more artistically controllable for toon).
  - CONSTANT interpolation when SpecialRimLightPower > 5 (hard cel edge);
    LINEAR kept for softer power values.
  - WorldFresnelIntensity scalar multiplied into rim Emission Strength.
* Enriched skip labels: _skip_reason() returns SkipInfo(node_label, log_reason).
  The short node_label is written onto the Blender node (visible in Shader Editor);
  the long log_reason goes to the System Console.  Node graphs are now educational.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import bpy

from .material_db import (
    MaterialRecord,
    MaterialType,
    TextureEntry,
    UNCERTAIN_CHANNELS,
    WIRED_CHANNELS,
)


# ---------------------------------------------------------------------------
# Custom property tag used to identify auto-generated nodes
# ---------------------------------------------------------------------------
_AUTOGEN_TAG = "khazan_autogen"
_AUTOGEN_VALUE = "1"

# Labels used for rim-light nodes (allows reuse on subsequent runs)
_LABEL_RIM_LW       = "Khazan_Rim_LayerWeight"
_LABEL_RIM_RAMP     = "Khazan_Rim_ColorRamp"
_LABEL_RIM_SCALE    = "Khazan_Rim_IntensityScale"
_LABEL_RIM_EMISSION = "Khazan_Rim_Emission"
_LABEL_RIM_ADD      = "Khazan_Rim_AddShader"

# Labels used for procedural eye nodes
_LABEL_EYE_TEXCO    = "Khazan_Eye_TexCoord"
_LABEL_EYE_MAPPING  = "Khazan_Eye_Mapping"
_LABEL_EYE_SUBTRACT = "Khazan_Eye_Center"
_LABEL_EYE_LENGTH   = "Khazan_Eye_Length"
_LABEL_EYE_SCALE    = "Khazan_Eye_Normalize"
_LABEL_EYE_RAMP     = "Khazan_Eye_Ramp"
_LABEL_EYE_HIGHLIGHT= "Khazan_Eye_Highlight"
_LABEL_EYE_MIX      = "Khazan_Eye_Mix"

# Labels used for cel specular nodes
_LABEL_SPEC_LW      = "Khazan_CelSpec_LayerWeight"
_LABEL_SPEC_RAMP    = "Khazan_CelSpec_ColorRamp"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class BuildResult:
    material_name: str
    json_matched: Optional[str] = None
    mat_type: str = "unknown"
    textures_loaded: List[str] = field(default_factory=list)
    textures_skipped: List[Tuple[str, str]] = field(default_factory=list)
    textures_missing: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    was_dry_run: bool = False

    def has_issues(self) -> bool:
        return bool(self.warnings or self.errors or self.textures_missing)


# ---------------------------------------------------------------------------
# Node layout constants
# ---------------------------------------------------------------------------
_COL_TEX = -800.0     # Image Texture nodes
_COL_NMAP = -430.0    # Normal Map node
_COL_PBSDF = -150.0   # Principled BSDF (Blender default)
_ROW_STEP = 280.0     # Vertical spacing between texture rows

# Rim light nodes sit above the main stack
_COL_RIM_LW = -800.0
_COL_RIM_RAMP = -530.0
_COL_RIM_EMISSION = -260.0
_COL_RIM_ADD = 60.0   # Add Shader, just right of PBSDF
_ROW_RIM = 480.0      # Above all texture rows


# ---------------------------------------------------------------------------
# Per-material-type Principled BSDF defaults
# ---------------------------------------------------------------------------
# MSM_BBQCartoon is a toon shader. Standard PBR values produce overly shiny,
# metallic-looking results. These defaults approximate the matte, cel-shaded
# look of the game.
#
# Reasoning:
# - Metallic = 0.0: Toon shaders never use PBR metallic workflow.
# - Specular low (0.2): The custom shader handles specular highlights via its
#   own toon specular model; Blender Specular would add a second, incorrect
#   highlight layer.
# - Roughness high (0.8–0.85): BBQCartoon materials are intentionally matte.
#   Individual roughness textures (_R) will override this per-texel anyway.
# - Sheen/Clearcoat = 0.0: Unused in toon shaders.
# - Anisotropic for hair: Confirmed by JSON AnisotropicStrength/Power scalars.

_PBSDF_DEFAULTS: Dict[str, Dict[str, float]] = {
    MaterialType.CLOTHING.value: {
        "Metallic": 0.0,
        "Specular": 0.0,    # 0.0 eliminates glossy HDRI vinyl/plastic sheen on clothing
        "Roughness": 0.90,
        "Sheen": 0.0,
        "Clearcoat": 0.0,
    },
    MaterialType.SKIN.value: {
        "Metallic": 0.0,
        "Specular": 0.0,    # matte toon skin
        "Roughness": 0.85,
        "Sheen": 0.0,
        "Clearcoat": 0.0,
    },
    MaterialType.HAIR.value: {
        "Metallic": 0.0,
        "Specular": 0.05,   # subtle sheen
        "Roughness": 0.65,
        "Anisotropic": 0.6, # anime anisotropic hair highlight
        "Sheen": 0.0,
        "Clearcoat": 0.0,
    },
    MaterialType.FACE.value: {
        "Metallic": 0.0,
        "Specular": 0.0,
        "Roughness": 0.85,
        "Sheen": 0.0,
        "Clearcoat": 0.0,
    },
    MaterialType.EYE.value: {
        "Metallic": 0.0,
        "Specular": 0.5,    # glossy wet eyeball
        "Roughness": 0.05,
        "Sheen": 0.0,
        "Clearcoat": 0.0,
    },
    MaterialType.EYESHADOW.value: {
        "Metallic": 0.0,
        "Specular": 0.0,
        "Roughness": 1.0,   # matte decal
        "Sheen": 0.0,
        "Clearcoat": 0.0,
    },
    MaterialType.ITEM.value: {
        "Metallic": 0.0,
        "Specular": 0.05,
        "Roughness": 0.85,
        "Sheen": 0.0,
        "Clearcoat": 0.0,
    },
    MaterialType.UNKNOWN.value: {
        "Metallic": 0.0,
        "Specular": 0.0,
        "Roughness": 0.90,
        "Sheen": 0.0,
        "Clearcoat": 0.0,
    },
}


def _apply_pbsdf_defaults(pbsdf: bpy.types.Node, mat_type: MaterialType) -> None:
    """
    Apply per-material-type Principled BSDF defaults to approximate the matte,
    cel-shaded look of Khazan's toon shader and eliminate glossy HDRI reflections.
    """
    mat_type_str = mat_type.value if isinstance(mat_type, MaterialType) else str(mat_type)
    defaults = _PBSDF_DEFAULTS.get(mat_type_str, _PBSDF_DEFAULTS[MaterialType.CLOTHING.value])

    for key, val in defaults.items():
        _set_input_safe(pbsdf, key, val)
        if key == "Specular":
            _set_input_safe(pbsdf, "Specular IOR Level", val)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tag_node(node: bpy.types.Node) -> None:
    """Mark a node as auto-generated."""
    node[_AUTOGEN_TAG] = _AUTOGEN_VALUE


def _is_autogen(node: bpy.types.Node) -> bool:
    return node.get(_AUTOGEN_TAG) == _AUTOGEN_VALUE


def _find_autogen_by_label(
    tree: bpy.types.NodeTree,
    label: str,
) -> Optional[bpy.types.Node]:
    """Return the first autogen node with the given label, or None."""
    for node in tree.nodes:
        if _is_autogen(node) and node.label == label:
            return node
    return None


def _find_principled(nodes: bpy.types.Nodes) -> Optional[bpy.types.Node]:
    for node in nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    return None


def _find_output(nodes: bpy.types.Nodes) -> Optional[bpy.types.Node]:
    for node in nodes:
        if node.type == "OUTPUT_MATERIAL" and node.is_active_output:
            return node
    for node in nodes:
        if node.type == "OUTPUT_MATERIAL":
            return node
    return None


def _ensure_node_tree(mat: bpy.types.Material) -> bpy.types.NodeTree:
    if not mat.use_nodes:
        mat.use_nodes = True
    return mat.node_tree


def _ensure_principled(tree: bpy.types.NodeTree) -> bpy.types.Node:
    """Return or create the Principled BSDF node."""
    pbsdf = _find_principled(tree.nodes)
    if pbsdf is None:
        pbsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
        pbsdf.location = (_COL_PBSDF, 0.0)
        output = _find_output(tree.nodes)
        if output:
            tree.links.new(pbsdf.outputs["BSDF"], output.inputs["Surface"])
    return pbsdf


def _set_input_safe(
    node: bpy.types.Node,
    socket_name: str,
    value: float,
) -> None:
    """Set a scalar input default_value, silently ignoring absent sockets."""
    sock = node.inputs.get(socket_name)
    if sock is not None:
        sock.default_value = value


# ---------------------------------------------------------------------------
# Node reuse helpers
# ---------------------------------------------------------------------------
def _find_autogen_tex_node(
    tree: bpy.types.NodeTree,
    abs_path: str,
) -> Optional[bpy.types.Node]:
    """
    Find an existing autogen Image Texture node pointing at *abs_path*.
    Uses Blender's own abspath resolution so relative paths still match.
    """
    target = bpy.path.abspath(abs_path).lower()
    for node in tree.nodes:
        if node.type == "TEX_IMAGE" and _is_autogen(node) and node.image:
            if bpy.path.abspath(node.image.filepath).lower() == target:
                return node
    return None


def _find_autogen_nmap_node(
    tree: bpy.types.NodeTree,
    tex_node: bpy.types.Node,
) -> Optional[bpy.types.Node]:
    """Find an existing autogen Normal Map node wired from *tex_node*."""
    for link in tree.links:
        if (
            link.from_node == tex_node
            and link.to_node.type == "NORMAL_MAP"
            and _is_autogen(link.to_node)
        ):
            return link.to_node
    return None


# ---------------------------------------------------------------------------
# Cleanup: remove only unclaimed autogen nodes
# ---------------------------------------------------------------------------
def _remove_unclaimed_autogen(
    tree: bpy.types.NodeTree,
    claimed: Set[bpy.types.Node],
) -> int:
    to_remove = [n for n in tree.nodes if _is_autogen(n) and n not in claimed]
    for node in to_remove:
        tree.nodes.remove(node)
    return len(to_remove)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def _load_image(abs_path: str) -> bpy.types.Image:
    """Load image with check_existing=True – never creates duplicates."""
    return bpy.data.images.load(abs_path, check_existing=True)


# ---------------------------------------------------------------------------
# Per-type PBSDF defaults
# ---------------------------------------------------------------------------
def _apply_pbsdf_defaults(
    pbsdf: bpy.types.Node,
    mat_type: MaterialType,
) -> None:
    """
    Apply per-material-type Principled BSDF defaults.

    These values compensate for the mismatch between MSM_BBQCartoon (a toon
    shader) and Blender's Principled BSDF (a PBR shader).  Individual texture
    channels (_R for roughness, etc.) will override these per-texel.
    """
    defaults = _PBSDF_DEFAULTS.get(mat_type.value, _PBSDF_DEFAULTS[MaterialType.UNKNOWN.value])
    for socket_name, value in defaults.items():
        _set_input_safe(pbsdf, socket_name, value)


# ---------------------------------------------------------------------------
# Rim light approximation
# ---------------------------------------------------------------------------
def _build_rim_light(
    tree: bpy.types.NodeTree,
    pbsdf: bpy.types.Node,
    record: MaterialRecord,
    claimed: Set[bpy.types.Node],
) -> bool:
    """
    Approximate SpecialRimLight via Layer Weight + ColorRamp + Emission BSDF
    inserted as an Add Shader before the Material Output.

    MSM_BBQCartoon uses three scalars/colors for its rim lighting:
      SpecialRimLightColor     → colour of the rim highlight
      SpecialRimLightPower     → sharpness / falloff exponent (3–10 typical)
      SpecialRimLightWidthAdd  → rim angular width additive offset (0–0.5)
      WorldFresnelIntensity    → global intensity multiplier (0–1, default 1)

    v1.6 improvements over v1.5:
      * Uses Layer Weight FACING output instead of FRESNEL.
        Facing = dot(view_dir, normal) = 0 at perpendicular, 1 at grazing.
        This is simpler and more artistically controllable than physical Fresnel
        for a toon shader where the rim is a stylistic effect, not optics.
      * ColorRamp interpolation: CONSTANT when power > 5 (hard cel step) or
        LINEAR for softer values. This matches the two visual styles seen in
        Khazan: sharp rims on clothing, soft rims on skin/hair.
      * WorldFresnelIntensity from JSON is used to scale the rim Emission
        Strength via a Math.MULTIPLY node, matching the UE parameter's role.

    :returns: True if rim light was built, False if no rim color in JSON.
    """
    rim_color_dict = record.colors.get("SpecialRimLightColor")
    if rim_color_dict is None:
        return False

    rim_power      = float(record.scalars.get("SpecialRimLightPower", 3.0))
    rim_width      = float(record.scalars.get("SpecialRimLightWidthAdd", 0.15))
    rim_intensity  = float(record.scalars.get("WorldFresnelIntensity", 1.0))
    rim_r = float(rim_color_dict.get("R", 1.0))
    rim_g = float(rim_color_dict.get("G", 0.9))
    rim_b = float(rim_color_dict.get("B", 0.7))

    # ---- Layer Weight (Facing output) ----
    # Facing is simpler than Fresnel for toon rim: it is the raw cosine of the
    # view-to-normal angle.  Blend input shifts the boundary inward/outward.
    # rim_width (0–0.5) widens the rim: map to Blend range 0.3–0.7.
    lw = _find_autogen_by_label(tree, _LABEL_RIM_LW)
    if lw is None:
        lw = tree.nodes.new("ShaderNodeLayerWeight")
        _tag_node(lw)
    lw.label = _LABEL_RIM_LW
    lw.location = (_COL_RIM_LW, _ROW_RIM)
    blend_value = max(0.3, min(0.75, 0.5 + rim_width * 0.6))
    _set_input_safe(lw, "Blend", blend_value)
    claimed.add(lw)

    # ---- ColorRamp: shapes sharpness and cel-ness ----
    ramp = _find_autogen_by_label(tree, _LABEL_RIM_RAMP)
    if ramp is None:
        ramp = tree.nodes.new("ShaderNodeValToRGB")
        _tag_node(ramp)
    ramp.label = _LABEL_RIM_RAMP
    ramp.location = (_COL_RIM_RAMP, _ROW_RIM)
    cr = ramp.color_ramp
    # Transition position: higher power = narrower rim = transition starts later.
    # sqrt formula gives a better perceptual spread than 1-1/power for Khazan's
    # typical range (power 3–8).
    import math as _math
    transition_start = max(0.45, min(0.97, 1.0 - _math.sqrt(1.0 / max(1.0, rim_power))))
    cr.elements[0].position = transition_start
    cr.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    cr.elements[1].position = min(transition_start + 0.05, 1.0)
    cr.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    # Hard step for high power (sharp cel rim); soft gradient for lower power.
    cr.interpolation = "CONSTANT" if rim_power > 5.0 else "LINEAR"
    claimed.add(ramp)
    tree.links.new(lw.outputs["Facing"], ramp.inputs["Fac"])

    # ---- WorldFresnelIntensity scalar multiplier ----
    # Inserted between ColorRamp and Emission Strength so that the overall
    # rim brightness can be tuned from JSON without touching the ramp shape.
    scale_node = _find_autogen_by_label(tree, _LABEL_RIM_SCALE)
    if scale_node is None:
        scale_node = tree.nodes.new("ShaderNodeMath")
        _tag_node(scale_node)
    scale_node.label = _LABEL_RIM_SCALE
    scale_node.operation = "MULTIPLY"
    scale_node.location = ((_COL_RIM_RAMP + _COL_RIM_EMISSION) / 2, _ROW_RIM)
    scale_node.inputs[1].default_value = max(0.0, rim_intensity)
    claimed.add(scale_node)
    tree.links.new(ramp.outputs["Color"], scale_node.inputs[0])

    # ---- Emission BSDF (rim colour) ----
    em = _find_autogen_by_label(tree, _LABEL_RIM_EMISSION)
    if em is None:
        em = tree.nodes.new("ShaderNodeEmission")
        _tag_node(em)
    em.label = _LABEL_RIM_EMISSION
    em.location = (_COL_RIM_EMISSION, _ROW_RIM)
    em.inputs["Color"].default_value = (rim_r, rim_g, rim_b, 1.0)
    _set_input_safe(em, "Strength", 1.0)
    claimed.add(em)
    tree.links.new(scale_node.outputs["Value"], em.inputs["Strength"])

    # ---- Add Shader: main PBSDF + rim emission ----
    add = _find_autogen_by_label(tree, _LABEL_RIM_ADD)
    if add is None:
        add = tree.nodes.new("ShaderNodeAddShader")
        _tag_node(add)
    add.label = _LABEL_RIM_ADD
    add.location = (_COL_RIM_ADD, 0.0)
    claimed.add(add)
    tree.links.new(pbsdf.outputs["BSDF"], add.inputs[0])
    tree.links.new(em.outputs["Emission"], add.inputs[1])

    # Rewire Material Output to receive Add Shader output.
    output = _find_output(tree.nodes)
    if output is not None:
        for link in list(tree.links):
            if link.to_node == output and link.to_socket.name == "Surface":
                tree.links.remove(link)
        tree.links.new(add.outputs["Shader"], output.inputs["Surface"])
        output.location = (_COL_RIM_ADD + 250, 0.0)

    return True


# ---------------------------------------------------------------------------
# Cel Specular Highlight
# ---------------------------------------------------------------------------
def _build_cel_specular(
    tree: bpy.types.NodeTree,
    pbsdf: bpy.types.Node,
    record: MaterialRecord,
    claimed: Set[bpy.types.Node],
) -> bool:
    """
    Approximate MSM_BBQCartoon cel specular highlight band using Layer Weight (Facing)
    and a CONSTANT ColorRamp.
    """
    spec_amount = float(record.scalars.get("SpecularAmount", 0.0))
    if spec_amount <= 0.0:
        return False

    lw = _find_autogen_by_label(tree, _LABEL_SPEC_LW)
    if lw is None:
        lw = tree.nodes.new("ShaderNodeLayerWeight")
        _tag_node(lw)
        lw.label = _LABEL_SPEC_LW
        lw.location = (_COL_RIM_LW, _ROW_RIM - 300)
    _set_input_safe(lw, "Blend", 0.85)
    claimed.add(lw)

    ramp = _find_autogen_by_label(tree, _LABEL_SPEC_RAMP)
    if ramp is None:
        ramp = tree.nodes.new("ShaderNodeValToRGB")
        _tag_node(ramp)
        ramp.label = _LABEL_SPEC_RAMP
        ramp.location = (_COL_RIM_RAMP, _ROW_RIM - 300)
    cr = ramp.color_ramp
    cr.interpolation = "CONSTANT"
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    cr.elements[1].position = max(0.85, 0.98 - spec_amount * 0.1)
    cr.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    claimed.add(ramp)
    tree.links.new(lw.outputs["Facing"], ramp.inputs["Fac"])

    spec_sock = pbsdf.inputs.get("Specular") or pbsdf.inputs.get("Specular IOR Level")
    if spec_sock:
        tree.links.new(ramp.outputs["Color"], spec_sock)
    return True


# ---------------------------------------------------------------------------
# Special material handlers
# ---------------------------------------------------------------------------
def _build_procedural_eye(
    tree: bpy.types.NodeTree,
    pbsdf: bpy.types.Node,
    record: MaterialRecord,
    claimed: Set[bpy.types.Node],
) -> List[str]:
    """
    Build a procedural anime eye using concentric circle zones.

    MSM_BBQCartoon generates the eye iris and pupil entirely procedurally;
    no textures exist (IsNull=True).  This function approximates the same
    concentric-zone structure using standard Blender nodes.

    JSON parameters used
    --------------------
    EyeUScale, EyeVScale   → UV tiling (default 1.0)
    PupilScale             → pupil radius as fraction of visible iris (0.3–0.6)
    Pupil_Circle0          → pupil dark center colour
    Pupil_Lens0            → iris main colour
    Pupil_Ring0            → limbal ring colour (darker ring at iris–sclera border)
    EyeWhiteColor0         → sclera colour

    Node graph
    ----------
    TexCoord.UV
        → Mapping (Scale = EyeUScale, EyeVScale, 1)
        → VectorMath.SUBTRACT (0.5, 0.5, 0)   ← centre UV at origin
        → VectorMath.LENGTH                    ← scalar radius 0.0 – ~0.5
        → Math.MULTIPLY (x2.0)                 ← normalise to 0.0 – 1.0
        → ColorRamp (6 stops)                  ← pupil → iris → limbal → sclera
        → PBSDF Base Color

    All nodes are tagged khazan_autogen and reused on subsequent builds.
    :returns: list of warning strings for the logger.
    """
    warnings: List[str] = []

    # --- 3 Eye Preset Modes (Phase 7) ---
    eye_mode = "MODE_C_TRAILER"
    try:
        settings = bpy.context.scene.khazan_settings
        eye_mode = settings.eye_mode
    except Exception:
        pass

    if eye_mode == "MODE_A_RAW":
        # Mode A: Exact exported JSON values (Raw Slate Blue, zero interpretation)
        d_white  = record.colors.get("EyeWhiteColor0", {})
        d_lens   = record.colors.get("Pupil_Lens0", {})
        d_circle = record.colors.get("Pupil_Circle0", {})
        d_ring   = record.colors.get("Pupil_Ring0", {})
        white_col  = (float(d_white.get("R", 0.698)),  float(d_white.get("G", 0.563)),  float(d_white.get("B", 0.563)), 1.0)
        lens_col   = (float(d_lens.get("R", 0.417)),   float(d_lens.get("G", 0.483)),   float(d_lens.get("B", 0.599)), 1.0)
        circle_col = (float(d_circle.get("R", 0.294)), float(d_circle.get("G", 0.340)), float(d_circle.get("B", 0.422)), 1.0)
        ring_col   = (float(d_ring.get("R", 0.115)),   float(d_ring.get("G", 0.186)),   float(d_ring.get("B", 0.328)), 1.0)
    elif eye_mode == "MODE_B_CONCEPT":
        # Mode B: Concept Art Amber Gold Preset (Artistic Override)
        white_col  = (0.92, 0.88, 0.88, 1.0)
        lens_col   = (0.83, 0.61, 0.29, 1.0)
        circle_col = (0.36, 0.21, 0.06, 1.0)
        ring_col   = (0.17, 0.08, 0.01, 1.0)
    else:
        # Mode C: Game Trailer Approximation (Default / Primary Research Target)
        # Muted pale gray-slate blue matching trailer screenshot
        white_col  = (0.90, 0.86, 0.86, 1.0)
        lens_col   = (0.48, 0.55, 0.65, 1.0)
        circle_col = (0.15, 0.18, 0.25, 1.0)
        ring_col   = (0.22, 0.28, 0.36, 1.0)

    pupil_scale = float(record.scalars.get("PupilScale", 0.4))
    eye_u       = float(record.scalars.get("EyeUScale",  1.0))
    eye_v       = float(record.scalars.get("EyeVScale",  1.0))

    # Compute ColorRamp zone boundaries in normalised radius space [0, 1].
    # PupilScale in Unreal controls pupil radius relative to overall iris.
    # Radius 0.0 - 0.20: Pupil dark center
    # Radius 0.20 - 0.50: Main Iris body
    # Radius 0.50 - 0.60: Limbal ring border
    # Radius 0.60 - 1.00: Sclera (eyeball white)
    eff_pupil    = max(0.15, min(0.65, 0.22 * pupil_scale))
    pupil_end    = eff_pupil * 0.85
    iris_start   = eff_pupil
    limbal_start = 0.48
    sclera_start = 0.60

    # Clamp everything to valid range
    def _c(v: float) -> float:
        return max(0.001, min(0.999, v))

    y_base = _ROW_RIM + 300  # row above rim nodes

    # ---- 1. Texture Coordinate ----
    texco = _find_autogen_by_label(tree, _LABEL_EYE_TEXCO)
    if texco is None:
        texco = tree.nodes.new("ShaderNodeTexCoord")
        _tag_node(texco)
    texco.label = _LABEL_EYE_TEXCO
    texco.location = (_COL_TEX - 300, y_base)
    claimed.add(texco)

    # ---- 2. Mapping (apply UV tiling) ----
    mapping = _find_autogen_by_label(tree, _LABEL_EYE_MAPPING)
    if mapping is None:
        mapping = tree.nodes.new("ShaderNodeMapping")
        _tag_node(mapping)
    mapping.label = _LABEL_EYE_MAPPING
    mapping.location = (_COL_TEX - 100, y_base)
    mapping.vector_type = "POINT"
    mapping.inputs["Scale"].default_value = (eye_u, eye_v, 1.0)
    claimed.add(mapping)
    tree.links.new(texco.outputs["UV"], mapping.inputs["Vector"])

    # ---- 3. Subtract (0.5, 0.5, 0) to centre UV at origin ----
    sub = _find_autogen_by_label(tree, _LABEL_EYE_SUBTRACT)
    if sub is None:
        sub = tree.nodes.new("ShaderNodeVectorMath")
        _tag_node(sub)
    sub.label = _LABEL_EYE_SUBTRACT
    sub.operation = "SUBTRACT"
    sub.location = (_COL_TEX + 100, y_base)
    sub.inputs[1].default_value = (0.5, 0.5, 0.0)
    claimed.add(sub)
    tree.links.new(mapping.outputs["Vector"], sub.inputs[0])

    # ---- 4. Vector Length (scalar radius) ----
    vlen = _find_autogen_by_label(tree, _LABEL_EYE_LENGTH)
    if vlen is None:
        vlen = tree.nodes.new("ShaderNodeVectorMath")
        _tag_node(vlen)
    vlen.label = _LABEL_EYE_LENGTH
    vlen.operation = "LENGTH"
    vlen.location = (_COL_TEX + 300, y_base)
    claimed.add(vlen)
    tree.links.new(sub.outputs["Vector"], vlen.inputs[0])

    # ---- 5. Multiply ×2 to normalise 0–0.5 range → 0–1 ----
    norm = _find_autogen_by_label(tree, _LABEL_EYE_SCALE)
    if norm is None:
        norm = tree.nodes.new("ShaderNodeMath")
        _tag_node(norm)
    norm.label = _LABEL_EYE_SCALE
    norm.operation = "MULTIPLY"
    norm.location = (_COL_TEX + 500, y_base)
    norm.inputs[1].default_value = 2.0
    claimed.add(norm)
    tree.links.new(vlen.outputs["Value"], norm.inputs[0])

    # ---- 6. ColorRamp: 6-stop zone definition ----
    ramp = _find_autogen_by_label(tree, _LABEL_EYE_RAMP)
    if ramp is None:
        ramp = tree.nodes.new("ShaderNodeValToRGB")
        _tag_node(ramp)
    ramp.label = _LABEL_EYE_RAMP
    ramp.location = (_COL_TEX + 750, y_base)
    claimed.add(ramp)

    cr = ramp.color_ramp
    cr.interpolation = "LINEAR"

    # Rebuild to exactly 6 stops (ColorRamp starts with 2).
    while len(cr.elements) > 2:
        cr.elements.remove(cr.elements[-1])

    # Stop 0: pupil centre core (darkest black-navy)
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.03, 0.03, 0.06, 1.0)
    # Stop 5 (last): sclera
    cr.elements[1].position = 1.0
    cr.elements[1].color = white_col

    # Insert intermediate stops in ascending position order
    e1 = cr.elements.new(_c(pupil_end))
    e1.color = circle_col
    e2 = cr.elements.new(_c(iris_start))
    e2.color = lens_col
    e3 = cr.elements.new(_c(limbal_start))
    e3.color = ring_col
    e4 = cr.elements.new(_c(sclera_start))
    e4.color = white_col

    tree.links.new(norm.outputs["Value"], ramp.inputs["Fac"])

    # ---- 7. Anime Specular Highlight Spot ----
    # Creates a crisp off-center white highlight dot typical in anime eyes
    hl_sub = _find_autogen_by_label(tree, _LABEL_EYE_HIGHLIGHT + "_Sub")
    if hl_sub is None:
        hl_sub = tree.nodes.new("ShaderNodeVectorMath")
        _tag_node(hl_sub)
    hl_sub.label = _LABEL_EYE_HIGHLIGHT + "_Sub"
    hl_sub.operation = "SUBTRACT"
    hl_sub.location = (_COL_TEX + 750, y_base + 220)
    # Offset highlight position to upper-right quadrant (-0.12, 0.15)
    hl_sub.inputs[1].default_value = (-0.12, 0.15, 0.0)
    claimed.add(hl_sub)
    tree.links.new(sub.outputs["Vector"], hl_sub.inputs[0])

    hl_len = _find_autogen_by_label(tree, _LABEL_EYE_HIGHLIGHT + "_Len")
    if hl_len is None:
        hl_len = tree.nodes.new("ShaderNodeVectorMath")
        _tag_node(hl_len)
    hl_len.label = _LABEL_EYE_HIGHLIGHT + "_Len"
    hl_len.operation = "LENGTH"
    hl_len.location = (_COL_TEX + 950, y_base + 220)
    claimed.add(hl_len)
    tree.links.new(hl_sub.outputs["Vector"], hl_len.inputs[0])

    hl_ramp = _find_autogen_by_label(tree, _LABEL_EYE_HIGHLIGHT)
    if hl_ramp is None:
        hl_ramp = tree.nodes.new("ShaderNodeValToRGB")
        _tag_node(hl_ramp)
    hl_ramp.label = _LABEL_EYE_HIGHLIGHT
    hl_ramp.location = (_COL_TEX + 1150, y_base + 220)
    claimed.add(hl_ramp)

    hl_cr = hl_ramp.color_ramp
    hl_cr.interpolation = "CONSTANT"  # Hard cel highlight dot
    hl_cr.elements[0].position = 0.0
    hl_cr.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    hl_cr.elements[1].position = 0.065  # highlight radius
    hl_cr.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    tree.links.new(hl_len.outputs["Value"], hl_ramp.inputs["Fac"])

    # ---- 8. Mix Iris Base Color with Specular Highlight ----
    mix_node = _find_autogen_by_label(tree, _LABEL_EYE_MIX)
    if mix_node is None:
        mix_node = tree.nodes.new("ShaderNodeMixRGB")
        _tag_node(mix_node)
    mix_node.label = _LABEL_EYE_MIX
    mix_node.blend_type = "MIX"
    mix_node.location = (_COL_TEX + 1350, y_base)
    claimed.add(mix_node)

    tree.links.new(hl_ramp.outputs["Color"], mix_node.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], mix_node.inputs["Color1"])
    # Highlight color = crisp white
    mix_node.inputs["Color2"].default_value = (1.0, 1.0, 1.0, 1.0)

    # ---- 9. Upper Eyelid Cast Shadow Overlay ----
    shadow_ramp = _find_autogen_by_label(tree, "Khazan_Eye_UpperShadow")
    if shadow_ramp is None:
        shadow_ramp = tree.nodes.new("ShaderNodeValToRGB")
        _tag_node(shadow_ramp)
        shadow_ramp.label = "Khazan_Eye_UpperShadow"
        shadow_ramp.location = (_COL_TEX + 1550, y_base + 150)
    claimed.add(shadow_ramp)

    s_cr = shadow_ramp.color_ramp
    s_cr.elements[0].position = 0.0
    s_cr.elements[0].color = (0.78, 0.78, 0.85, 1.0)  # shadow tint at top
    s_cr.elements[1].position = 0.38
    s_cr.elements[1].color = (1.0, 1.0, 1.0, 1.0)   # full brightness below

    tree.links.new(sub.outputs["Vector"], shadow_ramp.inputs["Fac"])

    mix_shadow = _find_autogen_by_label(tree, "Khazan_Eye_ShadowMix")
    if mix_shadow is None:
        mix_shadow = tree.nodes.new("ShaderNodeMixRGB")
        _tag_node(mix_shadow)
        mix_shadow.label = "Khazan_Eye_ShadowMix"
        mix_shadow.blend_type = "MULTIPLY"
        mix_shadow.location = (_COL_TEX + 1750, y_base)
        mix_shadow.inputs["Fac"].default_value = 0.60
    claimed.add(mix_shadow)

    tree.links.new(mix_node.outputs["Color"], mix_shadow.inputs["Color1"])
    tree.links.new(shadow_ramp.outputs["Color"], mix_shadow.inputs["Color2"])

    # ---- Connect to PBSDF Base Color ----
    base_sock = pbsdf.inputs.get("Base Color")
    if base_sock:
        tree.links.new(mix_shadow.outputs["Color"], base_sock)

    warnings.append(
        f"EYE: Procedural iris/pupil + anime highlight + eyelid shadow built. "
        f"PupilScale={pupil_scale:.2f}, EyeU={eye_u:.1f}, EyeV={eye_v:.1f}. "
        f"Approximates MSM_BBQCartoon procedural eye rendering."
    )
    return warnings


def _apply_eyeshadow_setup(
    mat: bpy.types.Material,
    pbsdf: bpy.types.Node,
    record: MaterialRecord,
) -> List[str]:
    """
    EYESHADOW materials use MSM_Unlit with BLEND_Translucent.
    They render as a flat colour overlay with a fixed opacity.

    JSON parameters used:
      Colors.Color   → base colour of the shadow (usually black)
      Scalars.Opacity → translucency (0 = invisible, 1 = opaque)

    Blender approximation: Principled BSDF with matching color and alpha.
    """
    warnings: List[str] = []

    color_dict = record.colors.get("Color", {})
    r = float(color_dict.get("R", 0.0))
    g = float(color_dict.get("G", 0.0))
    b = float(color_dict.get("B", 0.0))

    base_sock = pbsdf.inputs.get("Base Color")
    if base_sock:
        base_sock.default_value = (r, g, b, 1.0)

    opacity = float(record.scalars.get("Opacity", 0.6))
    alpha_sock = pbsdf.inputs.get("Alpha")
    if alpha_sock:
        alpha_sock.default_value = opacity

    mat.blend_method = "BLEND"
    mat.shadow_method = "HASHED"

    warnings.append(
        f"EYESHADOW (MSM_Unlit): applied Color=({r:.2f},{g:.2f},{b:.2f}) "
        f"and Opacity={opacity:.2f} from JSON. No textures."
    )
    return warnings


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------
def build_material(
    mat: bpy.types.Material,
    record: MaterialRecord,
    settings,           # KhazanSettings PropertyGroup (avoids circular import)
    mat_type: MaterialType = MaterialType.UNKNOWN,
    dry_run: bool = False,
) -> BuildResult:
    """
    Rebuild the node tree of *mat* from *record*.

    Steps:
      1. Ensure nodes are active, Principled BSDF + Output exist.
      2. Apply per-type PBSDF defaults (Metallic=0, correct Specular/Roughness).
      3. For each texture: reuse existing autogen node or create new one.
         - D: Base Color (+Alpha for BLEND_Masked)
         - N: Normal Map (Non-Color)
         - R: Roughness (Non-Color)
         - E/S/I/NTP/F2: loaded, labelled, disconnected.
      4. Build rim light (if SpecialRimLightColor present).
      5. Special handling for EYE and EYESHADOW types.
      6. Remove leftover unclaimed autogen nodes.
      7. Apply blend mode + two-sided from JSON.
    """
    result = BuildResult(
        material_name=mat.name,
        json_matched=record.json_stem,
        mat_type=mat_type.value,
        was_dry_run=dry_run,
    )

    if dry_run:
        # Mirror the wiring logic so dry-run output matches what the actual
        # import would produce.  Textures with uncertain channels (E, S, I,
        # NTP, F2) are always reference-only, never wired → report as Skipped,
        # not Loaded.  Without this distinction the dry-run falsely shows _E
        # as "Loaded" for materials where EmissiveAmount > 0 was previously
        # used as the wiring heuristic.
        for entry in record.textures:
            if entry.abs_path is None:
                result.textures_missing.append(entry.filename)
            elif entry.canonical in WIRED_CHANNELS:
                result.textures_loaded.append(entry.filename)
            else:
                # Loaded as reference only — not wired to any PBSDF input
                reason = _skip_reason(entry.canonical)
                result.textures_skipped.append((entry.filename, reason))
        return result

    tree = _ensure_node_tree(mat)
    pbsdf = _ensure_principled(tree)

    # ---- Step 2: Apply per-type PBSDF defaults ----
    _apply_pbsdf_defaults(pbsdf, mat_type)

    # ---- Special: EYESHADOW has no textures – apply color/opacity and exit early ----
    if mat_type == MaterialType.EYESHADOW:
        ws = _apply_eyeshadow_setup(mat, pbsdf, record)
        result.warnings.extend(ws)
        return result

    # Track claimed nodes for cleanup
    claimed_nodes: Set[bpy.types.Node] = set()

    # Determine if this is a masked material (needs alpha channel from D tex)
    is_masked = record.blend_mode == 1  # BLEND_Masked

    # Sort: wired channels (D, N, R) first
    def _sort_key(e: TextureEntry) -> int:
        return {"D": 0, "N": 1, "R": 2}.get(e.canonical, 99)

    sorted_entries = sorted(record.textures, key=_sort_key)

    for row_index, entry in enumerate(sorted_entries):
        y = -(_ROW_STEP * row_index)

        if entry.abs_path is None:
            result.textures_missing.append(entry.filename)
            continue

        # Load image
        try:
            img = _load_image(entry.abs_path)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"Could not load {entry.filename}: {exc}")
            continue

        # Canonical channel never escalates to E_WIRE anymore
        canonical = entry.canonical
        should_wire = canonical in WIRED_CHANNELS

        if not should_wire and settings.ignore_unknown_maps:
            result.textures_skipped.append(
                (entry.filename, "ignored (unknown map, setting active)")
            )
            continue

        # ---- Reuse or create Image Texture node ----
        tex_node = _find_autogen_tex_node(tree, entry.abs_path)
        if tex_node is None:
            tex_node = tree.nodes.new("ShaderNodeTexImage")
            _tag_node(tex_node)
            tex_node.image = img
            tex_node.location = (_COL_TEX, y)
        else:
            tex_node.image = img  # refresh in case file moved
        claimed_nodes.add(tex_node)

        if not should_wire:
            skip_reason = _skip_reason(canonical, record)
            result.textures_skipped.append((entry.filename, skip_reason))
            # Use the rich node label so artists can understand why this
            # texture is disconnected directly from the Shader Editor.
            tex_node.label = _skip_node_label(canonical)
            continue

        result.textures_loaded.append(entry.filename)

        # ---- Channel-specific wiring ----
        if canonical == "D":
            tex_node.label = f"[Base Color] {entry.filename}"
            tree.links.new(tex_node.outputs["Color"], pbsdf.inputs["Base Color"])
            # For cutout materials, also wire the alpha channel from the
            # diffuse texture (UV-mapped transparency mask used by hair, etc.)
            if is_masked:
                alpha_sock = pbsdf.inputs.get("Alpha")
                if alpha_sock is not None:
                    tree.links.new(tex_node.outputs["Alpha"], alpha_sock)
                    tex_node.label = f"[Base Color + Alpha] {entry.filename}"

        elif canonical == "N":
            img.colorspace_settings.name = "Non-Color"
            tex_node.label = f"[Normal] {entry.filename}"

            nmap_node = _find_autogen_nmap_node(tree, tex_node)
            if nmap_node is None:
                nmap_node = tree.nodes.new("ShaderNodeNormalMap")
                _tag_node(nmap_node)
                nmap_node.location = (_COL_NMAP, y)
                nmap_node.label = "Normal Map (Khazan)"
            claimed_nodes.add(nmap_node)

            tree.links.new(tex_node.outputs["Color"], nmap_node.inputs["Color"])
            tree.links.new(nmap_node.outputs["Normal"], pbsdf.inputs["Normal"])

        elif canonical == "R":
            img.colorspace_settings.name = "Non-Color"
            tex_node.label = f"[Glossiness -> Inverted Roughness] {entry.filename}"

            inv_node = _find_autogen_by_label(tree, "Khazan_Invert_Roughness")
            if inv_node is None:
                inv_node = tree.nodes.new("ShaderNodeInvert")
                _tag_node(inv_node)
                inv_node.location = (_COL_TEX + 250, y)
                inv_node.label = "Toon Glossiness -> PBR Roughness (Invert)"
            claimed_nodes.add(inv_node)

            tree.links.new(tex_node.outputs["Color"], inv_node.inputs["Color"])
            tree.links.new(inv_node.outputs["Color"], pbsdf.inputs["Roughness"])

        elif canonical == "S_PACKED":
            img.colorspace_settings.name = "Non-Color"
            tex_node.label = f"[Packed SpecularMasks] {entry.filename}"

            sep_node = _find_autogen_by_label(tree, "Khazan_Unpack_SpecularMasks")
            if sep_node is None:
                sep_node = tree.nodes.new("ShaderNodeSeparateColor")
                _tag_node(sep_node)
                sep_node.location = (_COL_TEX + 250, y)
                sep_node.label = "Unpack SpecularMasks (R=Spec, G=Shadow, B=Rim)"
            claimed_nodes.add(sep_node)

            tree.links.new(tex_node.outputs["Color"], sep_node.inputs["Color"])
            # Channel R -> Specular intensity (ONLY for props/metals or wet eyes to avoid HDRI cloth sheen)
            if mat_type in (MaterialType.ITEM, MaterialType.EYE):
                spec_sock = pbsdf.inputs.get("Specular") or pbsdf.inputs.get("Specular IOR Level")
                if spec_sock:
                    tree.links.new(sep_node.outputs["Red"], spec_sock)

        elif canonical == "I_INDIRECT":
            img.colorspace_settings.name = "sRGB"
            tex_node.label = f"[Indirect Lighting] {entry.filename}"

            mix_indirect = _find_autogen_by_label(tree, "Khazan_Indirect_Mix")
            if mix_indirect is None:
                mix_indirect = tree.nodes.new("ShaderNodeMixRGB")
                _tag_node(mix_indirect)
                mix_indirect.blend_type = "MULTIPLY"
                mix_indirect.location = (_COL_TEX + 350, y)
                mix_indirect.label = "Indirect Ambient Overlay (35%)"
                mix_indirect.inputs["Fac"].default_value = 0.35
            claimed_nodes.add(mix_indirect)

            base_sock = pbsdf.inputs.get("Base Color")
            if base_sock and base_sock.is_linked:
                orig_link = base_sock.links[0]
                from_socket = orig_link.from_socket
                tree.links.remove(orig_link)
                tree.links.new(from_socket, mix_indirect.inputs["Color1"])
                tree.links.new(tex_node.outputs["Color"], mix_indirect.inputs["Color2"])
                tree.links.new(mix_indirect.outputs["Color"], base_sock)
            elif base_sock:
                tree.links.new(tex_node.outputs["Color"], base_sock)

    # ---- EYE: procedural iris/pupil node graph ----
    if mat_type == MaterialType.EYE:
        ws = _build_procedural_eye(tree, pbsdf, record, claimed_nodes)
        result.warnings.extend(ws)

    # ---- Rim light (only for materials that define SpecialRimLightColor) ----
    built_rim = _build_rim_light(tree, pbsdf, record, claimed_nodes)
    if built_rim and settings.verbose_logging:
        result.warnings.append(
            "Rim light nodes built from SpecialRimLightColor/Power/WidthAdd."
        )

    # ---- Cel specular highlight band ----
    _build_cel_specular(tree, pbsdf, record, claimed_nodes)

    # ---- Cleanup leftover autogen nodes ----
    removed = _remove_unclaimed_autogen(tree, claimed_nodes)
    if removed and settings.verbose_logging:
        result.warnings.append(f"Removed {removed} stale auto-generated node(s).")

    # ---- Blend mode + two-sided ----
    _apply_blend_mode(mat, record)

    return result


# ---------------------------------------------------------------------------
# Blend mode + two-sided
# ---------------------------------------------------------------------------
def _apply_blend_mode(mat: bpy.types.Material, record: MaterialRecord) -> None:
    """
    Map Unreal BlendMode / IsTranslucent → Blender material alpha settings.

    Unreal BlendMode int values (from UE source):
        0 = BLEND_Opaque
        1 = BLEND_Masked  (alpha clip)
        2 = BLEND_Translucent
        3 = BLEND_Additive
        4 = BLEND_Modulate
    """
    blend_int = record.blend_mode

    if record.is_translucent or blend_int == 2:
        mat.blend_method = "BLEND"
        mat.shadow_method = "HASHED"
    elif blend_int == 1:
        mat.blend_method = "CLIP"
        mat.shadow_method = "CLIP"
        mat.alpha_threshold = 0.3333  # matches OpacityMaskClipValue in JSON
    else:
        mat.blend_method = "OPAQUE"
        mat.shadow_method = "OPAQUE"

    # TwoSided from JSON: disable backface culling only when the material
    # explicitly sets it (hair, clothing) vs. solid surfaces (book, eye).
    mat.use_backface_culling = not record.two_sided


# ---------------------------------------------------------------------------
# Skip-reason text for logging + node labelling
# ---------------------------------------------------------------------------
class SkipInfo(NamedTuple):
    """
    Information about why a texture channel is left disconnected.

    node_label  : Short string set directly on the Blender Image Texture node.
                  Visible at a glance in the Shader Editor; keeps the node graph
                  educational without needing to open the Console.
    log_reason  : Full technical explanation written to the System Console.
    """
    node_label: str
    log_reason: str


_SKIP_INFO: Dict[str, SkipInfo] = {
    "S_UNKNOWN": SkipInfo(
        node_label=(
            "[SpecularMasks] Packed RGBA\n"
            "MSM_BBQCartoon packed mask\n"
            "R=Specular  G=Shadow width\n"
            "B=Rim mask  A=unknown\n"
            "Use Separate Color node to unpack"
        ),
        log_reason=(
            "PM_SpecularMasks is a PACKED RGBA mask texture. "
            "MSM_BBQCartoon channels: R=Specular intensity, G=Toon shadow width, "
            "B=Rim light mask, A=unknown. These channels cannot be collapsed into "
            "a single Principled BSDF input without losing information. "
            "Artist: add a Separate Color node to extract individual channels."
        ),
    ),
    "E_UNKNOWN": SkipInfo(
        node_label=(
            "[Emissive] BBQCartoon toon effect\n"
            "NOT a PBR emission glow\n"
            "Wiring washes out cel shadows\n"
            "Leave disconnected"
        ),
        log_reason=(
            "PM_Emissive / Tex_E: MSM_BBQCartoon uses BaseEmissiveAmount and "
            "EmissiveAmount scalars to composite a toon self-illumination tint "
            "with cel shadow bands. Connecting to Blender Emission adds flat "
            "view-independent brightness that destroys shadow depth. "
            "Leave disconnected unless creating an explicit glow (e.g. magic runes)."
        ),
    ),
    "I_UNKNOWN": SkipInfo(
        node_label=(
            "[Indirect Lighting] Baked UE illumination\n"
            "Pre-baked toon indirect map\n"
            "No Principled BSDF equivalent\n"
            "Reference only — do not wire"
        ),
        log_reason=(
            "Tex_I: Almost certainly a pre-baked IndirectColor / IlluminationColor "
            "map used by MSM_BBQCartoon's toon lighting model. _I textures are "
            "3–5× larger than the diffuse, confirming they store high-detail baked "
            "lighting data. There is no equivalent Principled BSDF input. "
            "Do NOT wire; keep as a reference for shader reconstruction."
        ),
    ),
    "NTP_UNKNOWN": SkipInfo(
        node_label=(
            "[Normal-Thickness-Porosity?]\n"
            "Face skin detail map\n"
            "Purpose unconfirmed\n"
            "Loaded but NOT wired"
        ),
        log_reason=(
            "Tex_NTP: Face-specific map, possibly Normal-Thickness-Porosity or a "
            "skin detail / SSS map used for the toon shader's skin pass. "
            "Purpose unconfirmed by analysis. Loaded but NOT wired."
        ),
    ),
    "F2_UNKNOWN": SkipInfo(
        node_label=(
            "[Face Detail / Blush]\n"
            "Toon skin compositing\n"
            "Purpose unconfirmed\n"
            "Loaded but NOT wired"
        ),
        log_reason=(
            "Tex_F2: Face secondary detail / blush / makeup overlay. "
            "Likely composited by MSM_BBQCartoon's skin rendering pass. "
            "Loaded but NOT wired."
        ),
    ),
}


def _skip_reason(canonical: str, record: Optional[MaterialRecord] = None) -> str:
    """
    Return the log_reason string for a skipped channel.
    Used by the dry-run path and the System Console logger.
    """
    info = _SKIP_INFO.get(canonical)
    if info:
        return info.log_reason
    return f"Unknown channel '{canonical}' – NOT wired."


def _skip_node_label(canonical: str) -> str:
    """
    Return the short node_label for a skipped channel.
    Set directly on the Blender Image Texture node for in-editor visibility.
    """
    info = _SKIP_INFO.get(canonical)
    if info:
        return info.node_label
    return f"[Unknown channel: {canonical}]\nNot wired"


# ---------------------------------------------------------------------------
# Phase 5A: Visual Debug Node Preview Toggle
# ---------------------------------------------------------------------------
def toggle_debug_texture_preview(mat: bpy.types.Material, channel_to_preview: Optional[str] = None) -> Tuple[bool, str]:
    """
    Temporarily connect a disconnected reference texture node (Tex_S, Tex_E, Tex_I, Tex_NTP, Tex_F2)
    directly into Base Color for rapid visual inspection in Blender viewport, or revert back to Tex_D.

    :param mat: Target Blender Material.
    :param channel_to_preview: Channel canonical code e.g. "S_UNKNOWN", "E_UNKNOWN", "I_UNKNOWN", "D", or None.
    :returns: (success_bool, status_message)
    """
    if not mat.use_nodes or not mat.node_tree:
        return False, "Material has no node tree."

    tree = mat.node_tree
    pbsdf = _find_principled(tree.nodes)
    if not pbsdf:
        return False, "Material has no Principled BSDF node."

    base_color_sock = pbsdf.inputs.get("Base Color")
    if not base_color_sock:
        return False, "Principled BSDF has no Base Color socket."

    # Remove existing links to Base Color
    for link in list(tree.links):
        if link.to_socket == base_color_sock:
            tree.links.remove(link)

    # Find image nodes
    img_nodes = [n for n in tree.nodes if n.type == "TEX_IMAGE"]
    if not img_nodes:
        return False, "No Image Texture nodes found in material."

    if not channel_to_preview or channel_to_preview == "D":
        # Revert to standard Diffuse (D) texture
        d_nodes = [n for n in img_nodes if "[Base Color" in n.label or "_D" in (n.image.name if n.image else "")]
        target = d_nodes[0] if d_nodes else img_nodes[0]
        tree.links.new(target.outputs["Color"], base_color_sock)
        return True, f"Reverted Base Color to standard diffuse ({target.name})."

    # Find matching disconnected channel node
    matching = [n for n in img_nodes if channel_to_preview in n.label or (n.image and channel_to_preview.split("_")[0] in n.image.name)]
    if not matching:
        # Fallback to any node with matching label keyword
        matching = [n for n in img_nodes if n != _find_principled(tree.nodes)]

    if not matching:
        return False, f"No texture node found matching channel '{channel_to_preview}'."

    target_node = matching[0]
    tree.links.new(target_node.outputs["Color"], base_color_sock)
    return True, f"Previewing '{target_node.label}' on Base Color."


def setup_base_eye_e_debug_preview(
    mat: bpy.types.Material,
    channel_mode: str = "ALL",
    texture_path: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Forensic experiment: Wire UV -> BASE_Eye_E.png -> (Separate Color R/G/B/A) -> Principled BSDF Base Color
    directly on the eye material to visually inspect mesh UV alignment and channel mapping in Viewport.

    :param mat: Target eye material (e.g. C_NPC_Daprona_Eye).
    :param channel_mode: "ALL", "R", "G", "B", "A", or "RESET".
    :param texture_path: Path to BASE_Eye_E.png image file.
    """
    if not mat or not mat.use_nodes or not mat.node_tree:
        return False, "Material has no node tree."

    tree = mat.node_tree
    pbsdf = _find_principled(tree.nodes)
    if not pbsdf:
        return False, "Material has no Principled BSDF node."

    base_sock = pbsdf.inputs.get("Base Color")
    if not base_sock:
        return False, "Principled BSDF has no Base Color socket."

    # Disconnect existing Base Color links
    for link in list(tree.links):
        if link.to_socket == base_sock:
            tree.links.remove(link)

    if channel_mode == "RESET":
        eye_ramp = _find_autogen_by_label(tree, "Khazan_Eye_ColorRamp")
        if eye_ramp and "Color" in eye_ramp.outputs:
            tree.links.new(eye_ramp.outputs["Color"], base_sock)
            return True, "Reverted eye material to standard procedural shader."
        return True, "Disconnected debug preview."

    # Load or find BASE_Eye_E image
    img = None
    if texture_path and os.path.exists(texture_path):
        img = bpy.data.images.load(texture_path, check_existing=True)
    else:
        for image_item in bpy.data.images:
            if "base_eye_e" in image_item.name.lower():
                img = image_item
                break

    if not img:
        return False, "BASE_Eye_E.png texture not loaded in Blender data."

    # Find or create Texture Image, UV, and Separate Color nodes
    tex_node = _find_autogen_by_label(tree, "Khazan_Debug_BASE_Eye_E")
    if not tex_node:
        tex_node = tree.nodes.new("ShaderNodeTexImage")
        _tag_node(tex_node)
        tex_node.label = "Khazan_Debug_BASE_Eye_E"
        tex_node.location = (-600, 300)

    tex_node.image = img
    if tex_node.image:
        tex_node.image.colorspace_settings.name = "sRGB" if channel_mode == "ALL" else "Non-Color"

    uv_node = _find_autogen_by_label(tree, "Khazan_Debug_Eye_UV")
    if not uv_node:
        uv_node = tree.nodes.new("ShaderNodeTexCoord")
        _tag_node(uv_node)
        uv_node.label = "Khazan_Debug_Eye_UV"
        uv_node.location = (-850, 300)

    sep_node = _find_autogen_by_label(tree, "Khazan_Debug_Eye_Separate")
    if not sep_node:
        sep_node = tree.nodes.new("ShaderNodeSeparateColor")
        _tag_node(sep_node)
        sep_node.label = "Khazan_Debug_Eye_Separate"
        sep_node.location = (-350, 300)

    # Wire UV -> Image
    tree.links.new(uv_node.outputs["UV"], tex_node.inputs["Vector"])
    tree.links.new(tex_node.outputs["Color"], sep_node.inputs["Color"])

    if channel_mode == "ALL":
        tree.links.new(tex_node.outputs["Color"], base_sock)
        return True, "Wired UV -> BASE_Eye_E (Full RGBA) -> Base Color."
    elif channel_mode in ("R", "RED"):
        tree.links.new(sep_node.outputs["Red"], base_sock)
        return True, "Wired UV -> BASE_Eye_E -> Separate Color [RED] -> Base Color."
    elif channel_mode in ("G", "GREEN"):
        tree.links.new(sep_node.outputs["Green"], base_sock)
        return True, "Wired UV -> BASE_Eye_E -> Separate Color [GREEN] -> Base Color."
    elif channel_mode in ("B", "BLUE"):
        tree.links.new(sep_node.outputs["Blue"], base_sock)
        return True, "Wired UV -> BASE_Eye_E -> Separate Color [BLUE] -> Base Color."
    elif channel_mode in ("A", "ALPHA"):
        tree.links.new(sep_node.outputs["Alpha"], base_sock)
        return True, "Wired UV -> BASE_Eye_E -> Separate Color [ALPHA] -> Base Color."

    return False, f"Unknown channel mode '{channel_mode}'."

