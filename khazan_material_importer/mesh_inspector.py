"""
mesh_inspector.py
=================
Mesh & UV Relationship Inspector for the Khazan Material Importer.

Provides forensic tools for inspecting mesh topology, material assignment,
UV coordinates (including iris UV centroid alignment), and detecting
co-located multi-mesh setups (e.g. Eye vs Cornea vs Lashes).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import bpy


def inspect_uv_layer(mesh: bpy.types.Mesh, uv_layer_name: str) -> Dict:
    """
    Inspect a specific UV layer on a mesh and return statistical bounds,
    centroid, aspect ratio, and iris alignment analysis.
    """
    uv_layer = mesh.uv_layers.get(uv_layer_name)
    if not uv_layer or not mesh.loops:
        return {
            "name": uv_layer_name,
            "valid": False,
            "reason": "Empty or missing UV layer",
        }

    u_vals: List[float] = []
    v_vals: List[float] = []

    for loop in mesh.loops:
        uv = uv_layer.data[loop.index].uv
        u_vals.append(float(uv[0]))
        v_vals.append(float(uv[1]))

    if not u_vals:
        return {"name": uv_layer_name, "valid": False, "reason": "No UV loop data"}

    u_min, u_max = min(u_vals), max(u_vals)
    v_min, v_max = min(v_vals), max(v_vals)
    u_center = (u_min + u_max) / 2.0
    v_center = (v_min + v_max) / 2.0

    u_range = max(1e-6, u_max - u_min)
    v_range = max(1e-6, v_max - v_min)
    aspect_ratio = u_range / v_range

    # Iris Alignment Check: check if centroid is near (0.5, 0.5) within ±0.15
    iris_aligned = (abs(u_center - 0.5) < 0.15) and (abs(v_center - 0.5) < 0.15)

    return {
        "name": uv_layer_name,
        "valid": True,
        "active": (uv_layer == mesh.uv_layers.active),
        "loop_count": len(u_vals),
        "u_min": round(u_min, 4),
        "u_max": round(u_max, 4),
        "v_min": round(v_min, 4),
        "v_max": round(v_max, 4),
        "u_center": round(u_center, 4),
        "v_center": round(v_center, 4),
        "aspect_ratio": round(aspect_ratio, 3),
        "iris_aligned_center": iris_aligned,
        "alignment_notes": (
            "UV centroid centered near (0.5, 0.5) — compatible with procedural iris origin."
            if iris_aligned
            else f"UV centroid offset at ({u_center:.2f}, {v_center:.2f}) — may require UV mapping offset."
        ),
    }


def inspect_mesh_object(obj: bpy.types.Object) -> Optional[Dict]:
    """
    Gather forensic details on a mesh object, its material slots, UV maps,
    vertex colors, vertex groups, and parent hierarchy.
    """
    if not obj or obj.type != "MESH" or not obj.data:
        return None

    mesh: bpy.types.Mesh = obj.data

    # Material slots
    mat_slots: List[Dict] = []
    shared_mats: List[str] = []
    for idx, slot in enumerate(obj.material_slots):
        mat_name = slot.material.name if slot.material else "(empty slot)"
        mat_slots.append({
            "slot_index": idx,
            "material_name": mat_name,
            "is_assigned": slot.material is not None,
        })
        if slot.material and slot.material.users > 1:
            shared_mats.append(f"{mat_name} ({slot.material.users} users)")

    # UV inspection
    uv_reports: List[Dict] = []
    for uv_l in mesh.uv_layers:
        uv_reports.append(inspect_uv_layer(mesh, uv_l.name))

    # Vertex colors / Attributes
    vcol_names = [attr.name for attr in mesh.color_attributes] if hasattr(mesh, "color_attributes") else []
    vgroup_names = [vg.name for vg in obj.vertex_groups]
    attr_names = [a.name for a in mesh.attributes] if hasattr(mesh, "attributes") else []

    # Hierarchy
    parent_name = obj.parent.name if obj.parent else "None"
    children_names = [c.name for c in obj.children]

    return {
        "object_name": obj.name,
        "mesh_name": mesh.name,
        "polygons": len(mesh.polygons),
        "vertices": len(mesh.vertices),
        "edges": len(mesh.edges),
        "material_slot_count": len(mat_slots),
        "material_slots": mat_slots,
        "shared_materials": shared_mats,
        "uv_layer_count": len(uv_reports),
        "uv_layers": uv_reports,
        "vertex_color_layers": vcol_names,
        "vertex_groups": vgroup_names,
        "custom_attributes": attr_names,
        "parent": parent_name,
        "children": children_names,
    }


def detect_colocated_meshes(objects: List[bpy.types.Object]) -> List[Dict]:
    """
    Detect co-located or overlay meshes that share near-identical spatial
    positions (e.g. Eye sphere vs Cornea vs Lashes vs Highlight overlay).
    """
    mesh_objs = [o for o in objects if o.type == "MESH"]
    overlaps: List[Dict] = []

    for i in range(len(mesh_objs)):
        for j in range(i + 1, len(mesh_objs)):
            o1, o2 = mesh_objs[i], mesh_objs[j]
            # Check centroid distance
            dist = (o1.matrix_world.translation - o2.matrix_world.translation).length
            name1, name2 = o1.name.lower(), o2.name.lower()

            # Name keywords for eye/face overlay setups
            overlay_tokens = ("eye", "cornea", "lens", "lash", "brow", "face", "shadow", "highlight")
            has_token = any(t in name1 for t in overlay_tokens) or any(t in name2 for t in overlay_tokens)

            if dist < 0.05 or (dist < 0.2 and has_token):
                overlaps.append({
                    "object_1": o1.name,
                    "object_2": o2.name,
                    "distance": round(dist, 4),
                    "relationship": (
                        "Co-located Eye/Face overlay mesh"
                        if has_token
                        else "Spatially overlapping mesh"
                    ),
                })

    return overlaps


def print_mesh_diagnostic_report(obj: bpy.types.Object, all_objects: Optional[List[bpy.types.Object]] = None) -> None:
    """Print formatted forensic mesh report to System Console."""
    data = inspect_mesh_object(obj)
    if not data:
        print(f"  [ERROR] Selected object '{obj.name}' is not a mesh.")
        return

    SEP = "=" * 60
    SEP2 = "-" * 60
    print()
    print(SEP)
    print("  KHAZAN MESH & GEOMETRY FORENSIC REPORT")
    print(SEP)
    print(f"  Object Name     : {data['object_name']}")
    print(f"  Mesh Data       : {data['mesh_name']}")
    print(f"  Parent          : {data['parent']}")
    print(f"  Children        : {', '.join(data['children']) if data['children'] else 'None'}")
    print(f"  Polygons / Verts: {data['polygons']} polys / {data['vertices']} verts")
    print(SEP2)

    print("  Material Slots:")
    for slot in data["material_slots"]:
        print(f"    Slot {slot['slot_index']}: {slot['material_name']}")
    if data["shared_materials"]:
        print("  Shared Materials across objects:")
        for sm in data["shared_materials"]:
            print(f"    • {sm}")

    print(SEP2)
    print("  UV Layers & Iris Alignment Inspection:")
    for uv in data["uv_layers"]:
        if not uv.get("valid"):
            print(f"    UV '{uv['name']}': {uv.get('reason')}")
            continue
        print(f"    UV '{uv['name']}' (active={uv['active']}):")
        print(f"      Bounds U    : [{uv['u_min']:.4f}, {uv['u_max']:.4f}]")
        print(f"      Bounds V    : [{uv['v_min']:.4f}, {uv['v_max']:.4f}]")
        print(f"      Centroid    : ({uv['u_center']:.4f}, {uv['v_center']:.4f})")
        print(f"      Aspect Ratio: {uv['aspect_ratio']:.3f}")
        print(f"      Alignment   : {uv['alignment_notes']}")

    print(SEP2)
    print(f"  Vertex Color Layers : {', '.join(data['vertex_color_layers']) if data['vertex_color_layers'] else 'None'}")
    print(f"  Vertex Groups ({len(data['vertex_groups'])}): {', '.join(data['vertex_groups'][:10]) if data['vertex_groups'] else 'None'}")
    print(f"  Custom Attributes  : {', '.join(data['custom_attributes']) if data['custom_attributes'] else 'None'}")

    if all_objects:
        overlaps = detect_colocated_meshes(all_objects)
        if overlaps:
            print(SEP2)
            print("  Co-Located Multi-Mesh Overlay Analysis:")
            for ov in overlaps:
                if ov["object_1"] == obj.name or ov["object_2"] == obj.name:
                    print(f"    • Pair: {ov['object_1']} <-> {ov['object_2']} (dist={ov['distance']:.4f}u) — {ov['relationship']}")

    print(SEP)
    import sys
    sys.stdout.flush()
