"""
Khazan Material Importer – Blender 3.6 Add-on
==============================================
Rebuilds Principled BSDF materials from The First Berserker: Khazan
extracted assets (NekoPixil DeviantArt releases).

Compatible: Blender 3.6.x  (NOT 4.x)
Author   : Antigravity / Google DeepMind
License  : MIT
"""

bl_info = {
    "name": "Khazan Material Importer",
    "author": "Antigravity",
    "version": (1, 3, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Khazan",
    "description": (
        "Rebuilds materials from The First Berserker: Khazan "
        "FBX assets using exported Unreal material JSON data."
    ),
    "category": "Material",
}

# ---------------------------------------------------------------------------
# Sub-module registration – each module exposes register() / unregister()
# ---------------------------------------------------------------------------
from . import operators   # noqa: E402
from . import panels      # noqa: E402
from . import properties  # noqa: E402


def register() -> None:
    properties.register()
    operators.register()
    panels.register()


def unregister() -> None:
    panels.unregister()
    operators.unregister()
    properties.unregister()
