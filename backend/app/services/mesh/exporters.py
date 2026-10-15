"""Mesh export to web- and CAD-friendly formats.

The GLB writer is deliberately hand-rolled for the *material* only through
trimesh's PBR material support, so the colour set in the UI survives the round
trip into three.js.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from ...core.errors import MeshGenerationError
from ...core.logging_ import get_logger
from ...models import MeshData

logger = get_logger("mesh.exporters")


def hex_to_rgba(color: str, alpha: float = 1.0) -> list[float]:
    value = (color or "#8ab4f8").lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    try:
        r, g, b = (int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    except (ValueError, IndexError):
        r, g, b = 0.54, 0.71, 0.97
    return [r, g, b, float(alpha)]


def build_trimesh(mesh: MeshData, *, color: str = "#8ab4f8", alpha: float = 1.0, name: str = "structure") -> trimesh.Trimesh:
    tm = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )
    tm.metadata["name"] = name
    rgba = hex_to_rgba(color, alpha)
    try:
        material = trimesh.visual.material.PBRMaterial(
            name=f"{name}_material",
            baseColorFactor=rgba,
            metallicFactor=0.05,
            roughnessFactor=0.55,
            doubleSided=False,
        )
        tm.visual = trimesh.visual.TextureVisuals(material=material)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not attach a PBR material (%s); falling back to vertex colours.", exc)
        vertex_colors = np.tile(np.array(rgba) * 255.0, (len(tm.vertices), 1)).astype(np.uint8)
        tm.visual = trimesh.visual.ColorVisuals(mesh=tm, vertex_colors=vertex_colors)
    return tm


def export_glb(mesh: MeshData, path: Path, *, color: str, name: str) -> Path:
    tm = build_trimesh(mesh, color=color, name=name)
    scene = trimesh.Scene()
    scene.add_geometry(tm, node_name=name, geom_name=name)
    data = scene.export(file_type="glb")
    if not data:
        raise MeshGenerationError("GLB export produced no data.")
    path.write_bytes(data)
    return path


def export_stl(mesh: MeshData, path: Path) -> Path:
    tm = build_trimesh(mesh, name="structure")
    data = tm.export(file_type="stl")
    path.write_bytes(data)
    return path


def export_obj(mesh: MeshData, path: Path) -> Path:
    tm = build_trimesh(mesh, name="structure")
    data = tm.export(file_type="obj")
    path.write_bytes(data if isinstance(data, bytes) else str(data).encode("utf-8"))
    return path


def export_ply(mesh: MeshData, path: Path) -> Path:
    tm = build_trimesh(mesh, name="structure")
    data = tm.export(file_type="ply")
    path.write_bytes(data)
    return path


def export_mesh(mesh: MeshData, path: Path, fmt: str, *, color: str = "#8ab4f8", name: str = "structure") -> Path:
    fmt = fmt.lower().lstrip(".")
    if fmt == "glb":
        return export_glb(mesh, path, color=color, name=name)
    if fmt == "stl":
        return export_stl(mesh, path)
    if fmt == "obj":
        return export_obj(mesh, path)
    if fmt == "ply":
        return export_ply(mesh, path)
    raise MeshGenerationError(f"Unsupported mesh format '{fmt}'. Supported: glb, stl, obj, ply.")


def estimate_glb_bytes(mesh: MeshData) -> dict[str, Any]:
    return {
        "vertices": int(mesh.n_vertices),
        "faces": int(mesh.n_faces),
        "approx_uncompressed_bytes": int(mesh.n_vertices * 12 + mesh.n_faces * 12),
    }
