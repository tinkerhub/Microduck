"""Build part STLs from build/parts.json without Fusion, using manifold3d mesh booleans.

Same primitives and the same (solids - cuts) & clip recipe as ``fusion_build.py``, so a part can be checked
(``check_rom.py``) and simulated before it goes through Fusion. Cylinders are 96-sided polygons rather than exact
B-rep surfaces; Fusion stays the reference export for printing. It does not rebuild the servo meshes.

Run: uv run --with manifold3d python cad/sts3215/mesh_build.py [part ...]      (no names: every part)
"""

import argparse
import json
import pathlib

import manifold3d as mf
import numpy as np
import trimesh

HERE = pathlib.Path(__file__).parent
SEGMENTS = 96


def _frame(z):
    """Rotation whose third column is the unit vector z."""
    z = np.asarray(z, float) / np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0] if abs(z[1]) < 0.9 else [1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _place(m, R, c):
    return m.transform(np.column_stack([np.asarray(R, float), np.asarray(c, float)]).tolist())


def prim(q):
    if q["t"] == "box":
        return _place(mf.Manifold.cube([2 * v for v in q["h"]], center=True), q["R"], q["c"])
    m = mf.Manifold.cylinder(2 * q["h"], q["r"], q["r"], SEGMENTS, center=True)
    return _place(m, _frame(q["a"]), q["c"])


def build(part):
    body = mf.Manifold.batch_boolean([prim(q) for q in part["solids"]], mf.OpType.Add)
    if part["cuts"]:
        body = body - mf.Manifold.batch_boolean([prim(q) for q in part["cuts"]], mf.OpType.Add)
    if part.get("clip"):
        body = body ^ mf.Manifold.batch_boolean([prim(q) for q in part["clip"]], mf.OpType.Add)
    mesh = body.to_mesh()
    tm = trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=False)
    return tm, len(body.decompose())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="parts to build (default: all)")
    ap.add_argument("--out", default=str(HERE / "stl"), help="output folder (default: stl/)")
    args = ap.parse_args()
    spec = json.loads((HERE / "build" / "parts.json").read_text())
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for part in spec["parts"]:
        if args.names and part["name"] not in args.names:
            continue
        m, bodies = build(part)
        m.export(out / (part["name"] + ".stl"))
        print(f'{part["name"]}: {m.volume:.0f} mm3, bodies {bodies}')


if __name__ == "__main__":
    main()
