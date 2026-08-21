#!/usr/bin/env python3
"""
Generate an EdelweissFE-compatible 3D mesh (one element layer through the
2 mm thickness) for the dogbone specimen in ../geometry/dogbone.stl.

Reuses the exact same 2D cross-section mesh as generate_mesh.py (same outline
extraction, RDP simplification, and gmsh fragment/quad-mesh pipeline), then
extrudes it into a single layer of 8-node hexahedra by duplicating every node
at z=0 and z=THICKNESS and stacking the corresponding rings -- this needs no
extra gmsh 3D meshing step since the source quad mesh is already a clean,
correctly-oriented (CCW/positive-area) cross-section: a right-prism extrusion
of a positive-area quad is always a positive-volume hex with the standard
Abaqus/EdelweissFE C3D8 corner ordering (ring 0-3 at zeta=-1, ring 4-7 at
zeta=+1, both rings in the same order).

Requires: numpy, numpy-stl, gmsh (``pip install numpy-stl gmsh``).

Usage:
    python generate_mesh_3d.py [--stl ../geometry/dogbone.stl] [--out ../meshes/dogbone_mesh_3d.inp]
"""

import argparse
from pathlib import Path

import numpy as np

from generate_mesh import (
    GAUGE_HALF_LENGTH,
    GRIP_CLAMP_X,
    build_full_polygon,
    extract_top_boundary,
    mesh_polygon,
    signed_area,
    simplify_outline,
)

THICKNESS = 2.0  # mm, out-of-plane extrusion length (one element layer)

# 8-node hexahedron, updated-Lagrangian element -- the finite-strain hex
# formulation this repo's own Marmot testfiles pair with COMPRESSIBLENEOHOOKE
# (e.g. testfiles/marmot/RigidBodyConstraintLargeDeformations3D/test.inp).
ELEMENT_TYPE = "C3D8UL"
ELEMENT_PROVIDER = "marmot"


def extrude_to_hex8(nodeCoords2d, elNodeTags2d, thickness: float):
    """Duplicate the 2D node set at z=0 and z=thickness, and stack each
    (already CCW-fixed) quad into an 8-node hex: [bottom ring, top ring]."""
    n2d = len(nodeCoords2d)

    bottom = np.column_stack([nodeCoords2d[:, :2], np.zeros(n2d)])
    top = np.column_stack([nodeCoords2d[:, :2], np.full(n2d, thickness)])
    nodeCoords3d = np.vstack([bottom, top])  # node i -> row i (bottom), row i+n2d (top)

    hexElements = np.hstack([elNodeTags2d, elNodeTags2d + n2d])
    return nodeCoords3d, hexElements


def write_inp_3d(out_path: Path, nodeCoords3d, hexElements, grip_clamp_x: float = GRIP_CLAMP_X):
    tol = 1e-4
    n2d = len(nodeCoords3d) // 2

    def nodes_where(cond):
        return sorted(i + 1 for i, c in enumerate(nodeCoords3d) if cond(c))

    grip_clamp_left = nodes_where(lambda c: c[0] <= -grip_clamp_x + tol)
    grip_clamp_right = nodes_where(lambda c: c[0] >= grip_clamp_x - tol)
    gauge_left_edge = nodes_where(lambda c: abs(c[0] - (-GAUGE_HALF_LENGTH)) < tol)
    gauge_right_edge = nodes_where(lambda c: abs(c[0] - GAUGE_HALF_LENGTH) < tol)
    # Removes the single otherwise-unconstrained out-of-plane (z) rigid-body
    # mode; contributes zero reaction since nothing loads the model in z.
    z0_layer = nodes_where(lambda c: abs(c[2]) < tol)

    gauge_elset = [
        i + 1
        for i, conn in enumerate(hexElements)
        if -GAUGE_HALF_LENGTH - tol
        <= np.mean([nodeCoords3d[int(n) - 1][0] for n in conn[:4]])
        <= GAUGE_HALF_LENGTH + tol
    ]

    lines = ["*node"]
    lines += [f"{i + 1}, {c[0]:.8f}, {c[1]:.8f}, {c[2]:.8f}" for i, c in enumerate(nodeCoords3d)]

    lines.append(f"*element, type={ELEMENT_TYPE}, elSet=all, provider={ELEMENT_PROVIDER}")
    lines += [f"{i + 1}, " + ", ".join(str(n) for n in conn) for i, conn in enumerate(hexElements)]

    def write_set(kind, name, items):
        lines.append(f"*{kind}, {kind}={name}")
        for i in range(0, len(items), 10):
            lines.append(", ".join(str(x) for x in items[i : i + 10]))

    write_set("nSet", "gripClampLeft", grip_clamp_left)
    write_set("nSet", "gripClampRight", grip_clamp_right)
    write_set("nSet", "gaugeLeftEdge", gauge_left_edge)
    write_set("nSet", "gaugeRightEdge", gauge_right_edge)
    write_set("nSet", "z0Layer", z0_layer)
    write_set("elSet", "gaugeRegion", gauge_elset)

    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path} ({len(nodeCoords3d)} nodes, {len(hexElements)} elements, {n2d} nodes/layer)")


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", type=Path, default=here / "../geometry/dogbone.stl")
    parser.add_argument("--out", type=Path, default=here / "../meshes/dogbone_mesh_3d.inp")
    parser.add_argument("--thickness", type=float, default=THICKNESS)
    parser.add_argument("--grip-clamp-x", type=float, default=GRIP_CLAMP_X)
    args = parser.parse_args()

    top = extract_top_boundary(args.stl)
    top_final = simplify_outline(top)
    polygon = build_full_polygon(top_final)

    nodeTags, nodeCoords, elTags, elNodeTags = mesh_polygon(polygon, grip_clamp_x=args.grip_clamp_x)
    coord_by_tag = {int(t): c for t, c in zip(nodeTags, nodeCoords)}

    # Same CCW/positive-area fix as generate_mesh.py -- required so the
    # extruded hex has the correct (positive-volume) orientation.
    fixed_conn = []
    for conn in elNodeTags:
        coords = np.array([coord_by_tag[int(n)] for n in conn])[:, :2]
        fixed_conn.append(conn[::-1] if signed_area(coords) < 0 else conn)
    fixed_conn = np.array(fixed_conn)

    # Re-index nodes/elements to a dense 0-based range matching nodeCoords'
    # row order, since gmsh node tags are not guaranteed contiguous.
    tag_to_idx = {int(t): i for i, t in enumerate(nodeTags)}
    elNodeIdx = np.vectorize(tag_to_idx.get)(fixed_conn)

    nodeCoords3d, hexElements0 = extrude_to_hex8(nodeCoords, elNodeIdx, args.thickness)
    hexElements = hexElements0 + 1  # -> 1-based node labels matching *node block

    write_inp_3d(args.out, nodeCoords3d, hexElements, grip_clamp_x=args.grip_clamp_x)


if __name__ == "__main__":
    main()
