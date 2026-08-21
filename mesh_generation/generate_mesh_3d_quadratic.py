#!/usr/bin/env python3
"""
Generate an EdelweissFE-compatible 3D mesh of 20-node ("serendipity")
hexahedra (one element layer through the 2 mm thickness) for the dogbone
specimen, for use with Marmot's reduced-integration finite-strain element
C3D20RUL -- unlike the linear C3D8UL element (generate_mesh_3d.py), C3D20RUL
does not suffer severe volumetric locking at high Poisson's ratio, since it
has no reduced-integration variant... wait, C3D8UL has none; C3D20RUL is
exactly the reduced-integration option that *does* exist for the quadratic
hex family (see edelweissfe_build_staleness memory / the large-strain-study
report's locking diagnostic for why this was needed).

Builds on the exact same corner-node hex8 mesh as generate_mesh_3d.py (same
outline extraction, RDP simplification, gmsh fragment/quad-mesh, and
right-prism extrusion), then adds one midside node per hex edge (at the
geometric edge midpoint -- the elements stay straight-sided, only the
displacement field becomes quadratic) to build 20-node connectivity in the
standard Abaqus C3D20 node order:

    corners 0-3: bottom ring (zeta=-1), corners 4-7: top ring (zeta=+1)
      -- same as C3D8
    midside  8-11: bottom ring edges  (0,1) (1,2) (2,3) (3,0)
    midside 12-15: top ring edges     (4,5) (5,6) (6,7) (7,4)
    midside 16-19: vertical edges     (0,4) (1,5) (2,6) (3,7)

Requires: numpy, numpy-stl, gmsh (``pip install numpy-stl gmsh``).

Usage:
    python generate_mesh_3d_quadratic.py [--stl ../geometry/dogbone.stl] [--out ../meshes/dogbone_mesh_3d_quad.inp]
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
from generate_mesh_3d import THICKNESS, extrude_to_hex8

ELEMENT_TYPE = "C3D20RUL"
ELEMENT_PROVIDER = "marmot"

# (corner_a, corner_b) pairs for each of the 12 hex edges, in the order the
# resulting midside nodes must appear (indices 8..19) -- standard Abaqus
# C3D20 convention.
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom ring -> midside 8-11
    (4, 5), (5, 6), (6, 7), (7, 4),  # top ring    -> midside 12-15
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical    -> midside 16-19
]


def upgrade_to_hex20(nodeCoords3d, hexElements0based):
    """Add one midside node per unique hex edge (shared edges get exactly
    one shared midside node) to build 20-node connectivity."""
    midside_cache = {}  # frozenset({cornerA, cornerB}) -> new node index
    extra_coords = []
    next_idx = len(nodeCoords3d)

    hex20 = np.zeros((len(hexElements0based), 20), dtype=int)
    hex20[:, :8] = hexElements0based

    for e, conn in enumerate(hexElements0based):
        for k, (a, b) in enumerate(EDGES):
            na, nb = conn[a], conn[b]
            key = frozenset((na, nb))
            if key not in midside_cache:
                midside_cache[key] = next_idx
                extra_coords.append(0.5 * (nodeCoords3d[na] + nodeCoords3d[nb]))
                next_idx += 1
            hex20[e, 8 + k] = midside_cache[key]

    nodeCoords3d_full = np.vstack([nodeCoords3d, np.array(extra_coords)])
    return nodeCoords3d_full, hex20


def write_inp_3d_quad(out_path: Path, nodeCoords3d, hexElements):
    tol = 1e-4

    def nodes_where(cond):
        return sorted(i + 1 for i, c in enumerate(nodeCoords3d) if cond(c))

    grip_clamp_left = nodes_where(lambda c: c[0] <= -GRIP_CLAMP_X + tol)
    grip_clamp_right = nodes_where(lambda c: c[0] >= GRIP_CLAMP_X - tol)
    gauge_left_edge = nodes_where(lambda c: abs(c[0] - (-GAUGE_HALF_LENGTH)) < tol)
    gauge_right_edge = nodes_where(lambda c: abs(c[0] - GAUGE_HALF_LENGTH) < tol)
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
    print(f"wrote {out_path} ({len(nodeCoords3d)} nodes, {len(hexElements)} elements)")


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", type=Path, default=here / "../geometry/dogbone.stl")
    parser.add_argument("--out", type=Path, default=here / "../meshes/dogbone_mesh_3d_quad.inp")
    parser.add_argument("--thickness", type=float, default=THICKNESS)
    args = parser.parse_args()

    top = extract_top_boundary(args.stl)
    top_final = simplify_outline(top)
    polygon = build_full_polygon(top_final)

    nodeTags, nodeCoords, elTags, elNodeTags = mesh_polygon(polygon)
    coord_by_tag = {int(t): c for t, c in zip(nodeTags, nodeCoords)}

    fixed_conn = []
    for conn in elNodeTags:
        coords = np.array([coord_by_tag[int(n)] for n in conn])[:, :2]
        fixed_conn.append(conn[::-1] if signed_area(coords) < 0 else conn)
    fixed_conn = np.array(fixed_conn)

    tag_to_idx = {int(t): i for i, t in enumerate(nodeTags)}
    elNodeIdx = np.vectorize(tag_to_idx.get)(fixed_conn)

    nodeCoords3d, hexElements0 = extrude_to_hex8(nodeCoords, elNodeIdx, args.thickness)
    nodeCoords3d, hex20_0based = upgrade_to_hex20(nodeCoords3d, hexElements0)
    hex20 = hex20_0based + 1  # -> 1-based node labels

    write_inp_3d_quad(args.out, nodeCoords3d, hex20)


if __name__ == "__main__":
    main()
