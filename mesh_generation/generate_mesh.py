#!/usr/bin/env python3
"""
Generate an EdelweissFE-compatible 2D plane-stress mesh (CPS4) for the dogbone
specimen in ../geometry/dogbone.stl.

The STL is a prismatic (constant-thickness) extrusion, so the 3D surface mesh
is reduced to its 2D cross-section outline, remeshed as an all-quad mesh with
gmsh, and written out in Abaqus-like *node/*element/*nSet/*elSet syntax as a
pure mesh file (../meshes/) -- kept separate from the simulation input files
(../simulations/) that *include it and add material/BCs/solver/output setup.

Requires: numpy, numpy-stl, gmsh (``pip install numpy-stl gmsh``).

Usage:
    python generate_mesh.py [--stl ../geometry/dogbone.stl] [--out ../meshes/dogbone_mesh.inp]
"""

import argparse
from pathlib import Path

import gmsh
import numpy as np
from stl import mesh as stl_mesh

from rdp import rdp

# Geometry anchors (mm), read off the flat/constant-width sections of the STL
# cross-section: grip corners at +/-57.5/+/-42.5, gauge corners at +/-16.5.
GRIP_HALF_LENGTH = 57.5
GRIP_TO_FILLET_X = 42.5
GAUGE_HALF_LENGTH = 16.5
GRIP_CLAMP_X = 47.5  # 10 mm Dirichlet clamp band, inset from the 57.5 mm end
MESH_SIZE = 1.0  # mm
RDP_TOLERANCE = 0.03  # mm, max deviation allowed when simplifying the STL outline


def extract_top_boundary(stl_path: Path) -> np.ndarray:
    """Return the ordered (x, y>=0) half of the STL's bottom-face boundary loop.

    The part is prismatic (2 mm thick, flat top/bottom caps), so the bottom
    cap's 2D boundary is the true in-plane outline of the specimen, and by
    construction it is symmetric about y=0.
    """
    m = stl_mesh.Mesh.from_file(str(stl_path))
    tris = m.vectors

    bottom = np.array([tri[:, :2] for tri in tris if np.all(np.abs(tri[:, 2]) < 1e-6)])

    edge_count: dict[tuple, int] = {}
    for tri in bottom:
        pts = [tuple(np.round(p, 5)) for p in tri]
        for i in range(3):
            key = tuple(sorted([pts[i], pts[(i + 1) % 3]]))
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary_edges = [k for k, v in edge_count.items() if v == 1]

    adjacency: dict[tuple, list] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    start = boundary_edges[0][0]
    loop = [start]
    prev, cur = None, start
    while True:
        nbrs = adjacency[cur]
        nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt

    loop = np.array(loop)
    top = loop[loop[:, 1] >= 0]
    return top[np.argsort(top[:, 0])]


def simplify_outline(top: np.ndarray) -> np.ndarray:
    """RDP-simplify the (dense, STL-tessellation-limited) top boundary, while
    keeping the exact geometric anchor points (grip/fillet/gauge corners) that
    later steps rely on."""
    anchors_x = [
        -GRIP_HALF_LENGTH,
        -GRIP_TO_FILLET_X,
        -GAUGE_HALF_LENGTH,
        GAUGE_HALF_LENGTH,
        GRIP_TO_FILLET_X,
        GRIP_HALF_LENGTH,
    ]

    pieces = []
    for i in range(len(anchors_x) - 1):
        a, b = anchors_x[i], anchors_x[i + 1]
        segment = top[(top[:, 0] >= a - 1e-6) & (top[:, 0] <= b + 1e-6)]
        simplified = rdp(segment, RDP_TOLERANCE)
        pieces.append(simplified[1:] if i > 0 else simplified)

    return np.vstack(pieces)


def build_full_polygon(top_final: np.ndarray) -> np.ndarray:
    """Mirror the (simplified) top half about y=0 to close the full outline."""
    bottom = top_final[::-1].copy()
    bottom[:, 1] *= -1
    return np.vstack([top_final, bottom])


def mesh_polygon(polygon: np.ndarray, grip_clamp_x: float = GRIP_CLAMP_X):
    """Build the 2D geometry in gmsh's OCC kernel, fragment in cut lines at
    the Dirichlet-clamp and gauge-section boundaries (so the mesh has exact
    conformal node lines there for BC application and cross-section
    sampling), and generate an all-quad mesh.

    grip_clamp_x need not fall within the flat 25 mm-wide grip block -- the
    cut line's y-extent is taken from the polygon's own boundary (by
    interpolating the top-half outline) at that x, not assumed to be the
    grip's nominal half-width, so this is correct even if the clamp line
    lands inside the fillet.

    Returns (nodeTags, nodeCoords[N,3], elTags, elNodeTags[N,4]).
    """

    def build_geometry():
        gmsh.initialize()
        gmsh.model.add("dogbone")
        occ = gmsh.model.occ

        n = len(polygon)
        pts = [occ.addPoint(x, y, 0.0, MESH_SIZE) for x, y in polygon]
        lines = [occ.addLine(pts[i], pts[(i + 1) % n]) for i in range(n)]
        loop = occ.addCurveLoop(lines)
        surf = occ.addPlaneSurface([loop])
        occ.synchronize()

        top_boundary = polygon[polygon[:, 1] > 0]
        order = np.argsort(top_boundary[:, 0])
        top_boundary = top_boundary[order]

        def half_width_at(x):
            return float(np.interp(x, top_boundary[:, 0], top_boundary[:, 1]))

        cut_lines = []
        for cx in (-grip_clamp_x, -GAUGE_HALF_LENGTH, GAUGE_HALF_LENGTH, grip_clamp_x):
            ytop = half_width_at(cx)
            p_top = occ.addPoint(cx, ytop, 0.0, MESH_SIZE)
            p_bot = occ.addPoint(cx, -ytop, 0.0, MESH_SIZE)
            cut_lines.append(occ.addLine(p_bot, p_top))

        occ.fragment([(2, surf)], [(1, l) for l in cut_lines])
        occ.synchronize()

        # Guarantees a 100% quad mesh (EdelweissFE's element library has no
        # triangles) regardless of the base triangulation.
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", MESH_SIZE * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", MESH_SIZE)

    build_geometry()
    try:
        # Algorithm 8 (Frontal-Delaunay for Quads) + direct recombination is
        # what produced every mesh in this repo's history; prefer it for
        # exact reproducibility of the default (10 mm clamp) mesh.
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        for dim, tag in gmsh.model.getEntities(2):
            gmsh.model.mesh.setRecombine(dim, tag)
        gmsh.model.mesh.generate(2)
    except Exception:
        # Both algorithm 8 and its direct recombination step require an even
        # boundary-edge count per fragment region and fail ("1D mesh cannot
        # be divided by 2") for some grip_clamp_x cut positions. gmsh's
        # internal 1D mesh state from the failed attempt isn't cleanly
        # undone by mesh.clear() alone, so start a fresh gmsh session and
        # rebuild the geometry before falling back to plain triangulation
        # (algorithm 6) with recombination disabled, relying purely on
        # SubdivisionAlgorithm=1 (subdivides each triangle into 3 quads,
        # with no parity constraint) to still guarantee an all-quad mesh.
        gmsh.finalize()
        build_geometry()
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)

    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    nodeCoords = nodeCoords.reshape(-1, 3)

    triTags, _ = gmsh.model.mesh.getElementsByType(2)
    assert len(triTags) == 0, f"{len(triTags)} leftover triangles in the mesh"

    elTags, elNodeTags = gmsh.model.mesh.getElementsByType(3)  # quad4
    elNodeTags = elNodeTags.reshape(-1, 4)

    gmsh.finalize()
    return nodeTags, nodeCoords, elTags, elNodeTags


def signed_area(coords: np.ndarray) -> float:
    x, y = coords[:, 0], coords[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def write_inp(out_path: Path, nodeTags, nodeCoords, elTags, elNodeTags, grip_clamp_x: float = GRIP_CLAMP_X):
    coord_by_tag = {int(t): c for t, c in zip(nodeTags, nodeCoords)}
    tol = 1e-4

    # EdelweissFE's CPS4 shape functions assume standard CCW-from-(-1,-1)
    # parametric node ordering; flip any element gmsh emitted clockwise.
    fixed_conn = []
    for conn in elNodeTags:
        coords = np.array([coord_by_tag[int(n)] for n in conn])[:, :2]
        fixed_conn.append(conn[::-1] if signed_area(coords) < 0 else conn)

    def nodes_where(cond):
        return sorted(int(t) for t, c in coord_by_tag.items() if cond(c))

    grip_clamp_left = nodes_where(lambda c: c[0] <= -grip_clamp_x + tol)
    grip_clamp_right = nodes_where(lambda c: c[0] >= grip_clamp_x - tol)
    gauge_left_edge = nodes_where(lambda c: abs(c[0] - (-GAUGE_HALF_LENGTH)) < tol)
    gauge_right_edge = nodes_where(lambda c: abs(c[0] - GAUGE_HALF_LENGTH) < tol)

    gauge_elset = [
        int(tag)
        for tag, conn in zip(elTags, fixed_conn)
        if -GAUGE_HALF_LENGTH - tol <= np.mean([coord_by_tag[int(n)][0] for n in conn]) <= GAUGE_HALF_LENGTH + tol
    ]

    lines = ["*node"]
    lines += [f"{int(t)}, {c[0]:.8f}, {c[1]:.8f}" for t, c in zip(nodeTags, nodeCoords)]

    lines.append("*element, type=CPS4, elSet=all, provider=edelweiss")
    lines += [f"{int(tag)}, {int(c[0])}, {int(c[1])}, {int(c[2])}, {int(c[3])}" for tag, c in zip(elTags, fixed_conn)]

    def write_set(kind, name, items):
        lines.append(f"*{kind}, {kind}={name}")
        for i in range(0, len(items), 10):
            lines.append(", ".join(str(x) for x in items[i : i + 10]))

    write_set("nSet", "gripClampLeft", grip_clamp_left)
    write_set("nSet", "gripClampRight", grip_clamp_right)
    write_set("nSet", "gaugeLeftEdge", gauge_left_edge)
    write_set("nSet", "gaugeRightEdge", gauge_right_edge)
    write_set("elSet", "gaugeRegion", gauge_elset)

    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path} ({len(nodeTags)} nodes, {len(elTags)} elements)")


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", type=Path, default=here / "../geometry/dogbone.stl")
    parser.add_argument("--out", type=Path, default=here / "../meshes/dogbone_mesh.inp")
    parser.add_argument(
        "--grip-clamp-x",
        type=float,
        default=GRIP_CLAMP_X,
        help="x-coordinate of the Dirichlet grip-clamp boundary (default: 47.5, i.e. a 10 mm clamp "
        "band inset from the 57.5 mm end). Need not fall within the flat grip block.",
    )
    args = parser.parse_args()

    top = extract_top_boundary(args.stl)
    top_final = simplify_outline(top)
    polygon = build_full_polygon(top_final)

    nodeTags, nodeCoords, elTags, elNodeTags = mesh_polygon(polygon, grip_clamp_x=args.grip_clamp_x)
    write_inp(args.out, nodeTags, nodeCoords, elTags, elNodeTags, grip_clamp_x=args.grip_clamp_x)


if __name__ == "__main__":
    main()
