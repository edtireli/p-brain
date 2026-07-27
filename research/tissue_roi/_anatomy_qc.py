"""Does this parcellation actually sit on the anatomy? — a backend-agnostic check.

Segmentation fails *silently*. A registration that optimised from a transposed frame,
a NN handed a volume in the wrong orientation, a labelmap resampled through a bad
affine: all of them return a full, plausible-looking label volume. Nothing downstream
can tell. On this project a 90°-transposed Bruker affine produced labels that scored
**27/28 slices covered** while painting them into empty air; the true answer was 14/28.

So do not score overlap — ask where the structures *are*. In RAS+ world coordinates the
mammalian brain has a fixed layout, and these relations hold for mouse and human alike:

* the olfactory bulbs are **anterior** to the cerebellum,
* the cortex is **superior** to the hypothalamus,
* the cerebellum is **posterior** to the cortex.

Every one of them is orientation-sensitive by construction, so a swapped or flipped
axis breaks at least one. They are also cheap — three centroids — and they are what was
used by hand to confirm the orientation fix. Automating them means the next bad affine
is caught on the first subject instead of after a week.

Deliberately a **warning**, not an error: an unusual preparation could legitimately
violate one, and refusing to run would be worse than saying so loudly. Region names are
matched case-insensitively by substring, so this works for any provider whose groups are
named recognisably — including a future NN backend, not just the bundled atlas.
"""

from __future__ import annotations

import numpy as np

# (structure A, axis, must be, structure B, why) — axis 1 = anterior+, 2 = superior+
# in RAS world coordinates, which is what the affine delivers regardless of the voxel
# axis order the data happens to be stored in.
MAMMALIAN_LAYOUT: tuple[tuple[str, int, str, str, str], ...] = (
    ("olfactory",  1, ">", "cerebellum",   "olfactory bulbs are anterior to cerebellum"),
    ("cortex",     2, ">", "hypothalamus", "cortex is superior to hypothalamus"),
    ("cerebellum", 1, "<", "cortex",       "cerebellum is posterior to cortex"),
)
AXIS_NAME = {0: "left→right", 1: "posterior→anterior", 2: "inferior→superior"}

# Ordered most-specific first, so 'cortex' resolves to a cerebral cortex group rather
# than to a 'cerebellar cortex' one. A group, once claimed, is not offered again.
ALIASES: dict[str, tuple[str, ...]] = {
    "olfactory":    ("olfactory", "olfactory bulb", "olf"),
    "cerebellum":   ("cerebellum", "cerebellar"),
    "cortex":       ("cerebral cortex", "isocortex", "neocortex", "cortex"),
    "hypothalamus": ("hypothalamus", "hypothalamic"),
}


def _centroid_world(parc: np.ndarray, affine: np.ndarray,
                    ids: list[int]) -> np.ndarray | None:
    """Centre of mass of a set of labels, in world (RAS+) millimetres."""
    sel = np.isin(parc, ids)
    if not sel.any():
        return None
    vox = np.argwhere(sel).mean(axis=0)
    return (np.asarray(affine, dtype=float) @ np.append(vox, 1.0))[:3]


def _resolve(region_map: dict) -> dict[str, list[int]]:
    """Assign LUT groups to the structures we know about.

    Substring matching alone is ambiguous — 'cerebellar cortex' contains 'cortex' — so
    resolve every structure together, trying aliases most-specific first and removing
    each group from the pool once claimed. Deterministic regardless of dict order.
    """
    pool = {str(n): list(ids) for n, ids in region_map.items() if ids}
    found: dict[str, list[int]] = {}
    for structure, aliases in ALIASES.items():
        for alias in aliases:
            hits = [n for n in pool if alias in n.lower()]
            if hits:
                pick = min(hits, key=len)       # shortest name = least qualified
                found[structure] = pool.pop(pick)
                break
    return found


def _extent_world(parc: np.ndarray, affine: np.ndarray) -> np.ndarray | None:
    """Bounding-box size of all labelled voxels, along each world axis (mm)."""
    sel = np.argwhere(parc > 0)
    if not sel.size:
        return None
    aff = np.asarray(affine, dtype=float)
    corners = np.array([[sel[:, i].min() for i in range(3)],
                        [sel[:, i].max() for i in range(3)]])
    # Project every box corner, since the affine may rotate.
    pts = np.array([[x, y, z] for x in corners[:, 0]
                    for y in corners[:, 1] for z in corners[:, 2]])
    world = (aff[:3, :3] @ pts.T).T + aff[:3, 3]
    return world.max(axis=0) - world.min(axis=0)


def check_orientation(parc: np.ndarray, affine: np.ndarray, region_map: dict,
                      *, margin_mm: float = 0.0) -> dict:
    """Verify a parcellation against the mammalian layout.

    ``margin_mm`` requires the separation to exceed a threshold before a relation
    counts as satisfied — useful when two structures genuinely nearly coincide.

    Returns ``{"status": "ok"|"warn"|"skipped", "checks": [...], "summary": str}``.
    ``skipped`` means the region names were not recognisable, which is information
    about the LUT rather than about the anatomy.
    """
    parc = np.asarray(parc)
    checks: list[dict] = []

    # Shape, not just order. Two structures can keep their relative order under an
    # axis swap when the anatomy happens to be similarly arranged along both — the
    # A-P/S-I transposition slips past a pure ordering test that way. The brain being
    # longer front-to-back than top-to-bottom is true of every mammal and cannot
    # survive that swap, so it closes the gap.
    extent = _extent_world(parc, affine)
    if extent is not None and min(extent[1], extent[2]) > 0:
        checks.append({
            "relation": "brain is longer anterior-posterior than inferior-superior",
            "axis": "extent", "separation_mm": round(float(extent[1] - extent[2]), 2),
            "passed": bool(extent[1] > extent[2] + margin_mm),
        })

    resolved = _resolve(region_map)
    for a_name, axis, rel, b_name, why in MAMMALIAN_LAYOUT:
        a_ids, b_ids = resolved.get(a_name), resolved.get(b_name)
        if a_ids is None or b_ids is None:
            continue
        a_c, b_c = (_centroid_world(parc, affine, a_ids),
                    _centroid_world(parc, affine, b_ids))
        if a_c is None or b_c is None:
            continue
        delta = float(a_c[axis] - b_c[axis])
        ok = delta > margin_mm if rel == ">" else delta < -margin_mm
        checks.append({
            "relation": why, "axis": AXIS_NAME[axis],
            "separation_mm": round(delta, 2), "passed": bool(ok),
        })

    if not checks:
        return {"status": "skipped", "checks": [],
                "summary": "no recognisable region names to check orientation against"}
    failed = [c for c in checks if not c["passed"]]
    if not failed:
        plural = "relation" if len(checks) == 1 else "relations"
        return {"status": "ok", "checks": checks,
                "summary": f"{len(checks)} anatomical {plural} hold"}
    return {
        "status": "warn", "checks": checks,
        "summary": "; ".join(
            f"{c['relation']} — but measured {c['separation_mm']:+.1f} mm along "
            f"{c['axis']}" for c in failed),
    }
