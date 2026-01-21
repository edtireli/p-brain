"""Structural connectome + graph topology metrics.

This module turns tractography streamlines and an atlas/parcellation volume into a
connectivity matrix (streamline counts per node-pair) and computes common
graph-theoretic measures used in structural connectomics.

Outputs are designed to be consumed by p-brain-web as static artifacts under
`Analysis/diffusion/`.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import nibabel as nib
import numpy as np

try:
    import networkx as nx
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "networkx is required for connectome topology metrics (pip install networkx)."
    ) from exc

try:
    from nibabel.processing import resample_from_to
except Exception:  # pragma: no cover
    resample_from_to = None

from dipy.tracking.utils import connectivity_matrix


def _write_connectome_circular_png(path: str, *, weights: np.ndarray, names: Sequence[str]) -> None:
    """Write a simple circular (chord-like) connectome visualization.

    Best-effort: if matplotlib isn't available, silently skip.
    """

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch
    except Exception:
        return

    n = int(len(names))
    if n <= 1:
        return

    W = np.asarray(weights, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1] or W.shape[0] != n:
        return

    # Select strongest edges to avoid an unreadable hairball.
    iu = np.triu_indices(n, k=1)
    vals = W[iu]
    mask = np.isfinite(vals) & (vals > 0)
    if not bool(np.any(mask)):
        return

    edges = list(zip(iu[0][mask].tolist(), iu[1][mask].tolist(), vals[mask].tolist()))
    edges.sort(key=lambda t: float(t[2]), reverse=True)
    max_edges = int(min(350, len(edges)))
    edges = edges[:max_edges]
    max_w = float(edges[0][2]) if edges else 1.0
    if max_w <= 0:
        max_w = 1.0

    # Circular layout (clockwise, starting at top).
    angles = np.linspace(np.pi / 2.0, np.pi / 2.0 - 2.0 * np.pi, n, endpoint=False)
    xs = np.cos(angles)
    ys = np.sin(angles)

    fig = plt.figure(figsize=(9.5, 9.5), dpi=160)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw chords.
    for i, j, w in edges:
        wn = float(w) / max_w
        wn = 0.0 if not np.isfinite(wn) else max(0.0, min(1.0, wn))
        alpha = 0.06 + 0.55 * wn
        lw = 0.35 + 2.0 * wn
        path = Path(
            [(float(xs[i]), float(ys[i])), (0.0, 0.0), (float(xs[j]), float(ys[j]))],
            [Path.MOVETO, Path.CURVE3, Path.CURVE3],
        )
        patch = PathPatch(
            path,
            facecolor="none",
            edgecolor=(0.12, 0.45, 0.85, alpha),
            lw=lw,
            zorder=1,
        )
        ax.add_patch(patch)

    # Nodes.
    ax.scatter(xs, ys, s=18, c=[(0.12, 0.12, 0.12, 0.85)], zorder=3)

    # Labels (skip if too many to keep legible).
    if n <= 90:
        for idx, name in enumerate(names):
            label = str(name)
            if len(label) > 22:
                label = label[:21] + "…"

            ang = float(angles[idx])
            deg = float(np.degrees(ang))

            # Place slightly outside unit circle.
            r = 1.10
            x = float(r * np.cos(ang))
            y = float(r * np.sin(ang))

            # Keep text roughly upright.
            rot = deg
            ha = "left"
            if deg < -90.0 or deg > 90.0:
                rot = deg + 180.0
                ha = "right"

            ax.text(
                x,
                y,
                label,
                fontsize=6.5,
                rotation=rot,
                rotation_mode="anchor",
                ha=ha,
                va="center",
                color=(0.15, 0.15, 0.15, 0.95),
                zorder=4,
            )

    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


@dataclass(frozen=True)
class ConnectomeOutputs:
    matrix_csv: str
    labels_csv: str
    metrics_json: str


def _ensure_diffusion_dir(analysis_directory: str) -> str:
    path = os.path.join(analysis_directory, "diffusion")
    os.makedirs(path, exist_ok=True)
    return path


def _voxel_volume_mm3(affine: np.ndarray) -> float:
    return float(abs(np.linalg.det(np.asarray(affine, dtype=float)[:3, :3])))


def _load_reference_img(nifti_directory: str, diffusion_filename: str) -> nib.Nifti1Image:
    path = diffusion_filename
    if not os.path.isabs(path):
        path = os.path.join(nifti_directory, diffusion_filename)
    img = nib.load(path)
    data_shape = img.shape
    if len(data_shape) > 3:
        # Use spatial grid only.
        img = nib.Nifti1Image(np.asarray(img.dataobj)[..., 0], img.affine, img.header)
    return img


def _load_atlas_in_diffusion_space(
    nifti_directory: str,
    reference_img: nib.Nifti1Image,
    *,
    atlas_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """Return (atlas_data_int32, atlas_labels_int32, label_lookup)."""

    # Reuse the same atlas discovery + LUT logic as diffusion metrics.
    from modules import opt08_fa

    label_lookup = opt08_fa._load_label_lookup()  # type: ignore[attr-defined]

    if atlas_path:
        atlas_img = nib.load(atlas_path)
        atlas_data = np.asarray(atlas_img.get_fdata(), dtype=np.float32)
        if atlas_data.ndim > 3:
            atlas_data = np.squeeze(atlas_data)
            if atlas_data.ndim > 3:
                raise ValueError(f"Atlas has unsupported dimensionality: {atlas_path}")
            atlas_img = nib.Nifti1Image(atlas_data, atlas_img.affine, atlas_img.header)

        ref_shape = reference_img.shape[:3]
        ref_affine = np.asarray(reference_img.affine)
        if atlas_img.shape[:3] != ref_shape or not np.allclose(atlas_img.affine, ref_affine):
            if resample_from_to is None:
                raise RuntimeError(
                    "Cannot resample atlas segmentation (nibabel.processing.resample_from_to unavailable)."
                )
            atlas_img = resample_from_to(atlas_img, (ref_shape, ref_affine), order=0)
            atlas_data = np.asarray(atlas_img.get_fdata(), dtype=np.float32)

        labels = np.unique(atlas_data)
        labels = labels[labels != 0]
        if labels.size == 0:
            raise ValueError(f"Atlas contains no non-zero labels: {atlas_path}")

        return atlas_data.astype(np.int32), labels.astype(np.int32), label_lookup

    loaded = opt08_fa._load_atlas_segmentation(nifti_directory, reference_img)  # type: ignore[attr-defined]
    if loaded is None:
        raise FileNotFoundError(
            "Atlas/parcellation not found in NIfTI directory. "
            "Expected e.g. nifti_directory/segmentation/.../aparc.DKTatlas+aseg.deep(.nii.gz)"
        )

    atlas_data, atlas_labels = loaded
    return atlas_data.astype(np.int32), atlas_labels.astype(np.int32), label_lookup


def _write_matrix_csv(path: str, labels: Sequence[int], matrix: np.ndarray) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"] + [str(int(l)) for l in labels])
        for idx, label in enumerate(labels):
            writer.writerow([str(int(label))] + [str(float(v)) for v in matrix[idx, :]])


def _write_labels_csv(path: str, labels: Sequence[int], names: Sequence[str], volumes_mm3: Sequence[float]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "name", "volume_mm3"])
        for label, name, vol in zip(labels, names, volumes_mm3):
            writer.writerow([str(int(label)), name, f"{float(vol):.6f}"])


def _largest_component_subgraph(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    if nx.is_connected(G):
        return G
    comp = max(nx.connected_components(G), key=len)
    return G.subgraph(comp).copy()


def _global_efficiency_weighted(G: nx.Graph, *, distance: str = "distance") -> float:
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0

    # Average over ordered pairs (i != j): 1/(n*(n-1)) * sum_{i!=j} 1/d_ij
    total = 0.0
    for source, lengths in nx.all_pairs_dijkstra_path_length(G, weight=distance):
        for target, d in lengths.items():
            if source == target:
                continue
            if d <= 0:
                continue
            total += 1.0 / float(d)
    return float(total / (n * (n - 1)))


def _local_efficiency_weighted(G: nx.Graph, *, distance: str = "distance") -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    values: list[float] = []
    for node in G.nodes:
        nbrs = list(G.neighbors(node))
        if len(nbrs) < 2:
            values.append(0.0)
            continue
        sub = G.subgraph(nbrs).copy()
        values.append(_global_efficiency_weighted(sub, distance=distance))
    return float(np.mean(values)) if values else 0.0


def _small_worldness_sigma(
    G_bin: nx.Graph,
    *,
    n_random: int = 20,
    seed: int = 0,
) -> dict[str, float | int | None]:
    """Compute small-worldness sigma using degree-preserving randomization.

    Sigma = (C / C_rand) / (L / L_rand)
    where C is average clustering coefficient and L is characteristic path length.

    Uses the largest connected component for L when disconnected.
    """

    if G_bin.number_of_nodes() < 3 or G_bin.number_of_edges() == 0:
        return {"sigma": None, "n_random": 0, "C": float(nx.average_clustering(G_bin)), "L": None}

    rng = np.random.default_rng(seed)

    C = float(nx.average_clustering(G_bin))
    G_core = _largest_component_subgraph(G_bin)
    try:
        L = float(nx.average_shortest_path_length(G_core))
    except Exception:
        L = None

    if L is None or L <= 0:
        return {"sigma": None, "n_random": 0, "C": C, "L": L}

    Cr: list[float] = []
    Lr: list[float] = []

    # networkx.double_edge_swap mutates the graph in-place.
    swaps = max(1, 10 * G_bin.number_of_edges())

    for _ in range(int(max(0, n_random))):
        H = G_bin.copy()
        try:
            nx.double_edge_swap(H, nswap=swaps, max_tries=swaps * 20, seed=int(rng.integers(0, 2**31 - 1)))
        except Exception:
            # Fallback: Erdos-Renyi with same density.
            p = nx.density(G_bin)
            H = nx.erdos_renyi_graph(G_bin.number_of_nodes(), p, seed=int(rng.integers(0, 2**31 - 1)))

        Cr.append(float(nx.average_clustering(H)))
        H_core = _largest_component_subgraph(H)
        try:
            Lr.append(float(nx.average_shortest_path_length(H_core)))
        except Exception:
            continue

    if not Lr or not Cr:
        return {"sigma": None, "n_random": 0, "C": C, "L": L}

    C_rand = float(np.mean(Cr))
    L_rand = float(np.mean(Lr))
    if C_rand <= 0 or L_rand <= 0:
        sigma = None
    else:
        sigma = float((C / C_rand) / (L / L_rand))

    return {
        "sigma": sigma,
        "n_random": int(len(Lr)),
        "C": C,
        "L": L,
        "C_rand": C_rand,
        "L_rand": L_rand,
    }


def _compute_topology_metrics(
    weights: np.ndarray,
    *,
    min_streamlines: int = 1,
    small_world_random: int = 20,
    seed: int = 0,
) -> dict[str, object]:
    """Compute topology metrics from a streamline-count matrix."""

    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("weights must be a square 2D matrix")

    # Binary graph for classic 'small-worldness' definitions.
    A = (weights >= float(max(1, min_streamlines))).astype(np.uint8)
    np.fill_diagonal(A, 0)

    G_bin = nx.from_numpy_array(A)

    # Weighted graph (distance = 1/weight).
    G_w = nx.Graph()
    G_w.add_nodes_from(range(weights.shape[0]))
    for i in range(weights.shape[0]):
        for j in range(i + 1, weights.shape[0]):
            w = float(weights[i, j])
            if w <= 0:
                continue
            G_w.add_edge(i, j, weight=w, distance=(1.0 / w))

    metrics: dict[str, object] = {
        "nodes": int(weights.shape[0]),
        "edges": int(G_bin.number_of_edges()),
        "components": int(nx.number_connected_components(G_bin)) if G_bin.number_of_nodes() else 0,
        "density": float(nx.density(G_bin)) if G_bin.number_of_nodes() else 0.0,
        "clustering_coefficient": float(nx.average_clustering(G_bin)) if G_bin.number_of_nodes() else 0.0,
        "transitivity": float(nx.transitivity(G_bin)) if G_bin.number_of_nodes() else 0.0,
        "assortativity_coefficient": float(nx.degree_assortativity_coefficient(G_bin))
        if G_bin.number_of_edges() > 0
        else 0.0,
    }

    # Path length / efficiency (use largest connected component).
    G_bin_core = _largest_component_subgraph(G_bin)
    if G_bin_core.number_of_nodes() > 1 and G_bin_core.number_of_edges() > 0:
        try:
            metrics["characteristic_path_length"] = float(
                nx.average_shortest_path_length(G_bin_core)
            )
        except Exception:
            metrics["characteristic_path_length"] = None
    else:
        metrics["characteristic_path_length"] = None

    metrics["global_efficiency"] = float(nx.global_efficiency(G_bin)) if G_bin.number_of_nodes() else 0.0

    # Local efficiency (binary): mean global efficiency of neighbor subgraphs.
    if G_bin.number_of_nodes():
        local_vals: list[float] = []
        for node in G_bin.nodes:
            nbrs = list(G_bin.neighbors(node))
            if len(nbrs) < 2:
                local_vals.append(0.0)
                continue
            sub = G_bin.subgraph(nbrs)
            local_vals.append(float(nx.global_efficiency(sub)))
        metrics["local_efficiency"] = float(np.mean(local_vals)) if local_vals else 0.0
    else:
        metrics["local_efficiency"] = 0.0

    # Weighted efficiencies.
    metrics["global_efficiency_weighted"] = _global_efficiency_weighted(G_w)
    metrics["local_efficiency_weighted"] = _local_efficiency_weighted(G_w)

    metrics["small_worldness"] = _small_worldness_sigma(
        G_bin, n_random=small_world_random, seed=seed
    )

    return metrics


def compute_connectome(
    nifti_directory: str,
    analysis_directory: str,
    *,
    diffusion_filename: str,
    atlas_path: Optional[str] = None,
    tractography_path: Optional[str] = None,
    streamlines: Optional[Iterable[np.ndarray]] = None,
    min_streamlines: int = 1,
    normalize_by_nodepair_volume: bool = False,
    small_world_random: int = 20,
    seed: int = 0,
    output_prefix: str = "connectome",
) -> ConnectomeOutputs:
    """Compute a structural connectome and topology metrics.

    Parameters
    ----------
    diffusion_filename:
        Diffusion NIfTI filename (relative to `nifti_directory` or absolute) used
        as the reference grid for streamlines and atlas resampling.
    atlas_path:
        Optional explicit atlas/parcellation path. If omitted, p-brain's
        FreeSurfer/FastSurfer segmentation candidates are searched under
        `nifti_directory`.
    tractography_path:
        Optional `.trk`/`.tck` path. If omitted, defaults to
        `analysis_directory/diffusion/tractography.trk`.
    streamlines:
        Optional explicit streamlines in world (mm) coordinates. When provided,
        `tractography_path` is ignored.
    normalize_by_nodepair_volume:
        When True, edge weights become streamlines / (vol_i + vol_j) in mm^3.
    """

    diffusion_dir = _ensure_diffusion_dir(analysis_directory)

    ref_img = _load_reference_img(nifti_directory, diffusion_filename)
    atlas_data, atlas_labels, label_lookup = _load_atlas_in_diffusion_space(
        nifti_directory, ref_img, atlas_path=atlas_path
    )

    labels_sorted = [int(v) for v in sorted(set(int(x) for x in atlas_labels.tolist()))]
    if not labels_sorted:
        raise ValueError("Atlas has no non-zero labels")

    names = [label_lookup.get(int(lbl), str(int(lbl))) for lbl in labels_sorted]

    voxel_vol = _voxel_volume_mm3(ref_img.affine)
    volumes_mm3 = [float(np.count_nonzero(atlas_data == int(lbl)) * voxel_vol) for lbl in labels_sorted]

    if streamlines is None:
        if tractography_path is None:
            tractography_path = os.path.join(diffusion_dir, "tractography.trk")
        if not os.path.isfile(tractography_path):
            raise FileNotFoundError(f"Tractography file not found: {tractography_path}")
        tract = nib.streamlines.load(tractography_path)
        streamlines = tract.tractogram.streamlines

    # dipy returns a matrix sized (max_label+1, max_label+1)
    M_full = connectivity_matrix(
        streamlines,
        affine=np.asarray(ref_img.affine),
        label_volume=atlas_data,
        inclusive=True,
        symmetric=True,
        return_mapping=False,
        mapping_as_streamlines=False,
    )

    # Reduce to only present atlas labels.
    idx = np.asarray(labels_sorted, dtype=int)
    weights = np.asarray(M_full[np.ix_(idx, idx)], dtype=np.float64)
    np.fill_diagonal(weights, 0.0)

    if normalize_by_nodepair_volume:
        vols = np.asarray(volumes_mm3, dtype=np.float64)
        denom = vols[:, None] + vols[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(denom > 0, weights / denom, 0.0)
        np.fill_diagonal(weights, 0.0)

    metrics = _compute_topology_metrics(
        weights,
        min_streamlines=min_streamlines,
        small_world_random=small_world_random,
        seed=seed,
    )

    metrics.update(
        {
            "min_streamlines": int(max(1, min_streamlines)),
            "normalize_by_nodepair_volume": bool(normalize_by_nodepair_volume),
            "atlas_labels": labels_sorted,
        }
    )

    base = os.path.join(diffusion_dir, output_prefix)
    matrix_csv = base + "_matrix.csv"
    labels_csv = base + "_labels.csv"
    metrics_json = base + "_metrics.json"
    circular_png = base + "_circular.png"

    _write_matrix_csv(matrix_csv, labels_sorted, weights)
    _write_labels_csv(labels_csv, labels_sorted, names, volumes_mm3)
    with open(metrics_json, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    # Best-effort visualization output for UI consumption.
    try:
        _write_connectome_circular_png(circular_png, weights=weights, names=names)
    except Exception:
        pass

    return ConnectomeOutputs(matrix_csv=matrix_csv, labels_csv=labels_csv, metrics_json=metrics_json)
