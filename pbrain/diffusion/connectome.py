"""Structural connectome from tractography streamlines and a parcellation.

Given whole-brain streamlines (in world / RASMM space) and an integer label
volume with its voxel-to-world ``affine``, count the streamlines connecting
each pair of parcels -> an ``N x N`` symmetric structural connectivity matrix.
This is the graph the ``--diffusion tractography`` track emits when a
parcellation is available; it is built with dipy's
:func:`dipy.tracking.utils.connectivity_matrix`.

p-Brain deliberately stops at the raw streamline-count matrix (plus a
log-scaled figure and a CSV) and does not compute graph-theoretic summaries
on top -- those are left to the user's network-analysis tool of choice.
"""

from __future__ import annotations

import numpy as np


def structural_connectome(streamlines, affine, label_volume, *, symmetric: bool = True):
    """Return ``(matrix, region_labels)``.

    Parameters
    ----------
    streamlines
        Iterable of ``(P, 3)`` streamline coordinate arrays in world/RASMM space.
    affine
        ``(4, 4)`` voxel-to-world affine of ``label_volume``.
    label_volume
        3-D integer parcellation; label 0 is treated as background.
    symmetric
        Count A->B and B->A together (an undirected connectome). Default True.

    Returns
    -------
    matrix : np.ndarray
        ``(N, N)`` integer streamline-count matrix over the ``N`` non-zero
        labels present, in ascending label order.
    region_labels : list[int]
        The parcel label ids corresponding to the matrix rows/columns.
    """
    from dipy.tracking.utils import connectivity_matrix

    lv = np.rint(np.asarray(label_volume)).astype(np.int64)
    lv[lv < 0] = 0
    present = [int(x) for x in np.unique(lv) if x > 0]
    if not present:
        return np.zeros((0, 0), dtype=np.int64), []

    # connectivity_matrix uses label values as matrix indices, so remap the
    # (possibly sparse) FreeSurfer label ids to a dense 1..N range (0 = bg).
    remap = np.zeros(int(lv.max()) + 1, dtype=np.int64)
    for i, lab in enumerate(present, start=1):
        remap[lab] = i
    lv_dense = remap[lv]

    M = connectivity_matrix(
        list(streamlines), affine, lv_dense,
        return_mapping=False, mapping_as_streamlines=False, symmetric=symmetric,
    )
    M = np.asarray(M, dtype=np.int64)[1:, 1:]   # drop background row/col
    return M, present


def plot_connectome(matrix, out_png, *, names=None, title="Structural connectome"):
    """Log-scaled heat-map of a connectivity matrix -> ``out_png``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = np.asarray(matrix, dtype=float)
    disp = np.log10(M + 1.0)
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(disp, cmap="magma", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\log_{10}(1 + \mathrm{streamline\ count})$")
    ax.set_title(f"{title}  ({M.shape[0]} regions)")
    ax.set_xlabel("parcel"); ax.set_ylabel("parcel")
    if names and len(names) == M.shape[0] and M.shape[0] <= 60:
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=4)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


# ── anatomical grouping of FreeSurfer/DKT labels for the connectogram ────────
_GROUP_ORDER = ["L cortex", "L subcortical", "Brainstem", "Cerebellum",
                "R subcortical", "R cortex"]
_GROUP_COLOR = {
    "L cortex": "#1f77b4", "R cortex": "#d62728",
    "L subcortical": "#2ca02c", "R subcortical": "#ff7f0e",
    "Cerebellum": "#9467bd", "Brainstem": "#8c564b",
}
# ventricle / CSF labels are omitted from the connectogram (not GM/WM nodes)
_SKIP = {4, 5, 14, 15, 24, 43, 44, 72, 63, 31, 30, 62}


def _group_of(lab: int) -> str | None:
    if lab in _SKIP:
        return None
    if 1000 <= lab < 2000 or lab == 2:
        return "L cortex"
    if 2000 <= lab < 3000 or lab == 41:
        return "R cortex"
    if lab in (7, 8):
        return "Cerebellum"
    if lab in (46, 47):
        return "Cerebellum"
    if lab == 16:
        return "Brainstem"
    if lab in (10, 11, 12, 13, 17, 18, 26, 28):
        return "L subcortical"
    if lab in (49, 50, 51, 52, 53, 54, 58, 60):
        return "R subcortical"
    return "L subcortical" if lab % 2 == 0 else "R subcortical"


def plot_connectome_circular(matrix, labels, out_png, *, names=None,
                             top_fraction=0.18, title="Structural connectome",
                             highlight=None):
    """Circular connectogram (nodes on a ring grouped by hemisphere/structure,
    chords for the strongest connections coloured by streamline count).

    ``labels`` are the parcel ids for the matrix rows; ``names`` (optional,
    aligned to ``labels``) give the node text. Mirrors the classic
    "connectogram" layout without requiring external circular-plot packages.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch, Wedge
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    M = np.asarray(matrix, dtype=float)
    labels = list(labels)
    names = list(names) if names else [str(l) for l in labels]

    # keep only nodes with a group (drop CSF/ventricles), order round the ring
    keep = [(i, l) for i, l in enumerate(labels) if _group_of(int(l))]
    keep.sort(key=lambda il: (_GROUP_ORDER.index(_group_of(int(il[1]))), int(il[1])))
    order = [i for i, _ in keep]
    n = len(order)
    if n < 3:
        return None
    Msub = M[np.ix_(order, order)]
    node_names = [names[i] for i in order]
    node_groups = [_group_of(int(labels[i])) for i in order]

    # angular position of each node (small gaps between groups)
    gap = np.deg2rad(4.0)
    ang = np.zeros(n)
    boundaries = [k for k in range(1, n) if node_groups[k] != node_groups[k - 1]]
    total = 2 * np.pi - gap * (len(set(node_groups)))
    step = total / n
    a = np.pi / 2                    # start at top
    for k in range(n):
        if k in boundaries:
            a -= gap
        ang[k] = a
        a -= step
    xy = np.column_stack([np.cos(ang), np.sin(ang)])

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.95, 1.95); ax.set_ylim(-1.95, 1.95)

    # highlight one region: ring index of the label to emphasise (or None)
    hi = None
    if highlight is not None:
        hi = next((k for k in range(n) if int(labels[order[k]]) == int(highlight)), None)

    # edges (upper triangle). Default view: strongest chords only. Highlight
    # view: ALL of the selected region's connections coloured, the rest ghosted.
    iu, ju = np.triu_indices(n, k=1)
    w = Msub[iu, ju]
    nz = w > 0
    iu, ju, w = iu[nz], ju[nz], w[nz]
    inc = (iu == hi) | (ju == hi) if hi is not None else np.ones(w.shape, bool)
    if hi is None and w.size:
        thr = np.quantile(w, 1.0 - float(top_fraction))
        keepE = w >= thr
        iu, ju, w = iu[keepE], ju[keepE], w[keepE]
        inc = np.ones(w.shape, bool)
    lw = np.log10(w + 1.0)
    norm = Normalize(vmin=float(lw.min()) if lw.size else 0.0,
                     vmax=float(lw.max()) if lw.size else 1.0)
    cmap = plt.get_cmap("magma")
    for e in np.argsort(w):
        i, j = iu[e], ju[e]
        verts = [xy[i], [0, 0], xy[j]]                 # bend through centre
        pth = Path(verts, [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        if hi is not None and not inc[e]:
            ax.add_patch(PathPatch(pth, fill=False, lw=0.3, edgecolor="0.8",
                                   alpha=0.06, zorder=1))
        else:
            ax.add_patch(PathPatch(pth, fill=False, lw=0.4 + 1.9 * norm(lw[e]),
                                   edgecolor=cmap(norm(lw[e])),
                                   alpha=(0.9 if hi is not None else 0.25 + 0.6 * norm(lw[e])),
                                   zorder=2))
    nbr = set(np.concatenate([iu[inc], ju[inc]]).tolist()) if hi is not None else None

    # outer group arcs + group labels
    for grp in dict.fromkeys(node_groups):
        idx = [k for k in range(n) if node_groups[k] == grp]
        a0, a1 = np.rad2deg(ang[idx[-1]]) - 1.5, np.rad2deg(ang[idx[0]]) + 1.5
        ax.add_patch(Wedge((0, 0), 1.05, a0, a1, width=0.045,
                           facecolor=_GROUP_COLOR[grp], edgecolor="none", zorder=3))
        amid = np.deg2rad((a0 + a1) / 2)
        ax.text(1.72 * np.cos(amid), 1.72 * np.sin(amid), grp, ha="center",
                va="center", fontsize=13, fontweight="bold", color=_GROUP_COLOR[grp])

    # node markers (on the ring) + radial labels (short names, no L/R prefix —
    # the group already encodes hemisphere), rotated to radiate outward.
    def _short(nm: str) -> str:
        for pre in ("ctx-lh-", "ctx-rh-", "ctx-", "lh-", "rh-", "Left-", "Right-"):
            if nm.startswith(pre):
                nm = nm[len(pre):]
                break
        return nm.replace("-", " ")

    for k in range(n):
        hot = hi is None or k == hi or (nbr is not None and k in nbr)
        ms = 7.0 if k == hi else (3.6 if hot else 2.0)
        ax.plot([xy[k, 0]], [xy[k, 1]], "o", ms=ms,
                color=_GROUP_COLOR[node_groups[k]], zorder=5 if k == hi else 4,
                alpha=1.0 if hot else 0.35)
        deg = np.rad2deg(ang[k])
        ha = "left" if -90 < deg <= 90 else "right"
        rot = deg if -90 < deg <= 90 else deg + 180
        ax.text(1.09 * xy[k, 0], 1.09 * xy[k, 1], _short(node_names[k]),
                rotation=rot, rotation_mode="anchor", ha=ha, va="center",
                fontsize=(8.0 if k == hi else 6.0),
                fontweight=("bold" if k == hi else "normal"),
                color=(_GROUP_COLOR[node_groups[k]] if (hi is not None and hot) else "0.2"),
                alpha=1.0 if hot else 0.3)

    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.5)
    cb.set_label(r"$\log_{10}(1+\mathrm{streamline\ count})$", fontsize=11)
    if hi is not None:
        nconn = int((Msub[hi] > 0).sum())
        ax.set_title(f"{title}: {_short(node_names[hi])}  ({nconn} connections)",
                     fontsize=16, pad=18)
    else:
        ax.set_title(f"{title}  ({n} regions)", fontsize=16, pad=18)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


_HTML_GROUP_COLOR = {
    "L cortex": "#2f7ed8", "R cortex": "#d1495b", "L subcortical": "#2e9e5b",
    "R subcortical": "#e8802b", "Cerebellum": "#8258c4", "Brainstem": "#8c6d55",
}


def _short_name(nm: str) -> str:
    for pre in ("ctx-lh-", "ctx-rh-", "ctx-", "lh-", "rh-", "Left-", "Right-"):
        if nm.startswith(pre):
            return nm[len(pre):]
    return nm


def write_connectome_html(matrix, labels, out_html, *, names=None,
                          title="p-Brain structural connectome"):
    """Write a self-contained, interactive HTML connectome (no server needed).

    Opens in any browser: click a region on the ring to lock its connections
    (chords light up, coloured by streamline count) and list them; hover to
    preview; click empty space to reset. Same data as :func:`structural_connectome`.
    """
    import json
    from pathlib import Path

    M = np.asarray(matrix, dtype=np.int64)
    labels = list(labels)
    names = list(names) if names else [str(l) for l in labels]
    keep = [(i, l) for i, l in enumerate(labels) if _group_of(int(l))]
    keep.sort(key=lambda il: (_GROUP_ORDER.index(_group_of(int(il[1]))), int(il[1])))
    order = [i for i, _ in keep]
    nodes = [{"name": _short_name(names[i]).replace("-", " "),
              "group": _group_of(int(labels[i]))} for i in order]
    Msub = M[np.ix_(order, order)]
    edges = []
    n = len(order)
    for a in range(n):
        for b in range(a + 1, n):
            w = int(Msub[a, b])
            if w > 0:
                edges.append([a, b, w])
    edges.sort(key=lambda e: -e[2])
    groups = [g for g in _GROUP_ORDER if any(nd["group"] == g for nd in nodes)]
    data = {"nodes": nodes, "edges": edges, "groups": groups}

    html = (_CONNECTOME_HTML
            .replace("__TITLE__", str(title))
            .replace("__DATA__", json.dumps(data))
            .replace("__COLORS__", json.dumps(_HTML_GROUP_COLOR)))
    Path(out_html).write_text(html, encoding="utf-8")
    return out_html


_CONNECTOME_HTML = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
 body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
   color:#1c1c1e;margin:0;padding:18px;background:#fff;}
 h1{font-size:18px;margin:0 0 2px;} .sub{color:#8a8a8e;font-size:13px;margin-bottom:10px;}
 .legend{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:12px;color:#48484a;margin-bottom:8px;}
 .wrap{display:flex;flex-wrap:wrap;gap:14px;} .svgbox{flex:1 1 480px;min-width:300px;}
 aside{flex:1 1 220px;min-width:190px;font-size:13px;}
 #t{font-weight:700;font-size:15px;margin-bottom:2px;} #s{color:#8a8a8e;margin-bottom:8px;}
 ol{list-style:none;padding:0;margin:0;max-height:520px;overflow:auto;}
 li{display:flex;justify-content:space-between;gap:8px;padding:2px 0;border-bottom:1px solid #ececec;}
 svg{width:100%;height:auto;cursor:pointer;} text{user-select:none;}
</style></head><body>
<h1>__TITLE__</h1>
<div class="sub">Click a region to lock its connections · hover to preview · click empty space to reset</div>
<div class="legend" id="legend"></div>
<div class="wrap">
 <div class="svgbox"><svg id="svg" viewBox="0 0 640 640" aria-label="Circular connectome"></svg></div>
 <aside><div id="t">Click a region</div><div id="s"></div><ol id="list"></ol></aside>
</div>
<script>
const DATA=__DATA__, GC=__COLORS__;
const SVGNS="http://www.w3.org/2000/svg",C=320,R=208,LR=216;
const svg=document.getElementById("svg"),N=DATA.nodes.length,gap=0.045;
let ang=new Array(N),a=-Math.PI/2,nG=new Set(DATA.nodes.map(n=>n.group)).size;
let step=(2*Math.PI-gap*nG)/N;
for(let k=0;k<N;k++){if(k>0&&DATA.nodes[k].group!==DATA.nodes[k-1].group)a+=gap;ang[k]=a;a+=step;}
const pos=k=>[C+R*Math.cos(ang[k]),C+R*Math.sin(ang[k])];
const ws=DATA.edges.map(e=>Math.log10(e[2]+1)),wmin=Math.min(...ws),wmax=Math.max(...ws);
const t=w=>(Math.log10(w+1)-wmin)/(wmax-wmin||1);
const MAG=[[0,'#150e37'],[.25,'#3b0f70'],[.5,'#8c2981'],[.7,'#de4968'],[.85,'#fe9f6d'],[1,'#fcfdbf']];
function lerp(c0,c1,f){const p=s=>[parseInt(s.slice(1,3),16),parseInt(s.slice(3,5),16),parseInt(s.slice(5,7),16)];const A=p(c0),B=p(c1);return 'rgb('+A.map((v,i)=>Math.round(v+(B[i]-v)*f)).join(',')+')';}
function magma(x){x=Math.max(0,Math.min(1,x));for(let i=1;i<MAG.length;i++){if(x<=MAG[i][0]){return lerp(MAG[i-1][1],MAG[i][1],(x-MAG[i-1][0])/(MAG[i][0]-MAG[i-1][0]||1));}}return MAG[MAG.length-1][1];}
const adj=Array.from({length:N},()=>[]);
DATA.edges.forEach((e,i)=>{adj[e[0]].push([e[1],e[2]]);adj[e[1]].push([e[0],e[2]]);});
const edgeEls=DATA.edges.map((e,i)=>{const p0=pos(e[0]),p1=pos(e[1]);const P=document.createElementNS(SVGNS,"path");P.setAttribute("d","M"+p0[0]+","+p0[1]+" Q"+C+","+C+" "+p1[0]+","+p1[1]);P.setAttribute("fill","none");svg.appendChild(P);return P;});
const nodeEls=[],labelEls=[];
for(let k=0;k<N;k++){const[x,y]=pos(k);const c=document.createElementNS(SVGNS,"circle");c.setAttribute("cx",x);c.setAttribute("cy",y);c.setAttribute("r",3.4);c.setAttribute("fill",GC[DATA.nodes[k].group]);c.dataset.k=k;svg.appendChild(c);nodeEls.push(c);
 const deg=ang[k]*180/Math.PI,flip=(deg>90||deg<-90),lx=C+LR*Math.cos(ang[k]),ly=C+LR*Math.sin(ang[k]);
 const tx=document.createElementNS(SVGNS,"text");tx.setAttribute("x",lx);tx.setAttribute("y",ly);tx.setAttribute("font-size","7");tx.setAttribute("fill","#9a9a9e");tx.setAttribute("dominant-baseline","central");tx.setAttribute("text-anchor",flip?"end":"start");tx.setAttribute("transform","rotate("+(flip?deg+180:deg)+","+lx+","+ly+")");tx.textContent=DATA.nodes[k].name;tx.dataset.k=k;svg.appendChild(tx);labelEls.push(tx);}
let locked=null;
function style(sel){const on=sel!=null;
 edgeEls.forEach((P,i)=>{const e=DATA.edges[i],inc=on&&(e[0]===sel||e[1]===sel);
  if(!on){P.setAttribute("stroke","#9a9a9e");P.setAttribute("stroke-opacity",0.13);P.setAttribute("stroke-width",0.4+1.1*t(e[2]));}
  else if(inc){P.setAttribute("stroke",magma(t(e[2])));P.setAttribute("stroke-opacity",0.92);P.setAttribute("stroke-width",0.8+2.4*t(e[2]));}
  else{P.setAttribute("stroke","#c8c8cc");P.setAttribute("stroke-opacity",0.04);P.setAttribute("stroke-width",0.4);}});
 const nbr=on?new Set(adj[sel].map(x=>x[0])):null;
 nodeEls.forEach((c,k)=>{const hot=on&&(k===sel||nbr.has(k));c.setAttribute("r",on?(k===sel?6:(nbr.has(k)?4.2:2.2)):3.4);c.setAttribute("fill-opacity",!on||hot?1:0.22);});
 labelEls.forEach((tx,k)=>{const hot=on&&(k===sel||nbr.has(k));tx.setAttribute("fill",hot?GC[DATA.nodes[k].group]:(on?"#c8c8cc":"#9a9a9e"));tx.setAttribute("font-weight",on&&k===sel?"700":"400");});}
function info(sel){const T=document.getElementById("t"),S=document.getElementById("s"),L=document.getElementById("list");
 if(sel==null){T.textContent="Click a region";S.textContent="";L.innerHTML="";return;}
 const nd=DATA.nodes[sel];T.innerHTML='<span style="color:'+GC[nd.group]+'">●</span> '+nd.name;
 const cs=adj[sel].slice().sort((x,y)=>y[1]-x[1]);S.textContent=nd.group+" · "+cs.length+" connections";
 L.innerHTML=cs.map(([j,w])=>'<li><span style="color:'+GC[DATA.nodes[j].group]+'">●</span><span style="flex:1;color:#48484a;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:0 6px;">'+DATA.nodes[j].name+'</span><span style="color:#8a8a8e;">'+w+'</span></li>').join("");}
function sel(k){locked=k;style(k);info(k);}
nodeEls.concat(labelEls).forEach(el=>{const k=+el.dataset.k;el.style.cursor="pointer";
 el.addEventListener("mouseenter",()=>{if(locked==null){style(k);info(k);}});
 el.addEventListener("mouseleave",()=>{if(locked==null){style(null);info(null);}});
 el.addEventListener("click",ev=>{ev.stopPropagation();sel(locked===k?null:k);});});
svg.addEventListener("click",()=>sel(null));
document.getElementById("legend").innerHTML=DATA.groups.map(g=>'<span><span style="color:'+GC[g]+'">●</span> '+g+'</span>').join("")+'<span style="margin-left:auto;color:#9a9a9e;">chord colour = log streamline count</span>';
style(null);info(null);
</script></body></html>"""
