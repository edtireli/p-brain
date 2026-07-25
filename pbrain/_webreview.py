"""Interactive web review for verify/manual mode.

At a decision checkpoint the pipeline pauses, serves a self-contained review page
on ``localhost`` (Python's stdlib ``http.server`` — no new dependency, works
offline on scanner data), opens the browser, and blocks until the user confirms.
The page is theme-aware (clay by default) and matches the CLI's look.

First checkpoint: **AIF selection + ROI** — inspect the candidate input-function
curves and the ROI voxels that formed them, pick the one to use, confirm.

Contract (payload → result):
    payload = {
        "subject": str, "checkpoint": "aif",
        "t_s": [float],                       # time axis (s)
        "vessels": ["rica","lica","sss"],     # candidate sources
        "stats": ["mean","median","max"],     # reduction over the ROI
        "selected": {"vessel": "sss", "stat": "max"},   # the default pick
        "curves": {"sss|max": [float], ...},  # a curve per vessel|stat
        "roi_curves": {"sss": [[float],...]},  # individual voxel curves per vessel (faint)
        "slice": {"png": "data:image/png;base64,…" | None, "n_slices": int, "idx": int,
                  "rois": {"sss": {"poly": [[x,y],…], "max": [x,y]}, …}},  # normalised 0..1
    }
    result  = {"vessel": str, "stat": str, "max": [x,y]|None, "accepted": bool}  # or None
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pbrain._banner import art as _banner_art
from pbrain._palette import palette

_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>p-Brain · verify</title>
<style>
  :root{--bg:#0b0c0e;--panel:#111317;--edge:#1e2127;--ink:#e8e8ea;--mut:#8a8f98;--dim:#5a5f66;
        --accent:#D97757;--deep:#B8543A;--lite:#EBA680;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;}
  .mono{font-family:ui-monospace,"SF Mono",Menlo,"Courier New",monospace;}
  header{display:flex;align-items:center;gap:16px;padding:26px 22px 14px;border-bottom:1px solid var(--edge);}
  header pre.brain{margin:0;font-family:Menlo,"Courier New",monospace;font-size:12px;line-height:1.02;
        color:var(--accent);letter-spacing:0;}
  header .mark{font-size:19px;font-weight:800;color:var(--accent);}
  header .cp{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--mut);margin-top:2px;}
  header .subj{margin-left:auto;font-size:12px;color:var(--mut);}
  main{display:grid;grid-template-columns:minmax(360px,1fr) 260px;gap:16px;padding:20px;max-width:1180px;margin:0 auto;}
  .card{background:var(--panel);border:1px solid var(--edge);border-radius:14px;padding:15px;
        animation:pop .5s cubic-bezier(.2,.7,.2,1) both;}
  @keyframes pop{from{opacity:0;transform:translateY(9px)}to{opacity:1;transform:none}}
  main .card:nth-child(2){animation-delay:.08s} main .card:nth-child(3){animation-delay:.16s}
  main .card:nth-child(4){animation-delay:.24s} main .card:nth-child(5){animation-delay:.32s}
  #dce,#plot,svg{transition:opacity .35s ease;}
  .card h2{margin:0 0 3px;font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--mut);font-weight:600;}
  .card .sub{font-size:11.5px;color:var(--dim);margin-bottom:10px;}
  #dce{width:100%;max-width:400px;margin:0 auto;border-radius:9px;display:block;cursor:crosshair;background:#000;}
  .scrub{display:flex;align-items:center;gap:10px;margin-top:10px;font-size:11px;color:var(--mut);}
  .scrub input{flex:1;accent-color:var(--accent);}
  .seg{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px;}
  .seg button{flex:1;min-width:60px;font-size:12px;padding:8px 6px;border-radius:8px;border:1px solid var(--edge);
        background:#15171b;color:var(--ink);cursor:pointer;text-transform:capitalize;}
  .seg button.on{background:var(--accent);border-color:var(--accent);color:#1a0d07;font-weight:700;}
  .seg button:not(.on):hover{border-color:var(--accent);}
  .lab{font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--mut);margin:14px 0 6px;}
  .peakinfo{margin-top:14px;font-size:12px;color:var(--mut);line-height:1.7;}
  .peakinfo b{color:var(--ink);font-weight:600;}
  svg{width:100%;height:auto;display:block;}
  footer{display:flex;gap:12px;align-items:center;justify-content:flex-end;padding:14px 22px;border-top:1px solid var(--edge);
          position:sticky;bottom:0;background:var(--bg);}
  button.act{font-family:inherit;font-size:13px;border-radius:9px;padding:10px 18px;cursor:pointer;border:1px solid var(--edge);
          background:#15171b;color:var(--ink);}
  button.primary{background:var(--accent);border-color:var(--accent);color:#1a0d07;font-weight:700;}
  button.primary:hover{background:var(--lite);}
  button.ghost:hover{border-color:var(--mut);}
  .hint{font-size:11.5px;color:var(--dim);margin-right:auto;}
  .done{padding:70px;text-align:center;color:var(--mut);font-size:15px;}
  .tip{position:fixed;pointer-events:none;background:#000;border:1px solid var(--accent);color:var(--ink);
       font-size:11px;padding:4px 9px;border-radius:6px;display:none;z-index:20;
       font-family:ui-monospace,Menlo,monospace;white-space:nowrap;}
</style></head><body>
<div id="tip" class="tip"></div>
<header><pre class="brain" id="brain"></pre>
  <div><div class="mark">p&#8209;Brain</div><div class="cp" id="cp">verify · AIF localisation</div></div>
  <span class="subj" id="subj"></span></header>
<main id="main">
  <div class="card"><h2>venous input localisation</h2>
    <div class="sub" id="vsub"></div>
    <canvas id="dce" width="460" height="460"></canvas>
    <div class="scrub"><span>slice</span><input type="range" id="slice" min="0" max="9" value="5">
      <span class="mono" id="slabel">5</span></div></div>
  <div class="card">
    <div class="lab" style="margin-top:0">vessel</div><div class="seg" id="vessels"></div>
    <div class="lab">AIF curve · motion correction</div><div class="seg" id="stats"></div>
    <button class="act ghost" id="drawbtn" style="width:100%;margin:10px 0 2px">✎ draw custom ROI</button>
    <div class="peakinfo" id="peakinfo"></div></div>
  <div class="card" style="grid-column:1/-1"><h2>concentration–time curve · <span id="curlbl" class="mono"></span></h2>
    <div class="sub">the input function fed to the kinetic models · faint = individual ROI voxels</div>
    <svg id="plot" viewBox="0 0 900 300" preserveAspectRatio="xMidYMid meet"></svg></div>
</main>
<footer><span class="hint">the whole vessel ROI is used; the dot marks the max voxel · click the image to move it</span>
  <button class="act ghost" onclick="send(false)">reject &amp; abort</button>
  <button class="act primary" onclick="send(true)">confirm selection</button></footer>
<script>
let D=null, V="sss", S="max", sliceIdx=5, maxPt=null, MANUAL=false, DRAW=false, poly=[];
const drawing=()=>DRAW||MANUAL;   // draw a custom ROI (a toggle in verify; always on in manual)
const VLBL={rica:"Right ICA",lica:"Left ICA",sss:"SSS",sss_shifted_to_rica:"Shifted SSS",custom:"Custom ROI"};
const SLBL={max:"Fixed voxel",adaptive:"Motion-corrected"};   // AIF curve method
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function key(){return V+"|"+S;}
// --- braille brain: decode → rotate 90° CCW (points up) → animate draw/erase ---
const _BDOT=[[0,0,1],[0,1,2],[0,2,4],[0,3,64],[1,0,8],[1,1,16],[1,2,32],[1,3,128]];
function _bdecode(rows){const R=rows.length,C=rows[0].length,H=R*4,W=C*2;
  const g=Array.from({length:H},()=>Array(W).fill(false));
  for(let cy=0;cy<R;cy++)for(let cx=0;cx<C;cx++){const b=rows[cy].codePointAt(cx)-0x2800;
    for(const d of _BDOT)if(b&d[2])g[cy*4+d[1]][cx*2+d[0]]=true;} return g;}
function _bencode(g){const H=g.length,W=g[0].length,R=Math.ceil(H/4),C=Math.ceil(W/2),rows=[];
  for(let cy=0;cy<R;cy++){let s='';for(let cx=0;cx<C;cx++){let b=0;
    for(const d of _BDOT){const y=cy*4+d[1],x=cx*2+d[0];if(y<H&&x<W&&g[y][x])b|=d[2];}s+=String.fromCharCode(0x2800+b);}rows.push(s);}return rows;}
let _bgrid=null,_border=null,_bstart=null,_btype=0,_bcyc=-1;
// reveal-order fields — each cycle draws in / out with a different one
const _BANIMS=[
  (x,y,cx,cy,mr,W,H)=>(Math.hypot(x-cx,y-cy)/mr+(Math.atan2(y-cy,x-cx)+Math.PI)/(2*Math.PI))/2, // spiral
  (x,y,cx,cy,mr,W,H)=>Math.hypot(x-cx,y-cy)/mr,                                                  // ripple
  (x,y,cx,cy,mr,W,H)=>{const p=(y%2===0)?x:(W-1-x);return (y*W+p)/(H*W);},                        // zig-zag
  (x,y,cx,cy,mr,W,H)=>Math.abs(x-cx)/(cx||1),                                                     // split
  (x,y,cx,cy,mr,W,H)=>(H-1-y)/((H-1)||1),                                                         // bottom-up
];
function _setBrainOrder(){
  const g=_bgrid,H=g.length,W=g[0].length,cx=(W-1)/2,cy=(H-1)/2,mr=Math.hypot(cx,cy)||1;
  const f=_BANIMS[_btype%_BANIMS.length]; let mn=1e9,mx=-1e9;
  const raw=g.map((row,y)=>row.map((on,x)=>{if(!on)return null;const v=f(x,y,cx,cy,mr,W,H);if(v<mn)mn=v;if(v>mx)mx=v;return v;}));
  const rng=(mx-mn)||1; _border=raw.map(row=>row.map(v=>v===null?2:(v-mn)/rng));
}
function setupBrain(){
  const src=(D.braille&&D.braille.length)?D.braille:["⢠⣴⣿⣿⣿⣷⣦⠁"];
  _bgrid=_bdecode(src); _setBrainOrder(); requestAnimationFrame(_banim);
}
function _banim(ts){
  if(_bstart===null)_bstart=ts; const t=(ts-_bstart)/1000;
  const DR=1.2,HO=0.6,ER=1.0,GA=0.4,CY=DR+HO+ER+GA,ph=t%CY,cyc=Math.floor(t/CY);
  if(cyc!==_bcyc){_bcyc=cyc; if(cyc>0){_btype=(_btype+1)%_BANIMS.length;_setBrainOrder();}}  // next type each cycle
  // erase past 0 so the LAST dot (order 0) also clears — fully empty during the gap
  let p; if(ph<DR)p=ph/DR; else if(ph<DR+HO)p=1; else if(ph<DR+HO+ER)p=1-1.06*((ph-DR-HO)/ER); else p=-0.06;
  const out=_bgrid.map((row,y)=>row.map((on,x)=>on&&_border[y][x]<=p));
  const el=document.getElementById('brain'); if(el)el.textContent=_bencode(out).join("\n");
  requestAnimationFrame(_banim);
}
async function boot(){
  D=await (await fetch('/data')).json();
  const t=D.theme||{}; for(const k in t) document.documentElement.style.setProperty('--'+k,t[k]);
  setupBrain();
  document.getElementById('subj').textContent=D.subject||'';
  const CP=(D.checkpoint||'aif');
  document.getElementById('cp').textContent=(D.mode||'verify')+' · '+(CP==='aif'?'AIF localisation':(D.title||CP));
  if(CP==='baseline'){ renderBaseline(); return; }
  if(CP==='tissue'){ renderTissue(); return; }
  if(CP!=='aif'){ renderGeneric(); return; }   // model etc.
  MANUAL=(D.mode==='manual'); DRAW=false;
  const DRAW_HINT='draw your ROI — click to add points, double-click to close, right-click to clear';
  const AIF_HINT='the whole vessel ROI is used; the dot marks the max voxel · click the image to move it';
  if(MANUAL){document.querySelector('.hint').textContent=DRAW_HINT;}
  const db=document.getElementById('drawbtn');
  if(db){ if(MANUAL){db.style.display='none';}   // manual already draws
    db.onclick=()=>{DRAW=!DRAW; db.classList.toggle('primary',DRAW); poly=[]; maxPt=null;
      db.textContent=DRAW?'✓ drawing custom ROI (click to cancel)':'✎ draw custom ROI';
      document.querySelector('.hint').textContent=DRAW?DRAW_HINT:AIF_HINT; drawDCE();}; }
  V=(D.selected||{}).vessel||"sss"; S=(D.selected||{}).stat||"max";
  sliceIdx=(D.slice||{}).idx!=null?D.slice.idx:5;
  const sl=document.getElementById('slice'); sl.max=((D.slice||{}).n_slices||10)-1; sl.value=sliceIdx;
  sl.oninput=()=>{sliceIdx=+sl.value;document.getElementById('slabel').textContent=sliceIdx;drawDCE();};
  seg('vessels',D.vessels||["rica","lica","sss"],()=>V,v=>{V=v;refresh();},x=>VLBL[x]||x);
  seg('stats',D.stats||["max"],()=>S,v=>{S=v;refresh();},x=>SLBL[x]||x);
  const cv=document.getElementById('dce');
  const at=e=>{const r=cv.getBoundingClientRect();return [(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height];};
  cv.onclick=e=>{ if(drawing()){poly.push(at(e));} else {maxPt=at(e);} drawDCE(); };
  cv.ondblclick=e=>{ e.preventDefault(); drawDCE(); };
  cv.oncontextmenu=e=>{ e.preventDefault(); if(drawing()){poly=[];drawDCE();} };
  refresh();
}
function seg(id,items,cur,set,lbl){
  const el=document.getElementById(id); el.innerHTML='';
  items.forEach(it=>{const b=document.createElement('button'); b.textContent=lbl?lbl(it):it;
    b.className=(it===cur()?'on':''); b.onclick=()=>{set(it);}; el.appendChild(b);});
}
function jumpToVesselSlice(){
  const r=(D.rois||{})[V]; if(r&&r.max_slice!=null){sliceIdx=r.max_slice;
    const sl=document.getElementById('slice'); if(sl)sl.value=sliceIdx;
    const lb=document.getElementById('slabel'); if(lb)lb.textContent=sliceIdx;}
}
function refresh(){
  seg('vessels',D.vessels,()=>V,v=>{V=v;maxPt=null;jumpToVesselSlice();refresh();},x=>VLBL[x]||x);
  seg('stats',D.stats,()=>S,v=>{S=v;refresh();},x=>SLBL[x]||x);
  document.getElementById('vsub').textContent=(VLBL[V]||V)+" · slice "+sliceIdx;
  document.getElementById('curlbl').textContent=(VLBL[V]||V)+" · "+S;
  const c=(D.curves||{})[key()]||[]; const peak=Math.max(...c.filter(isFinite));
  const roi=(D.rois||{})[V]||{};
  document.getElementById('peakinfo').innerHTML=
    `<b>${VLBL[V]||V}</b> · ${S}<br>peak <b>${isFinite(peak)?peak.toFixed(2):'–'} mM</b>`+
    `<br>ROI <b>${(roi.n||((D.roi_curves||{})[V]||[]).length)}</b> voxels`;
  drawDCE(); drawCurve();
}
function drawDCE(){
  const cv=document.getElementById('dce'), x=cv.getContext('2d'), W=cv.width, H=cv.height;
  x.fillStyle='#000'; x.fillRect(0,0,W,H);
  const roi=(D.rois||{})[V]||{}, vsl=(roi.slices||{})[sliceIdx]||{}, S=D.slice||{};
  const png=vsl.png||(S.pngs?S.pngs[sliceIdx]:null)||S.png;   // vessel-baked slice, else plain
  if(png){const im=new Image(); im.onload=()=>{x.drawImage(im,0,0,W,H);overlay(x,W,H);};im.src=png;return;}
  overlay(x,W,H);
}
function overlay(x,W,H){
  const A0=css('--accent');
  if(drawing()){
    if(poly.length){
      x.beginPath(); poly.forEach((p,i)=>{const PX=p[0]*W,PY=p[1]*H; i?x.lineTo(PX,PY):x.moveTo(PX,PY);});
      if(poly.length>=3)x.closePath();
      x.fillStyle=hexA(A0,.26); if(poly.length>=3)x.fill();
      x.strokeStyle=A0; x.lineWidth=2; x.stroke();
      poly.forEach(p=>{x.beginPath();x.arc(p[0]*W,p[1]*H,3.2,0,6.283);x.fillStyle=A0;x.fill();});
    }
    return;
  }
  const roi=(D.rois||{})[V]; if(!roi)return;
  const A=css('--accent');
  // the grown vessel region is baked into the slice image; just the 2px max dot
  if(maxPt||sliceIdx===roi.max_slice){
    const m=maxPt||roi.max||[0.5,0.8]; const MX=m[0]*W,MY=m[1]*H;
    x.beginPath(); x.arc(MX,MY,3,0,6.283); x.fillStyle=A; x.fill(); x.strokeStyle='#fff'; x.lineWidth=1; x.stroke();
  }
}
function hexA(hex,a){const h=hex.replace('#','');const r=parseInt(h.slice(0,2),16),g=parseInt(h.slice(2,4),16),b=parseInt(h.slice(4,6),16);return `rgba(${r},${g},${b},${a})`;}
function drawCurve(){
  const svg=document.getElementById('plot'), c=(D.curves||{})[key()]||[], T=D.t_s;
  const roi=(D.roi_curves||{})[V]||[];
  const tmax=Math.max(...T),tmin=Math.min(...T); let ymax=0;
  // shared y-axis across every vessel so their amplitudes compare directly
  (D.vessels||[V]).forEach(v2=>{((D.curves||{})[v2+'|'+S]||[]).forEach(val=>{if(isFinite(val)&&val>ymax)ymax=val;});});
  roi.forEach(cu=>cu.forEach(v=>{if(isFinite(v)&&v>ymax)ymax=v;})); ymax=ymax*1.08||1;
  const L=52,R=16,Tp=12,B=32,W=900,H=300;
  const X=t=>L+(t-tmin)/(tmax-tmin)*(W-L-R), Y=v=>H-B-(v/ymax)*(H-B-Tp);
  const P=cu=>{let d='';T.forEach((t,i)=>{const v=cu[i];if(!isFinite(v))return;d+=(d?'L':'M')+X(t).toFixed(1)+' '+Y(v).toFixed(1)+' ';});return d;};
  let g=`<line x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}" stroke="var(--edge)"/><line x1="${L}" y1="${Tp}" x2="${L}" y2="${H-B}" stroke="var(--edge)"/>`;
  for(let k=0;k<=4;k++){const v=ymax*k/4,y=Y(v);g+=`<line x1="${L}" y1="${y}" x2="${W-R}" y2="${y}" stroke="var(--edge)" stroke-opacity=".4"/><text x="${L-7}" y="${y+3}" text-anchor="end" font-size="10" fill="var(--dim)" class="mono">${v.toFixed(1)}</text>`;}
  g+=`<text x="${W/2}" y="${H-6}" text-anchor="middle" font-size="11" fill="var(--mut)">time (s)</text>`;
  g+=`<text x="13" y="${H/2}" text-anchor="middle" font-size="11" fill="var(--mut)" transform="rotate(-90 13 ${H/2})">[Gd] (mM)</text>`;
  roi.forEach(cu=>{g+=`<path d="${P(cu)}" fill="none" stroke="var(--dim)" stroke-width="1" stroke-opacity=".26"/>`;});
  // other vessels (same statistic) — faint, hover to label, click to select
  (D.vessels||[]).filter(v2=>v2!==V).forEach(v2=>{
    const cu=(D.curves||{})[v2+'|'+S]||[]; if(!cu.length)return;
    g+=`<path d="${P(cu)}" fill="none" stroke="var(--mut)" stroke-width="1.5" stroke-opacity=".55" style="cursor:pointer"`+
       ` onmousemove="hov(event,'${(VLBL[v2]||v2)} · ${S}')" onmouseout="unhov(event)"`+
       ` onclick="V='${v2}';maxPt=null;refresh()"/>`;
  });
  g+=`<path d="${P(c)}" fill="none" stroke="var(--accent)" stroke-width="2.8"/>`;
  // bolus peak markers (dual-bolus → two): dashed line + ring + label
  (D.boluses||[]).forEach((bo,k)=>{const bx=X(bo.t),by=Y(bo.c);
    g+=`<line x1="${bx.toFixed(1)}" y1="${Tp}" x2="${bx.toFixed(1)}" y2="${H-B}" stroke="var(--lite)" stroke-width="1" stroke-dasharray="3 3" stroke-opacity=".5"/>`;
    g+=`<circle cx="${bx.toFixed(1)}" cy="${by.toFixed(1)}" r="4" fill="none" stroke="var(--lite)" stroke-width="2"/>`;
    g+=`<text x="${bx.toFixed(1)}" y="${(Tp+11)}" text-anchor="middle" font-size="10" fill="var(--lite)">bolus ${k+1}</text>`;});
  svg.innerHTML=g;
}
function hov(e,label){const t=document.getElementById('tip');t.textContent=label;t.style.display='block';
  t.style.left=(e.clientX+13)+'px';t.style.top=(e.clientY-6)+'px';
  e.target.setAttribute('stroke-width','2.6');e.target.setAttribute('stroke-opacity','1');e.target.setAttribute('stroke','var(--lite)');}
function unhov(e){document.getElementById('tip').style.display='none';
  if(e&&e.target){e.target.setAttribute('stroke-width','1.5');e.target.setAttribute('stroke-opacity','.55');e.target.setAttribute('stroke','var(--mut)');}}
// --- tissue segmentation review: scroll slices; manual = draw exclusion polygons ---
function renderTissue(){
  const main=document.getElementById('main'); main.style.display='block';
  const MAN=(D.mode==='manual');
  const hint=document.querySelector('.hint');
  if(hint)hint.textContent=MAN?'draw over voxels to EXCLUDE from tissue — click to add points, double-click to close, right-click to clear; then confirm'
                              :'check the tissue segmentation across slices, then confirm or reject';
  window._tslice=(D.idx!=null?D.idx:Math.floor((D.n_slices||10)/2)); window._texcl=[]; window._tpoly=[];
  main.innerHTML=
    `<div class="card"><h2>tissue segmentation</h2><div class="sub" id="tsub"></div>`+
    `<canvas id="tdce" width="460" height="460"></canvas>`+
    `<div class="scrub"><span>slice</span><input type="range" id="tslice" min="0" max="${(D.n_slices||10)-1}" value="${window._tslice}">`+
    `<span class="mono" id="tslab">${window._tslice}</span></div></div>`+
    `<div class="card"><div class="lab" style="margin-top:0">segmentation</div><div class="peakinfo" id="tinfo"></div></div>`;
  const sl=document.getElementById('tslice');
  sl.oninput=()=>{window._tslice=+sl.value;document.getElementById('tslab').textContent=window._tslice;window._tpoly=[];drawTissue();};
  const cv=document.getElementById('tdce');
  const at=e=>{const r=cv.getBoundingClientRect();return [(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height];};
  cv.onclick=e=>{ if(MAN){window._tpoly.push(at(e));drawTissue();} };
  cv.ondblclick=e=>{ e.preventDefault(); if(MAN&&window._tpoly.length>=3){window._texcl.push({slice:window._tslice,polygon:window._tpoly.slice()});window._tpoly=[];drawTissue();} };
  cv.oncontextmenu=e=>{ e.preventDefault(); if(MAN){window._tpoly=[];drawTissue();} };
  cv.onmousemove=e=>{const [u,v]=at(e); const g=(D.label_grid||{})[window._tslice]; const tip=document.getElementById('tip');
    if(!g||!g.length){tip.style.display='none';return;}
    const r=Math.min(g.length-1,Math.max(0,Math.floor(v*g.length))), c=Math.min(g[0].length-1,Math.max(0,Math.floor(u*g[0].length)));
    const lab=g[r][c], nm=(D.label_names||{})[lab];
    if(lab&&nm){tip.textContent=nm; tip.style.display='block'; tip.style.left=(e.clientX+13)+'px'; tip.style.top=(e.clientY-6)+'px';}
    else tip.style.display='none';};
  cv.onmouseleave=()=>{document.getElementById('tip').style.display='none';};
  drawTissue();
}
function drawTissue(){
  const cv=document.getElementById('tdce'),x=cv.getContext('2d'),W=cv.width,H=cv.height;
  x.fillStyle='#000';x.fillRect(0,0,W,H);
  const paint=()=>{
    const R='#e5484d';   // exclusions in a clear "remove" red
    (window._texcl||[]).filter(e=>e.slice===window._tslice).forEach(e=>{
      x.beginPath(); e.polygon.forEach((p,i)=>{const PX=p[0]*W,PY=p[1]*H;i?x.lineTo(PX,PY):x.moveTo(PX,PY);}); x.closePath();
      x.fillStyle=hexA(R,.34);x.fill();x.strokeStyle=R;x.lineWidth=2;x.stroke();});
    const p=window._tpoly||[];
    if(p.length){x.beginPath();p.forEach((q,i)=>{const PX=q[0]*W,PY=q[1]*H;i?x.lineTo(PX,PY):x.moveTo(PX,PY);});
      if(p.length>=3)x.closePath(); x.fillStyle=hexA(css('--accent'),.22); if(p.length>=3)x.fill();
      x.strokeStyle=css('--accent');x.lineWidth=2;x.stroke();
      p.forEach(q=>{x.beginPath();x.arc(q[0]*W,q[1]*H,3,0,6.283);x.fillStyle=css('--accent');x.fill();});}
    const inf=document.getElementById('tinfo');
    if(inf)inf.innerHTML=`tissue <b>${(D.n_voxels||0).toLocaleString()}</b> voxels · <b>${D.n_regions||0}</b> regions`+
      `<br><span style="color:var(--dim)">hover a region for its name</span>`+
      (window._texcl.length?`<br><b>${window._texcl.length}</b> exclusion region(s) drawn`:'');
    const su=document.getElementById('tsub'); if(su)su.textContent='slice '+window._tslice+' of '+((D.n_slices||1)-1);
  };
  const png=(D.slices||{})[window._tslice];
  if(png){const im=new Image();im.onload=()=>{x.drawImage(im,0,0,W,H);paint();};im.src=png;} else paint();
}
// --- generic plug-in review: a matplotlib figure and/or declarative panels ---
function renderGeneric(){
  const main=document.getElementById('main'); main.style.display='block';
  const editable=(D.mode==='manual')&&(D.controls||[]).length;
  const hint=document.querySelector('.hint');
  if(hint)hint.textContent=editable?'adjust the parameters, then confirm to re-fit — or reject to abort'
                                    :'review the validation below, then confirm or reject';
  let h='';
  if(D.figure) h+=`<div class="card"><img src="${D.figure}" style="width:100%;border-radius:9px;display:block"></div>`;
  (D.panels||[]).forEach(p=>{h+=`<div class="card">`+renderPanel(p)+`</div>`;});
  if(editable) h+=`<div class="card">`+renderControls(D.controls)+`</div>`;
  main.innerHTML=h||'<div class="done" style="grid-column:1/-1">nothing to review</div>';
}
function renderControls(ctrls){
  const IST='background:#0f1113;color:var(--ink);border:1px solid var(--edge);border-radius:7px;padding:6px 10px;font-size:13px;min-width:130px';
  let h='<h2>parameters · re-fit</h2><div style="display:flex;flex-direction:column;gap:11px">';
  (ctrls||[]).forEach((c,i)=>{const id='ctl_'+i;
    const inp=c.options
      ? `<select id="${id}" data-key="${c.key}" style="${IST}">`+c.options.map(o=>`<option${o==c.value?' selected':''}>${o}</option>`).join('')+`</select>`
      : `<input id="${id}" data-key="${c.key}" type="number" value="${c.value}"${c.min!=null?` min="${c.min}"`:''}${c.max!=null?` max="${c.max}"`:''}${c.step!=null?` step="${c.step}"`:''} style="${IST}">`;
    h+=`<label style="display:flex;align-items:center;justify-content:space-between;gap:14px;font-size:13px;color:var(--mut)"><span>${c.label||c.key}</span>${inp}</label>`;});
  return h+'</div>';
}
function renderPanel(p){
  let h=`<h2>${p.title||p.kind||''}</h2>`;
  if(p.kind==='values'){
    h+='<table style="width:100%;font-size:13px;border-collapse:collapse">';
    const it=p.items||{}; for(const k in it) h+=`<tr><td style="color:var(--mut);padding:5px 0">${k}</td><td style="text-align:right;color:var(--ink);font-weight:600">${it[k]}</td></tr>`;
    return h+'</table>';
  }
  if(p.kind==='image') return h+`<img src="${p.png}" style="width:100%;border-radius:9px;display:block">`;
  return h+plotSVG(p);   // scatter | curve
}
function plotSVG(p){
  const W=900,H=320,L=58,R=18,Tp=14,B=42; let xs=[],ys=[];
  const push=s=>{(s.x||[]).forEach(v=>{if(isFinite(v))xs.push(v)});(s.y||[]).forEach(v=>{if(isFinite(v))ys.push(v)});};
  (p.series||[]).forEach(push); (p.lines||[]).forEach(push);
  if(!xs.length && !p.xlim) return '<div style="color:var(--dim);padding:18px">no data</div>';
  // explicit xlim/ylim win over auto-fit — lets a plot keep full x context while
  // scaling y to the fitted region (out-of-range points are clipped, below).
  const xmin=p.xlim?p.xlim[0]:Math.min(...xs), xmax=p.xlim?p.xlim[1]:Math.max(...xs);
  const ymin=p.ylim?p.ylim[0]:Math.min(...ys,0), ymax=p.ylim?p.ylim[1]:Math.max(...ys);
  const xr=(xmax-xmin)||1,yr=(ymax-ymin)||1;
  const X=v=>L+(v-xmin)/xr*(W-L-R),Y=v=>H-B-(v-ymin)/yr*(H-B-Tp);
  const cid='pc'+(window.__pcid=(window.__pcid||0)+1);
  let g=`<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto"><defs><clipPath id="${cid}"><rect x="${L}" y="${Tp}" width="${(W-L-R).toFixed(1)}" height="${(H-B-Tp).toFixed(1)}"/></clipPath></defs>`;
  g+=`<line x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}" stroke="var(--edge)"/><line x1="${L}" y1="${Tp}" x2="${L}" y2="${H-B}" stroke="var(--edge)"/>`;
  for(let k=0;k<=4;k++){const v=ymin+yr*k/4,y=Y(v);g+=`<line x1="${L}" y1="${y.toFixed(1)}" x2="${W-R}" y2="${y.toFixed(1)}" stroke="var(--edge)" stroke-opacity=".4"/><text x="${L-7}" y="${(y+3).toFixed(1)}" text-anchor="end" font-size="10" fill="var(--dim)">${v.toFixed(2)}</text>`;}
  for(let k=0;k<=4;k++){const v=xmin+xr*k/4,x=X(v);g+=`<text x="${x.toFixed(1)}" y="${H-B+16}" text-anchor="middle" font-size="10" fill="var(--dim)">${v.toFixed(1)}</text>`;}
  if(p.xlabel)g+=`<text x="${W/2}" y="${H-5}" text-anchor="middle" font-size="11" fill="var(--mut)">${p.xlabel}</text>`;
  if(p.ylabel)g+=`<text x="14" y="${H/2}" text-anchor="middle" font-size="11" fill="var(--mut)" transform="rotate(-90 14 ${H/2})">${p.ylabel}</text>`;
  const COLS=[css('--accent'),css('--lite'),'#6aa1e0','#7bd389'];
  const scol=(s,si)=>s.role==='muted'?css('--dim'):(s.color?css(s.color):COLS[si%COLS.length]);
  const nS=(p.series||[]).length;
  let gd='';   // data layer — clipped to the axes box so out-of-range points vanish
  (p.series||[]).forEach((s,si)=>{const A=s.x||[],Bb=s.y||[],col=scol(s,si),mu=s.role==='muted';
    if(p.kind==='scatter'){const r=mu?1.7:2.6,op=mu?.5:.78;for(let i=0;i<A.length;i++)if(isFinite(A[i])&&isFinite(Bb[i]))gd+=`<circle cx="${X(A[i]).toFixed(1)}" cy="${Y(Bb[i]).toFixed(1)}" r="${r}" fill="${col}" fill-opacity="${op}"/>`;}
    else{let d='';for(let i=0;i<A.length;i++)if(isFinite(A[i])&&isFinite(Bb[i]))d+=(d?'L':'M')+X(A[i]).toFixed(1)+' '+Y(Bb[i]).toFixed(1)+' ';gd+=`<path d="${d}" fill="none" stroke="${col}" stroke-width="2"/>`;}});
  (p.lines||[]).forEach(s=>{const A=s.x||[],Bb=s.y||[];let d='';for(let i=0;i<A.length;i++)if(isFinite(A[i])&&isFinite(Bb[i]))d+=(d?'L':'M')+X(A[i]).toFixed(1)+' '+Y(Bb[i]).toFixed(1)+' ';gd+=`<path d="${d}" fill="none" stroke="var(--lite)" stroke-width="2" stroke-dasharray="5 4"/>`;});
  g+=`<g clip-path="url(#${cid})">${gd}</g>`;
  // legends sit outside the clip (top-right), stacked: series then fit line(s)
  (p.series||[]).forEach((s,si)=>{if(s.label)g+=`<text x="${W-R}" y="${(Tp+12+si*14).toFixed(0)}" text-anchor="end" font-size="11" fill="${scol(s,si)}">${s.label}</text>`;});
  (p.lines||[]).forEach((s,li)=>{if(s.label)g+=`<text x="${W-R}" y="${(Tp+12+(nS+li)*14).toFixed(0)}" text-anchor="end" font-size="11" fill="var(--lite)">${s.label}</text>`;});
  return g+'</svg>';
}
// --- baseline checkpoint: zoomed curve + draggable baseline point ---
function blGeom(){const t=D.t||[];const xl=D.xlim||[t[0],t[t.length-1]];return {W:900,H:340,L:58,R:18,Tp:16,B:42,xmin:xl[0],xmax:xl[1]};}
function blNearest(px){const t=D.t||[],G=blGeom();const val=G.xmin+(px-G.L)/(G.W-G.L-G.R)*(G.xmax-G.xmin);
  let best=0,bd=1e18;for(let i=0;i<t.length;i++){const d=Math.abs(t[i]-val);if(d<bd){bd=d;best=i;}}return best;}
function drawBaseline(){
  const svg=document.getElementById('blsvg');if(!svg)return;
  const t=D.t||[],c=D.curve||[],G=blGeom(),W=G.W,H=G.H,L=G.L,R=G.R,Tp=G.Tp,B=G.B,xmin=G.xmin,xmax=G.xmax;
  let ymin=Infinity,ymax=-Infinity;
  for(let i=0;i<t.length;i++){if(t[i]>=xmin&&t[i]<=xmax&&isFinite(c[i])){if(c[i]<ymin)ymin=c[i];if(c[i]>ymax)ymax=c[i];}}
  if(!isFinite(ymin)){const f=c.filter(isFinite);ymin=Math.min(...f);ymax=Math.max(...f);}
  const pad=(ymax-ymin)*0.1||1;ymin-=pad;ymax+=pad;
  const xr=(xmax-xmin)||1,yr=(ymax-ymin)||1;const X=v=>L+(v-xmin)/xr*(W-L-R),Y=v=>H-B-(v-ymin)/yr*(H-B-Tp);
  const A=css('--accent'),Dp=css('--deep')||A;
  let g=`<clipPath id="blc"><rect x="${L}" y="${Tp}" width="${W-L-R}" height="${H-B-Tp}"/></clipPath>`;
  g+=`<line x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}" stroke="var(--edge)"/><line x1="${L}" y1="${Tp}" x2="${L}" y2="${H-B}" stroke="var(--edge)"/>`;
  for(let k=0;k<=4;k++){const v=xmin+xr*k/4,x=X(v);g+=`<text x="${x.toFixed(1)}" y="${H-B+16}" text-anchor="middle" font-size="10" fill="var(--dim)">${v.toFixed(0)}</text>`;}
  g+=`<text x="${W/2}" y="${H-4}" text-anchor="middle" font-size="11" fill="var(--mut)">time (s)</text><text x="14" y="${H/2}" text-anchor="middle" font-size="11" fill="var(--mut)" transform="rotate(-90 14 ${H/2})">signal</text>`;
  let d='';for(let i=0;i<t.length;i++)if(isFinite(c[i]))d+=(d?'L':'M')+X(t[i]).toFixed(1)+' '+Y(c[i]).toFixed(1)+' ';
  g+=`<path clip-path="url(#blc)" d="${d}" fill="none" stroke="${A}" stroke-width="2"/>`;
  if(D.onset_frame!=null){const ox=X(t[D.onset_frame]);g+=`<line clip-path="url(#blc)" x1="${ox.toFixed(1)}" y1="${Tp}" x2="${ox.toFixed(1)}" y2="${H-B}" stroke="var(--mut)" stroke-dasharray="2 4" stroke-opacity=".5"/><text x="${ox.toFixed(1)}" y="${(Tp+10).toFixed(1)}" text-anchor="middle" font-size="9" fill="var(--mut)">onset</text>`;}
  const bf=window._blf,bx=X(t[bf]),by=Y(c[bf]);
  g+=`<line x1="${bx.toFixed(1)}" y1="${Tp}" x2="${bx.toFixed(1)}" y2="${(H-B).toFixed(1)}" stroke="${Dp}" stroke-dasharray="3 3" stroke-opacity=".6"/>`;
  g+=`<circle cx="${bx.toFixed(1)}" cy="${by.toFixed(1)}" r="7" fill="${A}" stroke="#fff" stroke-width="2"/>`;
  g+=`<text x="${bx.toFixed(1)}" y="${(Tp-2).toFixed(1)}" text-anchor="middle" font-size="10" fill="${A}">baseline</text>`;
  svg.innerHTML=g;
  const info=document.getElementById('blinfo');if(info)info.innerHTML=`baseline ends at frame <b>${bf}</b> · t=<b>${(t[bf]||0).toFixed(1)} s</b> · drag to adjust`;
}
function renderBaseline(){
  const main=document.getElementById('main');main.style.display='block';
  const hint=document.querySelector('.hint');if(hint)hint.textContent='drag the circle to set where the baseline ends (just before the first peak), then confirm';
  main.innerHTML='<div class="card"><h2>baseline point · '+(D.method||'auto')+'</h2>'+
    '<div class="sub">the pre-contrast baseline ends at the circle — drag it to adjust, or confirm the detected point</div>'+
    '<svg id="blsvg" viewBox="0 0 900 340" style="width:100%;height:auto;cursor:ew-resize"></svg>'+
    '<div class="peakinfo" id="blinfo"></div></div>';
  window._blf=D.baseline_frame|0;drawBaseline();
  const svg=document.getElementById('blsvg');let drag=false;
  const upd=cx=>{const r=svg.getBoundingClientRect();window._blf=blNearest((cx-r.left)/r.width*900);drawBaseline();};
  svg.onmousedown=e=>{drag=true;upd(e.clientX);};
  window.addEventListener('mousemove',e=>{if(drag)upd(e.clientX);});
  window.addEventListener('mouseup',()=>{drag=false;});
}
async function send(accepted){
  const CP=(D.checkpoint||'aif');
  let body={accepted};
  if(CP==='aif'){body={vessel:V,stat:S,max:maxPt,slice:sliceIdx,accepted};
    if(drawing()&&poly.length>=3){body.polygon=poly; body.vessel='custom';}}
  else if(CP==='baseline'){body={baseline_frame:window._blf,accepted};}
  else if(CP==='tissue'){body={accepted, exclusions:(window._texcl||[])};}
  else{const cs=document.querySelectorAll('#main [data-key]');   // model: manual param edits → re-fit
    if(accepted&&cs.length){const p={}; cs.forEach(el=>{p[el.getAttribute('data-key')]=el.value;}); body.params=p;}}
  await fetch('/confirm',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)});
  const what = CP==='aif' ? (body.vessel==='custom'?'custom ROI':(VLBL[V]||V)+' · '+S)+' sent'
             : CP==='baseline' ? 'baseline point set'
             : CP==='tissue' ? 'segmentation confirmed'+(( (window._texcl||[]).length)?' ('+window._texcl.length+' exclusion'+(window._texcl.length>1?'s':'')+')':'')
             : (D.title||'review')+' confirmed';
  document.getElementById('main').innerHTML=
    `<div class="done" style="grid-column:1/-1">${accepted?'✓ '+what+' — you can close this tab.':'✕ aborted — you can close this tab.'}</div>`;
  document.querySelector('footer').style.display='none';
}
boot();
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):        # keep the terminal cockpit clean
        pass

    def do_GET(self):
        if self.path.startswith("/data"):
            body = json.dumps(self.server.payload).encode()
            self._send(200, "application/json", body)
        else:
            self._send(200, "text/html; charset=utf-8", _PAGE.encode())

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            data = {}
        self.server.result = data
        self._send(200, "application/json", b'{"ok":true}')
        self.server.done.set()

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def review(payload: dict, *, port: int = 8731, open_browser: bool = True,
           timeout: float | None = None) -> dict | None:
    """Serve the review page, block until the user confirms/aborts, return the
    result dict (or None on timeout). Injects the active theme colours."""
    base, deep, lite = palette()
    payload = dict(payload)
    payload["theme"] = {"accent": base, "deep": deep, "lite": lite}
    payload.setdefault("braille", list(_banner_art()))

    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    except OSError:
        # the preferred port is busy (a lingering review, or another app) — let
        # the OS assign any free port; the browser is told the real one below.
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = httpd.server_address[1]
    httpd.payload = payload
    httpd.result = None
    httpd.done = threading.Event()
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{port}/"
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    try:
        from pbrain import _clock
        with _clock.review_pause():   # freeze the run/stage timers while we wait
            httpd.done.wait(timeout)
    finally:
        httpd.shutdown()
        httpd.server_close()   # release the socket so the port frees immediately
    return httpd.result
