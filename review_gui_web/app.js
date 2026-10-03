"use strict";

const $ = id => document.getElementById(id);
const canvas = $("editor"), ctx = canvas.getContext("2d");
const overlay = document.createElement("canvas"), overlayCtx = overlay.getContext("2d");
const colors = ["#fc927d", "#76baff", "#d7b7ff", "#ffe180"];
const statusNames = {pending: "검수 대기", approved: "승인 완료", rejected: "제외"};
const S = {
  config: null, frames: [], filtered: [], listPage: 0, pageSize: 80, frame: null,
  selectedFrames: new Set(), selectionAnchor: null,
  mask: null, boxes: [], selected: -1, rgb: null, nir: null,
  tool: "select", maskValue: 1, brushSize: 24, scale: 1, ox: 0, oy: 0,
  busy: false, drag: null, polygon: [], cursor: null, space: false,
  undo: [], redo: [], serial: 0, stateId: 0, savedId: 0, note: "", raf: 0,
  samPoints: [], samLabel: 1, samProposal: null, samRunning: false,
};
let toastTimer;
function toast(message, error = false) {
  $("toast").textContent = message;
  $("toast").className = "visible" + (error ? " error" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("toast").className = "", error ? 12000 : 3500);
}
async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
function dirty() { return S.stateId !== S.savedId; }
function snapshot() { return {mask: S.mask.slice(), boxes: structuredClone(S.boxes), selected: S.selected, note: S.note, stateId: S.stateId}; }
function beginEdit() {
  S.undo.push(snapshot());
  if (S.undo.length > 30) S.undo.shift();
  S.redo = [];
  S.stateId = ++S.serial;
}
function restore(snapshot) {
  Object.assign(S, snapshot);
  $("note").value = S.note;
  updateOverlay(); updateBoxes(); updateState(); draw();
}
function undo() {
  if (S.busy || !S.frame || S.drag) return;
  if (S.samProposal) { discardSam(); return; }
  if (S.polygon.length) { S.polygon.pop(); draw(); return; }
  if (!S.undo.length) return;
  S.redo.push(snapshot()); restore(S.undo.pop());
}
function redo() {
  if (S.busy || !S.frame || S.drag || !S.redo.length) return;
  if (S.samProposal) { discardSam(); return; }
  S.undo.push(snapshot()); restore(S.redo.pop());
}
function updateState() {
  $("dirtyBadge").textContent = S.samProposal ? "SAM 미리보기 · 미적용" : dirty() ? "미저장 변경" : "저장됨";
  $("dirtyBadge").className = "badge" + (dirty() || S.samProposal ? " dirty" : "");
  $("statusBadge").textContent = S.frame ? statusNames[S.frame.status] : "—";
  $("statusBadge").className = "badge " + (S.frame?.status || "");
  $("undo").disabled = S.busy || !S.undo.length;
  $("redo").disabled = S.busy || !S.redo.length;
  for (const id of ["saveDraft", "approve", "approveNext", "reject", "previous", "next"]) $(id).disabled = S.busy || !S.frame;
  $("loading").classList.toggle("hidden", !S.busy);
  $("loading").textContent = S.samRunning ? "SAM 분석 중… 첫 실행은 모델을 불러옵니다." : "불러오는 중…";
  $("note").disabled = S.busy || !S.frame;
  $("finishPolygon").disabled = S.busy || S.polygon.length < 3;
  $("cancelPolygon").disabled = !S.polygon.length;
  const position = S.filtered.findIndex(f => f.id === S.frame?.id);
  $("framePosition").textContent = position < 0 ? "현재 필터 밖" : `${position + 1} / ${S.filtered.length}`;
  updateSelection();
  updateSam();
}
function updateSelection() {
  const editingSelected = S.selectedFrames.has(S.frame?.id) && (dirty() || S.polygon.length || S.drag || S.samProposal);
  $("selectionCount").textContent = `${S.selectedFrames.size.toLocaleString()}장 선택`;
  $("clearSelection").disabled = S.busy || !S.selectedFrames.size;
  $("rejectSelected").disabled = S.busy || !S.selectedFrames.size || Boolean(editingSelected);
  $("selectionHint").textContent = editingSelected ? (S.samProposal ? "SAM 미리보기를 적용하거나 취소하세요." : "현재 편집을 완료하고 저장하면 일괄 제외할 수 있습니다.") : "";
}
function clearFrameSelection() { S.selectedFrames.clear(); S.selectionAnchor = null; }
async function selectFrame(id, event) {
  if (S.busy || S.drag) return;
  if (event.shiftKey) {
    const ids = S.filtered.map(f => f.id);
    const start = ids.indexOf(S.selectionAnchor), end = ids.indexOf(id);
    if (end < 0) return;
    const anchor = start < 0 ? end : start;
    const range = ids.slice(Math.min(anchor, end), Math.max(anchor, end) + 1);
    S.selectedFrames = new Set(event.ctrlKey || event.metaKey ? [...S.selectedFrames, ...range] : range);
    S.selectionAnchor = ids[anchor];
  } else if (event.ctrlKey || event.metaKey) {
    if (S.selectedFrames.has(id)) S.selectedFrames.delete(id); else S.selectedFrames.add(id);
    S.selectionAnchor = id;
  } else if (id === S.frame?.id) {
    S.selectedFrames = new Set([id]); S.selectionAnchor = id;
  } else {
    await openFrame(id); return;
  }
  renderFrames(); updateSelection();
}
async function rejectSelected() {
  if (S.busy || !S.selectedFrames.size || S.drag) return;
  const activeSelected = S.selectedFrames.has(S.frame?.id);
  if (activeSelected && (dirty() || S.polygon.length || S.samProposal)) return toast("현재 편집과 미리보기를 완료하고 저장한 뒤 일괄 제외하세요.", true);
  const entries = S.frames.filter(f => S.selectedFrames.has(f.id)).map(f => ({
    id: f.id, annotation_revision: f.annotation_revision,
  }));
  S.busy = true; updateState();
  try {
    const result = await api("/api/frames/reject", {
      method: "POST", headers: {"Content-Type": "application/json", "X-Review-Token": S.config.token},
      body: JSON.stringify({frames: entries}),
    });
    const changed = new Map(result.frames.map(f => [f.id, f]));
    S.frames.forEach(f => { if (changed.has(f.id)) Object.assign(f, changed.get(f.id)); });
    clearFrameSelection(); filterFrames();
    if (activeSelected) {
      // Reload the active frame's revision as well as its metadata. The editor
      // was clean; never keep old boxes with a newer revision after another save.
      const activeId = S.frame.id;
      Object.assign(S.frame, changed.get(activeId));
      S.busy = false;
      await openFrame(activeId, true);
      clearFrameSelection(); renderFrames();
    }
    toast(`${result.count.toLocaleString()}개 프레임을 학습에서 제외했습니다.`);
  } catch (error) { toast(`일괄 제외 오류: ${error.message}`, true); }
  finally { S.busy = false; updateState(); }
}
function bytesToBase64(bytes) {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 32768) binary += String.fromCharCode(...bytes.subarray(i, i + 32768));
  return btoa(binary);
}
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("이미지를 읽지 못했습니다."));
    image.src = url;
  });
}
function canLeave() {
  return (!dirty() && !S.polygon.length && !S.samProposal) || confirm("저장하지 않은 변경 또는 미리보기가 있습니다. 변경을 버리고 이동할까요?");
}
async function openFrame(id, skipConfirm = false) {
  if (S.busy || (!skipConfirm && !canLeave())) return;
  S.busy = true; updateState();
  try {
    const [frame, rgb] = await Promise.all([api(`/api/frame/${id}`), loadImage(`/api/frame/${id}/rgb`)]);
    const mask = Uint8Array.from(atob(frame.mask), c => c.charCodeAt(0));
    if (mask.length !== frame.width * frame.height) throw new Error("마스크 크기가 잘못되었습니다.");
    S.frame = frame; S.rgb = rgb; S.nir = null; S.mask = mask; S.boxes = frame.boxes;
    S.selectedFrames = new Set([id]); S.selectionAnchor = id;
    S.selected = -1; S.undo = []; S.redo = []; S.serial = 0; S.stateId = 0; S.savedId = 0;
    S.note = frame.note; S.polygon = []; S.drag = null;
    S.samPoints = []; S.samProposal = null; $("samStatus").textContent = "";
    $("note").value = S.note; $("showNir").checked = false; $("showNir").disabled = !frame.nir_available;
    $("frameName").textContent = frame.name;
    $("frameDetail").textContent = `${frame.date} / ${frame.split} / ${frame.sequence} · ${frame.width} × ${frame.height}`;
    canvas.dataset.frameId = id;
    try { localStorage.setItem(`dy-review:${S.config.root}:last-frame`, id); } catch (_) { /* Private browsing may disable storage. */ }
    const item = S.frames.find(f => f.id === id); if (item) Object.assign(item, frame, {mask: undefined, boxes: undefined, teacher_boxes: undefined});
    const position = S.filtered.findIndex(f => f.id === id);
    if (position >= 0) S.listPage = Math.floor(position / S.pageSize);
    overlay.width = frame.width; overlay.height = frame.height;
    updateOverlay(); updateBoxes(); renderFrames(); fit();
  } catch (error) { toast(error.message, true); }
  finally { S.busy = false; updateState(); }
}
async function navigate(delta) {
  if (S.busy || !S.filtered.length) return;
  const position = S.filtered.findIndex(f => f.id === S.frame?.id);
  const next = position < 0 ? 0 : position + delta;
  if (next < 0 || next >= S.filtered.length) return toast("이 방향에 더 이상 프레임이 없습니다.");
  await openFrame(S.filtered[next].id);
}
async function save(status, next = false) {
  if (S.busy || !S.frame) return;
  if (S.samProposal) return toast("SAM 미리보기를 적용하거나 취소한 뒤 저장하세요.", true);
  if (S.drag || S.polygon.length) return toast("진행 중인 편집을 완료하거나 Esc로 취소하세요.", true);
  const position = S.filtered.findIndex(f => f.id === S.frame.id);
  const nextId = S.filtered[position + 1]?.id;
  S.busy = true; updateState();
  try {
    const result = await api(`/api/frame/${S.frame.id}`, {
      method: "POST", headers: {"Content-Type": "application/json", "X-Review-Token": S.config.token},
      body: JSON.stringify({revision: S.frame.revision, boxes: S.boxes, mask: bytesToBase64(S.mask), status, note: S.note}),
    });
    S.frame.revision = result.revision; Object.assign(S.frame, result.frame);
    Object.assign(S.frames.find(f => f.id === S.frame.id), result.frame);
    S.savedId = S.stateId; filterFrames();
    toast(`${S.frame.name} · ${statusNames[status]} 저장 완료`);
    S.busy = false; updateState();
    if (next && nextId) await openFrame(nextId, true);
    else if (next) toast("현재 목록의 마지막 프레임까지 저장했습니다.");
  } catch (error) { toast(`저장하지 못했습니다. 변경 내용은 화면에 유지됩니다.\n${error.message}`, true); }
  finally { S.busy = false; updateState(); }
}

function filterFrames() {
  const query = $("search").value.trim().toLowerCase();
  S.filtered = S.frames.filter(f => (!$("splitFilter").value || f.split === $("splitFilter").value)
    && (!$("statusFilter").value || f.status === $("statusFilter").value)
    && (!$("sequenceFilter").value || `${f.date}/${f.sequence}` === $("sequenceFilter").value)
    && (!query || `${f.name} ${f.sequence} ${f.date}`.toLowerCase().includes(query)));
  const order = $("sortOrder").value;
  S.filtered.sort((a, b) => (order === "uncertain" ? b.uncertain_fraction - a.uncertain_fraction
    : order === "empty" ? Number(b.box_count === 0) - Number(a.box_count === 0) : 0) || a.image_file.localeCompare(b.image_file, undefined, {numeric: true}));
  S.listPage = Math.min(S.listPage, Math.max(0, Math.ceil(S.filtered.length / S.pageSize) - 1));
  const visible = new Set(S.filtered.map(f => f.id));
  S.selectedFrames = new Set([...S.selectedFrames].filter(id => visible.has(id)));
  if (!visible.has(S.selectionAnchor)) S.selectionAnchor = null;
  renderFrames(); updateState();
}
function renderFrames() {
  $("filterCount").textContent = `${S.filtered.length.toLocaleString()}장`;
  const counts = {pending: 0, approved: 0, rejected: 0};
  S.frames.forEach(f => counts[f.status]++);
  $("progress").textContent = `${S.frames.length.toLocaleString()}장 · 승인 ${counts.approved.toLocaleString()} · 대기 ${counts.pending.toLocaleString()} · 제외 ${counts.rejected.toLocaleString()}`;
  $("frameList").replaceChildren();
  for (const f of S.filtered.slice(S.listPage * S.pageSize, (S.listPage + 1) * S.pageSize)) {
    const button = document.createElement("button");
    button.className = "frame-row" + (f.id === S.frame?.id ? " selected" : "") + (S.selectedFrames.has(f.id) ? " batch-selected" : "");
    button.dataset.frameId = f.id;
    button.setAttribute("role", "option"); button.setAttribute("aria-selected", String(S.selectedFrames.has(f.id)));
    button.title = `${f.image_file}\n불확실 마스크 ${(f.uncertain_fraction * 100).toFixed(2)}%`;
    const dot = document.createElement("span"); dot.className = `dot ${f.status}`;
    const copy = document.createElement("span"); copy.className = "frame-copy";
    const title = document.createElement("strong"); title.textContent = f.name;
    const detail = document.createElement("small"); detail.textContent = `${f.split} · ${f.sequence}`;
    copy.append(title, detail);
    const count = document.createElement("span"); count.className = "count"; count.textContent = `□ ${f.box_count}`;
    button.append(dot, copy, count); button.onclick = event => selectFrame(f.id, event);
    button.onmousedown = event => { if (event.shiftKey) event.preventDefault(); };
    $("frameList").append(button);
  }
  const pages = Math.max(1, Math.ceil(S.filtered.length / S.pageSize));
  $("listPage").textContent = `${S.listPage + 1} / ${pages}`;
  $("listPrev").disabled = S.listPage === 0; $("listNext").disabled = S.listPage >= pages - 1;
}

function updateOverlay() {
  if (!S.frame) return;
  const image = overlayCtx.createImageData(S.frame.width, S.frame.height), data = image.data;
  let road = 0, ignore = 0;
  const shownMask = S.samProposal ? samResultMask() : S.mask;
  for (let i = 0; i < shownMask.length; i++) {
    const p = i * 4;
    if (shownMask[i] === 1) { data[p] = 40; data[p+1] = S.samProposal ? 180 : 218; data[p+2] = S.samProposal ? 255 : 138; data[p+3] = 255; road++; }
    else if (shownMask[i] === 255) { data[p] = 255; data[p+1] = 179; data[p+2] = 53; data[p+3] = 255; ignore++; }
  }
  overlayCtx.putImageData(image, 0, 0);
  $("maskStats").textContent = `${S.samProposal ? "미리보기 · " : ""}도로 ${(road / S.mask.length * 100).toFixed(1)}% · 불확실 ${(ignore / S.mask.length * 100).toFixed(1)}%`;
}

function updateSam() {
  const available = Boolean(S.config?.sam?.available);
  const blocked = S.busy || !S.frame || !available || Boolean(S.drag) || Boolean(S.polygon.length);
  $("samPointCount").textContent = `포함 ${S.samPoints.filter(p=>p.label===1).length} · 제외 ${S.samPoints.filter(p=>p.label===0).length}`;
  for (const id of ["samPositive","samNegative","samMode"]) $(id).disabled = blocked;
  document.querySelector('[data-tool="sam"]').disabled = blocked;
  $("samUndoPoint").disabled = blocked || !S.samPoints.length;
  $("samClear").disabled = blocked || (!S.samPoints.length && !S.samProposal);
  $("samRun").disabled = blocked || !S.samPoints.some(p=>p.label===1);
  $("samApply").disabled = blocked || !S.samProposal;
  $("samDiscard").disabled = S.busy || !S.samProposal;
  $("samPositive").classList.toggle("active", S.samLabel===1);
  $("samNegative").classList.toggle("active", S.samLabel===0);
  if (S.config && !available) $("samStatus").textContent = S.config.sam?.message || "SAM이 연결되지 않았습니다.";
}
function samResultMask() {
  const mode = $("samMode").value;
  if (mode === "replace") return S.samProposal.slice();
  const mask = S.mask.slice();
  for (let i=0;i<mask.length;i++) if (S.samProposal[i]) mask[i] = mode === "erase" ? 0 : 1;
  return mask;
}
function discardSam(clearPoints=false) {
  if (S.busy) return;
  S.samProposal = null;
  if (clearPoints) S.samPoints = [];
  $("samStatus").textContent = "";
  updateOverlay(); updateState(); draw();
}
function addSamPoint(p, event) {
  if (event.shiftKey) {
    let index=-1, distance=12/S.scale;
    S.samPoints.forEach((q,i)=>{const d=Math.hypot(q.x-p.x,q.y-p.y);if(d<distance){index=i;distance=d;}});
    if (index>=0) { S.samPoints.splice(index,1); discardSam(); }
    return;
  }
  const x=Math.min(S.frame.width-1,Math.max(0,p.x)), y=Math.min(S.frame.height-1,Math.max(0,p.y));
  const label = event.button===2 ? 0 : S.samLabel;
  const existing = S.samPoints.find(q=>Math.round(q.x)===Math.round(x) && Math.round(q.y)===Math.round(y));
  if (existing) existing.label=label;
  else {
    if (S.samPoints.length>=64) return toast("SAM 점은 최대 64개입니다. 불필요한 점을 삭제하세요.",true);
    S.samPoints.push({x,y,label});
  }
  discardSam();
}
async function runSam() {
  if (S.busy || !S.frame || S.drag || S.polygon.length || !S.samPoints.some(p=>p.label===1)) return;
  const id=S.frame.id;
  S.busy=true; S.samRunning=true; updateState();
  try {
    const result=await api(`/api/frame/${id}/sam`,{method:"POST",headers:{"Content-Type":"application/json","X-Review-Token":S.config.token},body:JSON.stringify({points:S.samPoints})});
    const mask=Uint8Array.from(atob(result.mask),c=>c.charCodeAt(0));
    if (result.width!==S.frame.width || result.height!==S.frame.height || mask.length!==S.mask.length || mask.some(v=>v!==0 && v!==1)) throw new Error("SAM 마스크 형식이 올바르지 않습니다.");
    S.samProposal=mask;
    $("showMask").checked=true;
    if (Number($("opacity").value)===0) $("opacity").value=38;
    $("samStatus").textContent = result.prompt_violations ? `점 ${result.prompt_violations}개가 결과와 일치하지 않습니다. 점을 조정하거나 미리보기를 확인하세요.` : "파란색 도로 미리보기를 확인하고 적용하세요. 아직 저장되지 않았습니다.";
    updateOverlay(); draw();
  } catch(error) { toast(`SAM 실행 실패: ${error.message}`,true); }
  finally { S.busy=false; S.samRunning=false; updateState(); }
}
function applySam() {
  if (S.busy || !S.samProposal || S.drag || S.polygon.length) return;
  const mask=samResultMask();
  beginEdit(); S.mask=mask; S.samProposal=null;
  $("samStatus").textContent="적용했습니다. Ctrl+Z로 되돌리거나 Ctrl+S로 저장 · 승인하세요.";
  updateOverlay(); updateState(); draw();
}
function fit() {
  if (!S.frame) return;
  const rect = canvas.getBoundingClientRect();
  S.scale = Math.min((rect.width - 32) / S.frame.width, (rect.height - 32) / S.frame.height);
  S.ox = (rect.width - S.frame.width * S.scale) / 2; S.oy = (rect.height - S.frame.height * S.scale) / 2;
  draw();
}
function draw() {
  if (S.raf) return;
  S.raf = requestAnimationFrame(() => { S.raf = 0; renderCanvas(); });
}
function renderCanvas() {
  const rect = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  const pixelW = Math.round(rect.width * dpr), pixelH = Math.round(rect.height * dpr);
  if (canvas.width !== pixelW || canvas.height !== pixelH) { canvas.width = pixelW; canvas.height = pixelH; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, rect.width, rect.height);
  if (!S.frame || !S.rgb) return;
  ctx.translate(S.ox, S.oy); ctx.scale(S.scale, S.scale);
  const w = S.frame.width, h = S.frame.height;
  ctx.drawImage($("showNir").checked && S.nir ? S.nir : S.rgb, 0, 0, w, h);
  if ($("showMask").checked) { ctx.globalAlpha = Number($("opacity").value) / 100; ctx.imageSmoothingEnabled = false; ctx.drawImage(overlay, 0, 0); ctx.globalAlpha = 1; ctx.imageSmoothingEnabled = true; }
  if ($("showBoxes").checked) S.boxes.forEach((box, i) => drawBox(box, i === S.selected, i + 1));
  if (S.tool === "sam") S.samPoints.forEach((p,i)=>{
    ctx.beginPath(); ctx.arc(p.x,p.y,5/S.scale,0,Math.PI*2);
    ctx.fillStyle=p.label ? "#63f5b0" : "#ff7588"; ctx.fill();
    ctx.strokeStyle="#111";ctx.lineWidth=1.5/S.scale;ctx.stroke();
    ctx.font=`${12/S.scale}px system-ui`; ctx.lineWidth=3/S.scale;
    const text=`${p.label?"+":"−"}${i+1}`;ctx.strokeText(text,p.x+8/S.scale,p.y-5/S.scale);ctx.fillText(text,p.x+8/S.scale,p.y-5/S.scale);
  });
  if (S.drag?.type === "box") drawBox({class_id: Number($("boxClass").value), bbox_xyxy: normalizedRect(S.drag.start, S.drag.end)}, true, "+");
  if (S.polygon.length) {
    ctx.beginPath(); S.polygon.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    if (S.cursor) ctx.lineTo(S.cursor.x, S.cursor.y);
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5 / S.scale; ctx.setLineDash([5/S.scale, 4/S.scale]); ctx.stroke(); ctx.setLineDash([]);
    for (const p of S.polygon) { ctx.fillStyle = "#80e9bb"; ctx.fillRect(p.x-3/S.scale, p.y-3/S.scale, 6/S.scale, 6/S.scale); }
  }
  if (S.tool === "brush" && S.cursor && !S.space) {
    ctx.beginPath(); ctx.arc(S.cursor.x, S.cursor.y, S.brushSize/2, 0, 2*Math.PI);
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5/S.scale; ctx.stroke();
    ctx.strokeStyle = "#000"; ctx.lineWidth = .5/S.scale; ctx.stroke();
  }
  $("zoomLabel").textContent = `${Math.round(S.scale * 100)}%`;
  canvas.style.cursor = S.space || S.tool === "pan" ? (S.drag ? "grabbing" : "grab") : S.tool === "select" ? "default" : "crosshair";
}
function drawBox(box, selected, index) {
  const [x1, y1, x2, y2] = box.bbox_xyxy;
  ctx.strokeStyle = selected ? "#fff" : colors[box.class_id]; ctx.lineWidth = (selected ? 2.5 : 1.5)/S.scale;
  ctx.strokeRect(x1, y1, x2-x1, y2-y1);
  ctx.font = `${11/S.scale}px system-ui`;
  const text = `${index} ${S.config.classes[box.class_id]}`;
  const textWidth = ctx.measureText(text).width;
  const labelY = Math.max(0, y1 - 19/S.scale);
  ctx.fillStyle = "#111c2be6"; ctx.fillRect(x1, labelY, textWidth+8/S.scale, 18/S.scale);
  ctx.fillStyle = colors[box.class_id]; ctx.fillText(text, x1+4/S.scale, labelY+13/S.scale);
  if (selected) for (const [x, y] of corners(box)) { ctx.fillStyle = "#fff"; ctx.fillRect(x-4/S.scale, y-4/S.scale, 8/S.scale, 8/S.scale); }
}
function corners(box) { const [x1, y1, x2, y2] = box.bbox_xyxy; return [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]; }
function point(event, clamp = true) {
  const r = canvas.getBoundingClientRect(); let x = (event.clientX-r.left-S.ox)/S.scale, y = (event.clientY-r.top-S.oy)/S.scale;
  if (clamp) { x = Math.max(0, Math.min(S.frame.width, x)); y = Math.max(0, Math.min(S.frame.height, y)); }
  return {x, y};
}
function normalizedRect(a, b) { return [Math.min(a.x,b.x),Math.min(a.y,b.y),Math.max(a.x,b.x),Math.max(a.y,b.y)]; }
function hitBox(p) {
  // Smaller overlapping boxes should remain selectable inside large objects.
  return S.boxes.map((box, i) => ({box, i})).filter(({box}) => { const [a,b,c,d]=box.bbox_xyxy; return a<=p.x && p.x<=c && b<=p.y && p.y<=d; })
    .sort((a,b) => ((a.box.bbox_xyxy[2]-a.box.bbox_xyxy[0])*(a.box.bbox_xyxy[3]-a.box.bbox_xyxy[1]))-((b.box.bbox_xyxy[2]-b.box.bbox_xyxy[0])*(b.box.bbox_xyxy[3]-b.box.bbox_xyxy[1])))[0]?.i ?? -1;
}
function paint(from, to) {
  const radius = S.brushSize / 2, steps = Math.max(1, Math.ceil(Math.hypot(to.x-from.x, to.y-from.y)/Math.max(1, radius/3)));
  const w = S.frame.width, h = S.frame.height;
  for (let i=0; i<=steps; i++) {
    const cx = Math.min(w-.5, Math.floor(from.x + (to.x-from.x)*i/steps)+.5);
    const cy = Math.min(h-.5, Math.floor(from.y + (to.y-from.y)*i/steps)+.5);
    for (let y=Math.max(0,Math.floor(cy-radius)); y<Math.min(h,Math.ceil(cy+radius)); y++)
      for (let x=Math.max(0,Math.floor(cx-radius)); x<Math.min(w,Math.ceil(cx+radius)); x++)
        if ((x+.5-cx)**2+(y+.5-cy)**2 <= radius**2) S.mask[y*w+x] = S.maskValue;
  }
  updateOverlay(); draw(); updateState();
}
function finishPolygon() {
  if (S.busy || S.polygon.length < 3) return;
  beginEdit();
  const points = S.polygon, w = S.frame.width, h = S.frame.height;
  const minY = Math.max(0, Math.floor(Math.min(...points.map(p=>p.y)))), maxY = Math.min(h, Math.ceil(Math.max(...points.map(p=>p.y))));
  for (let y=minY; y<maxY; y++) {
    const intersections = [], py = y+.5;
    for (let i=0,j=points.length-1; i<points.length; j=i++) {
      const a=points[i], b=points[j];
      if ((a.y>py) !== (b.y>py)) intersections.push(a.x+(py-a.y)*(b.x-a.x)/(b.y-a.y));
    }
    intersections.sort((a,b)=>a-b);
    for (let i=0; i+1<intersections.length; i+=2) {
      const start=Math.max(0,Math.ceil(intersections[i]-.5)), end=Math.min(w,Math.ceil(intersections[i+1]-.5));
      S.mask.fill(S.maskValue, y*w+start, y*w+end);
    }
  }
  S.polygon=[]; updateOverlay(); updateState(); draw();
}
function setTool(tool) {
  if (S.busy || S.drag) return;
  if (S.samProposal && tool !== "sam") { toast("SAM 미리보기를 적용하거나 취소한 뒤 도구를 바꾸세요."); return; }
  if (tool === "sam" && !S.config?.sam?.available) return toast("SAM이 연결되지 않았습니다.",true);
  if (S.polygon.length && tool !== "polygon") { toast("다각형을 Enter로 채우거나 Esc로 취소하세요."); return; }
  S.tool = tool;
  if (tool === "box") selectBox(-1);
  document.querySelectorAll("[data-tool]").forEach(b => b.classList.toggle("active", b.dataset.tool === tool));
  const hints = {sam:"RGB 기준 영역 선택 · 클릭: 선택한 점 종류 · 우클릭: 제외점 · Shift+클릭: 점 삭제",select:"박스를 드래그해 이동하고, 선택한 박스의 모서리를 드래그해 크기를 조절합니다.", box:"추가할 클래스를 선택한 뒤 이미지에서 드래그하세요.", brush:"드래그해서 마스크를 칠합니다. 1 도로 · 2 비도로 · 3 불확실", polygon:"꼭짓점 클릭 → Enter / 더블클릭으로 채우기 · Backspace 마지막 점 삭제", pan:"드래그로 화면 이동 · 휠 확대/축소 · F 전체 보기"};
  $("toolHint").textContent = hints[tool]; draw();
}
function selectBox(index) { S.selected=index; if(index>=0) $("boxClass").value=S.boxes[index].class_id; updateBoxes(); draw(); }
function updateBoxes() {
  $("boxCount").textContent=S.boxes.length;
  $("boxList").replaceChildren();
  S.boxes.forEach((box,i)=>{
    const button=document.createElement("button"); button.className="box-row"+(i===S.selected?" active":"");
    const swatch=document.createElement("span"); swatch.className="box-color"; swatch.style.background=colors[box.class_id];
    const name=document.createElement("span"); name.textContent=`${i+1}. ${S.config.classes[box.class_id]}`;
    const size=document.createElement("small"); size.textContent=`${Math.round(box.bbox_xyxy[2]-box.bbox_xyxy[0])}×${Math.round(box.bbox_xyxy[3]-box.bbox_xyxy[1])}`;
    button.append(swatch,name,size); button.onclick=()=>{if(!S.busy) selectBox(i);}; $("boxList").append(button);
  });
  const box=S.boxes[S.selected];
  ["x1","y1","x2","y2"].forEach((id,i)=>{ $(id).value=box?Number(box.bbox_xyxy[i].toFixed(2)):""; $(id).disabled=!box || S.busy; });
  $("applyBox").disabled=!box || S.busy; $("deleteBox").disabled=!box || S.busy;
}
function deleteBox() {
  if(S.busy || S.selected<0 || S.drag) return;
  beginEdit(); S.boxes.splice(S.selected,1); S.selected=-1; updateBoxes(); updateState(); draw();
}
function cancelOperation() {
  if (S.samProposal && !S.busy) { discardSam(); return; }
  if(S.drag) {
    const drag=S.drag; S.drag=null;
    if(drag.started) { restore(S.undo.pop()); S.redo=[]; }
    else draw();
  }
  S.polygon=[]; updateState(); draw();
}

canvas.addEventListener("pointerdown", event=>{
  if(S.busy || !S.frame || S.drag || (![0,1].includes(event.button) && !(S.tool==="sam" && event.button===2))) return;
  event.preventDefault(); canvas.focus();
  const p=point(event), raw=point(event,false);
  if(S.space || S.tool==="pan" || event.button===1) {
    S.drag={type:"pan",x:event.clientX,y:event.clientY,ox:S.ox,oy:S.oy};
  } else {
    if(raw.x<0 || raw.y<0 || raw.x>S.frame.width || raw.y>S.frame.height) return;
    if(S.tool==="sam") { addSamPoint(p,event); return; }
    if(S.tool==="polygon") { S.polygon.push(p); updateState(); draw(); return; }
    if(S.tool==="brush") { beginEdit(); S.drag={type:"brush",last:p,started:true}; paint(p,p); }
    else if(S.tool==="box") { S.drag={type:"box",start:p,end:p}; }
    else {
      let handle=-1;
      if(S.selected>=0) handle=corners(S.boxes[S.selected]).findIndex(([x,y])=>Math.hypot(x-p.x,y-p.y)<9/S.scale);
      if(handle>=0) { S.drag={type:"resize",handle,start:p,original:[...S.boxes[S.selected].bbox_xyxy]}; }
      else { selectBox(hitBox(p)); if(S.selected>=0) S.drag={type:"move",start:p,original:[...S.boxes[S.selected].bbox_xyxy]}; }
    }
  }
  if(S.drag) canvas.setPointerCapture(event.pointerId);
  draw();
});
canvas.addEventListener("pointermove", event=>{
  if(!S.frame) return;
  const p=point(event); S.cursor=p;
  $("coordinates").textContent=`x ${Math.floor(p.x)}, y ${Math.floor(p.y)}`;
  if(S.busy) return;
  const drag=S.drag;
  if(drag?.type==="pan") { S.ox=drag.ox+event.clientX-drag.x; S.oy=drag.oy+event.clientY-drag.y; }
  else if(drag?.type==="brush") { paint(drag.last,p); drag.last=p; }
  else if(drag?.type==="box") drag.end=p;
  else if(drag && ["move","resize"].includes(drag.type)) {
    if(!drag.started && Math.hypot(p.x-drag.start.x,p.y-drag.start.y)*S.scale>=1) { beginEdit(); drag.started=true; }
    if(drag.started) {
      const [x1,y1,x2,y2]=drag.original;
      if(drag.type==="move") {
        const dx=Math.max(-x1,Math.min(S.frame.width-x2,p.x-drag.start.x));
        const dy=Math.max(-y1,Math.min(S.frame.height-y2,p.y-drag.start.y));
        S.boxes[S.selected].bbox_xyxy=[x1+dx,y1+dy,x2+dx,y2+dy];
      } else {
        const opposite=[[x2,y2],[x1,y2],[x1,y1],[x2,y1]][drag.handle];
        const box=normalizedRect(p,{x:opposite[0],y:opposite[1]});
        if(box[2]-box[0]>=1 && box[3]-box[1]>=1) S.boxes[S.selected].bbox_xyxy=box;
      }
      updateBoxes(); updateState();
    }
  }
  draw();
});
canvas.addEventListener("pointerup", event=>{
  const drag=S.drag;
  if(!drag) return;
  if(drag.type==="box") {
    const box=normalizedRect(drag.start,point(event));
    if(box[2]-box[0]>=1 && box[3]-box[1]>=1) { beginEdit(); S.boxes.push({class_id:Number($("boxClass").value),bbox_xyxy:box}); selectBox(S.boxes.length-1); }
  }
  S.drag=null;
  if(canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  updateState(); draw();
});
canvas.addEventListener("pointercancel", cancelOperation);
canvas.addEventListener("lostpointercapture", ()=>{if(S.drag) cancelOperation();});
canvas.addEventListener("pointerleave", ()=>{if(!S.drag){S.cursor=null;draw();}});
canvas.addEventListener("dblclick", event=>{event.preventDefault();if(S.tool==="polygon")finishPolygon();});
canvas.addEventListener("contextmenu", event=>{if(S.tool==="sam")event.preventDefault();});
canvas.addEventListener("wheel", event=>{
  if(!S.frame || S.busy || S.drag) return;
  event.preventDefault(); const p=point(event,false), rect=canvas.getBoundingClientRect();
  const fitScale=Math.min(rect.width/S.frame.width,rect.height/S.frame.height);
  S.scale=Math.max(fitScale*.2,Math.min(12,S.scale*Math.exp(-event.deltaY*.0015)));
  S.ox=event.clientX-rect.left-p.x*S.scale; S.oy=event.clientY-rect.top-p.y*S.scale; draw();
},{passive:false});

document.querySelectorAll("[data-tool]").forEach(b=>b.onclick=()=>setTool(b.dataset.tool));
document.querySelectorAll("[data-mask]").forEach(b=>b.onclick=()=>{
  if(S.drag) return;
  S.maskValue=Number(b.dataset.mask);
  document.querySelectorAll("[data-mask]").forEach(x=>x.classList.toggle("active",x===b));
});
$("brushSize").oninput=()=>{S.brushSize=Number($("brushSize").value);$("brushSizeLabel").textContent=`${S.brushSize} px`;draw();};
$("boxClass").onchange=()=>{
  if(S.busy || !S.frame || S.selected<0) return;
  const value=Number($("boxClass").value); if(S.boxes[S.selected].class_id===value) return;
  beginEdit();S.boxes[S.selected].class_id=value;updateBoxes();updateState();draw();
};
$("applyBox").onclick=()=>{
  if(S.busy || S.selected<0) return;
  const ids=["x1","y1","x2","y2"], box=ids.map(id=>Number($(id).value));
  if(ids.some(id=>$(id).value==="") || !box.every(Number.isFinite) || !(0<=box[0] && box[0]<box[2] && box[2]<=S.frame.width && 0<=box[1] && box[1]<box[3] && box[3]<=S.frame.height)) return toast("이미지 범위 안의 유효한 좌표를 입력하세요.",true);
  beginEdit();S.boxes[S.selected].bbox_xyxy=box;updateBoxes();updateState();draw();
};
$("note").oninput=()=>{if(!S.busy && S.frame){beginEdit();S.note=$("note").value;updateState();}};
$("deleteBox").onclick=deleteBox; $("undo").onclick=undo; $("redo").onclick=redo;
$("finishPolygon").onclick=finishPolygon; $("cancelPolygon").onclick=cancelOperation; $("fit").onclick=fit;
$("previous").onclick=()=>navigate(-1); $("next").onclick=()=>navigate(1);
$("saveDraft").onclick=()=>save("pending"); $("approve").onclick=()=>save("approved");
$("approveNext").onclick=()=>save("approved",true); $("reject").onclick=()=>save("rejected");
$("rejectSelected").onclick=rejectSelected;
$("clearSelection").onclick=()=>{if(!S.busy){clearFrameSelection();renderFrames();updateSelection();}};
$("samPositive").onclick=()=>{setTool("sam");S.samLabel=1;updateSam();};
$("samNegative").onclick=()=>{setTool("sam");S.samLabel=0;updateSam();};
$("samUndoPoint").onclick=()=>{if(!S.busy){S.samPoints.pop();discardSam();}};
$("samClear").onclick=()=>discardSam(true);
$("samRun").onclick=runSam; $("samApply").onclick=applySam; $("samDiscard").onclick=()=>discardSam();
$("samMode").onchange=()=>{if(S.samProposal){updateOverlay();draw();}};
for(const id of ["showMask","showBoxes","opacity"]) $(id).oninput=draw;
$("showNir").onchange=async()=>{
  if(S.busy || !S.frame) return;
  if($("showNir").checked && !S.nir) {
    S.busy=true;updateState();
    try{S.nir=await loadImage(`/api/frame/${S.frame.id}/nir`);}catch(error){$("showNir").checked=false;toast(error.message,true);}
    finally{S.busy=false;updateState();}
  }
  draw();
};
for(const id of ["search","splitFilter","statusFilter","sequenceFilter","sortOrder"]) $(id).oninput=()=>{clearFrameSelection();S.listPage=0;filterFrames();};
$("listPrev").onclick=()=>{S.listPage--;renderFrames();}; $("listNext").onclick=()=>{S.listPage++;renderFrames();};
$("refresh").onclick=async()=>{
  if(S.busy) return;
  try{const result=await api("/api/frames?refresh=1");S.frames=result.frames;filterFrames();toast("목록을 갱신했습니다. 현재 편집 내용은 유지됩니다.");}catch(error){toast(error.message,true);}
};
document.addEventListener("keydown", event=>{
  const typing=["INPUT","SELECT","TEXTAREA"].includes(event.target.tagName);
  const modifier=event.ctrlKey || event.metaKey;
  if(modifier && event.key.toLowerCase()==="s") {event.preventDefault();save("approved");return;}
  if(modifier && event.key==="Enter") {event.preventDefault();save("approved",true);return;}
  if(typing || S.busy) return;
  if(modifier && ["z","y"].includes(event.key.toLowerCase())) {event.preventDefault();event.shiftKey || event.key.toLowerCase()==="y"?redo():undo();return;}
  if(modifier || event.altKey) return;
  if(event.code==="Space") {event.preventDefault();S.space=true;draw();return;}
  const key=event.key.toLowerCase();
  const tools={v:"select",b:"box",r:"brush",p:"polygon",h:"pan",g:"sam"};
  if(tools[key]) setTool(tools[key]);
  else if(key==="f") fit();
  else if(key==="a") navigate(-1);
  else if(key==="d") navigate(1);
  else if(key==="delete") deleteBox();
  else if(key==="escape") cancelOperation();
  else if(key==="enter") S.tool==="sam" ? runSam() : finishPolygon();
  else if(key==="backspace" && S.tool==="sam") {event.preventDefault();S.samPoints.pop();discardSam();}
  else if(key==="backspace" && S.polygon.length) {event.preventDefault();S.polygon.pop();updateState();draw();}
  else if(["1","2","3"].includes(key)) document.querySelector(`[data-mask="${{1:1,2:0,3:255}[key]}"]`).click();
  else if(["[","]"].includes(key)) {$("brushSize").value=Math.max(1,Math.min(160,S.brushSize+(key==="["?-4:4)));$("brushSize").oninput();}
});
document.addEventListener("keyup", event=>{if(event.code==="Space"){S.space=false;draw();}});
window.addEventListener("blur",()=>{S.space=false;if(S.drag)cancelOperation();});
window.addEventListener("beforeunload",event=>{if(dirty() || S.polygon.length || S.samProposal){event.preventDefault();event.returnValue="";}});
new ResizeObserver(draw).observe($("canvasWrap"));

(async()=>{
  S.busy=true;updateState();
  try{
    const [config,result]=await Promise.all([api("/api/config"),api("/api/frames")]);
    S.config=config;S.frames=result.frames;
    config.classes.forEach((name,i)=>$("boxClass").add(new Option(`${i} · ${name}`,String(i))));
    const sequences=[...new Set(S.frames.map(f=>`${f.date}/${f.sequence}`))].sort();
    sequences.forEach(name=>$("sequenceFilter").add(new Option(name,name)));
    filterFrames();S.busy=false;
    let previous;
    try { previous=localStorage.getItem(`dy-review:${config.root}:last-frame`); } catch (_) {}
    if(S.filtered.length) await openFrame(S.frames.some(f=>f.id===previous)?previous:S.filtered[0].id,true);
  }catch(error){S.busy=false;updateState();toast(error.message,true);}
})();
