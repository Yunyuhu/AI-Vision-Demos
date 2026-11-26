/* 手寫轉成 MNIST 形式後在瀏覽器推論；筆跡只在本機處理。 */
"use strict";
const $ = id => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";
const INK = "#1f2f5c";

// 看得到的畫布：深藍墨水畫在透明紙上；看不到的遮罩：白筆畫在黑底上，給模型用
const canvas = $("drawing"), ctx = canvas.getContext("2d");
const mask = document.createElement("canvas"); mask.width = mask.height = 280;
const mctx = mask.getContext("2d", { willReadFrequently: true });
const preview = $("processed"), pctx = preview.getContext("2d");
let model, samples, drawing = false, hasInk = false, samplePixels = null, sampleLabel = null, lastPixels = null, throttle = 0;
const sampleCounters = Array(10).fill(0);

for (let digit = 0; digit < 10; digit++) {
  const col = document.createElement("div"); col.className = "col";
  const bar = document.createElement("div"); bar.className = "col-bar";
  const fill = document.createElement("div"); fill.className = "col-fill";
  const score = document.createElement("span"); score.className = "col-score"; score.textContent = "0%";
  const label = document.createElement("span"); label.className = "col-label"; label.textContent = digit;
  bar.append(fill); col.append(score, bar, label); $("bars").append(col);
}

function status(message, error = false) { $("status").textContent = message; $("status").classList.toggle("error", error); }
function pen(c, color) { c.strokeStyle = color; c.lineWidth = 18; c.lineCap = "round"; c.lineJoin = "round"; }

function reset() {
  ctx.clearRect(0, 0, 280, 280); mctx.fillStyle = "#000"; mctx.fillRect(0, 0, 280, 280);
  pen(ctx, INK); pen(mctx, "#fff");
  pctx.clearRect(0, 0, 28, 28); hasInk = false; samplePixels = null; sampleLabel = null; lastPixels = null;
  $("hint").hidden = false; $("answer").textContent = "?"; $("stamp").classList.remove("sure", "unsure", "pop", "wrong");
  $("result-text").textContent = "等你寫第一個數字";
  $("confidence").textContent = "模型會給 0～9 每個答案一個分數";
  for (const col of $("bars").children) { col.classList.remove("best"); col.querySelector(".col-fill").style.height = "0%"; col.querySelector(".col-score").textContent = "0%"; }
  if (model) status("寫一個數字，或從下面挑一張範例。");
}

function point(event) {
  const rect = canvas.getBoundingClientRect();
  return [(event.clientX - rect.left) * 280 / rect.width, (event.clientY - rect.top) * 280 / rect.height];
}
function stroke(fn) { fn(ctx); fn(mctx); }
canvas.addEventListener("pointerdown", event => {
  if (samplePixels) reset();
  drawing = true; hasInk = true; $("hint").hidden = true; canvas.setPointerCapture(event.pointerId);
  const [x, y] = point(event); stroke(c => { c.beginPath(); c.moveTo(x, y); c.lineTo(x + .1, y + .1); c.stroke(); });
});
canvas.addEventListener("pointermove", event => {
  if (!drawing) return;
  const [x, y] = point(event); stroke(c => { c.lineTo(x, y); c.stroke(); });
  const now = performance.now(); if (now - throttle > 140) { throttle = now; recognize(true); }
});
const stop = () => { if (drawing) { drawing = false; recognize(); } };
canvas.addEventListener("pointerup", stop); canvas.addEventListener("pointercancel", stop);
canvas.addEventListener("keydown", event => { if (event.key === "Escape" || event.key === "Delete") reset(); });

function drawPixels(pixels) {
  const image = pctx.createImageData(28, 28);
  pixels.forEach((v, i) => { const g = 255 - Math.round(v * 220); image.data.set([g - 10 * v, g - 4 * v, 255 - 120 * v, 255], i * 4); });
  pctx.putImageData(image, 0, 0);
}

function preparePixels() {
  if (samplePixels) return samplePixels;
  const data = mctx.getImageData(0, 0, 280, 280).data;
  let minX = 280, minY = 280, maxX = -1, maxY = -1;
  for (let y = 0; y < 280; y++) for (let x = 0; x < 280; x++) {
    if (data[(y * 280 + x) * 4] > 25) { minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y); }
  }
  if (maxX < 0) return null;
  const width = maxX - minX + 1, height = maxY - minY + 1, scale = 20 / Math.max(width, height);
  const small = document.createElement("canvas"); small.width = small.height = 28;
  const sctx = small.getContext("2d", { willReadFrequently: true }); sctx.fillStyle = "black"; sctx.fillRect(0, 0, 28, 28);
  sctx.drawImage(mask, minX, minY, width, height, (28 - width * scale) / 2, (28 - height * scale) / 2, width * scale, height * scale);
  const raw = sctx.getImageData(0, 0, 28, 28).data;
  let mass = 0, cx = 0, cy = 0;
  for (let i = 0; i < 784; i++) { const v = raw[i * 4]; mass += v; cx += (i % 28) * v; cy += Math.floor(i / 28) * v; }
  if (mass < 255) return null;
  // 以筆跡重心置中，縮小手寫位置與 MNIST 訓練資料的差異
  const shifted = document.createElement("canvas"); shifted.width = shifted.height = 28;
  const tctx = shifted.getContext("2d"); tctx.fillStyle = "black"; tctx.fillRect(0, 0, 28, 28);
  tctx.drawImage(small, Math.round(13.5 - cx / mass), Math.round(13.5 - cy / mass));
  const result = tctx.getImageData(0, 0, 28, 28).data;
  return Array.from({ length: 784 }, (_, i) => result[i * 4] / 255);
}

function recognize(live = false) {
  if (!model) return;
  const pixels = hasInk ? preparePixels() : null;
  if (!pixels) return;
  lastPixels = pixels; drawPixels(pixels);
  const scores = DigitInference.predict(model, pixels), top = scores.indexOf(Math.max(...scores));
  const previous = $("answer").textContent;
  $("answer").textContent = top;
  const sure = scores[top] >= .6;
  $("stamp").classList.toggle("sure", sure); $("stamp").classList.toggle("unsure", !sure);
  if (String(top) !== previous) { $("stamp").classList.remove("pop"); void $("stamp").offsetWidth; $("stamp").classList.add("pop"); }
  const second = scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0])[1];
  $("result-text").textContent = sure ? `我猜是 ${top}` : `有點像 ${top}，也可能是 ${second[1]}`;
  if (sampleLabel !== null) $("result-text").textContent += top === sampleLabel ? "，答對了" : `，猜錯了（正確是 ${sampleLabel}）`;
  $("stamp").classList.toggle("wrong", sampleLabel !== null && top !== sampleLabel);
  $("confidence").textContent = `信心 ${(scores[top] * 100).toFixed(1)}%${live ? "・還在看你寫" : ""}`;
  [...$("bars").children].forEach((col, i) => {
    col.classList.toggle("best", i === top);
    col.querySelector(".col-fill").style.height = `${Math.max(scores[i] * 100, .5)}%`;
    col.querySelector(".col-score").textContent = scores[i] >= .995 ? "99%+" : `${Math.round(scores[i] * 100)}%`;
  });
  if (!live) status(samplePixels ? status.sampleNote : "寫完了！可以擦掉再寫一個。");
}

// 28×28 影像：滑過時顯示該格亮度
$("processed").addEventListener("mousemove", event => {
  if (!lastPixels) return;
  const rect = preview.getBoundingClientRect();
  const x = Math.min(27, Math.floor((event.clientX - rect.left) / rect.width * 28)), y = Math.min(27, Math.floor((event.clientY - rect.top) / rect.height * 28));
  const peek = $("peek"); peek.hidden = false;
  peek.textContent = `第 ${y + 1} 列・第 ${x + 1} 行：${lastPixels[y * 28 + x].toFixed(2)}`;
});
$("processed").addEventListener("mouseleave", () => { $("peek").hidden = true; });

function loadSample(digit, tile) {
  reset();
  const options = samples.filter(s => s.label === digit);
  const sample = options[sampleCounters[digit]++ % options.length];
  samplePixels = sample.pixels.map(v => v / 255); sampleLabel = sample.label; hasInk = true; $("hint").hidden = true;
  // 把 28×28 範例放大畫到紙上（深藍墨水）
  const tmp = document.createElement("canvas"); tmp.width = tmp.height = 28; const tc = tmp.getContext("2d"); const img = tc.createImageData(28, 28);
  samplePixels.forEach((v, i) => img.data.set([31, 47, 92, Math.round(v * 255)], i * 4)); tc.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true; ctx.drawImage(tmp, 0, 0, 280, 280);
  status.sampleNote = `這是 MNIST 測試資料第 ${sample.index} 筆，正確答案是 ${sample.label}。`;
  recognize();
  if (tile) { paintThumb(tile.querySelector("canvas"), sample.pixels); tile.querySelector(".nth").textContent = `第 ${(sampleCounters[digit] - 1) % options.length + 1}/${options.length} 張`; }
}
function paintThumb(c, pixels) {
  const x = c.getContext("2d"), img = x.createImageData(28, 28);
  pixels.forEach((v, i) => img.data.set([31, 47, 92, v], i * 4)); x.putImageData(img, 0, 0);
}

/* 評估圖表 */
function curve(history, test) {
  const svg = $("curve"), W = 320, H = 180, L = 34, R = 10, T = 12, B = 26;
  const xs = i => L + (W - L - R) * i / (history.length - 1), ys = v => T + (H - T - B) * (1 - v);
  const el = (tag, attrs, text) => { const e = document.createElementNS(SVG, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; svg.append(e); return e; };
  for (const v of [0, .25, .5, .75, 1]) { el("line", { x1: L, x2: W - R, y1: ys(v), y2: ys(v), class: "grid" }); el("text", { x: L - 6, y: ys(v) + 4, class: "axis", "text-anchor": "end" }, `${v * 100}%`); }
  el("text", { x: L, y: H - 6, class: "axis" }, "第 1 輪"); el("text", { x: W - R, y: H - 6, class: "axis", "text-anchor": "end" }, `第 ${history.length} 輪`);
  const d = history.map((h, i) => `${i ? "L" : "M"}${xs(i).toFixed(1)} ${ys(h.validation_accuracy).toFixed(1)}`).join(" ");
  el("path", { d: `${d} L${xs(history.length - 1)} ${ys(0)} L${xs(0)} ${ys(0)} Z`, class: "area" });
  el("path", { d, class: "line" });
  history.forEach((h, i) => { const c = el("circle", { cx: xs(i), cy: ys(h.validation_accuracy), r: 3.2, class: "dot" }); const t = document.createElementNS(SVG, "title"); t.textContent = `第 ${h.epoch} 輪：${(h.validation_accuracy * 100).toFixed(1)}%`; c.append(t); });
  el("line", { x1: L, x2: W - R, y1: ys(test), y2: ys(test), class: "target" });
  el("text", { x: W - R, y: ys(test) - 5, class: "axis target-label", "text-anchor": "end" }, `測試 ${(test * 100).toFixed(2)}%`);
}
function confusion(matrix) {
  const perDigit = matrix.map((row, i) => row[i] / row.reduce((a, b) => a + b, 0));
  perDigit.forEach((v, i) => {
    const row = document.createElement("div"); row.className = "pd";
    row.innerHTML = `<span class="pd-d">${i}</span><span class="pd-track"><span class="pd-fill"></span></span><span class="pd-v">${(v * 100).toFixed(1)}%</span>`;
    row.querySelector(".pd-fill").style.width = `${(v - .7) / .3 * 100}%`;
    if (v === Math.min(...perDigit)) row.classList.add("weak");
    $("per-digit").append(row);
  });
  const offMax = Math.max(...matrix.flatMap((row, i) => row.filter((_, j) => j !== i)));
  const corner = document.createElement("span"); corner.className = "m-head"; $("matrix").append(corner);
  for (let j = 0; j < 10; j++) { const h = document.createElement("span"); h.className = "m-head"; h.textContent = j; $("matrix").append(h); }
  matrix.forEach((row, i) => {
    const h = document.createElement("span"); h.className = "m-head"; h.textContent = i; $("matrix").append(h);
    row.forEach((n, j) => {
      const cell = document.createElement("span"); cell.className = i === j ? "m-cell diag" : "m-cell";
      cell.style.setProperty("--a", i === j ? .9 : (n / offMax).toFixed(3));
      if (i === j) cell.textContent = "✓";
      const text = i === j ? `正確答案 ${i}，答對 ${n} 次` : `正確答案 ${i}，被判成 ${j}：${n} 次`;
      cell.title = text;
      cell.addEventListener("mouseenter", () => { $("matrix-readout").textContent = text; });
      $("matrix").append(cell);
    });
  });
  const pairs = matrix.flatMap((row, i) => row.map((n, j) => ({ i, j, n }))).filter(p => p.i !== p.j).sort((a, b) => b.n - a.n).slice(0, 3);
  for (const p of pairs) { const li = document.createElement("li"); li.innerHTML = `<b>${p.i}</b> 常被看成 <b>${p.j}</b><span>${p.n} 次</span>`; $("top-confusions").append(li); }
}

$("clear").addEventListener("click", reset);
reset();

/* 一次讀取模型、範例與評估紀錄；失敗時保留可讀的錯誤訊息 */
Promise.all(["model.json", "samples.json", "metrics.json"].map(async url => {
  const response = await fetch(url); if (!response.ok) throw new Error(`無法讀取 ${url}`); return response.json();
})).then(([loaded, examples, metrics]) => {
  model = loaded; samples = examples;
  for (let digit = 0; digit < 10; digit++) {
    const first = samples.find(s => s.label === digit);
    const tile = document.createElement("button"); tile.type = "button"; tile.className = "tile";
    tile.setAttribute("aria-label", `試試資料集裡的數字 ${digit}`);
    const c = document.createElement("canvas"); c.width = c.height = 28; paintThumb(c, first.pixels);
    const tag = document.createElement("span"); tag.className = "tag"; tag.textContent = digit;
    const nth = document.createElement("span"); nth.className = "nth"; nth.textContent = "點我試試";
    tile.append(c, tag, nth); tile.addEventListener("click", () => loadSample(digit, tile));
    $("examples").append(tile);
  }
  $("accuracy").textContent = `${(metrics.test_accuracy * 100).toFixed(2)}%`;
  $("metric-detail").textContent = `在 ${metrics.test_count.toLocaleString()} 張沒看過的手寫數字上答對的比例`;
  $("training-info").textContent = `訓練資料 ${metrics.training_count.toLocaleString()} 張、驗證 ${metrics.validation_count.toLocaleString()} 張；SGD、MSE、batch size ${metrics.batch_size}。`;
  curve(metrics.validation_history, metrics.test_accuracy); confusion(metrics.confusion_matrix);
  status("寫一個數字，或從下面挑一張範例。"); canvas.focus({ preventScroll: true });
}).catch(error => { status(`${error.message}，請重新整理頁面。`, true); });
