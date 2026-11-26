/* 將手寫畫布轉成 MNIST 形式，再呼叫已載入權重；所有筆跡僅在本機處理。 */
"use strict";
const $ = id => document.getElementById(id);
const canvas = $("drawing"), ctx = canvas.getContext("2d", { willReadFrequently: true });
const preview = $("processed"), pctx = preview.getContext("2d");
let model, samples, drawing = false, hasInk = false, samplePixels = null;
const sampleCounters = Array(10).fill(0);

/* 建立固定十類的可讀分數列，避免後續更新時插入外部 HTML。 */
for (let digit = 0; digit < 10; digit++) {
  const row = document.createElement("div"); row.className = "bar";
  row.innerHTML = `<span>${digit}</span><div class="bar-track"><div class="bar-fill" style="width:0%"></div></div><span class="bar-score">0.0%</span>`;
  $("bars").append(row);
  const button = document.createElement("button"); button.textContent = digit;
  button.setAttribute("aria-label", `試用數字 ${digit} 的資料集範例`);
  button.disabled = true; button.addEventListener("click", () => loadSample(digit));
  $("examples").append(button);
}

function status(message, error = false) {
  $("status").textContent = message; $("status").classList.toggle("error", error);
}

function reset() {
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, 280, 280);
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 18; ctx.lineCap = "round"; ctx.lineJoin = "round";
  pctx.clearRect(0, 0, 28, 28); hasInk = false; samplePixels = null;
  $("hint").hidden = false; $("answer").textContent = "?";
  $("result-text").textContent = "等你寫下第一個數字";
  $("confidence").textContent = "這裡會顯示模型的相對分數";
  Array.from($("bars").children).forEach(row => {
    row.classList.remove("best"); row.querySelector(".bar-fill").style.width = "0%";
    row.querySelector(".bar-score").textContent = "0.0%";
  });
  status(model ? "模型已就緒。" : "模型下載完成後即可操作。");
}

function coordinates(event) {
  const rect = canvas.getBoundingClientRect();
  return [(event.clientX - rect.left) * 280 / rect.width, (event.clientY - rect.top) * 280 / rect.height];
}

canvas.addEventListener("pointerdown", event => {
  if (samplePixels) reset();
  drawing = true; hasInk = true; samplePixels = null; $("hint").hidden = true;
  canvas.setPointerCapture(event.pointerId);
  const [x, y] = coordinates(event); ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + .1, y + .1); ctx.stroke();
});
canvas.addEventListener("pointermove", event => {
  if (!drawing) return;
  const [x, y] = coordinates(event); ctx.lineTo(x, y); ctx.stroke();
});
canvas.addEventListener("pointerup", () => { drawing = false; });
canvas.addEventListener("pointercancel", () => { drawing = false; });

function drawPixels(pixels) {
  const image = pctx.createImageData(28, 28);
  pixels.forEach((v, i) => {
    image.data[i * 4] = image.data[i * 4 + 1] = image.data[i * 4 + 2] = Math.round(v * 255);
    image.data[i * 4 + 3] = 255;
  });
  pctx.putImageData(image, 0, 0);
}

function preparePixels() {
  if (samplePixels) return samplePixels;
  const data = ctx.getImageData(0, 0, 280, 280).data;
  let minX = 280, minY = 280, maxX = -1, maxY = -1;
  for (let y = 0; y < 280; y++) for (let x = 0; x < 280; x++) {
    if (data[(y * 280 + x) * 4] > 25) {
      minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
  }
  if (maxX < 0) return null;
  const width = maxX - minX + 1, height = maxY - minY + 1;
  const scale = 20 / Math.max(width, height);
  const small = document.createElement("canvas"); small.width = small.height = 28;
  const sctx = small.getContext("2d", { willReadFrequently: true });
  sctx.fillStyle = "black"; sctx.fillRect(0, 0, 28, 28);
  sctx.drawImage(canvas, minX, minY, width, height, (28 - width * scale) / 2, (28 - height * scale) / 2, width * scale, height * scale);
  const raw = sctx.getImageData(0, 0, 28, 28).data;
  let mass = 0, cx = 0, cy = 0;
  for (let i = 0; i < 784; i++) { const value = raw[i * 4]; mass += value; cx += (i % 28) * value; cy += Math.floor(i / 28) * value; }
  if (mass < 255) return null;
  // 以筆跡重心置中，縮小手寫位置與 MNIST 訓練資料的差異。
  const shifted = document.createElement("canvas"); shifted.width = shifted.height = 28;
  const tctx = shifted.getContext("2d"); tctx.fillStyle = "black"; tctx.fillRect(0, 0, 28, 28);
  tctx.drawImage(small, Math.round(13.5 - cx / mass), Math.round(13.5 - cy / mass));
  const result = tctx.getImageData(0, 0, 28, 28).data;
  return Array.from({ length: 784 }, (_, i) => result[i * 4] / 255);
}

function recognize() {
  if (!model) return status("模型尚未就緒，請稍候。", true);
  const pixels = hasInk ? preparePixels() : null;
  if (!pixels) return status("請先寫下一個數字，或選擇下方的資料集範例。", true);
  drawPixels(pixels);
  const scores = DigitInference.predict(model, pixels), top = scores.indexOf(Math.max(...scores));
  $("answer").textContent = top;
  $("result-text").textContent = scores[top] < .6 ? "這次的判斷較不確定" : `模型判斷為數字 ${top}`;
  $("confidence").textContent = `最高模型分數 ${(scores[top] * 100).toFixed(1)}%，仍可能辨識錯誤`;
  Array.from($("bars").children).forEach((row, i) => {
    row.classList.toggle("best", i === top); row.querySelector(".bar-fill").style.width = `${scores[i] * 100}%`;
    row.querySelector(".bar-score").textContent = `${(scores[i] * 100).toFixed(1)}%`;
  });
  status("辨識完成。可以清除重寫，或查看其他範例。");
}

function loadSample(digit) {
  reset();
  const options = samples.filter(sample => sample.label === digit);
  const sample = options[sampleCounters[digit]++ % options.length];
  samplePixels = sample.pixels.map(v => v / 255); hasInk = true; $("hint").hidden = true;
  drawPixels(samplePixels); ctx.imageSmoothingEnabled = false; ctx.drawImage(preview, 0, 0, 280, 280); ctx.imageSmoothingEnabled = true;
  recognize(); status(`MNIST 測試集第 ${sample.index} 筆，標準答案為 ${sample.label}。`);
}

$("clear").addEventListener("click", reset);
$("recognize").addEventListener("click", recognize);
reset();

/* 一次讀取模型與評估紀錄；失敗時保持可讀錯誤，避免畫面看似已完成辨識。 */
Promise.all(["model.json", "samples.json", "metrics.json"].map(async url => {
  const response = await fetch(url); if (!response.ok) throw new Error(`無法讀取 ${url}`); return response.json();
})).then(([loaded, examples, metrics]) => {
  model = loaded; samples = examples;
  $("recognize").disabled = false; $("recognize").textContent = "辨識數字";
  $("model-state").textContent = "瀏覽器本機推論";
  Array.from($("examples").children).forEach(button => { button.disabled = false; });
  $("accuracy").textContent = `${(metrics.test_accuracy * 100).toFixed(2)}%`;
  $("metric-detail").textContent = `MNIST 測試集 ${metrics.test_count.toLocaleString()} 筆 · ${metrics.trained_on} 重訓`;
  $("training-info").textContent = `訓練 ${metrics.training_count.toLocaleString()} 筆，驗證 ${metrics.validation_count.toLocaleString()} 筆；SGD、MSE、batch size ${metrics.batch_size}、${metrics.epochs} epochs。依驗證集選用第 ${metrics.best_epoch} 個 epoch 的權重。`;
  status("模型已就緒。寫下一個數字，或先試試範例。");
}).catch(error => { $("model-state").textContent = "載入失敗"; $("recognize").textContent = "模型載入失敗"; status(`${error.message}，請重新整理頁面。`, true); });
