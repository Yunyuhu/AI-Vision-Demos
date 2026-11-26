/* 以相同權重順序重現 NumPy Dense 推論；同時供瀏覽器與 Node 一致性檢查使用。 */
(function (root) {
  "use strict";
  function predict(model, pixels) {
    if (pixels.length !== 784) throw new Error("影像需要 784 個像素。");
    let values = Array.from(pixels);
    for (let layer = 0; layer < model.weights.length; layer++) {
      const w = model.weights[layer], b = model.biases[layer];
      const next = b.slice();
      for (let i = 0; i < values.length; i++) {
        for (let j = 0; j < next.length; j++) next[j] += values[i] * w[i][j];
      }
      if (layer < model.weights.length - 1) values = next.map(v => Math.max(0, v));
      else {
        const max = Math.max(...next), exp = next.map(v => Math.exp(v - max));
        const sum = exp.reduce((a, b) => a + b, 0);
        values = exp.map(v => v / sum);
      }
    }
    return values;
  }
  root.DigitInference = { predict };
  if (typeof module !== "undefined") module.exports = { predict };
})(typeof globalThis !== "undefined" ? globalThis : window);
