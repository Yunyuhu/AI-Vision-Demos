"""重現原練習的 Dense 架構，輸出可由瀏覽器執行的模型與評估紀錄。

2026 整理版使用 NumPy 明確實作反向傳播；權重重新訓練，與 2024 紀錄分開。
資料使用 MNIST 官方既定測試集；從訓練集固定保留 6,000 筆作為驗證資料。
"""

import argparse
import json
from pathlib import Path
import numpy as np


def forward(x, weights, biases):
    """依序計算 ReLU 隱藏層與穩定化 softmax，保留反向傳播需要的中間值。"""
    activations = [x]
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        z = activations[-1] @ weight + bias
        if i == len(weights) - 1:
            exp = np.exp(z - z.max(axis=1, keepdims=True))
            a = exp / exp.sum(axis=1, keepdims=True)
        else:
            a = np.maximum(z, 0)
        activations.append(a)
    return activations


def evaluate(x, y, weights, biases):
    """分批評估，降低記憶體使用；回傳準確率、MSE 與各筆機率。"""
    probs = np.concatenate([forward(x[i:i + 500], weights, biases)[-1]
                            for i in range(0, len(x), 500)])
    onehot = np.eye(10, dtype=np.float32)[y]
    return float((probs.argmax(1) == y).mean()), float(((probs - onehot) ** 2).mean()), probs


def main():
    """固定初始化與資料順序；以驗證集選模型，最後才評估測試集。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('dist'))
    args = parser.parse_args()
    data = np.load(args.data)
    rng = np.random.default_rng(42)
    x = data['x_train'].reshape(-1, 784).astype(np.float32) / 255
    y = data['y_train']
    order = rng.permutation(len(x))
    train, val = order[6000:], order[:6000]
    xt = data['x_test'].reshape(-1, 784).astype(np.float32) / 255
    yt = data['y_test']
    dims = [784, 11, 73, 52, 61, 10]
    weights = [rng.uniform(-np.sqrt(6 / (a + b)), np.sqrt(6 / (a + b)),
                           (a, b)).astype(np.float32) for a, b in zip(dims, dims[1:])]
    biases = [np.zeros(b, dtype=np.float32) for b in dims[1:]]
    history, best, checkpoint = [], -1, None
    eye = np.eye(10, dtype=np.float32)
    for epoch in range(30):
        for batch in np.array_split(rng.permutation(train), len(train) // 100):
            acts = forward(x[batch], weights, biases)
            p = acts[-1]
            # Keras MSE 對十個輸出取平均；softmax Jacobian 以向量形式計算。
            delta_p = 2 * (p - eye[y[batch]]) / (len(batch) * 10)
            delta = p * (delta_p - (delta_p * p).sum(axis=1, keepdims=True))
            grads_w, grads_b = [], []
            for layer in reversed(range(len(weights))):
                grads_w.append(acts[layer].T @ delta)
                grads_b.append(delta.sum(axis=0))
                if layer:
                    delta = (delta @ weights[layer].T) * (acts[layer] > 0)
            # 所有梯度均根據更新前的權重計算，避免各層更新時互相污染。
            for layer, (dw, db) in enumerate(zip(reversed(grads_w), reversed(grads_b))):
                weights[layer] -= 0.088 * dw
                biases[layer] -= 0.088 * db
        accuracy, loss, _ = evaluate(x[val], y[val], weights, biases)
        history.append({'epoch': epoch + 1, 'validation_accuracy': accuracy, 'validation_mse': loss})
        if accuracy > best:
            best = accuracy
            checkpoint = ([w.copy() for w in weights], [b.copy() for b in biases], epoch + 1)
        print(f'Epoch {epoch + 1:02d}: validation accuracy={accuracy:.4f}, MSE={loss:.5f}', flush=True)
    weights, biases, best_epoch = checkpoint
    accuracy, loss, probs = evaluate(xt, yt, weights, biases)
    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (yt, probs.argmax(1)), 1)
    args.output.mkdir(parents=True, exist_ok=True)
    model = {'architecture': dims, 'weights': [w.tolist() for w in weights],
             'biases': [b.tolist() for b in biases]}
    (args.output / 'model.json').write_text(json.dumps(model, separators=(',', ':')))
    metrics = {'trained_on': '2026-11-26', 'implementation': 'NumPy, SGD, MSE',
               'seed': 42, 'training_count': len(train), 'validation_count': len(val),
               'test_count': len(xt), 'epochs': 30, 'best_epoch': best_epoch,
               'batch_size': 100, 'learning_rate': 0.088,
               'test_accuracy': accuracy, 'test_mse': loss, 'validation_history': history,
               'confusion_matrix': confusion.tolist(),
               'dataset_source': 'https://keras.io/api/datasets/mnist/',
               'historical_2024_test_accuracy': 0.9309}
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    # 每類取五筆測試樣本，包含可能辨識錯誤的案例，不依預測結果挑選。
    samples = [{'label': digit, 'index': int(i), 'pixels': data['x_test'][i].flatten().tolist()}
               for digit in range(10) for i in np.flatnonzero(yt == digit)[:5]]
    (args.output / 'samples.json').write_text(json.dumps(samples, separators=(',', ':')))
    # 供 JavaScript 與 Python 推論一致性驗證使用。
    (args.output / 'reference.json').write_text(json.dumps(probs[:20].tolist()))
    print(json.dumps({'test_accuracy': accuracy, 'best_epoch': best_epoch}), flush=True)


if __name__ == '__main__':
    main()
