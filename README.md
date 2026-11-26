# AI 影像辨識展示

[直接開啟手寫數字辨識 Demo](https://yunyuhu.github.io/AI-Vision-Demos)

目前已完成手寫數字辨識。其他影像辨識內容完成後再加入。

由 [2024 MNIST 練習](https://github.com/Yunyuhu/AI-Practice-2024/blob/main/course/week05_mnist_dense_nn.ipynb)延伸的公開互動展示。可以直接手寫數字、試用測試樣本、查看模型輸入與十類分數，無須金鑰或登入。

## 原始實作與本次整理

- 原始版本使用 TensorFlow／Keras，隱藏層為 11、73、52、61，ReLU、十類 softmax、MSE、SGD，並記錄多組 batch size 與 epochs。
- 2026 整理版以 NumPy 重現相同 Dense 架構，重新訓練並匯出 JSON 權重。瀏覽器使用 JavaScript 推論，筆跡不會上傳。
- 固定 seed 42，從原訓練集切出 6,000 筆驗證資料，訓練 54,000 筆；測試集保留 10,000 筆。
- 訓練 30 epochs，batch size 100、learning rate 0.088，依驗證準確率選擇權重。
- 本次測試集準確率為 93.23%；2024 既有紀錄為 93.09%，兩次訓練設定與資料安排有差異，分別保留。
- `dist/metrics.json` 包含實際訓練紀錄、混淆矩陣與評估數字。

## 操作

使用 Python 啟動靜態網站伺服器後，開啟其網址。正式部署只需發布 `dist/`。

```sh
# 從專案根目錄提供靜態網站；在終端機中按 Ctrl+C 結束。
python3 -m http.server 8765 --directory dist
```

## 重新訓練

NumPy 版本為 2.5.5。使用獨立虛擬環境安裝 `requirements.txt`，並從 [TensorFlow 官方範例的資料網址](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz)下載資料。

```sh
# 在專案根目錄建立獨立環境，避免修改其他專案的套件。
python3 -m venv .venv
# 安裝訓練用依賴；靜態 Demo 無須安裝此套件。
.venv/bin/python -m pip install -r requirements.txt
# 將下載的 MNIST 檔案放在本機，明確指定資料位置後重訓。
.venv/bin/python train.py --data /absolute/path/to/mnist.npz --output dist
```

## 驗證與限制

- 以 50 筆樣本比較 NumPy 與 JavaScript 輸出，最大機率絕對誤差約 3.22e-7。
- 已在瀏覽器驗證模型載入、空白輸入提示、資料集範例、清除與實際手寫直線數字 1。
- 手寫輸入會先依筆跡邊界裁切、等比例縮放及重心置中。多個數字、字母、符號與差異較大的筆跡不在模型訓練範圍。
- softmax 分數尚未校準，不能當作經過校準的正確機率。

## 資料授權

[Keras MNIST 說明](https://keras.io/api/datasets/mnist/)記載資料集版權由 Yann LeCun 與 Corinna Cortes 持有，依 CC BY-SA 3.0 提供。展示所附樣本取自 MNIST，沿用相同授權；其原始資料衍生自 NIST。
