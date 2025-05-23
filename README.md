# **程式碼庫文件說明**

## **專案概述**
這個專案是一個使用 PyTorch 和 ResNet18 架構的影像分類系統。專案目標是將來自兩位特定 Pixiv 藝術家的圖像（ID 分別為 48631 和 57824462）進行分類。專案提供了一個完整的管道，涵蓋資料處理、模型訓練及推論。

**本專案中的大部分程式碼由 ChatGPT 輔助產出，並詳細記錄，以便未來回顧與學習。**

**本程式碼僅作為自我練習用途，並未經藝術家同意使用其插圖，因此只提供數據，並不附上任何圖像。**

<img src="https://img.shields.io/badge/Python-3.10-blue.svg?logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/ResNet18-%23000000.svg?logo=code&logoColor=white" alt="ResNet18">
<img src="https://img.shields.io/badge/Pixiv_Style_Classifier-%232F88F6.svg?logo=pixiv&logoColor=white" alt="Pixiv Style Classifier">
            

## **文件說明**

### 1. **`resnet.py`**
**目的**：定義並實現 ResNet18 架構，用於影像分類。

**工作原理**：
- **`BasicBlock`**：此類別創建 ResNet 架構的基本殘差塊。
- **`ResNet18`**：主類別，構建 ResNet18 模型，包含卷積層、批次正規化、ReLU 激活函數等。
- **`resnet18()`**：一個輔助函數，用來創建指定輸出類別數量的模型。
- 這個架構包含了 4 層殘差塊，這是 ResNet18 設計中的標準配置。

### 2. **`preprocess.py`**
**目的**：通過裁剪處理圖像，使其符合指定的 3:4 長寬比。

**工作原理**：
- 從 `pixiv_artist_48631` 文件夾中載入圖像。
- 篩選出長寬比介於 0.6 到 0.9 之間的圖像。
- 裁剪符合條件的圖像，使其達到 3:4 的長寬比。
- 將裁剪後的圖像保存到新文件夾，並以 "cropped_" 作為前綴。
- 跳過不符合長寬比條件的圖像。

### 3. **`split_dataset.py`**
**目的**：將資料集拆分為訓練集和驗證集。

**工作原理**：
- 接收包含不同類別（Pixiv 藝術家）的子文件夾的資料夾。
- 隨機將每個類別中的圖像拆分為訓練集（80%）和驗證集（20%）。
- 創建包含 `train/` 和 `val/` 子文件夾的新目錄結構，每個子文件夾中包含類別文件夾。
- 保留每個類別的標籤，並保持拆分的資料一致性。
- 使用固定的隨機種子（42）以保證可重現性。

### 4. **`train.py`**
**目的**：在準備好的資料集上訓練 ResNet18 模型。

**工作原理**：
- 定義自訂的 `CustomImageFolder` 類別來加載從資料集拆分後的圖像。
- 實現訓練和驗證函數來追蹤模型的性能。
- 設定數據加載器、模型初始化、損失函數（交叉熵損失）和優化器（Adam）。
- 訓練模型 10 個時期，並顯示每個時期的損失和準確度指標。
- 將訓練後的模型權重保存為 `'resnet18_custom.pth'`。

### 5. **`inference.py`**
**目的**：使用訓練好的模型來對新圖像進行分類。

**工作原理**：
- 從 `'resnet18_custom.pth'` 加載訓練好的 ResNet18 模型。
- 使用與訓練過程中相同的圖像轉換處理一張測試圖像（`'test_img.jpg'`）。
- 對圖像進行推論，並輸出預測的類別（Pixiv 藝術家 48631 或 57824462）。

### 6. **`tensorboard.py`**
**目的**：創建 ResNet18 模型架構的 TensorBoard 可視化。

**工作原理**：
- 初始化 TensorBoard 寫入器。
- 創建 ResNet18 模型的實例。
- 生成一個假數據張量，將其傳入模型。
- 將模型的計算圖添加到 TensorBoard 進行可視化。
- 將可視化結果輸出到目錄 `"runs/resnet18_demo"`。

### 7. **`requirements.txt`**
**目的**：列出專案所需的 Python 套件依賴。

**內容**：
- Pillow（圖像處理）
- torch（用於深度學習的 PyTorch）
- tensorboard（用於可視化）
- torchvision（提供視覺相關的工具）

## **完整工作流程**

1. **資料準備**：
   - 使用 `preprocess.py` 裁剪圖像至所需的長寬比。
   - 使用 `split_dataset.py` 將資料集拆分為訓練集和驗證集。

2. **模型訓練**：
   - 運行 `train.py` 在準備好的資料集上訓練 ResNet18 模型。
     ![image](https://github.com/user-attachments/assets/f02edb25-8026-4ee1-ac26-8c1bf96aaf3f)

   - （可選）使用 `tensorboard.py` 可視化模型架構。

3. **推論**：
   - 使用 `inference.py` 對新圖像進行分類。
     ![image](https://github.com/user-attachments/assets/7188c099-b9d2-4b09-8c7e-329c9c17bea3)


此專案提供了一個完整的管道，用於訓練圖像分類模型，區分來自不同 Pixiv 藝術家的作品，並使用 ResNet18 架構。

參考資料: 

Pytorch YT教學 : 

https://youtu.be/QJgjcnuQqNI?si=NzYXexXrGLXv8wcF

2 位Pixiv繪師 :

https://www.pixiv.net/users/48631

https://www.pixiv.net/users/57824462
