import torch
from torchvision import transforms
from PIL import Image
import os

from resnet import ResNet18, BasicBlock  # 假設你模型是寫在 resnet.py 中
# 驗證模型參數
# 1. 載入模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=2)  # 你的類別數（ex: 2）
model.load_state_dict(torch.load("resnet18_custom.pth", map_location=device))
model.to(device)
model.eval()

# 2. 測試圖片路徑（你可以換成別的）
img_path = "test_img.jpg"

# 3. 前處理 (和訓練時一樣)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 和訓練一樣
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)  # 加上 batch 維度

# 4. 推論
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)

# 5. 顯示結果
class_names = ["pixiv_artist_57824462", "pixiv_artist_48631"]  # 根據你的順序調整
print(f"這張圖片預測為：{class_names[predicted.item()]}")
