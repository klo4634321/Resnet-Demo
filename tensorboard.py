import torch
from torch.utils.tensorboard import SummaryWriter
# 假設在 resnet.py 裡面
from resnet import BasicBlock, ResNet18


# 匯入你自己寫的 ResNet18（這一行請換成你實際的 import）
#from resnet import ResNet18  

# 建立一個 writer
writer = SummaryWriter("runs/resnet18_demo")

# 建立模型
model = ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=10)  # 假設分類數為10

# 建立一個假的輸入（內容都是 1）
dummy_input = torch.ones((1, 3, 224, 224))  # batch_size=1, 3 channels, 224x224

# 加入 graph 到 TensorBoard
writer.add_graph(model, dummy_input)

# 關閉 writer
writer.close()
