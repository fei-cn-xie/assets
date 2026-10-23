import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)


batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # 创建展平层，将每个 28x28 的二维图像展平成一维向量（28*28=788 个像素值），以便输入全连接层
        self.linear_relu_stack = nn.Sequential( # 创建由多个线性层和 ReLU 激活层顺序堆叠而成的网络结构，数据会按定义顺序依次通过这些层，最终输出 10 个类别的 logits
            nn.Linear(28*28, 512), # 创建第一个线性层，将展平后的 28*28=784 个像素值映射到 512 个隐藏神经元
            nn.ReLU(), # 创建 ReLU 激活层，为第一个线性层的输出添加非线性变换，使网络能够学习更复杂的模式
            nn.Linear(512, 512), # 创建第二个线性层，将 512 个隐藏神经元映射到另一个 512 个隐藏神经元
            nn.ReLU(), # 创建 ReLU 激活层，为第二个线性层的输出添加非线性变换
            nn.Linear(512, 10) # 创建第三个线性层，将 512 个隐藏神经元映射到 10 个输出类别的 logits，表示每个类别的预测分数
        )

    def forward(self, x):
        x = self.flatten(x) # 将输入图像展平为一维向量
        logits = self.linear_relu_stack(x)  # 将展平后的向量输入到线性层和 ReLU 激活层堆叠的网络结构中，得到每个类别的预测分数（logits）
        return logits # 返回 logits，表示每个类别的预测分数

model = NeuralNetwork().to(device)
print(model)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# 定义训练函数，对模型执行一个完整 epoch 的训练
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) # 获取训练数据集的样本总数，用于打印训练进度
    model.train() # 将模型设置为训练模式（启用 Dropout、BatchNorm 等训练专用行为）
    for batch, (X, y) in enumerate(dataloader): # 遍历数据加载器，按批次取出输入图像 X 和标签 y
        X, y = X.to(device), y.to(device) # 将数据和标签移动到当前计算设备（GPU/NPU/CPU）

        # Compute prediction error
        pred = model(X) # 前向传播，得到模型对每个类别的预测分数（logits）
        loss = loss_fn(pred, y) # 计算预测结果与真实标签之间的交叉熵损失

        # Backpropagation
        loss.backward() # 反向传播，计算损失相对于各参数的梯度
        optimizer.step() # 根据梯度更新模型参数
        optimizer.zero_grad() # 清空梯度，避免梯度在下一次迭代中累积

        if batch % 100 == 0: # 每 100 个批次打印一次训练状态
            loss, current = loss.item(), (batch + 1) * len(X) # 取出当前损失值和已处理的样本数
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") # 打印当前损失及训练进度

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

print("==============================================")
print("Loading the model from model.pth")
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

# 使用加载的模型进行预测，验证模型是否正确加载
print("Model loaded from model.pth")
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device) # 将测试样本移动到当前计算设备（GPU/NPU/CPU）
    pred = model(x) # 前向传播，得到该样本在 10 个类别上的预测分数（logits）
    predicted, actual = classes[pred[0].argmax(0)], classes[y] # 取 logits 中分数最高的类别索引作为预测类别，并与真实标签对应的类别一起映射为可读的类别名称
    print(f'Predicted: "{predicted}", Actual: "{actual}"') # 打印模型的预测类别和真实类别，用于直观检查单样本的预测效果