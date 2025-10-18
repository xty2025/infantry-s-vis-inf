'''共享卷积特征：在Fast R-CNN中，输入图像通过一次卷积神经网络生成共享的特征图。
之后，选择性搜索生成的候选框在特征图上进行池化操作，而不是对每个候选框单独进行卷积。这样可以显著提高计算效率。
RoI Pooling：Fast R-CNN使用了一种叫做**RoI Pooling（Region of Interest Pooling）**的技术,
它将不同大小的候选框统一池化为固定大小的特征图（例如7x7）。这使得模型能够高效地处理不同大小的区域，并减少了计算成本。
端到端训练：Fast R-CNN通过端到端的方式进行训练，不再需要在提取特征后使用独立的分类器和回归器。相反，分类和回归的损失直接在网络中共同优化。'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
# 1. 加载MNIST数据集并进行预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 生成边界框（相对坐标）
def generate_bboxes(num_samples):
    bboxes = []
    for _ in range(num_samples):
        offset = np.random.uniform(-0.05, 0.05, size=2)
        x1, y1 = max(0.05, 0.1 + offset[0]), max(0.05, 0.1 + offset[1])
        x2, y2 = min(0.95, 0.9 + offset[0]), min(0.95, 0.9 + offset[1])
        bboxes.append([x1, y1, x2, y2])
    return np.array(bboxes, dtype=np.float32)

train_bboxes = generate_bboxes(len(train_dataset))
test_bboxes = generate_bboxes(len(test_dataset))

# 2. RoI Pooling 层（PyTorch 实现）
class RoIPooling(nn.Module):
    def __init__(self, output_size=(7, 7)):
        super(RoIPooling, self).__init__()
        self.output_size = output_size

    def forward(self, features, rois):
        """
        features: 卷积特征图，形状为 (batch_size, C, H, W)。
        rois: 边界框，形状为 (batch_size, 4)，格式为 [x1, y1, x2, y2]（相对坐标）。
        """
        batch_size, channels, H, W = features.size()
        x1 = (rois[:, 0] * W).long()
        y1 = (rois[:, 1] * H).long()
        x2 = (rois[:, 2] * W).long()
        y2 = (rois[:, 3] * H).long()

        pooled_features = []
        for i in range(batch_size):
            roi_feature = features[i:i+1, :, y1[i]:y2[i]+1, x1[i]:x2[i]+1]
            pooled_roi = nn.functional.adaptive_max_pool2d(roi_feature, self.output_size)
            pooled_features.append(pooled_roi)
        
        return torch.cat(pooled_features, dim=0)
# 3. 构建 Fast R-CNN 模型
class FastRCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(FastRCNN, self).__init__()
        # 特征提取网络
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.roi_pool = RoIPooling(output_size=(7, 7))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True)
        )
        
        # 输出层
        self.cls_score = nn.Linear(512, num_classes)
        self.bbox_pred = nn.Linear(512, num_classes * 4)

    def forward(self, x, rois):
        features = self.features(x)
        roi_feats = self.roi_pool(features, rois)
        
        roi_feats_flat = roi_feats.view(roi_feats.size(0), -1)
        roi_feats = self.classifier(roi_feats_flat)
        
        cls_pred = self.cls_score(roi_feats)
        bbox_pred = self.bbox_pred(roi_feats)
        
        return cls_pred, bbox_pred

# 4. 定义 Fast R-CNN 损失函数
def fast_rcnn_loss(cls_pred, bbox_pred, cls_target, bbox_target):
    cls_loss = nn.CrossEntropyLoss()(cls_pred, cls_target)
    bbox_loss = nn.SmoothL1Loss()(bbox_pred, bbox_target)
    return cls_loss + bbox_loss

# 5. 训练模型
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        rois = torch.tensor(generate_bboxes(images.size(0)), dtype=torch.float32).to(device)

        optimizer.zero_grad()
        
        cls_pred, bbox_pred = model(images, rois)
        
        # 构造边界框目标
        bbox_target = rois  # 简单起见，边界框目标就是输入的边界框
        
        loss = fast_rcnn_loss(cls_pred, bbox_pred, labels, bbox_target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(cls_pred, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy

# 6. 测试模型
def test(model, test_loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            rois = torch.tensor(generate_bboxes(images.size(0)), dtype=torch.float32).to(device)

            cls_pred, bbox_pred = model(images, rois)
            bbox_target = rois
            
            loss = fast_rcnn_loss(cls_pred, bbox_pred, labels, bbox_target)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(cls_pred, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy

# 7. 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastRCNN(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, device)
        test_loss, test_acc = test(model, test_loader, device)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}% - "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # 保存模型
    torch.save(model.state_dict(), "fast_rcnn_mnist.pth")
    print("Model saved as 'fast_rcnn_mnist.pth'")

if __name__ == "__main__":
    main()
