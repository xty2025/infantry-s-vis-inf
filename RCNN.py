##生成多个候选框，用选择性搜索。
'''
分类和边界框回归：
对于每一个提取的特征向量，RCNN会使用一个全连接层进行分类，预测该候选框属于哪个类别（例如，人、车、猫等）。
同时，RCNN还使用一个回归网络来微调候选框的位置，以提高框的准确度。回归网络预测的目标是为每个候选框的四个边（x1, y1, x2, y2）提供一个精确的坐标值。
后处理（Non-Maximum Suppression，NMS）：
RCNN生成了多个候选框及其对应的分类得分。为了去除冗余的框，RCNN采用了非极大值抑制（NMS）算法。
NMS的目标是根据得分选择每个物体的最佳框，并删除与之重叠度较大的其他框
'''
import torch
import numpy as np
from PIL import Image
import albumentations as A
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class HybridAugmentation:
    """混合增强管道：结合torchvision.transforms和albumentations"""
    def __init__(self, is_train=True):
        # 1. albumentations增强（空间变换+边界框同步）
        self.albu_transform = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                p=0.6,
                border_mode=0,  # 填充黑色（MNIST背景）
                value=0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3)
        ], bbox_params=A.BboxParams(
            format='albumentations',  # (xmin, ymin, xmax, ymax) 归一化坐标
            label_fields=['class_labels']
        ))
        
        # 2. torchvision基础转换（标准化）
        self.torch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.1307, std=0.3081)  # MNIST官方均值/标准差
        ])
        
        self.is_train = is_train

    def __call__(self, img, bbox, label):
        # PIL → numpy（albumentations需要numpy数组）
        img_np = np.array(img, dtype=np.uint8)
        img_np = np.expand_dims(img_np, axis=-1)  # (28,28) → (28,28,1)
        
        if self.is_train:
            # 训练时应用增强
            augmented = self.albu_transform(
                image=img_np,
                bboxes=[bbox],
                class_labels=[label]
            )
            img_augmented = augmented['image']
            bbox_augmented = augmented['bboxes'][0] if augmented['bboxes'] else bbox
        else:
            # 测试时仅格式转换
            img_augmented = img_np
            bbox_augmented = bbox
        
        # numpy → PIL → 张量（适配torchvision）
        img_pil = Image.fromarray(img_augmented.squeeze(), mode='L')  # 单通道灰度图
        img_tensor = self.torch_transform(img_pil)
        
        return img_tensor, bbox_augmented


class MNISTHybridDataset(Dataset):
    def __init__(self, root='./data', train=True):
        self.mnist = MNIST(root=root, train=train, download=True, transform=None)
        self.augmenter = HybridAugmentation(is_train=train)  # 训练/测试区分增强

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]  # img: PIL.Image, label: 0-9
        initial_bbox = [0.05, 0.05, 0.95, 0.95]  # 初始边界框（避开图像边缘）
        
        # 应用增强，返回张量格式
        img_tensor, bbox = self.augmenter(img, initial_bbox, label)
        return (
            img_tensor,
            torch.tensor([bbox], dtype=torch.float32),  # (1,4)：单样本单RoI
            torch.tensor([label], dtype=torch.long)     # (1,)：单样本标签
        )


class RoIPooling(nn.Module):
    """简化版RoI Pooling：将任意RoI转为固定尺寸(7,7)"""
    def __init__(self, output_size=(7, 7)):
        super(RoIPooling, self).__init__()
        self.output_size = output_size
        
    def forward(self, features, rois):
        """
        features: 卷积特征图 (batch_size, channels, H, W)
        rois: 边界框 (batch_size, 1, 4) → 展平为 (batch_size, 4)
        """
        batch_size, channels, H, W = features.shape
        rois = rois.squeeze(1)  # 适配数据集输出的(bs,1,4) → (bs,4)
        
        # 相对坐标 → 绝对坐标（像素级）
        rois_abs = rois * torch.tensor([W, H, W, H], device=rois.device)
        rois_abs = rois_abs.long()  # 转为整数坐标
        
        pooled_features = []
        for i in range(batch_size):
            x1, y1, x2, y2 = rois_abs[i]
            # 确保边界不越界
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W - 1, x2)
            y2 = min(H - 1, y2)
            
            # 提取RoI特征并池化到固定尺寸
            roi_feat = features[i:i+1, :, y1:y2+1, x1:x2+1]  # (1, C, h_roi, w_roi)
            pooled = nn.functional.adaptive_max_pool2d(roi_feat, self.output_size)
            pooled_features.append(pooled)
        
        return torch.cat(pooled_features, dim=0)  # (batch_size, C, 7, 7)


class FastRCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(FastRCNN, self).__init__()
        # 特征提取网络（适配MNIST 28×28输入）
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # (1,28,28)→(32,28,28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # →(32,14,14)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # →(64,14,14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # →(64,7,7)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # →(128,7,7)
            nn.ReLU(inplace=True),
        )
        
        self.roi_pool = RoIPooling(output_size=(7, 7))  # RoI Pooling层
        
        # 分类器（全连接层）
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),  # 128×7×7=6272 → 1024
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(1024, 512),  # 1024 → 512
            nn.ReLU(inplace=True),
        )
        
        # 双输出头
        self.cls_score = nn.Linear(512, num_classes)  # 分类：512→10（0-9）
        self.bbox_pred = nn.Linear(512, num_classes * 4)  # 回归：512→40（10类×4坐标）
        
    def forward(self, x, rois):
        """
        x: 输入图像 (batch_size, 1, 28, 28)
        rois: 边界框 (batch_size, 1, 4)
        """
        # 1. 共享卷积提取特征
        features = self.features(x)  # (bs, 128, 7, 7)
        
        # 2. RoI Pooling
        roi_feats = self.roi_pool(features, rois)  # (bs, 128, 7, 7)
        
        # 3. 分类器特征压缩
        roi_feats_flat = roi_feats.view(roi_feats.size(0), -1)  # (bs, 6272)
        roi_feats = self.classifier(roi_feats_flat)  # (bs, 512)
        
        # 4. 分类与回归
        cls_pred = self.cls_score(roi_feats)  # (bs, 10)
        bbox_pred = self.bbox_pred(roi_feats)  # (bs, 40)
        
        return cls_pred, bbox_pred


class FastRCNNLoss(nn.Module):
    """Fast R-CNN多任务损失：分类损失 + 边界框回归损失"""
    def __init__(self, lambda_reg=1.0):
        super(FastRCNNLoss, self).__init__()
        self.cls_criterion = nn.CrossEntropyLoss()  # 分类损失（交叉熵）
        self.reg_criterion = nn.SmoothL1Loss(reduction='mean')  # 回归损失（Smooth L1）
        self.lambda_reg = lambda_reg  # 回归损失权重

    def forward(self, cls_pred, bbox_pred, cls_target, bbox_target):
        """
        cls_pred: (batch_size, 10) → 分类预测
        bbox_pred: (batch_size, 40) → 边界框预测
        cls_target: (batch_size, 1) → 分类目标（需展平为(batch_size,)）
        bbox_target: (batch_size, 1, 4) → 边界框目标（需展平为(batch_size,4)）
        """
        # 标签格式适配（展平为1维）
        cls_target_flat = cls_target.squeeze(1)  # (bs,1) → (bs,)
        bbox_target_flat = bbox_target.squeeze(1)  # (bs,1,4) → (bs,4)
        
        # 1. 分类损失
        cls_loss = self.cls_criterion(cls_pred, cls_target_flat)
        
        # 2. 边界框回归损失（MNIST无背景，所有样本都是正样本）
        batch_size = cls_pred.size(0)
        # 提取每个样本对应类别的边界框预测（cls_target_flat为类别索引）
        bbox_pred_idx = cls_target_flat.unsqueeze(1) * 4  # (bs,1) → 每个类别的起始索引
        bbox_pred_pos = torch.gather(
            bbox_pred, dim=1, 
            index=bbox_pred_idx.repeat(1, 4) + torch.arange(4, device=bbox_pred.device)
        )  # (bs,4)：每个样本对应类别的4个坐标
        
        # 计算回归损失
        reg_loss = self.reg_criterion(bbox_pred_pos, bbox_target_flat)
        
        # 3. 总损失
        total_loss = cls_loss + self.lambda_reg * reg_loss
        return total_loss, cls_loss, reg_loss


def train(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, rois, labels) in enumerate(train_loader):
        # 数据送设备（CPU/GPU）
        images = images.to(device)
        rois = rois.to(device)
        labels = labels.to(device)
        
        # 前向传播
        cls_pred, bbox_pred = model(images, rois)
        
        # 计算损失
        loss, cls_loss, reg_loss = criterion(cls_pred, bbox_pred, labels, rois)
        
        # 反向传播+优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计指标
        total_loss += loss.item() * images.size(0)  # 按样本数加权
        _, predicted = torch.max(cls_pred, 1)  # 取分类概率最大的类别
        total_correct += (predicted == labels.squeeze(1)).sum().item()
        total_samples += images.size(0)
        
        # 打印批次进度
        if (batch_idx + 1) % 100 == 0:
            avg_batch_loss = loss.item()
            batch_acc = (predicted == labels.squeeze(1)).sum().item() / images.size(0) * 100
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {avg_batch_loss:.4f}, Acc: {batch_acc:.2f}%, "
                  f"Cls Loss: {cls_loss:.4f}, Reg Loss: {reg_loss:.4f}")
    
    # 计算 epoch 级指标
    avg_epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples * 100
    print(f"\nEpoch [{epoch+1}/{num_epochs}] Training Summary: "
          f"Avg Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%\n")
    return avg_epoch_loss, epoch_acc


def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # 测试时禁用梯度计算
        for images, rois, labels in test_loader:
            images = images.to(device)
            rois = rois.to(device)
            labels = labels.to(device)
            
            cls_pred, bbox_pred = model(images, rois)
            loss, _, _ = criterion(cls_pred, bbox_pred, labels, rois)
            
            # 统计
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(cls_pred, 1)
            total_correct += (predicted == labels.squeeze(1)).sum().item()
            total_samples += images.size(0)
    
    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples * 100
    print(f"Test Set Summary: Avg Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%\n")
    return avg_loss, acc


if __name__ == '__main__':
    # 1. 超参数设置
    batch_size = 32
    learning_rate = 1e-4  # 1e-4 比 0.001 更稳定，避免梯度爆炸
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    # 2. 加载数据集
    train_dataset = MNISTHybridDataset(root='./data', train=True)
    test_dataset = MNISTHybridDataset(root='./data', train=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # 3. 初始化模型、损失函数、优化器
    model = FastRCNN(num_classes=10).to(device)
    criterion = FastRCNNLoss(lambda_reg=1.0)  # 回归损失权重设为1.0
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # 加L2正则防过拟合

    # 4. 记录训练过程指标
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    # 5. 训练循环
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        # 保存指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    # 6. 保存模型
    torch.save(model.state_dict(), 'fast_rcnn_mnist.pth')
    print("Model saved as 'fast_rcnn_mnist.pth'")

    # 7. 绘制训练曲线
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体支持
    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="训练损失")
    plt.plot(test_losses, label="测试损失")
    plt.title("损失曲线")
    plt.xlabel("Epoch")
    plt.ylabel("损失")
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="训练准确率")
    plt.plot(test_accs, label="测试准确率")
    plt.title("准确率曲线")
    plt.xlabel("Epoch")
    plt.ylabel("准确率")
    plt.legend()

    plt.tight_layout()
    plt.show()