"""
改进的训练脚本
针对性解决四川话识别问题
"""

import os
import math
import argparse
import sys

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import CrossEntropyLoss
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'train&predict'))
from model import efficientnet_b0, efficientnet_b3, efficientnet_b4
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """保存混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('方言识别混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def evaluate_detailed(model, data_loader, device, class_names):
    """详细评估，返回混淆矩阵数据"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 计算总体准确率
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # 每个类别的准确率
    class_accuracies = {}
    for i, class_name in enumerate(class_names):
        mask = np.array(all_labels) == i
        if mask.sum() > 0:
            class_acc = (np.array(all_preds)[mask] == i).sum() / mask.sum()
            class_accuracies[class_name] = class_acc
    
    return accuracy, all_labels, all_preds, class_accuracies


def main(args):
    # 智能设备选择：优先MPS（Mac M系列芯片）> CUDA（NVIDIA GPU）> CPU
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用设备: MPS (Apple Silicon GPU加速)")
    elif 'cuda' in args.device and torch.cuda.is_available():
        device = torch.device(args.device)
        print(f"使用设备: {device}")
    else:
        device = torch.device("cpu")
        print("使用设备: CPU")
    
    # 创建输出目录
    os.makedirs("./weights", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    # TensorBoard
    tb_writer = SummaryWriter(log_dir=f"./runs/{args.exp_name}")
    
    # 读取数据
    print("读取数据集...")
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    
    # 类别名称
    class_names = ['changsha', 'hefei', 'kejia', 'nanchang', 'ningxia', 'shan3xi', 'sichuan']
    chinese_names = ['长沙话', '合肥话', '客家话', '南昌话', '宁夏话', '陕西话', '四川话']
    
    # 选择模型和输入尺寸
    img_size_dict = {"B0": 224, "B1": 240, "B2": 260, "B3": 300, "B4": 380}
    img_size = img_size_dict[args.model]
    
    # 数据增强（更强的增强）
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # 创建数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])
    
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    
    # 数据加载器
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'使用 {nw} 个dataloader workers')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn
    )
    
    # 创建模型
    print(f"创建模型: EfficientNet-{args.model}")
    if args.model == "B0":
        model = efficientnet_b0(num_classes=args.num_classes).to(device)
    elif args.model == "B3":
        model = efficientnet_b3(num_classes=args.num_classes).to(device)
    elif args.model == "B4":
        model = efficientnet_b4(num_classes=args.num_classes).to(device)
    else:
        raise ValueError(f"不支持的模型: {args.model}")
    
    # 加载预训练权重
    if args.weights != "":
        if os.path.exists(args.weights):
            print(f"加载预训练权重: {args.weights}")
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items() if 'classifier' not in k}
            model.load_state_dict(load_weights_dict, strict=False)
        else:
            print(f"警告: 找不到权重文件 {args.weights}，从头训练")
    
    # 冻结层（可选）
    if args.freeze_layers:
        print("冻结底层特征提取层")
        for name, para in model.named_parameters():
            if ("features.top" not in name) and ("classifier" not in name):
                para.requires_grad_(False)
    
    # 优化器
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    if args.scheduler == 'cosine':
        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 损失函数（带类别权重）
    if args.use_class_weights:
        # 计算类别权重
        class_counts = np.bincount(train_images_label)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = CrossEntropyLoss(weight=class_weights)
        print(f"使用类别权重: {class_weights.cpu().numpy()}")
    else:
        criterion = CrossEntropyLoss()
    
    # 训练循环
    best_acc = 0.0
    best_epoch = 0
    
    print("\n" + "="*60)
    print("开始训练!")
    print("="*60)
    
    for epoch in range(args.epochs):
        # 训练
        mean_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch
        )
        
        scheduler.step()
        
        # 验证
        if (epoch + 1) % args.eval_interval == 0:
            acc, true_labels, pred_labels, class_accs = evaluate_detailed(
                model, val_loader, device, class_names
            )
            
            print(f"\n[Epoch {epoch+1}] 总体准确率: {acc:.4f}")
            print("各类别准确率:")
            for class_name, chinese_name, class_acc in zip(class_names, chinese_names, class_accs.values()):
                print(f"  {chinese_name}({class_name}): {class_acc:.2%}")
            
            # 记录到TensorBoard
            tb_writer.add_scalar('train/loss', mean_loss, epoch)
            tb_writer.add_scalar('val/accuracy', acc, epoch)
            tb_writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], epoch)
            
            for i, (class_name, class_acc) in enumerate(class_accs.items()):
                tb_writer.add_scalar(f'val/acc_{class_name}', class_acc, epoch)
            
            # 保存最佳模型
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                torch.save(model.state_dict(), f"./weights/best_model_{args.exp_name}.pth")
                print(f"✓ 保存最佳模型 (准确率: {acc:.4f})")
                
                # 保存混淆矩阵
                save_confusion_matrix(
                    true_labels, pred_labels, chinese_names,
                    f"./results/confusion_matrix_epoch{epoch+1}.png"
                )
        
        # 定期保存
        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), f"./weights/model_{args.exp_name}_epoch{epoch+1}.pth")
    
    # 最终评估
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print(f"最佳准确率: {best_acc:.4f} (Epoch {best_epoch+1})")
    
    # 加载最佳模型并详细评估
    model.load_state_dict(torch.load(f"./weights/best_model_{args.exp_name}.pth"))
    acc, true_labels, pred_labels, class_accs = evaluate_detailed(
        model, val_loader, device, class_names
    )
    
    print("\n最终详细评估:")
    print(f"总体准确率: {acc:.4f}")
    print("\n各类别准确率:")
    for class_name, chinese_name, class_acc in zip(class_names, chinese_names, class_accs.values()):
        print(f"  {chinese_name}({class_name}): {class_acc:.2%}")
    
    # 生成分类报告
    report = classification_report(
        true_labels, pred_labels,
        target_names=chinese_names,
        digits=4
    )
    print("\n分类报告:")
    print(report)
    
    # 保存报告
    with open(f"./results/classification_report_{args.exp_name}.txt", 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 最终混淆矩阵
    save_confusion_matrix(
        true_labels, pred_labels, chinese_names,
        f"./results/confusion_matrix_final_{args.exp_name}.png"
    )
    
    tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 数据相关
    parser.add_argument('--data_path', type=str, 
                       default='./processed_data',
                       help='数据集路径')
    parser.add_argument('--num_classes', type=int, default=10)
    
    # 模型相关
    parser.add_argument('--model', type=str, default='B3',
                       choices=['B0', 'B3', 'B4'],
                       help='EfficientNet模型版本')
    parser.add_argument('--weights', type=str, default='',
                       help='预训练权重路径')
    parser.add_argument('--freeze_layers', action='store_true',
                       help='是否冻结底层')
    
    # 训练相关
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step'])
    parser.add_argument('--use_class_weights', action='store_true',
                       help='使用类别权重平衡')
    
    # 其他
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--exp_name', type=str, default='improved_training')
    parser.add_argument('--eval_interval', type=int, default=2,
                       help='评估间隔（epoch）')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='保存间隔（epoch）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("方言识别改进训练")
    print("="*60)
    print(f"数据路径: {args.data_path}")
    print(f"模型: EfficientNet-{args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"类别权重: {args.use_class_weights}")
    print("="*60 + "\n")
    
    main(args)

