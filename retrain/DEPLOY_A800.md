# 🚀 A800服务器部署指南（超快版）

## 你的服务器配置

```
🔥 8块 NVIDIA A800 80GB GPU
💾 640GB 总显存
⚡ CUDA 12.2
✅ PyTorch 2.4.1 已安装
💿 1.2TB 可用空间
```

**这是顶配AI训练服务器！训练速度比Mac M3快10-20倍！**

---

## 🎯 快速部署（3步，10分钟）

### 步骤1：上传已处理的数据（在Mac上）

```bash
# 方法A：使用rsync（推荐，支持断点续传）
cd /Users/wangbo/PycharmProjects/Chinese-dialect-recognition-with-efficientNet

# 上传代码
rsync -avz --progress retrain/ wangbo@172.22.0.35:~/dialect_training/retrain/
rsync -avz --progress train\&predict/ wangbo@172.22.0.35:~/dialect_training/train_predict/

# 上传已处理好的数据（~20GB，需要5-10分钟）
rsync -avz --progress retrain/processed_data/ \
    wangbo@172.22.0.35:~/dialect_training/retrain/processed_data/
```

### 步骤2：SSH到服务器并安装依赖

```bash
ssh wangbo@172.22.0.35
cd ~/dialect_training/retrain

# 安装依赖（只需要一次）
pip3 install tensorboard seaborn scikit-learn tqdm pillow scipy numpy pandas matplotlib
```

### 步骤3：启动训练🚀

```bash
# 推荐配置：单GPU B4模型
CUDA_VISIBLE_DEVICES=0 nohup python3 train_improved.py \
    --data_path ./processed_data \
    --model B4 \
    --epochs 50 \
    --batch_size 256 \
    --lr 0.001 \
    --exp_name a800_training \
    --use_class_weights \
    --device cuda:0 \
    > training.log 2>&1 &

echo "训练已启动！"
echo "查看进度: tail -f training.log"
```

---

## 📊 性能对比

| 配置 | 设备 | Batch Size | 速度 | 完成时间 |
|------|------|-----------|------|---------|
| Mac | M3 | 24 | 1.4s/batch | 48小时 |
| **服务器单GPU** | **A800** | **256** | **~0.2s/batch** | **3-4小时** ⚡ |
| **服务器2GPU** | **A800 x2** | **256** | **~0.1s/batch** | **1.5-2小时** ⚡⚡ |

---

## 🔍 训练监控

### 实时查看进度

```bash
# 查看最新日志
tail -f ~/dialect_training/retrain/training.log

# 监控GPU使用
watch -n 1 nvidia-smi

# 查看指定GPU
watch -n 1 "nvidia-smi | grep -A 5 'GPU  0'"
```

### 查看训练统计

```bash
# 查看epoch完成情况
grep "Epoch" ~/dialect_training/retrain/training.log | tail -20

# 查看loss变化
grep "mean loss" ~/dialect_training/retrain/training.log | tail -50
```

---

## 💡 推荐配置

### 🥇 推荐：单GPU B4（最佳平衡）

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_improved.py \
    --data_path ./processed_data \
    --model B4 \
    --epochs 50 \
    --batch_size 256 \
    --lr 0.001 \
    --use_class_weights \
    --device cuda:0
```

**优点**：
- 充分利用80GB显存
- 速度快（3-4小时）
- 不影响其他用户
- B4模型精度高

### 🥈 备选：单GPU B3（更快）

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_improved.py \
    --data_path ./processed_data \
    --model B3 \
    --epochs 50 \
    --batch_size 128 \
    --lr 0.001 \
    --use_class_weights \
    --device cuda:0
```

**优点**：
- 更快（2-3小时）
- 显存占用少
- 精度也不错

---

## 📥 下载训练好的模型

### 在Mac上运行：

```bash
# 下载最佳模型
scp wangbo@172.22.0.35:~/dialect_training/retrain/weights/best_model_a800_training.pth \
    /Users/wangbo/PycharmProjects/Chinese-dialect-recognition-with-efficientNet/GUI/weight/model-29.pth

# 或者下载所有模型
scp wangbo@172.22.0.35:~/dialect_training/retrain/weights/*.pth \
    /Users/wangbo/PycharmProjects/Chinese-dialect-recognition-with-efficientNet/retrain/weights/
```

---

## 🎯 完整流程示例

```bash
# === 在Mac上 ===
cd /Users/wangbo/PycharmProjects/Chinese-dialect-recognition-with-efficientNet

# 上传数据和代码（一次性）
rsync -avz --progress retrain/ wangbo@172.22.0.35:~/dialect_training/retrain/
rsync -avz --progress train\&predict/ wangbo@172.22.0.35:~/dialect_training/train_predict/

# === SSH到服务器 ===
ssh wangbo@172.22.0.35

# 安装依赖（首次）
cd ~/dialect_training/retrain
pip3 install tensorboard seaborn scikit-learn tqdm pillow scipy numpy pandas matplotlib

# 启动训练
CUDA_VISIBLE_DEVICES=0 nohup python3 train_improved.py \
    --data_path ./processed_data \
    --model B4 \
    --epochs 50 \
    --batch_size 256 \
    --lr 0.001 \
    --exp_name a800_b4_training \
    --use_class_weights \
    --device cuda:0 \
    > training.log 2>&1 &

# 查看进度
tail -f training.log

# 监控GPU（另开一个终端）
watch -n 1 nvidia-smi

# === 3-4小时后，在Mac上下载模型 ===
scp wangbo@172.22.0.35:~/dialect_training/retrain/weights/best_model_a800_b4_training.pth \
    /Users/wangbo/PycharmProjects/Chinese-dialect-recognition-with-efficientNet/GUI/weight/model-29.pth
```

---

## 🔧 常见问题

### Q: 如何检查训练是否在运行？

```bash
ps aux | grep train_improved
nvidia-smi  # 查看GPU使用率
```

### Q: 如何停止训练？

```bash
pkill -f train_improved.py
```

### Q: 如何使用多GPU？

```bash
# 使用2块GPU
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
    --nproc_per_node=2 \
    train_improved.py \
    --data_path ./processed_data \
    --model B4 \
    --batch_size 256 \
    --device cuda \
    [其他参数...]
```

### Q: 显存不足怎么办？

```bash
# 减小batch size
--batch_size 128  # 或 64

# 使用更小的模型
--model B3  # 或 B0
```

---

## 📈 预期结果

### 训练完成后应该看到：

```
Epoch 49/50 完成
验证准确率: 88-92%
训练损失: < 0.5
最佳模型已保存: weights/best_model_a800_b4_training.pth

各方言准确率:
  四川话: 90%+  ← 主要目标
  客家话: 88%+
  ...
```

### 改进对比：

| 指标 | 旧模型 | 新模型（预期）|
|------|--------|--------------|
| 四川话准确率 | ~0% | 90%+ ⬆️⬆️⬆️ |
| 总体准确率 | 78% | 88-92% ⬆️ |
| 数据量 | 250张/类 | 6500张/类 |

---

## 🎊 总结

### 你的优势：
1. ⚡ **A800 GPU** - 顶级训练卡
2. 💾 **80GB显存** - 可以用超大batch size
3. 🚀 **速度快10-20倍** - 3-4小时vs 48小时
4. 📊 **数据充足** - 65,000张声谱图

### 推荐行动：
1. ✅ 立即上传数据到服务器
2. ✅ 使用单GPU B4配置训练
3. ✅ 3-4小时后下载新模型
4. ✅ 测试四川话识别效果

**预计今晚就能完成训练！** 🎉

---

需要帮助？提供这些信息：
- `nvidia-smi` 输出
- `tail -100 training.log`
- 错误信息


