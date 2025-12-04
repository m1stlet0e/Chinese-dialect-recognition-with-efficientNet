# 🚀 A800服务器快速开始

## 当前情况
- ✅ 服务器: 8x A800 80GB (172.22.0.35)
- ✅ PyTorch: 2.4.1+cu121
- ✅ CUDA: 可用
- ⏸️ 数据: 待上传
- ⏸️ 训练: 待开始

---

## 📋 完整流程（清晰版）

### 🔹 阶段1：在Mac上上传数据

**打开Mac终端**（不是服务器），运行：

```bash
# 1. 上传代码
cd /Users/wangbo/PycharmProjects/Chinese-dialect-recognition-with-efficientNet

rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '*.log' \
    retrain/ \
    wangbo@172.22.0.35:~/dialect_training/retrain/

rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    train\&predict/ \
    wangbo@172.22.0.35:~/dialect_training/train_predict/

# 2. 上传处理好的数据（重要！约20GB，需要5-10分钟）
rsync -avz --progress \
    retrain/processed_data/ \
    wangbo@172.22.0.35:~/dialect_training/retrain/processed_data/
```

---

### 🔹 阶段2：在服务器上检查

**SSH到服务器**：
```bash
ssh wangbo@172.22.0.35
```

**检查数据是否上传成功**：
```bash
# 检查数据
ls ~/dialect_training/retrain/processed_data/
# 应该看到：changsha  hebei  hefei  kejia  minnan  nanchang  ningxia  shan3xi  shanghai  sichuan

# 统计文件数
find ~/dialect_training/retrain/processed_data/ -name "*.png" | wc -l
# 应该显示：65000
```

---

### 🔹 阶段3：在服务器上安装依赖

**还在服务器上**，运行：

```bash
cd ~/dialect_training/retrain

# 安装Python依赖（只需要一次）
pip3 install tensorboard seaborn scikit-learn tqdm pillow scipy numpy pandas matplotlib
```

---

### 🔹 阶段4：在服务器上启动训练

**推荐配置（B4模型，256 batch size）**：

```bash
cd ~/dialect_training/retrain

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

echo "✓ 训练已启动！"
echo "进程ID: $!"
```

**查看训练进度**：
```bash
tail -f training.log
# 按 Ctrl+C 退出查看

# 监控GPU
watch -n 1 nvidia-smi
```

---

### 🔹 阶段5：等待训练完成（3-4小时）

**可以关闭终端，训练会继续运行**

随时可以SSH回来查看：
```bash
ssh wangbo@172.22.0.35
cd ~/dialect_training/retrain
tail -50 training.log
```

---

### 🔹 阶段6：训练完成后下载模型

**在Mac终端上**（不是服务器）运行：

```bash
# 下载最佳模型
scp wangbo@172.22.0.35:~/dialect_training/retrain/weights/best_model_a800_b4_training.pth \
    /Users/wangbo/PycharmProjects/Chinese-dialect-recognition-with-efficientNet/GUI/weight/model-29.pth

echo "✓ 新模型已下载并替换旧模型"
```

---

## 🔍 常用检查命令

### 在服务器上检查训练状态

```bash
# 检查是否在运行
ps aux | grep train_improved

# 查看最新进度
tail -30 ~/dialect_training/retrain/training.log

# 查看GPU使用
nvidia-smi

# 查看训练了多少轮
grep "Epoch.*完成" ~/dialect_training/retrain/training.log | tail -5
```

### 在Mac上检查本地数据

```bash
# 检查处理好的数据是否存在
ls /Users/wangbo/PycharmProjects/Chinese-dialect-recognition-with-efficientNet/retrain/processed_data/

# 统计数量
find /Users/wangbo/PycharmProjects/Chinese-dialect-recognition-with-efficientNet/retrain/processed_data/ -name "*.png" | wc -l
```

---

## 📊 预计时间表

| 步骤 | 时间 | 位置 |
|------|------|------|
| 1. 上传数据 | 5-10分钟 | Mac |
| 2. 安装依赖 | 2分钟 | 服务器 |
| 3. 启动训练 | 1分钟 | 服务器 |
| 4. 训练运行 | **3-4小时** | 服务器（自动） |
| 5. 下载模型 | 1分钟 | Mac |

**总计：3-4小时（大部分时间无需人工干预）**

---

## ⚠️ 注意事项

### 区分Mac和服务器

```
Mac终端提示符：
wangbo@wangbodeMacBook-Pro-7 ~ %

服务器提示符：
wangbo@m7-2-5-a1-7-29U-AI:~$
```

### 命令执行位置

| 命令 | 在哪里执行 |
|------|-----------|
| `rsync ... wangbo@172.22.0.35:...` | **Mac** |
| `scp wangbo@172.22.0.35:... /Users/...` | **Mac** |
| `python3 train_improved.py` | **服务器** |
| `nvidia-smi` | **服务器** |
| `tail -f training.log` | **服务器** |

---

## 🎯 当前你需要做的

### 如果在服务器上：

1. **先退出到Mac**：
   ```bash
   exit  # 或按 Ctrl+D
   ```

2. **在Mac上上传数据**（见阶段1）

3. **再SSH回服务器启动训练**

### 如果在Mac上：

直接开始阶段1的上传命令

---

## 💡 快速命令参考

```bash
# === 在Mac上 ===
# 上传数据
rsync -avz --progress retrain/processed_data/ wangbo@172.22.0.35:~/dialect_training/retrain/processed_data/

# === SSH到服务器 ===
ssh wangbo@172.22.0.35

# === 在服务器上 ===
# 启动训练
cd ~/dialect_training/retrain
CUDA_VISIBLE_DEVICES=0 nohup python3 train_improved.py \
    --data_path ./processed_data \
    --model B4 \
    --epochs 50 \
    --batch_size 256 \
    --lr 0.001 \
    --exp_name a800_b4 \
    --use_class_weights \
    --device cuda:0 \
    > training.log 2>&1 &

# 查看进度
tail -f training.log

# === 训练完成后，回到Mac ===
exit

# === 在Mac上下载模型 ===
scp wangbo@172.22.0.35:~/dialect_training/retrain/weights/best_model_a800_b4.pth \
    /Users/wangbo/PycharmProjects/Chinese-dialect-recognition-with-efficientNet/GUI/weight/model-29.pth
```

---

需要帮助？告诉我你现在在哪一步！


