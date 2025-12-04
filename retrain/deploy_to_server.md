# 服务器训练部署指南

## 🖥️ 第一步：检查服务器配置

**在服务器上运行这些命令：**

```bash
# 1. 检查GPU
nvidia-smi

# 2. 检查Python和PyTorch
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"

# 3. 检查磁盘空间（需要至少30GB）
df -h ~
```

## 📦 第二步：上传数据和代码

### 方案A：使用scp上传（推荐）

**在你的Mac上运行：**

```bash
# 1. 打包项目
cd /Users/wangbo/PycharmProjects/Chinese-dialect-recognition-with-efficientNet
tar -czf dialect_training.tar.gz retrain/ train\&predict/

# 2. 上传到服务器
scp dialect_training.tar.gz wangbo@172.22.0.35:~/

# 3. 上传数据（如果服务器上没有）
cd /Users/wangbo/Desktop
tar -czf origin_data.tar.gz origin_data/
scp origin_data.tar.gz wangbo@172.22.0.35:~/
```

### 方案B：使用rsync（更快，支持断点续传）

```bash
# 上传项目
rsync -avz --progress /Users/wangbo/PycharmProjects/Chinese-dialect-recognition-with-efficientNet/retrain/ \
    wangbo@172.22.0.35:~/dialect_training/retrain/

rsync -avz --progress /Users/wangbo/PycharmProjects/Chinese-dialect-recognition-with-efficientNet/train\&predict/ \
    wangbo@172.22.0.35:~/dialect_training/train_predict/

# 上传数据
rsync -avz --progress /Users/wangbo/Desktop/origin_data/ \
    wangbo@172.22.0.35:~/origin_data/
```

## 🚀 第三步：在服务器上运行训练

**SSH到服务器后：**

```bash
# 1. 解压（如果使用tar包）
tar -xzf dialect_training.tar.gz
tar -xzf origin_data.tar.gz

# 2. 安装依赖
pip3 install torch torchvision tensorboard seaborn scikit-learn tqdm pillow scipy numpy

# 3. 检查processed_data是否已存在
ls -lh ~/dialect_training/retrain/processed_data/ 2>/dev/null

# 如果没有，需要上传Mac上已处理好的数据：
# rsync -avz --progress /Users/wangbo/.../retrain/processed_data/ \
#     wangbo@172.22.0.35:~/dialect_training/retrain/processed_data/

# 4. 开始训练
cd ~/dialect_training/retrain
nohup python3 train_improved.py \
    --data_path ./processed_data \
    --model B3 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0005 \
    --exp_name server_training \
    --use_class_weights \
    --device cuda:0 \
    > training.log 2>&1 &

# 5. 查看进度
tail -f training.log

# 6. 监控GPU使用
watch -n 1 nvidia-smi
```

## ⚡ 性能对比预估

| 设备 | 类型 | 预计速度 | 完成时间 |
|------|------|----------|----------|
| Mac M3 | MPS | 1.4秒/batch | ~48小时 |
| **服务器 (假设V100)** | **CUDA** | **0.3-0.5秒/batch** | **12-20小时** ⚡ |
| **服务器 (假设A100)** | **CUDA** | **0.2-0.3秒/batch** | **8-12小时** ⚡⚡ |

## 📊 训练监控

### 实时查看日志
```bash
# 查看最新日志
tail -f ~/dialect_training/retrain/training.log

# 查看最后100行
tail -100 ~/dialect_training/retrain/training.log

# 搜索特定epoch的结果
grep "Epoch.*完成" ~/dialect_training/retrain/training.log
```

### 监控GPU
```bash
# 实时GPU使用
watch -n 1 nvidia-smi

# 或者记录到文件
while true; do nvidia-smi >> gpu_monitor.log; sleep 60; done &
```

### 使用TensorBoard（可选）
```bash
# 在服务器上启动TensorBoard
tensorboard --logdir=~/dialect_training/retrain/runs/ --host=0.0.0.0 --port=6006 &

# 然后在Mac上通过SSH隧道访问
ssh -L 6006:localhost:6006 wangbo@172.22.0.35

# 浏览器访问: http://localhost:6006
```

## 📥 训练完成后下载模型

```bash
# 在Mac上运行
scp wangbo@172.22.0.35:~/dialect_training/retrain/weights/best_model_server_training.pth \
    /Users/wangbo/PycharmProjects/Chinese-dialect-recognition-with-efficientNet/GUI/weight/model-29.pth
```

## 🔧 常见问题

### 如果PyTorch未安装或不支持CUDA

```bash
# 查看CUDA版本
nvcc --version

# 安装对应版本的PyTorch（以CUDA 11.8为例）
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 如果磁盘空间不足

```bash
# 检查空间
du -sh ~/dialect_training/retrain/processed_data/

# 如果空间不够，可以只上传必要的数据
# 或者清理不需要的文件
```

### 如果训练中断

```bash
# 检查是否还在运行
ps aux | grep train_improved

# 重新启动（会从头开始）
cd ~/dialect_training/retrain
nohup python3 train_improved.py [参数...] > training.log 2>&1 &
```

## 🎯 推荐配置

### 标准配置（服务器有8GB+ GPU显存）
```bash
--batch_size 32    # 比Mac的24大
--model B3         # 或B4如果显存够
--epochs 50
```

### 高性能配置（服务器有16GB+ GPU显存）
```bash
--batch_size 64    # 更大的batch size
--model B4         # 更大的模型
--epochs 80        # 更多轮次
```

## 📞 需要帮助？

如果遇到问题，发送这些信息：
1. `nvidia-smi` 输出
2. `tail -100 training.log` 输出
3. 错误信息截图

---

**祝训练顺利！服务器GPU训练会比Mac快3-5倍！** 🚀



