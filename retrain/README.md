# 重新训练方言识别模型

## 📋 目的

解决当前模型将四川话误识别为客家话的问题，通过使用完整数据集（65,000个音频）重新训练，提高识别准确率。

## 🗂️ 数据集信息

```
位置: /Users/wangbo/Desktop/origin_data
格式: PCM音频文件
类别: 10个方言
  - changsha (长沙话)
  - hebei (河北话)
  - hefei (合肥话)
  - kejia (客家话)
  - minnan (闽南话)
  - nanchang (南昌话)
  - ningxia (宁夏话)
  - shan3xi (陕西话)
  - shanghai (上海话)
  - sichuan (四川话)

每类: 6500个文件
总计: 65,000个音频文件
结构: train/ 和 dev/ 目录
```

## 🚀 快速开始（推荐）

### 一键运行完整流程

```bash
cd /Users/wangbo/PycharmProjects/Chinese-dialect-recognition-with-efficientNet/retrain

# 添加执行权限
chmod +x full_pipeline.sh

# 运行完整流程
./full_pipeline.sh
```

脚本会引导你完成：
1. 数据预处理（PCM → WAV → 声谱图 → RGB）
2. 模型训练（可选不同配置）
3. 结果保存和评估

## 📝 分步执行

如果你想分步控制，可以手动执行：

### 步骤1: 数据预处理

```bash
python3 prepare_data.py \
    --input_dir /Users/wangbo/Desktop/origin_data \
    --output_dir ./processed_data
```

**预计时间**: 1-2小时（处理65,000个文件）

**输出**: `processed_data/` 目录，包含10个方言文件夹，每个包含RGB声谱图

### 步骤2: 模型训练

#### 配置1: 快速测试（10分钟）
```bash
python3 train_improved.py \
    --data_path ./processed_data \
    --model B0 \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.001 \
    --exp_name quick_test
```

#### 配置2: 标准训练（推荐，2-3小时）
```bash
python3 train_improved.py \
    --data_path ./processed_data \
    --model B3 \
    --epochs 30 \
    --batch_size 16 \
    --lr 0.001 \
    --exp_name standard_training \
    --use_class_weights
```

#### 配置3: 完整训练（最佳效果，4-6小时）
```bash
python3 train_improved.py \
    --data_path ./processed_data \
    --model B3 \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.0005 \
    --exp_name full_training \
    --use_class_weights
```

## 📊 训练监控

### 使用TensorBoard查看训练过程

```bash
tensorboard --logdir=./runs/full_training
```

然后访问: http://localhost:6006

可以看到：
- 训练损失曲线
- 验证准确率曲线
- 每个方言的准确率
- 学习率变化

## 📈 查看结果

训练完成后，会生成：

```
weights/
  └── best_model_full_training.pth  # 最佳模型

results/
  ├── confusion_matrix_final_full_training.png  # 混淆矩阵
  └── classification_report_full_training.txt   # 详细报告

runs/
  └── full_training/  # TensorBoard日志
```

### 混淆矩阵

混淆矩阵会显示哪些方言容易混淆，特别关注：
- 四川话 vs 客家话
- 其他容易混淆的方言对

### 分类报告

包含每个方言的：
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数
- 支持样本数

## 🔧 调试和优化

### 如果训练很慢

```bash
# 1. 减小batch_size
--batch_size 8

# 2. 使用更小的模型
--model B0

# 3. 减少epochs
--epochs 20
```

### 如果内存不足

```bash
# 减小batch_size
--batch_size 4

# 或使用B0模型
--model B0
```

### 如果想使用GPU

```bash
# 确保PyTorch支持CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# 训练时指定GPU
--device cuda:0
```

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `prepare_data.py` | 数据预处理脚本 |
| `train_improved.py` | 改进的训练脚本 |
| `full_pipeline.sh` | 一键运行脚本 |
| `README.md` | 本文件 |

## 🎯 预期改进

| 指标 | 当前 | 目标 |
|------|------|------|
| 总体准确率 | 78% | 85-92% |
| 四川话准确率 | ~0% (误判) | >85% |
| 数据量 | 250张/类 | 5000+张/类 |
| 训练轮数 | 30 | 50 |
| 模型 | B4 | B3/B4 |

## ⚠️ 注意事项

### 数据预处理
- 第一次运行需要1-2小时处理所有音频
- 需要约20GB磁盘空间存储声谱图
- 建议在空闲时间运行

### 训练过程
- 完整训练需要4-6小时（CPU）或1-2小时（GPU）
- 建议使用完整训练配置获得最佳效果
- 训练过程中可以通过TensorBoard实时监控

### 中断恢复
训练会定期保存模型，如果中断：
```bash
# 可以从最后的checkpoint继续
--weights ./weights/model_full_training_epoch40.pth
```

## 🐛 常见问题

### Q: 数据预处理失败？
```bash
# 检查输入目录
ls /Users/wangbo/Desktop/origin_data/

# 检查Python依赖
pip3 install numpy scipy pillow tqdm
```

### Q: 训练时OOM（内存不足）？
```bash
# 减小batch size
--batch_size 4

# 或使用更小的模型
--model B0
```

### Q: 如何对比新旧模型？
```bash
# 测试旧模型
python3 ../API/test_client.py your_audio.wav

# 替换API中的模型文件后测试新模型
cp weights/best_model_full_training.pth ../GUI/weight/model-29.pth
```

## 📞 下一步

训练完成后：

1. **评估新模型**
   ```bash
   cd ../API
   cp ../retrain/weights/best_model_full_training.pth ../GUI/weight/model-29.pth
   # 重启API服务
   python3 api_server.py
   ```

2. **测试四川话音频**
   ```bash
   curl -X POST -F "file=@sichuan_audio.wav" http://localhost:8000/api/predict
   ```

3. **对比结果**
   - 查看四川话识别准确率
   - 检查混淆矩阵
   - 评估总体性能提升

## 💡 进一步优化

如果结果仍不理想，可以尝试：

1. **数据增强** - 已内置在`train_improved.py`
2. **更大的模型** - 使用B4或B6
3. **更多训练轮数** - 增加到80-100 epochs
4. **集成学习** - 训练多个模型并投票
5. **调整类别权重** - 已通过`--use_class_weights`实现

## 📚 相关文档

- 原始训练代码: `../train&predict/`
- GUI应用: `../GUI/`
- API服务: `../API/`
- 改进建议: `../改进建议.md`

---

**祝训练顺利！如有问题随时查看本文档或日志文件。**

