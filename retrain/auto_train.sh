#!/bin/bash

# 自动训练脚本：等待数据预处理完成后自动开始训练

echo "=================================================="
echo "自动训练监控脚本"
echo "=================================================="
echo ""

# 等待数据预处理完成
echo "等待数据预处理完成..."
while pgrep -f "prepare_data.py" > /dev/null; do
    # 显示进度
    if [ -d "./processed_data" ]; then
        total_images=$(find ./processed_data -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
        progress=$((total_images * 100 / 65000))
        echo -ne "\r进度: $total_images/65000 ($progress%)   "
    fi
    sleep 10
done

echo ""
echo ""
echo "✓ 数据预处理完成！"
echo ""

# 检查数据
total_images=$(find ./processed_data -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
echo "生成的声谱图总数: $total_images"
echo ""

if [ $total_images -lt 50000 ]; then
    echo "⚠️ 警告: 声谱图数量少于预期，请检查数据预处理日志"
    echo "日志文件: data_preprocessing.log"
    exit 1
fi

echo "=================================================="
echo "开始模型训练"
echo "=================================================="
echo ""
echo "配置: M3优化 (EfficientNet-B3, 50轮)"
echo "预计时间: 8-12小时"
echo ""

# 等待5秒让用户看到信息
sleep 5

# 启动训练
export PYTORCH_ENABLE_MPS_FALLBACK=1

nohup python3 train_improved.py \
    --data_path ./processed_data \
    --model B3 \
    --epochs 50 \
    --batch_size 24 \
    --lr 0.0005 \
    --exp_name m3_optimized_training \
    --use_class_weights \
    --device mps \
    --eval_interval 2 \
    --save_interval 10 \
    > training.log 2>&1 &

TRAIN_PID=$!

echo "✓ 训练已启动！"
echo "进程ID: $TRAIN_PID"
echo "日志文件: training.log"
echo ""
echo "查看训练进度:"
echo "  tail -f training.log"
echo "  ./check_progress.sh"
echo ""
echo "TensorBoard:"
echo "  tensorboard --logdir=./runs/m3_optimized_training"
echo ""
echo "=================================================="


