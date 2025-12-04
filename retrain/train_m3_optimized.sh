#!/bin/bash

# M3芯片优化的训练脚本
# 充分利用Apple Silicon的性能

echo "=========================================="
echo "M3芯片优化训练配置"
echo "=========================================="
echo "CPU: Apple M3 (8核)"
echo "GPU: M3 GPU (10核)"
echo "内存: 16GB"
echo "=========================================="
echo ""

# 推荐配置
MODEL="B3"
EPOCHS=50
BATCH_SIZE=24  # M3可以用更大的batch size
LR=0.0005
EXP_NAME="m3_optimized_training"

echo "训练配置:"
echo "  模型: EfficientNet-$MODEL"
echo "  训练轮数: $EPOCHS"
echo "  批次大小: $BATCH_SIZE (针对M3优化)"
echo "  学习率: $LR"
echo ""

# 设置MPS后端（Metal Performance Shaders）
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 开始训练
python3 train_improved.py \
    --data_path ./processed_data \
    --model "$MODEL" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --exp_name "$EXP_NAME" \
    --use_class_weights \
    --device mps \
    --eval_interval 2 \
    --save_interval 10

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="
echo ""
echo "预计训练时间: 8-12小时"
echo "最佳模型: ./weights/best_model_$EXP_NAME.pth"
echo "TensorBoard: tensorboard --logdir=./runs/$EXP_NAME"
echo ""


