#!/bin/bash

# 完整训练流程脚本
# 从原始PCM音频到训练完成

set -e  # 遇到错误立即退出

echo "=========================================="
echo "方言识别完整训练流程"
echo "=========================================="

# 配置
INPUT_DIR="/Users/wangbo/Desktop/origin_data"
OUTPUT_DIR="./processed_data"
WEIGHTS_DIR="./weights"
RESULTS_DIR="./results"

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 找不到数据集目录 $INPUT_DIR"
    exit 1
fi

echo "数据集目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 询问用户
read -p "是否要处理数据？(如果已处理过可跳过) [y/n]: " process_data

if [ "$process_data" = "y" ] || [ "$process_data" = "Y" ]; then
    echo ""
    echo "=========================================="
    echo "步骤 1: 数据预处理"
    echo "=========================================="
    echo "这将需要较长时间（约1-2小时），处理65,000个音频文件..."
    echo ""
    
    read -p "确认开始处理？[y/n]: " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "已取消"
        exit 0
    fi
    
    python3 prepare_data.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR"
    
    if [ $? -ne 0 ]; then
        echo "数据处理失败!"
        exit 1
    fi
    
    echo ""
    echo "✓ 数据处理完成"
fi

# 检查处理后的数据
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "错误: 找不到处理后的数据目录 $OUTPUT_DIR"
    echo "请先运行数据处理步骤"
    exit 1
fi

echo ""
echo "=========================================="
echo "步骤 2: 模型训练"
echo "=========================================="
echo ""

# 选择训练配置
echo "请选择训练配置:"
echo "1. 快速测试 (B0模型, 10 epochs, 小数据集)"
echo "2. 标准训练 (B3模型, 30 epochs)"
echo "3. 完整训练 (B3模型, 50 epochs, 推荐)"
echo "4. 高精度训练 (B4模型, 50 epochs)"
read -p "选择 [1-4]: " choice

case $choice in
    1)
        MODEL="B0"
        EPOCHS=10
        BATCH_SIZE=32
        LR=0.001
        EXP_NAME="quick_test"
        echo "配置: 快速测试"
        ;;
    2)
        MODEL="B3"
        EPOCHS=30
        BATCH_SIZE=16
        LR=0.001
        EXP_NAME="standard_training"
        echo "配置: 标准训练"
        ;;
    3)
        MODEL="B3"
        EPOCHS=50
        BATCH_SIZE=16
        LR=0.0005
        EXP_NAME="full_training"
        echo "配置: 完整训练（推荐）"
        ;;
    4)
        MODEL="B4"
        EPOCHS=50
        BATCH_SIZE=8
        LR=0.0005
        EXP_NAME="high_precision"
        echo "配置: 高精度训练"
        ;;
    *)
        echo "无效选择，使用默认配置（完整训练）"
        MODEL="B3"
        EPOCHS=50
        BATCH_SIZE=16
        LR=0.0005
        EXP_NAME="full_training"
        ;;
esac

echo ""
echo "训练参数:"
echo "  模型: EfficientNet-$MODEL"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  学习率: $LR"
echo "  实验名称: $EXP_NAME"
echo ""

read -p "确认开始训练？[y/n]: " confirm_train
if [ "$confirm_train" != "y" ] && [ "$confirm_train" != "Y" ]; then
    echo "已取消"
    exit 0
fi

# 检查是否需要下载预训练权重
PRETRAIN_WEIGHTS=""
echo ""
read -p "是否使用预训练权重？(推荐) [y/n]: " use_pretrain
if [ "$use_pretrain" = "y" ] || [ "$use_pretrain" = "Y" ]; then
    PRETRAIN_PATH="../train&predict/efficientnet-b${MODEL:1:1}.pth"
    if [ -f "$PRETRAIN_PATH" ]; then
        PRETRAIN_WEIGHTS="--weights $PRETRAIN_PATH"
        echo "使用预训练权重: $PRETRAIN_PATH"
    else
        echo "警告: 找不到预训练权重，将从头训练"
    fi
fi

# 开始训练
echo ""
echo "=========================================="
echo "开始训练..."
echo "=========================================="
echo ""

python3 train_improved.py \
    --data_path "$OUTPUT_DIR" \
    --model "$MODEL" \
    $PRETRAIN_WEIGHTS \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --exp_name "$EXP_NAME" \
    --use_class_weights \
    --device "cuda:0"

if [ $? -ne 0 ]; then
    echo ""
    echo "训练失败!"
    exit 1
fi

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="
echo ""
echo "结果保存在:"
echo "  模型: $WEIGHTS_DIR/best_model_$EXP_NAME.pth"
echo "  混淆矩阵: $RESULTS_DIR/"
echo "  TensorBoard: ./runs/$EXP_NAME/"
echo ""
echo "查看TensorBoard:"
echo "  tensorboard --logdir=./runs/$EXP_NAME"
echo ""
echo "测试模型:"
echo "  python3 test_model.py --model_path $WEIGHTS_DIR/best_model_$EXP_NAME.pth"
echo ""


