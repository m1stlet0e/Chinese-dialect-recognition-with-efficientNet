#!/bin/bash

# A800服务器优化训练脚本
# 充分利用80GB显存和多GPU

echo "=========================================="
echo "A800服务器优化训练"
echo "=========================================="
echo "GPU: 8x NVIDIA A800 80GB"
echo "显存: 640GB total"
echo "=========================================="
echo ""

# 配置选项
echo "请选择训练配置:"
echo ""
echo "1. 🚀 单GPU快速训练（推荐新手）"
echo "   - 使用1块GPU"
echo "   - Batch Size: 128"
echo "   - 模型: B3"
echo "   - 预计: 2-3小时"
echo ""
echo "2. ⚡ 单GPU高性能（推荐）"
echo "   - 使用1块GPU"
echo "   - Batch Size: 256（充分利用80GB显存）"
echo "   - 模型: B4"
echo "   - 预计: 3-4小时"
echo ""
echo "3. 🔥 多GPU并行训练（极速）"
echo "   - 使用2块GPU"
echo "   - Batch Size: 256 (每GPU 128)"
echo "   - 模型: B4"
echo "   - 预计: 1.5-2小时"
echo ""
echo "4. 💎 8GPU全力训练（疯狂模式）"
echo "   - 使用全部8块GPU"
echo "   - Batch Size: 512 (每GPU 64)"
echo "   - 模型: B6（最大）"
echo "   - 预计: <1小时"
echo ""

read -p "请选择 (1-4): " config

case $config in
    1)
        echo ""
        echo "=== 启动单GPU快速训练 ==="
        CUDA_VISIBLE_DEVICES=0 nohup python3 train_improved.py \
            --data_path ./processed_data \
            --model B3 \
            --epochs 50 \
            --batch_size 128 \
            --lr 0.001 \
            --exp_name a800_single_fast \
            --use_class_weights \
            --device cuda:0 \
            > training.log 2>&1 &
        ;;
        
    2)
        echo ""
        echo "=== 启动单GPU高性能训练 ==="
        CUDA_VISIBLE_DEVICES=0 nohup python3 train_improved.py \
            --data_path ./processed_data \
            --model B4 \
            --epochs 50 \
            --batch_size 256 \
            --lr 0.001 \
            --exp_name a800_single_b4 \
            --use_class_weights \
            --device cuda:0 \
            > training.log 2>&1 &
        ;;
        
    3)
        echo ""
        echo "=== 启动2GPU并行训练 ==="
        CUDA_VISIBLE_DEVICES=0,1 nohup python3 -m torch.distributed.launch \
            --nproc_per_node=2 \
            train_improved.py \
            --data_path ./processed_data \
            --model B4 \
            --epochs 50 \
            --batch_size 256 \
            --lr 0.001 \
            --exp_name a800_multi2_b4 \
            --use_class_weights \
            --device cuda \
            > training.log 2>&1 &
        ;;
        
    4)
        echo ""
        echo "=== 启动8GPU全力训练 ==="
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python3 -m torch.distributed.launch \
            --nproc_per_node=8 \
            train_improved.py \
            --data_path ./processed_data \
            --model B6 \
            --epochs 80 \
            --batch_size 512 \
            --lr 0.001 \
            --exp_name a800_multi8_b6 \
            --use_class_weights \
            --device cuda \
            > training.log 2>&1 &
        ;;
        
    *)
        echo "无效选项"
        exit 1
        ;;
esac

TRAIN_PID=$!

echo ""
echo "✓ 训练已启动！"
echo "进程ID: $TRAIN_PID"
echo ""
echo "监控命令:"
echo "  查看日志: tail -f training.log"
echo "  监控GPU: watch -n 1 nvidia-smi"
echo ""
echo "预计完成: 根据配置1-4小时"
echo "=========================================="


