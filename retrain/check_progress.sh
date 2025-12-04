#!/bin/bash

# 检查训练进度的脚本

echo "=================================================="
echo "训练进度监控"
echo "=================================================="
echo ""

# 检查数据预处理进程
if pgrep -f "prepare_data.py" > /dev/null; then
    echo "✓ 数据预处理正在运行中..."
    echo ""
    echo "最新日志（最后20行）:"
    echo "---"
    tail -20 data_preprocessing.log
    echo ""
    
    # 统计已处理的文件
    if [ -d "./processed_data" ]; then
        total_images=$(find ./processed_data -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
        echo "已生成声谱图: $total_images / 65000"
        progress=$((total_images * 100 / 65000))
        echo "进度: $progress%"
    fi
else
    echo "○ 数据预处理已完成（或未启动）"
    
    # 检查是否有处理好的数据
    if [ -d "./processed_data" ]; then
        total_images=$(find ./processed_data -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
        echo "总声谱图数量: $total_images"
        
        echo ""
        echo "各方言数据量:"
        for dir in ./processed_data/*/; do
            if [ -d "$dir" ]; then
                dialect=$(basename "$dir")
                count=$(find "$dir" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
                echo "  $dialect: $count 张"
            fi
        done
    fi
fi

echo ""
echo "---"

# 检查训练进程
if pgrep -f "train_improved.py" > /dev/null; then
    echo "✓ 模型训练正在运行中..."
    echo ""
    if [ -f "training.log" ]; then
        echo "最新训练日志:"
        tail -20 training.log
    fi
elif [ -f "training.log" ]; then
    echo "○ 训练已完成"
    echo ""
    echo "查看完整日志: cat training.log"
    echo "查看TensorBoard: tensorboard --logdir=./runs/"
fi

echo ""
echo "=================================================="
echo "命令速查:"
echo "  查看数据预处理日志: tail -f data_preprocessing.log"
echo "  查看训练日志: tail -f training.log"
echo "  再次检查进度: ./check_progress.sh"
echo "=================================================="


