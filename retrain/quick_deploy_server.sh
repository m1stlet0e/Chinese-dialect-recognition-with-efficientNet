#!/bin/bash

# 快速部署到服务器的脚本

SERVER="wangbo@172.22.0.35"
PROJECT_DIR="/Users/wangbo/PycharmProjects/Chinese-dialect-recognition-with-efficientNet"
DATA_DIR="/Users/wangbo/Desktop/origin_data"

echo "=========================================="
echo "快速部署训练到服务器"
echo "=========================================="
echo ""
echo "服务器: $SERVER"
echo ""

# 询问用户要执行的步骤
echo "请选择操作："
echo "1. 只上传代码"
echo "2. 只上传已处理的数据（processed_data）"
echo "3. 上传代码和数据"
echo "4. 在服务器上启动训练"
echo "5. 查看服务器训练状态"
echo "6. 下载训练好的模型"
echo ""
read -p "请输入选项 (1-6): " choice

case $choice in
    1)
        echo ""
        echo "=== 上传代码到服务器 ==="
        rsync -avz --progress \
            --exclude '*.pyc' \
            --exclude '__pycache__' \
            --exclude 'processed_data' \
            --exclude 'weights' \
            --exclude '*.log' \
            $PROJECT_DIR/retrain/ \
            $SERVER:~/dialect_training/retrain/
        
        rsync -avz --progress \
            --exclude '*.pyc' \
            --exclude '__pycache__' \
            $PROJECT_DIR/train\&predict/ \
            $SERVER:~/dialect_training/train_predict/
        
        echo "✓ 代码上传完成"
        ;;
        
    2)
        echo ""
        echo "=== 上传已处理的数据 ==="
        if [ ! -d "$PROJECT_DIR/retrain/processed_data" ]; then
            echo "❌ 错误: 本地没有processed_data目录"
            echo "请先在Mac上完成数据预处理"
            exit 1
        fi
        
        echo "数据大小:"
        du -sh $PROJECT_DIR/retrain/processed_data
        echo ""
        read -p "确认上传？这可能需要较长时间 (y/n): " confirm
        
        if [ "$confirm" = "y" ]; then
            rsync -avz --progress \
                $PROJECT_DIR/retrain/processed_data/ \
                $SERVER:~/dialect_training/retrain/processed_data/
            echo "✓ 数据上传完成"
        fi
        ;;
        
    3)
        echo ""
        echo "=== 上传代码和数据 ==="
        
        # 上传代码
        echo "1/2 上传代码..."
        rsync -avz --progress \
            --exclude '*.pyc' \
            --exclude '__pycache__' \
            --exclude 'processed_data' \
            --exclude 'weights' \
            --exclude '*.log' \
            $PROJECT_DIR/retrain/ \
            $SERVER:~/dialect_training/retrain/
        
        rsync -avz --progress \
            --exclude '*.pyc' \
            --exclude '__pycache__' \
            $PROJECT_DIR/train\&predict/ \
            $SERVER:~/dialect_training/train_predict/
        
        # 上传数据
        echo "2/2 上传数据..."
        if [ -d "$PROJECT_DIR/retrain/processed_data" ]; then
            rsync -avz --progress \
                $PROJECT_DIR/retrain/processed_data/ \
                $SERVER:~/dialect_training/retrain/processed_data/
        else
            echo "⚠️  警告: 本地没有processed_data，跳过数据上传"
        fi
        
        echo "✓ 全部上传完成"
        ;;
        
    4)
        echo ""
        echo "=== 在服务器上启动训练 ==="
        echo ""
        echo "请手动SSH到服务器并运行："
        echo ""
        echo "ssh $SERVER"
        echo ""
        echo "然后执行："
        echo ""
        cat << 'EOF'
cd ~/dialect_training/retrain

# 安装依赖（首次运行需要）
pip3 install torch torchvision tensorboard seaborn scikit-learn tqdm pillow scipy numpy pandas matplotlib

# 启动训练
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

echo "训练已启动！"
echo "查看进度: tail -f training.log"
echo "监控GPU: watch -n 1 nvidia-smi"
EOF
        ;;
        
    5)
        echo ""
        echo "=== 查看服务器训练状态 ==="
        echo ""
        ssh $SERVER "cd ~/dialect_training/retrain && tail -50 training.log"
        ;;
        
    6)
        echo ""
        echo "=== 下载训练好的模型 ==="
        echo ""
        
        # 列出服务器上的模型
        echo "服务器上的模型:"
        ssh $SERVER "ls -lh ~/dialect_training/retrain/weights/ 2>/dev/null" || echo "没有找到模型文件"
        echo ""
        
        read -p "输入要下载的模型文件名 (例如: best_model_server_training.pth): " model_name
        
        if [ -n "$model_name" ]; then
            scp $SERVER:~/dialect_training/retrain/weights/$model_name \
                $PROJECT_DIR/GUI/weight/model-29.pth
            echo "✓ 模型已下载并替换旧模型"
        fi
        ;;
        
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "操作完成！"
echo "=========================================="


