python3 diagnose_audio.py #!/bin/bash

echo "=========================================="
echo "方言识别 API 服务启动脚本"
echo "=========================================="

# 检查依赖
echo "检查依赖..."
python3 -c "import flask; import flask_cors" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装依赖包..."
    pip3 install -r requirements.txt
fi

# 检查模型文件
echo ""
echo "检查模型文件..."
if [ ! -f "../GUI/weight/model-29.pth" ]; then
    echo "错误: 模型文件不存在!"
    echo "请确保 ../GUI/weight/model-29.pth 文件存在"
    exit 1
fi

if [ ! -f "../GUI/class_indices.json" ]; then
    echo "错误: 类别索引文件不存在!"
    echo "请确保 ../GUI/class_indices.json 文件存在"
    exit 1
fi

echo "✓ 所有文件检查通过"
echo ""
echo "=========================================="
echo "启动 API 服务..."
echo "=========================================="
echo ""

# 启动服务
python3 api_server.py


