#!/bin/bash

echo "=========================================="
echo "方言识别系统环境配置脚本"
echo "=========================================="

# 检查Python版本
echo "检查Python版本..."
python_version=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
echo "当前Python版本: $python_version"

# 安装依赖
echo ""
echo "安装Python依赖包..."
pip3 install -r requirements.txt

# 检查ffmpeg
echo ""
echo "检查ffmpeg..."
if command -v ffmpeg &> /dev/null
then
    echo "ffmpeg已安装: $(ffmpeg -version | head -n 1)"
else
    echo "警告: 未检测到ffmpeg!"
    echo "请手动安装ffmpeg:"
    echo "  macOS: brew install ffmpeg"
    echo "  Ubuntu: sudo apt-get install ffmpeg"
fi

# 检查必需文件
echo ""
echo "检查必需文件..."
files=("model.py" "class_indices.json" "weight/model-29.pth")
all_exist=true
for file in "${files[@]}"
do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (缺失)"
        all_exist=false
    fi
done

echo ""
if [ "$all_exist" = true ]; then
    echo "=========================================="
    echo "环境配置完成!"
    echo "=========================================="
    echo ""
    echo "启动GUI程序:"
    echo "  python3 打包方言识别.py"
    echo ""
    echo "或使用命令行测试:"
    echo "  python3 test_predict.py <音频文件.wav>"
else
    echo "=========================================="
    echo "警告: 缺少必需文件,请检查!"
    echo "=========================================="
fi


