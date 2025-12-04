#!/bin/bash

echo "=========================================="
echo "服务器配置检查"
echo "=========================================="
echo ""

echo "=== GPU信息 ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "未检测到NVIDIA GPU"
echo ""

echo "=== CPU信息 ==="
echo "CPU型号: $(lscpu | grep 'Model name' | cut -d ':' -f2 | xargs)"
echo "CPU核心数: $(nproc)"
echo ""

echo "=== 内存信息 ==="
free -h | grep Mem
echo ""

echo "=== 磁盘空间 ==="
df -h | grep -E '^/dev/' | head -3
echo ""

echo "=== Python环境 ==="
python3 --version 2>/dev/null || echo "Python3 未安装"
pip3 --version 2>/dev/null || echo "pip3 未安装"
echo ""

echo "=== PyTorch CUDA支持 ==="
python3 -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('CUDA版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU数量:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo "PyTorch 未安装"
echo ""

echo "=========================================="


