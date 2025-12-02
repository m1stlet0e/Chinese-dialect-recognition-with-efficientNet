# GUI 方言识别系统使用说明

## 环境要求

### 必需的Python包
```bash
pip install numpy scipy pillow torch torchvision wxpython pydub pyaudio
```

### 系统要求
- Python 3.7+
- 麦克风(用于录音功能)
- ffmpeg(用于音频处理,pydub依赖)

### 安装ffmpeg
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **Windows**: 从 https://ffmpeg.org/download.html 下载并添加到PATH

## 文件结构

确保以下文件存在:
```
GUI/
├── 打包方言识别.py     # 主GUI程序
├── test_predict.py      # 命令行测试脚本
├── model.py             # EfficientNet模型定义
├── class_indices.json   # 类别索引文件
├── weight/
│   └── model-29.pth     # 训练好的模型权重
└── temp/                # 临时文件目录(自动创建)
```

## 使用方法

### 方法1: GUI界面(推荐)

运行GUI程序:
```bash
cd GUI
python 打包方言识别.py
```

#### 功能1: 实时录音识别
1. 在"请输入录音时长"输入框输入5-10之间的数字(秒)
2. 点击右侧"确定"按钮
3. 听到提示后开始说话
4. 录音完成后自动识别并显示结果

#### 功能2: 上传音频文件识别
1. 点击"选择您的wav格式录音文件"旁的文件选择按钮
2. 选择一个WAV格式的音频文件
3. 点击右侧"确定"按钮
4. 等待处理,查看识别结果

### 方法2: 命令行测试

如果你有WAV文件想快速测试:
```bash
cd GUI
python test_predict.py <你的wav文件路径>
```

示例:
```bash
python test_predict.py test_audio.wav
```

## 支持的方言

本系统可以识别以下9种中国方言:
1. 长沙话
2. 河北话
3. 合肥话
4. 客家话
5. 南昌话
6. 宁夏话
7. 陕西话
8. 上海话
9. 四川话

## 音频要求

- **格式**: WAV
- **时长**: 5-10秒(推荐7秒左右)
- **内容**: 清晰的方言语音
- **质量**: 尽量减少背景噪音

## 常见问题

### 1. ModuleNotFoundError: No module named 'wx'
```bash
pip install wxpython
```

### 2. PyAudio安装失败
**macOS**:
```bash
brew install portaudio
pip install pyaudio
```

**Ubuntu/Debian**:
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

**Windows**:
下载对应版本的wheel文件: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

### 3. 找不到ffmpeg
确保ffmpeg已安装并在系统PATH中:
```bash
ffmpeg -version
```

### 4. 模型文件不存在
确保以下文件存在:
- `GUI/weight/model-29.pth` (模型权重)
- `GUI/class_indices.json` (类别索引)

### 5. 录音没有声音
- 检查麦克风权限
- 确保系统音频设备正常工作
- macOS用户需在"系统偏好设置-安全性与隐私-麦克风"中授权

## 技术说明

### 工作流程
1. **音频输入**: 录音或上传WAV文件
2. **预处理**: 
   - 转换为单声道
   - 采样率标准化
3. **特征提取**: 
   - 短时傅里叶变换(STFT)
   - 生成声谱图
   - 对数频率缩放
4. **模型预测**: 
   - 使用EfficientNet-B0
   - 输出9个类别的概率分布
5. **结果显示**: 显示最可能的方言及置信度

### 模型信息
- **架构**: EfficientNet-B0
- **输入尺寸**: 224×224 RGB图像
- **训练数据**: 2018年讯飞方言识别挑战赛数据集
- **类别数**: 9种方言
- **准确率**: ~78%

## 开发团队

组员: 赵子龙, 庞博, 张垚杰

## 许可证

本项目仅供学习和研究使用。


