"""
方言识别 REST API 服务
使用 Flask 提供 HTTP 接口
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import torch
from PIL import Image
from torchvision import transforms
import json
import numpy as np
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from werkzeug.utils import secure_filename
import tempfile
import traceback

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'GUI'))
from model import efficientnet_b0 as create_model

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 获取模型路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'GUI')
MODEL_WEIGHT_PATH = os.path.join(MODEL_DIR, 'weight', 'model-29.pth')
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, 'class_indices.json')

# 全局变量存储模型
model = None
device = None
class_indict = None
dialect_names = {
    "changsha": "长沙话",
    "hebei": "河北话",
    "hefei": "合肥话",
    "kejia": "客家话",
    "nanchang": "南昌话",
    "ningxia": "宁夏话",
    "shan3xi": "陕西话",
    "shanghai": "上海话",
    "sichuan": "四川话"
}


def init_model():
    """初始化模型"""
    global model, device, class_indict
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载类别索引
    with open(CLASS_INDICES_PATH, "r", encoding='utf-8') as f:
        class_indict = json.load(f)
    
    # 加载模型
    model = create_model(num_classes=9).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=device))
    model.eval()
    
    print("模型加载成功!")


def allowed_file(filename):
    """检查文件扩展名"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    """短时傅里叶变换"""
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    samples = np.append(np.zeros(int(np.floor(frameSize / 2.0))), sig)
    cols = int(np.ceil((len(samples) - frameSize) / float(hopSize)) + 1)
    samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames)


def logscale_spec(spec, sr=44100, factor=20., alpha=1.0, f0=0.9, fmax=1):
    """对数频率缩放"""
    spec = spec[:, 0:256]
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins)
    scale = np.array(list(
        map(lambda x: x * alpha if x <= f0 else (fmax - alpha * f0) / (fmax - f0) * (x - f0) + alpha * f0,
            scale)))
    scale *= (freqbins - 1) / max(scale)
    newspec = np.complex128(np.zeros([timebins, freqbins]))
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = [0.0 for i in range(freqbins)]
    totw = [0.0 for i in range(freqbins)]
    for i in range(0, freqbins):
        if (i < 1 or i + 1 >= freqbins):
            newspec[:, i] += spec[:, i]
            freqs[i] += allfreqs[i]
            totw[i] += 1.0
            continue
        else:
            w_up = scale[i] - np.floor(scale[i])
            w_down = 1 - w_up
            j = int(np.floor(scale[i]))
            newspec[:, j] += w_down * spec[:, i]
            freqs[j] += w_down * allfreqs[i]
            totw[j] += w_down
            newspec[:, j + 1] += w_up * spec[:, i]
            freqs[j + 1] += w_up * allfreqs[i]
            totw[j + 1] += w_up
    for i in range(len(freqs)):
        if (totw[i] > 1e-6):
            freqs[i] /= totw[i]
    return newspec, freqs


def audio_to_spectrogram(audiopath, spec_path):
    """将音频转换为声谱图"""
    samplerate, samples = wav.read(audiopath)
    samples = samples if len(samples.shape) <= 1 else samples[:, 0]
    s = stft(samples, 2 ** 10)
    sshow, freq = logscale_spec(s, factor=1, sr=samplerate, alpha=1.0)
    sshow = sshow[2:, :]
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)
    ims = np.transpose(ims)
    ims = ims[0:256, :]
    
    image = Image.fromarray(ims)
    image = image.convert('L')
    image.save(spec_path)


def predict_audio(audio_path):
    """预测音频文件的方言"""
    # 生成声谱图
    spec_path = os.path.join(tempfile.gettempdir(), 'temp_spec.png')
    audio_to_spectrogram(audio_path, spec_path)
    
    # 转换为RGB
    img = Image.open(spec_path).convert('RGB')
    img.save(spec_path)
    
    # 预处理图像
    img_size = 224
    data_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(spec_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    
    # 预测
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    
    # 构建结果
    predicted_class = class_indict[str(predict_cla)]
    predicted_name = dialect_names.get(predicted_class, predicted_class)
    probability = float(predict[predict_cla].numpy())
    
    # 所有类别概率
    all_probabilities = {}
    for i in range(len(predict)):
        class_name = class_indict[str(i)]
        dialect_name = dialect_names.get(class_name, class_name)
        all_probabilities[dialect_name] = float(predict[i].numpy())
    
    # 清理临时文件
    if os.path.exists(spec_path):
        os.remove(spec_path)
    
    return {
        'dialect': predicted_name,
        'confidence': probability,
        'all_probabilities': all_probabilities
    }


@app.route('/', methods=['GET'])
def index():
    """首页"""
    return jsonify({
        'service': '中国方言识别 API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'predict': '/api/predict (POST)',
            'health': '/api/health (GET)',
            'dialects': '/api/dialects (GET)'
        }
    })


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })


@app.route('/api/dialects', methods=['GET'])
def get_dialects():
    """获取支持的方言列表"""
    return jsonify({
        'dialects': list(dialect_names.values()),
        'count': len(dialect_names)
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    预测音频文件的方言
    
    请求:
        - 文件上传: 'file' 字段
        - 支持格式: WAV, MP3
        
    返回:
        {
            "success": true,
            "dialect": "客家话",
            "confidence": 0.8555,
            "all_probabilities": {
                "客家话": 0.8555,
                "长沙话": 0.0937,
                ...
            }
        }
    """
    try:
        # 检查文件
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': '请上传音频文件'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '文件名为空'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'不支持的文件格式，仅支持: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # 保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 预测
        result = predict_audio(filepath)
        
        # 删除上传的文件
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'success': True,
            **result
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():
    """
    批量预测多个音频文件
    
    请求:
        - 多个文件上传: 'files' 字段
        
    返回:
        {
            "success": true,
            "results": [
                {
                    "filename": "test1.wav",
                    "dialect": "客家话",
                    "confidence": 0.8555
                },
                ...
            ]
        }
    """
    try:
        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'error': '请上传音频文件'
            }), 400
        
        files = request.files.getlist('files')
        
        if len(files) == 0:
            return jsonify({
                'success': False,
                'error': '没有文件被上传'
            }), 400
        
        results = []
        
        for file in files:
            if file.filename == '':
                continue
            
            if not allowed_file(file.filename):
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': '不支持的文件格式'
                })
                continue
            
            try:
                # 保存并预测
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                result = predict_audio(filepath)
                
                results.append({
                    'filename': file.filename,
                    'success': True,
                    **result
                })
                
                # 删除文件
                if os.path.exists(filepath):
                    os.remove(filepath)
                    
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total': len(files),
            'results': results
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("初始化模型...")
    init_model()
    print("\n" + "="*50)
    print("方言识别 API 服务已启动!")
    print("="*50)
    print(f"访问地址: http://localhost:8000")
    print(f"API文档: http://localhost:8000")
    print(f"健康检查: http://localhost:8000/api/health")
    print(f"预测接口: http://localhost:8000/api/predict")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=8000, debug=False)

