"""
使用训练数据集验证模型准确性
"""
import sys
import os
sys.path.append('GUI')
from model import efficientnet_b4
import torch
from PIL import Image
from torchvision import transforms
import json
import numpy as np
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import struct
import tempfile

def pcm_to_wav(pcm_file, wav_file, channels=1, sample_width=2, frame_rate=16000):
    """将PCM文件转换为WAV文件"""
    with open(pcm_file, 'rb') as pcmfile:
        pcmdata = pcmfile.read()
    
    with open(wav_file, 'wb') as wavfile:
        # 写入WAV文件头
        wavfile.write(b'RIFF')
        wavfile.write(struct.pack('<I', len(pcmdata) + 36))
        wavfile.write(b'WAVE')
        wavfile.write(b'fmt ')
        wavfile.write(struct.pack('<I', 16))
        wavfile.write(struct.pack('<H', 1))  # PCM
        wavfile.write(struct.pack('<H', channels))
        wavfile.write(struct.pack('<I', frame_rate))
        wavfile.write(struct.pack('<I', frame_rate * channels * sample_width))
        wavfile.write(struct.pack('<H', channels * sample_width))
        wavfile.write(struct.pack('<H', sample_width * 8))
        wavfile.write(b'data')
        wavfile.write(struct.pack('<I', len(pcmdata)))
        wavfile.write(pcmdata)

def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)
    cols = int(np.ceil((len(samples) - frameSize) / float(hopSize)) + 1)
    samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), 
                                      strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames)

def logscale_spec(spec, sr=44100, factor=20., alpha=1.0, f0=0.9, fmax=1):
    """训练时使用的logscale_spec算法"""
    spec = spec[:, 0:256]
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins)
    scale = np.array(list(map(lambda x: x * alpha if x <= f0 else (fmax-alpha*f0)/(fmax-f0)*(x-f0)+alpha*f0, scale)))
    scale *= (freqbins-1)/max(scale)

    newspec = np.complex128(np.zeros([timebins, freqbins]))
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
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

def create_spectrogram(audiopath):
    """使用训练时相同的方法生成声谱图"""
    samplerate, samples = wav.read(audiopath)
    
    # 单声道
    if len(samples.shape) > 1:
        samples = samples[:, 0]
    
    # 生成STFT
    s = stft(samples, 1024)
    
    # Logscale变换
    sshow, freq = logscale_spec(s, factor=1, sr=samplerate, alpha=1.0)
    sshow = sshow[2:, :]
    
    # 转换为分贝
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)
    
    # 转置
    ims = np.transpose(ims)
    
    # 裁剪频率范围
    ims = ims[0:256, :]
    
    # 转换为图像
    image = Image.fromarray(ims)
    image = image.convert('L')
    image = image.convert('RGB')
    
    return image

def predict_audio(audio_path, model, device, class_indict, dialect_names):
    """预测音频"""
    # 生成声谱图
    spectrogram = create_spectrogram(audio_path)
    
    # 数据转换
    data_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = data_transform(spectrogram)
    img = torch.unsqueeze(img, dim=0)
    
    # 预测
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
    
    predict_cla = torch.argmax(predict).item()
    predicted_class = class_indict[str(predict_cla)]
    predicted_name = dialect_names.get(predicted_class, predicted_class)
    confidence = predict[predict_cla].item()
    
    # 获取top3
    sorted_indices = torch.argsort(predict, descending=True)[:3]
    top3 = []
    for idx in sorted_indices:
        class_name = class_indict[str(idx.item())]
        dialect_name = dialect_names.get(class_name, class_name)
        prob = predict[idx].item()
        top3.append(f"{dialect_name}:{prob:.1%}")
    
    return predicted_name, confidence, top3

def test_dialects():
    """测试所有方言"""
    # 加载模型
    device = torch.device("cpu")
    model = efficientnet_b4(num_classes=10).to(device)
    model.load_state_dict(torch.load('GUI/weight/model-29.pth', map_location=device))
    model.eval()
    
    # 加载类别
    with open('GUI/class_indices.json', 'r') as f:
        class_indict = json.load(f)
    
    dialect_names = {
        "changsha": "长沙话",
        "hebei": "河北话",
        "hefei": "合肥话",
        "kejia": "客家话",
        "minnan": "闽南话",
        "nanchang": "南昌话",
        "ningxia": "宁夏话",
        "shan3xi": "陕西话",
        "shanghai": "上海话",
        "sichuan": "四川话"
    }
    
    # 测试样本
    data_root = "/Users/wangbo/Desktop/origin_data"
    dialects = ["changsha", "hebei", "hefei", "kejia", "minnan", "nanchang", "ningxia", "shan3xi", "shanghai", "sichuan"]
    
    print("\n" + "="*70)
    print("使用训练数据集验证模型准确性")
    print("="*70)
    
    results = []
    total = 0
    correct = 0
    
    for dialect in dialects:
        # 找3个测试样本（从dev目录选，避免用训练数据）
        dev_dir = os.path.join(data_root, dialect, "dev", "speaker31", "short")
        if not os.path.exists(dev_dir):
            dev_dir = os.path.join(data_root, dialect, "train", "speaker01")
        
        # 递归查找PCM文件
        pcm_files = []
        if os.path.exists(dev_dir):
            for root, dirs, files in os.walk(dev_dir):
                pcm_files.extend([os.path.join(root, f) for f in files if f.endswith('.pcm')])
                if len(pcm_files) >= 3:
                    break
        pcm_files = pcm_files[:3]
        
        dialect_correct = 0
        dialect_total = len(pcm_files)
        
        for pcm_path in pcm_files:
            
            # 转换PCM到WAV
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                wav_path = tmp_wav.name
                pcm_to_wav(pcm_path, wav_path, frame_rate=16000)
                
                try:
                    # 预测
                    predicted, confidence, top3 = predict_audio(wav_path, model, device, class_indict, dialect_names)
                    
                    # 判断是否正确
                    is_correct = (predicted == dialect_names[dialect])
                    if is_correct:
                        correct += 1
                        dialect_correct += 1
                    total += 1
                    
                    # 显示结果
                    status = "✓" if is_correct else "✗"
                    print(f"{status} 真实:{dialect_names[dialect]:8s} | 预测:{predicted:8s} ({confidence:.1%}) | Top3: {', '.join(top3)}")
                    
                except Exception as e:
                    print(f"✗ 真实:{dialect_names[dialect]:8s} | 错误: {e}")
                finally:
                    os.unlink(wav_path)
        
        acc = dialect_correct / dialect_total if dialect_total > 0 else 0
        results.append((dialect_names[dialect], dialect_correct, dialect_total, acc))
        print(f"  {dialect_names[dialect]} 准确率: {dialect_correct}/{dialect_total} = {acc:.1%}\n")
    
    # 总体统计
    print("="*70)
    print("总体统计:")
    print("="*70)
    for name, correct_count, total_count, acc in results:
        bar = "█" * int(acc * 20)
        print(f"{name:8s}: {correct_count}/{total_count} = {acc:5.1%} {bar}")
    
    overall_acc = correct / total if total > 0 else 0
    print(f"\n总体准确率: {correct}/{total} = {overall_acc:.1%}")
    print("="*70)

if __name__ == '__main__':
    test_dialects()

