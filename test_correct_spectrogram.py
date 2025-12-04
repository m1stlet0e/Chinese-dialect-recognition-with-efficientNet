import sys
sys.path.append('GUI')
from model import efficientnet_b4
import torch
from PIL import Image
from torchvision import transforms
import json
import numpy as np
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from scipy import signal

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

def create_spectrogram_correct(audiopath):
    """使用训练时相同的方法生成声谱图"""
    samplerate, samples = wav.read(audiopath)
    print(f"原始采样率: {samplerate} Hz, 时长: {len(samples)/samplerate:.2f}秒")
    
    # 重采样到48000Hz（如果需要）
    target_sr = 48000
    if samplerate != target_sr:
        print(f"重采样: {samplerate} Hz -> {target_sr} Hz")
        num_samples = int(len(samples) * target_sr / samplerate)
        samples = signal.resample(samples, num_samples)
        samplerate = target_sr
    
    # 截取前5秒
    max_duration = 5
    if len(samples) > samplerate * max_duration:
        print(f"截取前{max_duration}秒")
        samples = samples[:int(samplerate * max_duration)]
    
    # 单声道
    if len(samples.shape) > 1:
        samples = samples[:, 0]
    
    # 生成STFT
    s = stft(samples, 1024)  # 注意：训练时用的是1024，不是512！
    
    # Logscale变换
    sshow, freq = logscale_spec(s, factor=1, sr=samplerate, alpha=1.0)
    sshow = sshow[2:, :]  # 去掉前2行
    
    # 转换为分贝（关键步骤！）
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)
    
    # 转置（关键步骤！）
    ims = np.transpose(ims)
    
    # 裁剪频率范围
    ims = ims[0:256, :]
    
    print(f"声谱图尺寸: {ims.shape}")
    
    # 转换为灰度图
    image = Image.fromarray(ims)
    image = image.convert('L')
    
    # 转换为RGB（模型需要3通道）
    image = image.convert('RGB')
    
    return image

def predict_with_correct_spectrogram(audio_path):
    # 生成正确的声谱图
    spectrogram = create_spectrogram_correct(audio_path)
    
    # 数据转换
    data_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = data_transform(spectrogram)
    img = torch.unsqueeze(img, dim=0)
    
    # 加载模型
    device = torch.device("cpu")
    model = efficientnet_b4(num_classes=10).to(device)
    model.load_state_dict(torch.load('GUI/weight/model-29.pth', map_location=device))
    model.eval()
    
    # 预测
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
    
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
    
    # 显示结果
    print(f"\n{'='*60}")
    print(f"测试音频: {audio_path.split('/')[-1]}")
    print(f"{'='*60}")
    
    sorted_indices = torch.argsort(predict, descending=True)
    for i, idx in enumerate(sorted_indices[:5]):
        class_name = class_indict[str(idx.item())]
        dialect_name = dialect_names.get(class_name, class_name)
        prob = predict[idx].item()
        marker = "👑" if i == 0 else "  "
        print(f"{marker} {dialect_name:8s}: {prob:6.2%}")
    
    predict_cla = torch.argmax(predict).item()
    predicted_class = class_indict[str(predict_cla)]
    predicted_name = dialect_names.get(predicted_class, predicted_class)
    print(f"\n✓ 识别结果: {predicted_name} (置信度: {predict[predict_cla].item():.2%})")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    print("="*60)
    print("使用训练时相同的声谱图生成方式测试")
    print("="*60)
    
    test_files = [
        '/Users/wangbo/Downloads/四川话标注样例/audio/recorder1238A.wav',
        '/Users/wangbo/Downloads/四川话标注样例/audio/recorder1239A.wav',
        '/Users/wangbo/Downloads/四川话标注样例/audio/recorder1240A.wav',
    ]
    
    for audio_file in test_files:
        try:
            predict_with_correct_spectrogram(audio_file)
        except Exception as e:
            print(f"处理 {audio_file} 时出错: {e}")
            import traceback
            traceback.print_exc()

