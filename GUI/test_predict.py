import os
import sys
import torch
from PIL import Image
from torchvision import transforms
import json
from model import efficientnet_b0 as create_model

def predict_wav(wav_path):
    """简单的命令行预测脚本"""
    import numpy as np
    import scipy.io.wavfile as wav
    from numpy.lib import stride_tricks
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 生成声谱图函数
    def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
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
    
    def plotstft(audiopath, name='temp.png'):
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
        image.save(name)
    
    print(f"正在处理音频文件: {wav_path}")
    
    if not os.path.exists(wav_path):
        print(f"错误: 文件不存在 {wav_path}")
        return
    
    # 生成声谱图
    temp_spec = os.path.join(SCRIPT_DIR, 'temp', 'test_spec.png')
    os.makedirs(os.path.dirname(temp_spec), exist_ok=True)
    
    plotstft(wav_path, temp_spec)
    
    # 转换为RGB
    img = Image.open(temp_spec).convert('RGB')
    img.save(temp_spec)
    
    # 预测
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    img_size = 224
    data_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(temp_spec)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    
    # 读取类别
    json_path = os.path.join(SCRIPT_DIR, 'class_indices.json')
    with open(json_path, "r", encoding='utf-8') as f:
        class_indict = json.load(f)
    
    # 加载模型
    model = create_model(num_classes=9).to(device)
    model_weight_path = os.path.join(SCRIPT_DIR, 'weight', 'model-29.pth')
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    
    # 预测
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    
    # 方言名称
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
    
    predicted_class = class_indict[str(predict_cla)]
    predicted_name = dialect_names.get(predicted_class, predicted_class)
    probability = predict[predict_cla].numpy()
    
    print("\n" + "="*50)
    print(f"识别结果: {predicted_name}")
    print(f"置信度: {probability:.2%}")
    print("="*50)
    print("\n所有类别概率:")
    
    sorted_indices = torch.argsort(predict, descending=True)
    for idx in sorted_indices:
        class_name = class_indict[str(idx.item())]
        dialect_name = dialect_names.get(class_name, class_name)
        prob = predict[idx].numpy()
        print(f"  {dialect_name}: {prob:.2%}")
    
    # 清理临时文件
    if os.path.exists(temp_spec):
        os.remove(temp_spec)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使用方法: python test_predict.py <wav文件路径>")
        print("示例: python test_predict.py test.wav")
        sys.exit(1)
    
    wav_file = sys.argv[1]
    predict_wav(wav_file)


