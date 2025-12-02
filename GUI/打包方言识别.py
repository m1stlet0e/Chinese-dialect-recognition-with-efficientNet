import numpy as np
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import json
import torch
from PIL import Image
from torchvision import transforms
from model import efficientnet_b0 as create_model
import os
import pyaudio
import wave
import wx
from pydub import AudioSegment

# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 临时文件目录
TEMP_DIR = os.path.join(SCRIPT_DIR, 'temp')
# 确保temp目录存在
os.makedirs(TEMP_DIR, exist_ok=True)

# 录音参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000

""" short time fourier transform of audio signal """


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize / 2.0))), sig)
    # cols for windowing
    cols = int(np.ceil((len(samples) - frameSize) / float(hopSize)) + 1)
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


""" scale frequency axis logarithmically """


def logscale_spec(spec, sr=44100, factor=20., alpha=1.0, f0=0.9, fmax=1):
    spec = spec[:, 0:256]
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins)  # ** factor

    # http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310
    scale = np.array(list(
        map(lambda x: x * alpha if x <= f0 else (fmax - alpha * f0) / (fmax - f0) * (x - f0) + alpha * f0,
            scale)))  # add list convert
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
            # scale[15] = 17.2
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


""" plot spectrogram"""


def plotstft(audiopath, binsize=2 ** 10, plotpath=None, colormap="gray", channel=0, name='tmp.png', alpha=1, offset=0):
    samplerate, samples = wav.read(audiopath)
    samples = samples if len(samples.shape) <= 1 else samples[:, channel]
    s = stft(samples, binsize)  # 431 * 513

    # sshow : 431 * 256,
    sshow, freq = logscale_spec(s, factor=1, sr=samplerate, alpha=alpha)
    sshow = sshow[2:, :]
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel
    timebins, freqbins = np.shape(ims)

    ims = np.transpose(ims)
    # ims = ims[0:256, offset:offset+768] # 0-11khz, ~9s interval
    ims = ims[0:256, :]  # 0-11khz, ~10s interval
    # print "ims.shape", ims.shape

    image = Image.fromarray(ims)
    image = image.convert('L')
    image.save(name)




# 录制声音的相关函数（参数1：录制的路径；参数2：录制的声音秒数）
def record_audio(wave_out_path, record_second):
    # 实例化相关的对象
    p = pyaudio.PyAudio()
    # 打开相关的流，然后传入响应参数
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    # 打开wav文件
    wf = wave.open(wave_out_path, 'wb')
    # 设置相关的声道
    wf.setnchannels(CHANNELS)
    # 设置采样位数8
    wf.setsampwidth(p.get_sample_size((FORMAT)))
    # 设置采样频率
    wf.setframerate(RATE)

    for _ in range(0, int(RATE * record_second / CHUNK)):
        data = stream.read(CHUNK)
        # 写入数据
        wf.writeframes(data)
    # 关闭流
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()

#GUI
class MyFrame(wx.Frame):
    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent, id, title="大创项目：基于机器学习的方言地域识别系统",
                          pos=(100, 100), size=(600, 400))
        panel = wx.Panel(self)  # 创建画板
        # 创建标题，并设置字体
        title = wx.StaticText(panel, label='基于机器学习的方言地域识别系统', pos=(100, 20))
        font = wx.Font(16, wx.DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        title.SetFont(font)
        # 创建文本和输入框
        self.label_user = wx.StaticText(panel, label="目前支持方言：长沙话，河北话，合肥话，客家话，南昌话，宁夏话，陕西话，上海话，四川话", pos=(50, 70))
        self.label_user = wx.StaticText(panel, label="组员：赵子龙，庞博，张垚杰", pos=(50, 320))
        self.label_user = wx.StaticText(panel, label="请输入录音时长(5秒到10秒):", pos=(50, 100))
        self.text_user = wx.TextCtrl(panel, pos=(230, 100), size=(235, 25), style=wx.TE_LEFT)
        self.picker=wx.FilePickerCtrl(panel, pos=(220, 250), size=(235, 25))
        # self.m_filePicker4 = wx.FileDialog(panel,pos=(250, 100), size=(235, 25))



        # 创建按钮
        self.bt_confirm = wx.Button(panel, label='确定', pos=(480, 100))  # 创建“确定”按钮
        self.bt_confirm.Bind(wx.EVT_BUTTON, self.OnclickSubmit1)
        # 创建文本
        wx.StaticText(panel, label='请点击右侧按钮开始录音', pos=(50, 150))
        wx.StaticText(panel, label='或者选择您的wav格式录音文件', pos=(50, 250))
        self.bt_confirm = wx.Button(panel, label='确定', pos=(480, 250))  # 创建“确定”按钮

        self.bt_confirm.Bind(wx.EVT_BUTTON, self.OnclickSubmit3)
        # 创建按钮
        self.bt_confirm = wx.Button(panel, label='确定', pos=(210, 150))  # 创建“确定”按钮

        self.bt_confirm.Bind(wx.EVT_BUTTON, self.OnclickSubmit2)


    def OnclickSubmit1(self, event): #第一个确定键
        """ 点击确定按钮，执行方法 """
        message = ""
        timeLength = self.text_user.GetValue()     # 获取输入的录音时长
        if  timeLength == "" :    # 判断录音时长是否为空
            message = '时长不能为空'
            wx.MessageBox(message)  # 弹出提示框
        elif int(timeLength) < 5 :  # 录音时长太短
            message = '录音时长太短'
            wx.MessageBox(message)  # 弹出提示框
        elif int(timeLength) > 10 :  # 录音时长太短
            message = '录音时长太长'
            wx.MessageBox(message)  # 弹出提示框

    def OnclickSubmit3(self, event):
        """上传WAV文件进行预测"""
        try:
            path = self.picker.GetPath()
            if not path or not os.path.exists(path):
                wx.MessageBox('请选择一个有效的文件', '错误', wx.ICON_ERROR)
                return
            
            if not path.lower().endswith('.wav'):
                wx.MessageBox('请选择WAV格式的音频文件', '错误', wx.ICON_ERROR)
                return
            
            wx.MessageBox('正在处理音频文件,请稍候...', '提示', wx.ICON_INFORMATION)
            
            # 生成声谱图
            input_dir = os.path.dirname(path)
            filename = os.path.basename(path).replace('.wav', '')
            spec_filename = filename + '_1.png'
            spec_path = os.path.join(TEMP_DIR, spec_filename)
            
            # 调用create_spec生成声谱图
            plotstft(path, channel=0, name=spec_path, alpha=1.0)
            
            # 转换为RGB
            img = Image.open(spec_path).convert("RGB")
            img.save(spec_path)
            
            # 预测
            result_text = main(spec_path)
            
            # 显示结果
            wx.MessageBox(result_text, '识别结果', wx.ICON_INFORMATION)
            
            # 清理临时文件
            if os.path.exists(spec_path):
                os.remove(spec_path)
                
        except Exception as e:
            wx.MessageBox(f'处理失败: {str(e)}', '错误', wx.ICON_ERROR)
            print(f"错误详情: {e}")
            import traceback
            traceback.print_exc()
    def OnclickSubmit2(self, event):
        """开始录音并预测"""
        try:
            timeLength = self.text_user.GetValue()
            
            if not timeLength:
                wx.MessageBox('请先输入录音时长', '提示', wx.ICON_WARNING)
                return
            
            try:
                duration = int(timeLength)
            except ValueError:
                wx.MessageBox('请输入有效的数字', '错误', wx.ICON_ERROR)
                return
            
            if duration < 5 or duration > 10:
                wx.MessageBox('录音时长必须在5-10秒之间', '错误', wx.ICON_ERROR)
                return
            
            wx.MessageBox(f'准备录音{duration}秒,点击确定开始...', '提示', wx.ICON_INFORMATION)
            
            # 定义临时文件路径
            raw_audio_path = os.path.join(TEMP_DIR, '录音样本.wav')
            converted_audio_path = os.path.join(TEMP_DIR, '转换声道后.wav')
            spec_image_path = os.path.join(TEMP_DIR, '录音样本_1.png')
            
            # 录音
            record_audio(raw_audio_path, duration)
            
            # 转换为单声道
            sound = AudioSegment.from_file(raw_audio_path, "wav")
            sound = sound.set_channels(1)
            sound.export(converted_audio_path, format="wav")
            
            # 生成声谱图
            plotstft(converted_audio_path, channel=0, name=spec_image_path, alpha=1.0)
            
            # 转换为RGB
            img = Image.open(spec_image_path).convert("RGB")
            img.save(spec_image_path)
            
            # 预测
            result_text = main(spec_image_path)
            
            # 显示结果
            wx.MessageBox(result_text, '识别结果', wx.ICON_INFORMATION)
            
            # 清理临时文件
            for temp_file in [raw_audio_path, converted_audio_path, spec_image_path]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                        
        except Exception as e:
            wx.MessageBox(f'录音或识别失败: {str(e)}', '错误', wx.ICON_ERROR)
            print(f"错误详情: {e}")
            import traceback
            traceback.print_exc()

def main(img_path):
    """预测声谱图对应的方言"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model]),
         transforms.CenterCrop(img_size[num_model]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 加载图像
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"图像文件不存在: {img_path}")
    
    img = Image.open(img_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # 读取类别索引
    json_path = os.path.join(SCRIPT_DIR, 'class_indices.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"类别索引文件不存在: {json_path}")
    
    with open(json_path, "r", encoding='utf-8') as f:
        class_indict = json.load(f)

    # 创建模型
    model = create_model(num_classes=9).to(device)
    
    # 加载模型权重
    model_weight_path = os.path.join(SCRIPT_DIR, 'weight', 'model-29.pth')
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f"模型权重文件不存在: {model_weight_path}")
    
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    
    # 预测
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # 方言名称映射
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
    
    # 构建结果文本
    result_text = f"识别结果: {predicted_name}\n"
    result_text += f"置信度: {probability:.2%}\n\n"
    result_text += "所有类别概率:\n"
    
    # 按概率排序
    sorted_indices = torch.argsort(predict, descending=True)
    for idx in sorted_indices:
        class_name = class_indict[str(idx.item())]
        dialect_name = dialect_names.get(class_name, class_name)
        prob = predict[idx].numpy()
        result_text += f"{dialect_name}: {prob:.2%}\n"
    
    print(result_text)
    return result_text


if __name__ == '__main__':
    try:
        app = wx.App()
        frame = MyFrame(parent=None, id=-1)
        frame.Show()
        app.MainLoop()
    except Exception as e:
        print(f"程序启动失败: {e}")
        import traceback
        traceback.print_exc()



