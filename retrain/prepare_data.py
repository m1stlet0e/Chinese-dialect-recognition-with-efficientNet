"""
数据预处理脚本
PCM音频 → WAV → 声谱图 → RGB图像
"""

import os
import sys
import wave
import numpy as np
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path


def pcm_to_wav(pcm_path, wav_path, channels=1, sample_width=2, framerate=16000):
    """PCM转WAV"""
    try:
        with open(pcm_path, 'rb') as pcm_file:
            pcm_data = pcm_file.read()
        
        with wave.open(wav_path, 'wb') as wav_file:
            wav_file.setparams((channels, sample_width, framerate, 0, 'NONE', 'NONE'))
            wav_file.writeframes(pcm_data)
        return True
    except Exception as e:
        print(f"转换失败 {pcm_path}: {e}")
        return False


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    """短时傅里叶变换"""
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    samples = np.append(np.zeros(int(np.floor(frameSize / 2.0))), sig)
    cols = int(np.ceil((len(samples) - frameSize) / float(hopSize)) + 1)
    samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(
        samples, 
        shape=(cols, frameSize),
        strides=(samples.strides[0] * hopSize, samples.strides[0])
    ).copy()
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


def wav_to_spectrogram(wav_path, spec_path, sr=16000):
    """WAV转声谱图"""
    try:
        samplerate, samples = wav.read(wav_path)
        
        # 转单声道
        if len(samples.shape) > 1:
            samples = samples[:, 0]
        
        # STFT
        s = stft(samples, 2 ** 10)
        
        # 对数缩放
        sshow, freq = logscale_spec(s, factor=1, sr=samplerate, alpha=1.0)
        sshow = sshow[2:, :]
        
        # 转分贝
        ims = 20. * np.log10(np.abs(sshow) / 10e-6)
        ims = np.transpose(ims)
        ims = ims[0:256, :]
        
        # 保存为灰度图
        image = Image.fromarray(ims)
        image = image.convert('L')
        image.save(spec_path)
        
        return True
    except Exception as e:
        print(f"声谱图生成失败 {wav_path}: {e}")
        return False


def grey_to_rgb(grey_path, rgb_path):
    """灰度转RGB"""
    try:
        img = Image.open(grey_path).convert("RGB")
        img.save(rgb_path)
        return True
    except Exception as e:
        print(f"RGB转换失败 {grey_path}: {e}")
        return False


def process_dialect(dialect_dir, output_dir, dialect_name, split='train'):
    """
    处理单个方言的数据
    
    Args:
        dialect_dir: 方言目录（包含train/dev）
        output_dir: 输出目录
        dialect_name: 方言名称（英文）
        split: 'train' 或 'val'
    """
    
    english_name = dialect_name
    
    # 创建输出目录
    output_dialect_dir = os.path.join(output_dir, english_name)
    os.makedirs(output_dialect_dir, exist_ok=True)
    
    # 临时目录
    temp_wav_dir = os.path.join(output_dir, '.temp_wav', english_name)
    temp_grey_dir = os.path.join(output_dir, '.temp_grey', english_name)
    os.makedirs(temp_wav_dir, exist_ok=True)
    os.makedirs(temp_grey_dir, exist_ok=True)
    
    # 获取PCM文件列表
    pcm_files = []
    source_dir = os.path.join(dialect_dir, split if split == 'train' else 'dev')
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.pcm'):
                pcm_files.append(os.path.join(root, file))
    
    print(f"\n处理 {dialect_name} ({english_name}) - {split} 集: {len(pcm_files)} 个文件")
    
    success_count = 0
    
    for pcm_file in tqdm(pcm_files, desc=f"{dialect_name}-{split}"):
        try:
            # 文件名
            base_name = os.path.splitext(os.path.basename(pcm_file))[0]
            
            # 1. PCM → WAV
            wav_file = os.path.join(temp_wav_dir, f"{base_name}.wav")
            if not pcm_to_wav(pcm_file, wav_file):
                continue
            
            # 2. WAV → 灰度声谱图
            grey_file = os.path.join(temp_grey_dir, f"{base_name}.png")
            if not wav_to_spectrogram(wav_file, grey_file):
                os.remove(wav_file)
                continue
            
            # 3. 灰度 → RGB
            rgb_file = os.path.join(output_dialect_dir, f"{base_name}.png")
            if not grey_to_rgb(grey_file, rgb_file):
                os.remove(wav_file)
                os.remove(grey_file)
                continue
            
            # 清理临时文件
            os.remove(wav_file)
            os.remove(grey_file)
            
            success_count += 1
            
        except Exception as e:
            print(f"处理失败 {pcm_file}: {e}")
            continue
    
    print(f"完成 {dialect_name}: {success_count}/{len(pcm_files)} 个文件")
    
    return success_count


def main():
    parser = argparse.ArgumentParser(description='方言数据预处理')
    parser.add_argument('--input_dir', type=str, 
                       default='/Users/wangbo/Desktop/origin_data',
                       help='输入数据集目录')
    parser.add_argument('--output_dir', type=str,
                       default='./processed_data',
                       help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                       help='训练集使用比例 (0-1)')
    parser.add_argument('--val_ratio', type=float, default=1.0,
                       help='验证集使用比例 (0-1)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("方言识别数据预处理")
    print("="*60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print("="*60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 需要处理的方言（处理所有10种）
    dialects_to_process = ['changsha', 'hebei', 'hefei', 'kejia', 'minnan', 
                          'nanchang', 'ningxia', 'shan3xi', 'shanghai', 'sichuan']
    
    # 处理每个方言
    total_stats = {'train': {}, 'val': {}}
    
    for dialect in dialects_to_process:
        dialect_dir = os.path.join(args.input_dir, dialect)
        
        if not os.path.exists(dialect_dir):
            print(f"警告: 找不到 {dialect} 目录，跳过")
            continue
        
        # 处理训练集
        train_count = process_dialect(dialect_dir, args.output_dir, dialect, 'train')
        total_stats['train'][dialect] = train_count
        
        # 处理验证集
        val_count = process_dialect(dialect_dir, args.output_dir, dialect, 'val')
        total_stats['val'][dialect] = val_count
    
    # 清理临时目录
    import shutil
    temp_wav_dir = os.path.join(args.output_dir, '.temp_wav')
    temp_grey_dir = os.path.join(args.output_dir, '.temp_grey')
    if os.path.exists(temp_wav_dir):
        shutil.rmtree(temp_wav_dir)
    if os.path.exists(temp_grey_dir):
        shutil.rmtree(temp_grey_dir)
    
    # 打印统计信息
    print("\n" + "="*60)
    print("数据处理完成!")
    print("="*60)
    print("\n训练集统计:")
    for dialect, count in total_stats['train'].items():
        print(f"  {dialect}: {count} 张")
    print(f"\n训练集总计: {sum(total_stats['train'].values())} 张")
    
    print("\n验证集统计:")
    for dialect, count in total_stats['val'].items():
        print(f"  {dialect}: {count} 张")
    print(f"\n验证集总计: {sum(total_stats['val'].values())} 张")
    
    print(f"\n数据保存在: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

