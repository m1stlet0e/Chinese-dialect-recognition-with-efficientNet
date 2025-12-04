#!/usr/bin/env python3
"""
音频格式转换工具
将任意音频转换为适合模型识别的格式
"""
import sys
import os
import argparse
from pydub import AudioSegment
from pydub.effects import normalize
import scipy.io.wavfile as wav
from scipy import signal
import numpy as np

def convert_audio(input_file, output_file=None, target_sr=16000, duration=5, denoise=False):
    """
    转换音频到模型要求的格式
    
    参数:
        input_file: 输入音频文件路径
        output_file: 输出文件路径（默认在同目录下生成）
        target_sr: 目标采样率（默认16000Hz）
        duration: 截取时长（秒，默认5秒）
        denoise: 是否降噪（简单降噪）
    """
    print(f"\n{'='*60}")
    print(f"音频转换工具")
    print(f"{'='*60}")
    
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        return False
    
    # 生成输出文件名
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_converted.wav"
    
    try:
        # 1. 读取音频
        print(f"\n📂 读取音频: {os.path.basename(input_file)}")
        audio = AudioSegment.from_file(input_file)
        
        print(f"   原始格式:")
        print(f"   - 采样率: {audio.frame_rate} Hz")
        print(f"   - 声道数: {audio.channels}")
        print(f"   - 位深: {audio.sample_width * 8} bit")
        print(f"   - 时长: {len(audio) / 1000:.2f} 秒")
        
        # 2. 截取指定时长
        if len(audio) > duration * 1000:
            print(f"\n✂️  截取前 {duration} 秒")
            audio = audio[:duration * 1000]
        
        # 3. 转单声道
        if audio.channels > 1:
            print(f"🔊 转换为单声道")
            audio = audio.set_channels(1)
        
        # 4. 设置采样率
        if audio.frame_rate != target_sr:
            print(f"🔄 重采样: {audio.frame_rate} Hz → {target_sr} Hz")
            audio = audio.set_frame_rate(target_sr)
        
        # 5. 设置位深为16-bit
        if audio.sample_width != 2:
            print(f"🔧 设置位深为 16-bit")
            audio = audio.set_sample_width(2)
        
        # 6. 音量归一化（关键：匹配训练数据的能量水平）
        print(f"📊 音量归一化（匹配训练数据能量）")
        
        # 获取音频样本
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        # 计算当前能量
        current_std = np.std(samples)
        
        # 目标能量水平（训练数据的典型值）
        target_std = 250.0  # 训练数据标准差约200-300
        
        # 缩放到目标能量
        if current_std > 0:
            scale_factor = target_std / current_std
            samples = samples * scale_factor
            
            # 确保不溢出
            max_val = 32767
            if np.abs(samples).max() > max_val:
                samples = samples * (max_val / np.abs(samples).max())
            
            # 转回AudioSegment
            samples = np.int16(samples)
            audio = AudioSegment(
                samples.tobytes(),
                frame_rate=target_sr,
                sample_width=2,
                channels=1
            )
        
        print(f"   原始能量: {current_std:.1f}")
        print(f"   目标能量: {target_std:.1f}")
        print(f"   缩放倍数: {scale_factor:.3f}x")
        
        # 7. 简单降噪（可选）
        if denoise:
            print(f"🔇 降噪处理")
            # 导出为numpy数组
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples = samples / (2**15)  # 归一化到 [-1, 1]
            
            # 简单的高通滤波去除低频噪音
            sos = signal.butter(5, 100, 'highpass', fs=target_sr, output='sos')
            samples = signal.sosfilt(sos, samples)
            
            # 转回AudioSegment
            samples = np.int16(samples * 32767)
            audio = AudioSegment(
                samples.tobytes(),
                frame_rate=target_sr,
                sample_width=2,
                channels=1
            )
        
        # 8. 导出
        print(f"\n💾 保存文件: {os.path.basename(output_file)}")
        audio.export(output_file, format="wav")
        
        print(f"\n✅ 转换成功！")
        print(f"   目标格式:")
        print(f"   - 采样率: {target_sr} Hz")
        print(f"   - 声道数: 1（单声道）")
        print(f"   - 位深: 16 bit")
        print(f"   - 时长: {len(audio) / 1000:.2f} 秒")
        print(f"   - 文件: {output_file}")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def batch_convert(input_dir, output_dir=None, **kwargs):
    """批量转换目录下的所有音频文件"""
    if output_dir is None:
        output_dir = os.path.join(input_dir, "converted")
    
    os.makedirs(output_dir, exist_ok=True)
    
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
    files = [f for f in os.listdir(input_dir) 
             if os.path.splitext(f)[1].lower() in audio_extensions]
    
    print(f"\n找到 {len(files)} 个音频文件")
    
    success = 0
    for i, filename in enumerate(files, 1):
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + "_converted.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"\n[{i}/{len(files)}] 处理: {filename}")
        if convert_audio(input_path, output_path, **kwargs):
            success += 1
    
    print(f"\n{'='*60}")
    print(f"批量转换完成: {success}/{len(files)} 成功")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='音频格式转换工具')
    parser.add_argument('input', help='输入音频文件或目录')
    parser.add_argument('-o', '--output', help='输出文件或目录')
    parser.add_argument('-sr', '--sample-rate', type=int, default=16000, 
                        help='目标采样率（默认16000Hz）')
    parser.add_argument('-d', '--duration', type=int, default=5,
                        help='截取时长（秒，默认5秒）')
    parser.add_argument('--denoise', action='store_true',
                        help='启用降噪')
    parser.add_argument('-b', '--batch', action='store_true',
                        help='批量转换模式')
    
    args = parser.parse_args()
    
    if args.batch or os.path.isdir(args.input):
        batch_convert(args.input, args.output, 
                     target_sr=args.sample_rate, 
                     duration=args.duration,
                     denoise=args.denoise)
    else:
        convert_audio(args.input, args.output,
                     target_sr=args.sample_rate,
                     duration=args.duration,
                     denoise=args.denoise)

