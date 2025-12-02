"""
音频文件诊断工具
检查音频文件是否符合识别要求
"""

import sys
import os
import wave
import numpy as np
import scipy.io.wavfile as wav
from PIL import Image
import matplotlib.pyplot as plt


def analyze_audio(audio_path):
    """分析音频文件"""
    print("\n" + "="*60)
    print(f"音频文件诊断: {os.path.basename(audio_path)}")
    print("="*60)
    
    if not os.path.exists(audio_path):
        print(f"❌ 错误: 文件不存在 {audio_path}")
        return
    
    try:
        # 读取WAV文件基本信息
        with wave.open(audio_path, 'rb') as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            duration = n_frames / framerate
            
        print(f"\n📊 基本信息:")
        print(f"  声道数: {channels} ({'单声道' if channels == 1 else '多声道'})")
        print(f"  采样位数: {sample_width * 8} bit")
        print(f"  采样率: {framerate} Hz")
        print(f"  总帧数: {n_frames}")
        print(f"  时长: {duration:.2f} 秒")
        
        # 检查时长
        if duration < 5:
            print(f"  ⚠️  警告: 时长太短 ({duration:.2f}秒)，建议5-10秒")
        elif duration > 10:
            print(f"  ⚠️  警告: 时长太长 ({duration:.2f}秒)，建议5-10秒")
        else:
            print(f"  ✓ 时长合适")
        
        # 检查采样率
        if framerate < 16000:
            print(f"  ⚠️  警告: 采样率较低 ({framerate} Hz)，建议16000 Hz以上")
        else:
            print(f"  ✓ 采样率合适")
        
        # 读取音频数据
        samplerate, samples = wav.read(audio_path)
        
        # 转换为单声道
        if len(samples.shape) > 1:
            samples = samples[:, 0]
        
        # 计算音量统计
        samples_float = samples.astype(float)
        max_amplitude = np.max(np.abs(samples_float))
        mean_amplitude = np.mean(np.abs(samples_float))
        
        # 归一化到0-1范围
        if sample_width == 2:  # 16-bit
            max_possible = 32768.0
        else:
            max_possible = 256.0
        
        max_volume = max_amplitude / max_possible
        mean_volume = mean_amplitude / max_possible
        
        print(f"\n🔊 音量分析:")
        print(f"  最大音量: {max_volume:.2%}")
        print(f"  平均音量: {mean_volume:.2%}")
        
        if max_volume < 0.1:
            print(f"  ⚠️  警告: 音量太小，可能影响识别")
        elif max_volume > 0.95:
            print(f"  ⚠️  警告: 音量可能过载")
        else:
            print(f"  ✓ 音量正常")
        
        # 计算信噪比估计
        # 使用能量法估计
        energy = np.sum(samples_float ** 2) / len(samples_float)
        noise_estimate = np.percentile(np.abs(samples_float), 10)  # 使用10%分位数估计噪声
        
        if noise_estimate > 0:
            snr_estimate = 20 * np.log10(max_amplitude / noise_estimate)
            print(f"\n📡 信噪比估计:")
            print(f"  SNR: {snr_estimate:.1f} dB")
            
            if snr_estimate < 10:
                print(f"  ⚠️  警告: 噪声较大，建议在安静环境录音")
            elif snr_estimate < 20:
                print(f"  ⚠️  注意: 有一定背景噪音")
            else:
                print(f"  ✓ 信噪比良好")
        
        # 静音检测
        silence_threshold = max_amplitude * 0.05
        silence_frames = np.sum(np.abs(samples_float) < silence_threshold)
        silence_ratio = silence_frames / len(samples_float)
        
        print(f"\n🔇 静音分析:")
        print(f"  静音比例: {silence_ratio:.1%}")
        
        if silence_ratio > 0.5:
            print(f"  ⚠️  警告: 静音过多 ({silence_ratio:.1%})，可能录音失败")
        elif silence_ratio > 0.3:
            print(f"  ⚠️  注意: 静音较多")
        else:
            print(f"  ✓ 语音内容充足")
        
        # 频率分析
        fft = np.fft.fft(samples_float)
        freqs = np.fft.fftfreq(len(samples_float), 1/samplerate)
        
        # 只看正频率
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft[:len(fft)//2])
        
        # 找到主要频率
        dominant_freq_idx = np.argmax(positive_fft)
        dominant_freq = positive_freqs[dominant_freq_idx]
        
        print(f"\n🎵 频率分析:")
        print(f"  主要频率: {abs(dominant_freq):.0f} Hz")
        
        # 人声一般在85-255 Hz (基频)
        if 85 <= abs(dominant_freq) <= 255:
            print(f"  ✓ 频率范围符合人声特征")
        else:
            print(f"  ⚠️  注意: 主频率不在典型人声范围")
        
        # 综合评分
        print(f"\n📈 综合评估:")
        score = 100
        issues = []
        
        if duration < 5 or duration > 10:
            score -= 20
            issues.append("时长不合适")
        
        if max_volume < 0.1:
            score -= 25
            issues.append("音量太小")
        
        if silence_ratio > 0.5:
            score -= 30
            issues.append("静音过多")
        
        if framerate < 16000:
            score -= 15
            issues.append("采样率偏低")
        
        print(f"  质量评分: {score}/100")
        
        if score >= 80:
            print(f"  ✓ 音频质量良好，适合识别")
        elif score >= 60:
            print(f"  ⚠️  音频质量一般，可能影响识别准确度")
            print(f"  问题: {', '.join(issues)}")
        else:
            print(f"  ❌ 音频质量较差，建议重新录制")
            print(f"  问题: {', '.join(issues)}")
        
        print("\n" + "="*60)
        
        # 建议
        print("\n💡 改进建议:")
        if duration < 5:
            print("  • 增加录音时长到5-10秒")
        if max_volume < 0.1:
            print("  • 增加麦克风音量或靠近麦克风")
        if silence_ratio > 0.3:
            print("  • 减少录音前后的停顿，保持连续说话")
        if framerate < 16000:
            print("  • 使用更高的采样率录音（建议16000 Hz或48000 Hz）")
        if noise_estimate / max_possible > 0.05:
            print("  • 在更安静的环境中录音")
        
        print("  • 说话清晰、发音标准")
        print("  • 使用典型的方言词汇和语调")
        print("  • 保持稳定的语速")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


def visualize_audio(audio_path):
    """可视化音频波形"""
    try:
        samplerate, samples = wav.read(audio_path)
        
        if len(samples.shape) > 1:
            samples = samples[:, 0]
        
        time = np.linspace(0, len(samples) / samplerate, num=len(samples))
        
        plt.figure(figsize=(12, 4))
        plt.plot(time, samples)
        plt.xlabel('时间 (秒)')
        plt.ylabel('振幅')
        plt.title(f'音频波形 - {os.path.basename(audio_path)}')
        plt.grid(True, alpha=0.3)
        
        output_path = audio_path.replace('.wav', '_waveform.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n波形图已保存: {output_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"波形可视化失败: {e}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使用方法: python diagnose_audio.py <音频文件.wav>")
        print("示例: python diagnose_audio.py test.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    analyze_audio(audio_file)
    
    # 询问是否生成波形图
    if len(sys.argv) > 2 and sys.argv[2] == '--visualize':
        visualize_audio(audio_file)


