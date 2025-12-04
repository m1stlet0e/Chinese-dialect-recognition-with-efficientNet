"""
长音频方言识别工具
将长音频切分为多个片段，分别识别后汇总结果
支持任意长度的音频文件（包括1小时以上）
"""

import os
import sys
import argparse
import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from collections import Counter

# 添加API目录到路径
sys.path.append(os.path.dirname(__file__))

# 导入预测函数
import torch
from PIL import Image
from torchvision import transforms
from model import efficientnet_b0 as create_model
from numpy.lib import stride_tricks


class LongAudioPredictor:
    """长音频预测器"""
    
    def __init__(self, model_path, class_indices_path, segment_length=7):
        """
        初始化预测器
        
        Args:
            model_path: 模型权重路径
            class_indices_path: 类别索引文件路径
            segment_length: 分段长度（秒）
        """
        self.segment_length = segment_length
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 加载类别映射
        with open(class_indices_path, 'r', encoding='utf-8') as f:
            self.class_indict = json.load(f)
        
        # 方言中文名称
        self.dialect_names = {
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
        
        # 加载模型
        self.model = create_model(num_classes=9).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # 数据转换
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"✓ 模型加载成功")
        print(f"✓ 使用设备: {self.device}")
        print(f"✓ 分段长度: {segment_length}秒")
    
    def load_audio(self, audio_path):
        """加载音频文件"""
        ext = audio_path.lower().split('.')[-1]
        
        if ext == 'wav':
            audio = AudioSegment.from_wav(audio_path)
        elif ext == 'mp3':
            audio = AudioSegment.from_mp3(audio_path)
        else:
            raise ValueError(f"不支持的音频格式: {ext}")
        
        # 转换为单声道，16kHz
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        return audio
    
    def split_audio(self, audio):
        """将音频切分为固定长度的片段"""
        duration_ms = len(audio)
        segment_ms = self.segment_length * 1000
        
        segments = []
        for start_ms in range(0, duration_ms, segment_ms):
            end_ms = min(start_ms + segment_ms, duration_ms)
            segment = audio[start_ms:end_ms]
            
            # 只保留长度足够的片段（至少3秒）
            if len(segment) >= 3000:
                segments.append(segment)
        
        return segments
    
    def audio_to_spectrogram(self, audio_segment):
        """将音频片段转换为声谱图"""
        # 转为numpy数组
        samples = np.array(audio_segment.get_array_of_samples())
        
        # STFT
        def stft(sig, frameSize=1024, overlapFac=0.5):
            win = np.hanning(frameSize)
            hopSize = int(frameSize - np.floor(overlapFac * frameSize))
            samples_padded = np.append(np.zeros(int(np.floor(frameSize / 2.0))), sig)
            cols = int(np.ceil((len(samples_padded) - frameSize) / float(hopSize)) + 1)
            samples_padded = np.append(samples_padded, np.zeros(frameSize))
            frames = stride_tricks.as_strided(
                samples_padded,
                shape=(cols, frameSize),
                strides=(samples_padded.strides[0] * hopSize, samples_padded.strides[0])
            ).copy()
            frames *= win
            return np.fft.rfft(frames)
        
        # 对数缩放
        def logscale_spec(spec, sr=16000):
            spec = spec[:, 0:256]
            timebins, freqbins = np.shape(spec)
            scale = np.linspace(0, 1, freqbins)
            scale = np.array([x if x <= 0.9 else (1 - 0.9 * 0.9) / (1 - 0.9) * (x - 0.9) + 0.9 * 0.9 
                            for x in scale])
            scale *= (freqbins - 1) / max(scale)
            
            newspec = np.complex128(np.zeros([timebins, freqbins]))
            for i in range(0, freqbins):
                if i < 1 or i + 1 >= freqbins:
                    newspec[:, i] = spec[:, i]
                else:
                    w_up = scale[i] - np.floor(scale[i])
                    w_down = 1 - w_up
                    j = int(np.floor(scale[i]))
                    newspec[:, j] += w_down * spec[:, i]
                    newspec[:, j + 1] += w_up * spec[:, i]
            
            return newspec
        
        s = stft(samples)
        sshow = logscale_spec(s)[2:, :]
        ims = 20. * np.log10(np.abs(sshow) / 10e-6)
        ims = np.transpose(ims)
        ims = ims[0:256, :]
        
        # 转为PIL Image
        image = Image.fromarray(ims).convert('L').convert('RGB')
        
        return image
    
    def predict_segment(self, image):
        """预测单个片段"""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = torch.squeeze(self.model(img_tensor)).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        
        return predict_cla, predict.numpy()
    
    def predict_long_audio(self, audio_path, show_progress=True):
        """
        预测长音频
        
        Returns:
            dict: 包含整体结果和详细片段信息
        """
        print(f"\n正在处理: {os.path.basename(audio_path)}")
        
        # 加载音频
        print("加载音频文件...")
        audio = self.load_audio(audio_path)
        duration_sec = len(audio) / 1000.0
        print(f"音频时长: {duration_sec:.1f}秒 ({duration_sec/60:.1f}分钟)")
        
        # 切分音频
        print(f"切分为{self.segment_length}秒的片段...")
        segments = self.split_audio(audio)
        print(f"共{len(segments)}个有效片段")
        
        # 逐个识别
        segment_results = []
        predictions_list = []
        
        iterator = tqdm(segments, desc="识别中") if show_progress else segments
        
        for i, segment in enumerate(iterator):
            try:
                # 转声谱图
                spec_image = self.audio_to_spectrogram(segment)
                
                # 预测
                pred_class, probs = self.predict_segment(spec_image)
                
                class_name = self.class_indict[str(pred_class)]
                dialect_name = self.dialect_names.get(class_name, class_name)
                
                segment_results.append({
                    'segment_id': i,
                    'start_time': i * self.segment_length,
                    'end_time': min((i + 1) * self.segment_length, duration_sec),
                    'predicted_class': class_name,
                    'predicted_dialect': dialect_name,
                    'confidence': float(probs[pred_class])
                })
                
                predictions_list.append(pred_class)
                
            except Exception as e:
                print(f"片段{i}处理失败: {e}")
                continue
        
        # 统计结果
        prediction_counts = Counter(predictions_list)
        
        # 计算每个方言的占比
        total_segments = len(predictions_list)
        dialect_stats = {}
        
        for pred_class, count in prediction_counts.items():
            class_name = self.class_indict[str(pred_class)]
            dialect_name = self.dialect_names.get(class_name, class_name)
            ratio = count / total_segments
            
            dialect_stats[dialect_name] = {
                'count': count,
                'ratio': ratio,
                'percentage': ratio * 100
            }
        
        # 主要方言（占比最高）
        main_dialect = max(dialect_stats.items(), key=lambda x: x[1]['ratio'])
        
        result = {
            'audio_file': os.path.basename(audio_path),
            'duration_seconds': duration_sec,
            'total_segments': len(segments),
            'valid_segments': total_segments,
            'main_dialect': main_dialect[0],
            'main_dialect_ratio': main_dialect[1]['ratio'],
            'dialect_distribution': dialect_stats,
            'segment_details': segment_results
        }
        
        return result
    
    def visualize_results(self, result, save_path=None):
        """可视化结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 方言分布饼图
        dialect_dist = result['dialect_distribution']
        labels = list(dialect_dist.keys())
        sizes = [d['percentage'] for d in dialect_dist.values()]
        colors = plt.cm.Set3(range(len(labels)))
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 10})
        ax1.set_title(f'方言分布\n主要方言: {result["main_dialect"]} '
                     f'({result["main_dialect_ratio"]*100:.1f}%)', 
                     fontsize=14, fontproperties='SimHei')
        
        # 时间轴分布
        segments = result['segment_details']
        times = [s['start_time'] for s in segments]
        dialects = [s['predicted_dialect'] for s in segments]
        
        # 为每个方言分配颜色
        unique_dialects = list(set(dialects))
        color_map = {d: colors[i % len(colors)] for i, d in enumerate(unique_dialects)}
        segment_colors = [color_map[d] for d in dialects]
        
        ax2.scatter(times, range(len(times)), c=segment_colors, s=100, alpha=0.6)
        ax2.set_xlabel('时间 (秒)', fontsize=12)
        ax2.set_ylabel('片段序号', fontsize=12)
        ax2.set_title('时间轴方言分布', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 添加图例
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=color_map[d], markersize=10, label=d)
                  for d in unique_dialects]
        ax2.legend(handles=handles, loc='best')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 可视化结果已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='长音频方言识别')
    parser.add_argument('audio_file', type=str, help='音频文件路径')
    parser.add_argument('--model', type=str,
                       default='../GUI/weight/model-29.pth',
                       help='模型权重路径')
    parser.add_argument('--class_indices', type=str,
                       default='../GUI/class_indices.json',
                       help='类别索引文件路径')
    parser.add_argument('--segment_length', type=int, default=7,
                       help='分段长度（秒）')
    parser.add_argument('--output', type=str, default=None,
                       help='结果保存路径（JSON）')
    parser.add_argument('--visualize', action='store_true',
                       help='生成可视化图表')
    
    args = parser.parse_args()
    
    # 检查文件
    if not os.path.exists(args.audio_file):
        print(f"错误: 找不到音频文件 {args.audio_file}")
        return
    
    # 创建预测器
    print("="*60)
    print("长音频方言识别工具")
    print("="*60)
    
    predictor = LongAudioPredictor(
        model_path=args.model,
        class_indices_path=args.class_indices,
        segment_length=args.segment_length
    )
    
    # 预测
    result = predictor.predict_long_audio(args.audio_file)
    
    # 打印结果
    print("\n" + "="*60)
    print("识别结果")
    print("="*60)
    print(f"音频文件: {result['audio_file']}")
    print(f"总时长: {result['duration_seconds']:.1f}秒 ({result['duration_seconds']/60:.1f}分钟)")
    print(f"分析片段: {result['valid_segments']}/{result['total_segments']}")
    print(f"\n主要方言: {result['main_dialect']} ({result['main_dialect_ratio']*100:.1f}%)")
    
    print(f"\n方言分布:")
    for dialect, stats in sorted(result['dialect_distribution'].items(), 
                                 key=lambda x: x[1]['ratio'], reverse=True):
        print(f"  {dialect}: {stats['count']}片段 ({stats['percentage']:.1f}%)")
    
    # 保存结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 详细结果已保存: {args.output}")
    
    # 可视化
    if args.visualize:
        vis_path = args.audio_file.replace('.wav', '_result.png').replace('.mp3', '_result.png')
        predictor.visualize_results(result, vis_path)
    
    print("="*60)


if __name__ == '__main__':
    main()


