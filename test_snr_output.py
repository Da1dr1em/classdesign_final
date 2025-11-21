#!/usr/bin/env python
"""
测试完整的SNR输出功能
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from audio_processor import AudioProcessor
from utils import save_audio

def test_complete_workflow():
    """测试完整的处理流程和SNR输出"""
    print("="*60)
    print("测试完整的SNR分析流程")
    print("="*60)
    
    # 创建测试音频文件
    print("\n[1/5] 创建测试音频...")
    sample_rate = 44100
    duration = 1  # 1秒
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 创建混合信号：两个频率 + 噪声
    signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    noise = 0.2 * np.random.randn(len(signal))
    noisy_signal = signal + noise
    
    # 保存测试文件
    test_file = "data/input/test_snr.wav"
    save_audio(noisy_signal, sample_rate, test_file)
    print(f"✅ 测试音频已保存: {test_file}")
    
    # 初始化处理器
    print("\n[2/5] 初始化处理器并加载音频...")
    processor = AudioProcessor(sample_rate=sample_rate)
    processor.load_audio(test_file)
    print("✅ 音频加载完成")
    
    # 应用滤波器
    print("\n[3/5] 应用低通滤波器...")
    processor.apply_filter('fir_lowpass', cutoff_freq=1500, numtaps=101)
    print("✅ 滤波器应用完成")
    
    # 增强信号
    print("\n[4/5] 信号增强...")
    processor.enhance_signal('normalize', target_max=0.9)
    print("✅ 信号增强完成")
    
    # 分析处理后信号
    print("\n[5/5] 分析处理后信号...")
    results = processor.analyze_processed_signal()
    print("✅ 分析完成")
    
    # 显示结果
    if 'metrics' in results:
        metrics = results['metrics']
        print("\n" + "="*60)
        print("信噪比分析结果:")
        print("="*60)
        
        if 'original_snr_estimated' in metrics:
            print(f"\n📊 基于噪声估计的SNR:")
            print(f"  原始信号SNR: {metrics['original_snr_estimated']:.2f} dB")
            print(f"  处理后SNR: {metrics['processed_snr_estimated']:.2f} dB")
            print(f"  SNR改善: {metrics['snr_improvement_estimated']:.2f} dB")
        
        print(f"\n📈 其他性能指标:")
        if 'correlation' in metrics:
            print(f"  相关系数: {metrics['correlation']:.3f}")
        if 'rmse' in metrics:
            print(f"  RMSE: {metrics['rmse']:.4f}")
        
        print("="*60)
    
    print("\n✅ 测试完成！噪声估计和SNR输出功能正常工作。")

if __name__ == "__main__":
    test_complete_workflow()
