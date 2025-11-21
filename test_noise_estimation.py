#!/usr/bin/env python
"""
测试噪声估计功能
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from audio_processor import AudioProcessor
from utils import estimate_noise_vad

def test_noise_estimation():
    """测试噪声估计功能"""
    print("="*60)
    print("测试噪声估计功能")
    print("="*60)
    
    # 创建测试信号
    print("\n1. 创建测试信号...")
    sample_rate = 44100
    duration = 2  # 2秒
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 纯信号：440Hz正弦波
    clean_signal = np.sin(2 * np.pi * 440 * t)
    
    # 添加噪声
    noise = 0.3 * np.random.randn(len(clean_signal))
    noisy_signal = clean_signal + noise
    
    print(f"✅ 生成信号长度: {len(noisy_signal)} 样本")
    
    # 测试噪声估计
    print("\n2. 测试噪声估计...")
    noise_estimate = estimate_noise_vad(noisy_signal, sample_rate)
    print(f"✅ 噪声估计完成，长度: {len(noise_estimate)}")
    
    # 计算信噪比
    print("\n3. 计算信噪比...")
    from analysis import FrequencyAnalysis
    freq_analysis = FrequencyAnalysis(sample_rate)
    
    # 真实SNR（与实际噪声对比）
    true_snr = freq_analysis.calculate_snr(clean_signal, noise)
    # 估计SNR（与估计噪声对比）
    estimated_snr = freq_analysis.calculate_snr(noisy_signal, noise_estimate)
    
    print(f"✅ 真实SNR: {true_snr:.2f} dB")
    print(f"✅ 估计SNR: {estimated_snr:.2f} dB")
    
    # 测试完整流程
    print("\n4. 测试AudioProcessor集成...")
    processor = AudioProcessor(sample_rate=sample_rate)
    
    # 手动设置音频数据（跳过文件加载）
    processor.audio_data = noisy_signal
    processor.original_data = noisy_signal.copy()
    
    # 执行噪声估计
    processor._estimate_noise()
    
    if processor.noise_estimate is not None:
        print("✅ AudioProcessor中噪声估计成功")
        snr = freq_analysis.calculate_snr(noisy_signal, processor.noise_estimate)
        print(f"✅ 集成后计算的SNR: {snr:.2f} dB")
    else:
        print("❌ AudioProcessor中噪声估计失败")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)

if __name__ == "__main__":
    test_noise_estimation()
