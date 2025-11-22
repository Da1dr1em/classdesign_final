#!/usr/bin/env python
"""
使用更好的噪声估计方法测试SNR
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from audio_processor import AudioProcessor
from analysis import FrequencyAnalysis

def test_with_better_method(input_file, method='spectral_floor'):
    """使用更好的噪声估计方法测试"""
    print("="*70)
    print(f"使用 {method} 方法进行SNR分析")
    print("="*70)
    
    # 创建处理器
    processor = AudioProcessor(sample_rate=44100)
    
    # 加载音频
    print("\n[1/4] 加载音频...")
    if not processor.load_audio(input_file):
        print("❌ 音频加载失败")
        return
    
    print(f"✅ 音频加载: {len(processor.audio_data)} 采样点")
    
    # 使用更好的方法估计噪声
    print(f"\n[2/4] 使用 {method} 方法估计噪声...")
    processor._estimate_noise(method=method)
    
    if processor.noise_estimate is None:
        print("❌ 噪声估计失败")
        return
    
    # 计算原始SNR
    freq_analysis = FrequencyAnalysis(processor.sample_rate)
    original_snr = freq_analysis.calculate_snr(processor.audio_data, processor.noise_estimate)
    print(f"✅ 原始信号SNR: {original_snr:.2f} dB")
    
    # 应用带通滤波器
    print("\n[3/4] 应用带通滤波器 (300-3400 Hz)...")
    processor.apply_filter('fir_bandpass', 
                          lowcut_freq=300, 
                          highcut_freq=3400, 
                          numtaps=101)
    print("✅ 滤波器应用完成")
    
    # 信号增强
    processor.enhance_signal('normalize', target_max=0.9)
    
    # 计算处理后的SNR
    print("\n[4/4] 计算性能指标...")
    processed_noise = processor.processed_data - processor.original_data
    processed_snr = freq_analysis.calculate_snr(processor.processed_data, processed_noise)
    snr_improvement = processed_snr - original_snr
    
    # 计算其他指标
    correlation = np.corrcoef(processor.original_data, processor.processed_data)[0, 1]
    rmse = np.sqrt(np.mean((processor.processed_data - processor.original_data) ** 2))
    
    # 显示结果
    print("\n" + "="*70)
    print("📊 信噪比分析结果 (使用 {} 方法)".format(method))
    print("="*70)
    print(f"  原始信号SNR:     {original_snr:.2f} dB")
    print(f"  处理后SNR:       {processed_snr:.2f} dB")
    print(f"  SNR改善:         {snr_improvement:+.2f} dB")
    
    print(f"\n📈 质量评估:")
    print(f"  相关系数:        {correlation:.3f}")
    print(f"  RMSE:           {rmse:.4f}")
    
    # 对比不同方法
    print("\n" + "="*70)
    print("📊 与VAD方法对比:")
    print("="*70)
    
    # 使用VAD方法
    from utils import estimate_noise_vad
    vad_noise = estimate_noise_vad(processor.audio_data, processor.sample_rate)
    vad_snr = freq_analysis.calculate_snr(processor.audio_data, vad_noise)
    
    print(f"  VAD方法SNR:      {vad_snr:.2f} dB")
    print(f"  {method}方法SNR: {original_snr:.2f} dB")
    print(f"  改善:            {original_snr - vad_snr:+.2f} dB")
    
    # 解释
    print("\n💡 分析:")
    if original_snr > vad_snr + 2:
        print(f"  ✅ {method}方法显著优于VAD方法！")
        print(f"     噪声估计更准确，SNR提升了 {original_snr - vad_snr:.2f} dB")
    elif original_snr > vad_snr:
        print(f"  ✓ {method}方法略优于VAD方法")
    else:
        print(f"  两种方法效果相近")
    
    if snr_improvement > 0:
        print(f"  ✅ 滤波器有效提升了信噪比")
    else:
        print(f"  ⚠️ 滤波器可能去除了部分有用信号")
    
    print("="*70)
    
    # 保存结果
    print("\n💾 保存处理结果...")
    processor.save_output(save_difference=True)
    print("✅ 完成！")

if __name__ == "__main__":
    input_file = r".\data\input\conversation_human.wav"
    
    print("\n【测试1: Spectral Floor方法（推荐）】\n")
    test_with_better_method(input_file, method='spectral_floor')
    
    print("\n\n【测试2: Minimum Statistics方法】\n")
    test_with_better_method(input_file, method='minimum_statistics')
