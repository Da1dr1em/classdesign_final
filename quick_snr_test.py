#!/usr/bin/env python
"""
快速SNR测试 - 跳过图表生成以加快速度
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from audio_processor import AudioProcessor

def quick_snr_test(input_file):
    """快速测试音频的SNR"""
    print("="*60)
    print(f"快速SNR分析: {input_file}")
    print("="*60)
    
    # 创建处理器
    processor = AudioProcessor(sample_rate=44100)
    
    # 加载音频（自动进行噪声估计）
    print("\n[1/4] 加载音频并估计噪声...")
    if not processor.load_audio(input_file):
        print("❌ 音频加载失败")
        return
    
    # 检查噪声估计
    if processor.noise_estimate is None:
        print("❌ 噪声估计失败")
        return
    
    # 计算原始SNR
    from analysis import FrequencyAnalysis
    freq_analysis = FrequencyAnalysis(processor.sample_rate)
    original_snr = freq_analysis.calculate_snr(processor.audio_data, processor.noise_estimate)
    
    print(f"✅ 音频加载: {len(processor.audio_data)} 采样点")
    print(f"✅ 原始信号SNR: {original_snr:.2f} dB")
    
    # 应用带通滤波器
    print("\n[2/4] 应用带通滤波器 (300-3400 Hz)...")
    processor.apply_filter('fir_bandpass', 
                          lowcut_freq=300, 
                          highcut_freq=3400, 
                          numtaps=101)
    print("✅ 滤波器应用完成")
    
    # 信号增强
    print("\n[3/4] 信号增强...")
    processor.enhance_signal('normalize', target_max=0.9)
    print("✅ 信号增强完成")
    
    # 计算处理后的SNR
    print("\n[4/4] 计算性能指标...")
    
    # 方法1: 残差法（旧方法）
    processed_noise_residual = processor.processed_data - processor.original_data
    processed_snr_residual = freq_analysis.calculate_snr(processor.processed_data, processed_noise_residual)
    
    # 方法2: 重新估计法（新方法）
    from utils import estimate_noise
    processed_noise_estimate = estimate_noise(
        processor.processed_data,
        processor.sample_rate,
        method='spectral_floor',
        percentile=10.0
    )
    processed_snr_estimated = freq_analysis.calculate_snr(processor.processed_data, processed_noise_estimate)
    
    snr_improvement_residual = processed_snr_residual - original_snr
    snr_improvement_estimated = processed_snr_estimated - original_snr
    
    # 计算其他指标
    correlation = np.corrcoef(processor.original_data, processor.processed_data)[0, 1]
    rmse = np.sqrt(np.mean((processor.processed_data - processor.original_data) ** 2))
    
    # 显示结果
    print("\n" + "="*60)
    print("📊 信噪比分析结果:")
    print("="*60)
    print(f"  原始信号SNR:     {original_snr:.2f} dB")
    print(f"\n  方法1 - 残差法 (旧):")
    print(f"    处理后SNR:     {processed_snr_residual:.2f} dB")
    print(f"    SNR改善:       {snr_improvement_residual:+.2f} dB")
    print(f"\n  方法2 - 重新估计法 (新) ⭐推荐:")
    print(f"    处理后SNR:     {processed_snr_estimated:.2f} dB")
    print(f"    SNR改善:       {snr_improvement_estimated:+.2f} dB")
    
    print(f"\n📈 质量评估:")
    print(f"  相关系数:        {correlation:.3f}")
    print(f"  RMSE:           {rmse:.4f}")
    
    # 解释结果
    print(f"\n💡 分析:")
    print(f"  两种方法的差异: {abs(processed_snr_estimated - processed_snr_residual):.2f} dB")
    
    if processed_snr_estimated > processed_snr_residual + 3:
        print(f"  ✅ 重新估计法更准确！残差法低估了处理后的SNR")
        print(f"     因为残差包含了被滤波器去除的有用信号")
    
    if snr_improvement_estimated > 3:
        print(f"  ✅ 显著改善！SNR提升了 {snr_improvement_estimated:.2f} dB")
    elif snr_improvement_estimated > 0:
        print(f"  ✓ 轻微改善，SNR提升了 {snr_improvement_estimated:.2f} dB")
    else:
        print(f"  → 滤波器在去除噪声的同时也去除了部分有用信号")
    
    if correlation > 0.8:
        print(f"  ✅ 信号保真度很好 (相关系数 {correlation:.3f})")
    elif correlation > 0.6:
        print(f"  ✓ 信号保真度尚可 (相关系数 {correlation:.3f})")
    else:
        print(f"  ⚠️ 信号失真较大 (相关系数 {correlation:.3f})")
    
    print("="*60)
    
    # 保存处理结果
    print("\n💾 保存处理结果...")
    processor.save_output(save_difference=True)
    print("✅ 完成！")

if __name__ == "__main__":
    input_file = r".\data\input\conversation_human.wav"
    quick_snr_test(input_file)
