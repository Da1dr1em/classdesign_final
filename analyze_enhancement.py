#!/usr/bin/env python
"""
分析信号增强算法的效果
"""
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from audio_processor import AudioProcessor
from utils import normalize_signal, estimate_noise
from analysis import FrequencyAnalysis

def analyze_enhancement(input_file):
    """分析增强算法的效果"""
    
    print("="*70)
    print("信号增强算法分析")
    print("="*70)
    
    # 创建处理器
    processor = AudioProcessor(sample_rate=44100, enable_plots=False)
    
    # 加载音频
    print("\n[1/5] 加载音频...")
    processor.load_audio(input_file)
    original_snr = processor.frequency_analysis.calculate_snr(
        processor.audio_data, processor.noise_estimate
    )
    print(f"✅ 原始信号SNR: {original_snr:.2f} dB")
    
    # 应用滤波器
    print("\n[2/5] 应用带通滤波器...")
    processor.apply_filter('fir_bandpass', lowcut_freq=300, highcut_freq=3400, numtaps=101)
    
    # 保存滤波后、增强前的状态
    filtered_only = processor.processed_data.copy()
    
    # 估计滤波后的噪声
    filtered_noise = estimate_noise(filtered_only, processor.sample_rate, 
                                   method='spectral_floor', percentile=10.0)
    filtered_snr = processor.frequency_analysis.calculate_snr(filtered_only, filtered_noise)
    print(f"✅ 滤波后SNR: {filtered_snr:.2f} dB")
    
    # 分析滤波后信号的统计特性
    print("\n[3/5] 分析滤波后信号特性...")
    filtered_max = np.max(np.abs(filtered_only))
    filtered_rms = np.sqrt(np.mean(filtered_only ** 2))
    filtered_mean = np.mean(filtered_only)
    filtered_std = np.std(filtered_only)
    
    print(f"  最大幅度: {filtered_max:.4f}")
    print(f"  RMS: {filtered_rms:.4f}")
    print(f"  均值: {filtered_mean:.6f}")
    print(f"  标准差: {filtered_std:.4f}")
    print(f"  峰值因子: {filtered_max/filtered_rms:.2f}")
    
    # 应用归一化增强
    print("\n[4/5] 应用归一化增强 (target_max=0.9)...")
    enhanced = normalize_signal(filtered_only, target_max=0.9)
    
    # 估计增强后的噪声
    enhanced_noise = estimate_noise(enhanced, processor.sample_rate,
                                   method='spectral_floor', percentile=10.0)
    enhanced_snr = processor.frequency_analysis.calculate_snr(enhanced, enhanced_noise)
    
    # 分析增强后信号
    enhanced_max = np.max(np.abs(enhanced))
    enhanced_rms = np.sqrt(np.mean(enhanced ** 2))
    enhanced_mean = np.mean(enhanced)
    enhanced_std = np.std(enhanced)
    
    print(f"✅ 增强后SNR: {enhanced_snr:.2f} dB")
    print(f"  最大幅度: {enhanced_max:.4f}")
    print(f"  RMS: {enhanced_rms:.4f}")
    print(f"  均值: {enhanced_mean:.6f}")
    print(f"  标准差: {enhanced_std:.4f}")
    print(f"  峰值因子: {enhanced_max/enhanced_rms:.2f}")
    
    # 计算增强因子
    amplitude_factor = enhanced_max / filtered_max
    rms_factor = enhanced_rms / filtered_rms
    
    print(f"\n  增强因子:")
    print(f"    幅度放大: {amplitude_factor:.2f}x")
    print(f"    RMS放大: {rms_factor:.2f}x")
    print(f"    功率放大: {rms_factor**2:.2f}x")
    
    # 分析SNR变化
    print("\n[5/5] 分析SNR变化...")
    snr_change = enhanced_snr - filtered_snr
    
    print("\n" + "="*70)
    print("📊 SNR变化分析")
    print("="*70)
    print(f"  原始信号SNR:   {original_snr:.2f} dB")
    print(f"  滤波后SNR:     {filtered_snr:.2f} dB")
    print(f"  增强后SNR:     {enhanced_snr:.2f} dB")
    print(f"  滤波改善:      {filtered_snr - original_snr:+.2f} dB")
    print(f"  增强改善:      {snr_change:+.2f} dB")
    print(f"  总体改善:      {enhanced_snr - original_snr:+.2f} dB")
    
    # 分析为什么SNR会变化
    print("\n" + "="*70)
    print("💡 归一化增强算法分析")
    print("="*70)
    
    print(f"\n算法原理:")
    print(f"  1. 找到信号的最大幅度: {filtered_max:.4f}")
    print(f"  2. 计算缩放因子: {amplitude_factor:.4f} = 0.9 / {filtered_max:.4f}")
    print(f"  3. 整个信号乘以缩放因子")
    print(f"  4. 信号和噪声都被同等放大")
    
    print(f"\n理论分析:")
    print(f"  归一化是线性操作，信号和噪声同比例缩放")
    print(f"  SNR = 10*log10(P_signal / P_noise)")
    print(f"  如果 signal' = k * signal, noise' = k * noise")
    print(f"  则 SNR' = 10*log10(k²*P_signal / k²*P_noise)")
    print(f"         = 10*log10(P_signal / P_noise)")
    print(f"         = SNR (不变)")
    
    print(f"\n实际结果:")
    if abs(snr_change) < 0.5:
        print(f"  ✅ SNR基本不变 ({snr_change:+.2f} dB)")
        print(f"     符合理论预期！归一化不会改变信噪比")
    elif snr_change > 0.5:
        print(f"  ⚠️ SNR略有提升 ({snr_change:+.2f} dB)")
        print(f"     可能原因:")
        print(f"     - 噪声估计的统计误差")
        print(f"     - 信号放大后，噪声估计更准确")
    else:
        print(f"  ⚠️ SNR略有下降 ({snr_change:+.2f} dB)")
        print(f"     可能原因:")
        print(f"     - 噪声估计的统计误差")
        print(f"     - 量化误差的影响")
    
    print(f"\n归一化的实际作用:")
    print(f"  1. ✅ 防止音频播放时削波失真")
    print(f"  2. ✅ 使不同音频的音量统一")
    print(f"  3. ✅ 充分利用数字音频的动态范围")
    print(f"  4. ❌ 不能改善信噪比（理论上）")
    print(f"  5. ⚠️ 可能略微影响量化噪声")
    
    # 检查是否有削波风险
    print(f"\n削波检查:")
    if filtered_max > 1.0:
        print(f"  ⚠️ 滤波后信号已经削波！最大值={filtered_max:.4f} > 1.0")
        print(f"     归一化可以修正这个问题")
    elif filtered_max > 0.95:
        print(f"  ⚠️ 滤波后信号接近削波边缘 (max={filtered_max:.4f})")
        print(f"     归一化可以防止潜在问题")
    elif filtered_max < 0.1:
        print(f"  ℹ️ 滤波后信号很小 (max={filtered_max:.4f})")
        print(f"     归一化可以提升音量到合适水平")
    else:
        print(f"  ✅ 滤波后信号幅度合理 (max={filtered_max:.4f})")
        print(f"     归一化主要起标准化作用")
    
    # 建议
    print("\n" + "="*70)
    print("🎯 建议")
    print("="*70)
    
    if abs(snr_change) < 0.5:
        print(f"  ✅ 当前增强算法合理")
        print(f"     - 归一化不会损害信号质量")
        print(f"     - 能标准化输出音量")
        print(f"     - 适合作为后处理步骤")
    
    if filtered_max < 0.3:
        print(f"\n  💡 可以考虑更激进的增强:")
        print(f"     - 使用 target_max=0.95 获得更大音量")
        print(f"     - 或添加动态范围压缩")
    
    if snr_change < -1.0:
        print(f"\n  ⚠️ 注意SNR下降较多")
        print(f"     - 检查是否引入了额外噪声")
        print(f"     - 考虑先降噪再增强")
    
    print("="*70)
    
    return {
        'original_snr': original_snr,
        'filtered_snr': filtered_snr,
        'enhanced_snr': enhanced_snr,
        'amplitude_factor': amplitude_factor,
        'rms_factor': rms_factor
    }

if __name__ == "__main__":
    input_file = r".\data\input\conversation_human.wav"
    results = analyze_enhancement(input_file)
