#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
音频增强原理详解及实际效果分析
针对 conversation_human.wav 进行完整处理流程演示
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import get_window, stft, istft
import matplotlib
from matplotlib import rcParams

# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from audio_processor import AudioProcessor
from utils import estimate_noise, normalize_signal
from analysis import FrequencyAnalysis

def analyze_audio_enhancement(input_file):
    """完整分析音频增强的原理和效果"""
    
    print("="*80)
    print("音频增强原理与效果分析".center(80))
    print("="*80)
    
    # ==================== 第一部分：理论原理 ====================
    print("\n【第一部分：音频增强的理论原理】")
    print("-"*80)
    
    print("\n1. 什么是音频增强？")
    print("   音频增强是指通过数字信号处理技术，改善音频信号的感知质量")
    print("   主要目标：")
    print("   • 降低背景噪声")
    print("   • 提高语音清晰度")
    print("   • 标准化音量水平")
    print("   • 改善听感")
    
    print("\n2. 音频增强的主要技术")
    print("   ┌─────────────────────────────────────────────┐")
    print("   │ (1) 噪声抑制 (Noise Suppression)          │")
    print("   │     - 频域滤波（带通/低通/高通）          │")
    print("   │     - 维纳滤波                             │")
    print("   │     - 谱减法                               │")
    print("   │                                            │")
    print("   │ (2) 动态范围处理 (Dynamic Range)          │")
    print("   │     - 归一化（Normalization）             │")
    print("   │     - 压缩（Compression）                 │")
    print("   │     - 限幅（Limiting）                    │")
    print("   │                                            │")
    print("   │ (3) 频谱整形 (Spectral Shaping)           │")
    print("   │     - 均衡（EQ）                          │")
    print("   │     - 去混响                               │")
    print("   │                                            │")
    print("   │ (4) 自适应滤波 (Adaptive Filtering)       │")
    print("   │     - LMS/NLMS算法                        │")
    print("   │     - 卡尔曼滤波                          │")
    print("   └─────────────────────────────────────────────┘")
    
    print("\n3. 本系统采用的增强策略")
    print("   完整处理流程：")
    print("   ")
    print("   原始音频")
    print("      ↓")
    print("   [步骤1] 噪声估计 (Spectral Floor法)")
    print("      ├─ STFT变换到频域")
    print("      ├─ 每个频率取10%百分位数")
    print("      └─ 得到噪声功率谱估计")
    print("      ↓")
    print("   [步骤2] 频域滤波 (带通滤波)")
    print("      ├─ 保留语音频段 (300-3400 Hz)")
    print("      ├─ 去除低频噪声 (<300 Hz)")
    print("      └─ 去除高频噪声 (>3400 Hz)")
    print("      ↓")
    print("   [步骤3] 信号增强 (归一化)")
    print("      ├─ 找到最大幅值")
    print("      ├─ 缩放到目标水平 (0.9)")
    print("      └─ 标准化输出音量")
    print("      ↓")
    print("   增强后音频")
    
    # ==================== 第二部分：实际数据分析 ====================
    print("\n" + "="*80)
    print("【第二部分：对 conversation_human.wav 的实际分析】")
    print("-"*80)
    
    # 创建处理器
    processor = AudioProcessor(sample_rate=44100, enable_plots=False)
    freq_analyzer = FrequencyAnalysis(sample_rate=44100)
    
    # 加载原始音频
    print("\n[阶段0] 加载原始音频...")
    processor.load_audio(input_file)
    original = processor.audio_data.copy()
    original_noise = processor.noise_estimate.copy()
    
    duration = len(original) / processor.sample_rate
    print(f"   ✓ 音频长度: {len(original)} 采样点 ({duration:.2f} 秒)")
    print(f"   ✓ 采样率: {processor.sample_rate} Hz")
    print(f"   ✓ 原始最大幅度: {np.max(np.abs(original)):.4f}")
    
    # 计算原始信号统计量
    original_power = np.mean(original ** 2)
    original_rms = np.sqrt(original_power)
    original_noise_power = np.mean(original_noise ** 2)
    original_snr = 10 * np.log10(original_power / original_noise_power)
    
    print(f"   ✓ 原始RMS: {original_rms:.6f}")
    print(f"   ✓ 原始噪声功率: {original_noise_power:.8f}")
    print(f"   ✓ 原始SNR: {original_snr:.2f} dB")
    
    # 步骤1：应用滤波器
    print("\n[阶段1] 应用带通滤波器 (300-3400 Hz)...")
    processor.apply_filter('fir_bandpass', lowcut_freq=300, highcut_freq=3400, numtaps=101)
    filtered = processor.processed_data.copy()
    
    # 重新估计滤波后的噪声
    filtered_noise = estimate_noise(filtered, processor.sample_rate, 
                                   method='spectral_floor', percentile=10.0)
    filtered_power = np.mean(filtered ** 2)
    filtered_rms = np.sqrt(filtered_power)
    filtered_noise_power = np.mean(filtered_noise ** 2)
    filtered_snr = 10 * np.log10(filtered_power / filtered_noise_power)
    
    print(f"   ✓ 滤波后最大幅度: {np.max(np.abs(filtered)):.4f}")
    print(f"   ✓ 滤波后RMS: {filtered_rms:.6f}")
    print(f"   ✓ 滤波后SNR: {filtered_snr:.2f} dB")
    print(f"   ✓ SNR改善: {filtered_snr - original_snr:+.2f} dB")
    
    # 步骤2：应用增强
    print("\n[阶段2] 应用归一化增强 (目标幅度 0.9)...")
    enhanced = normalize_signal(filtered, target_max=0.9)
    
    # 估计增强后的噪声
    enhanced_noise = estimate_noise(enhanced, processor.sample_rate,
                                   method='spectral_floor', percentile=10.0)
    enhanced_power = np.mean(enhanced ** 2)
    enhanced_rms = np.sqrt(enhanced_power)
    enhanced_noise_power = np.mean(enhanced_noise ** 2)
    enhanced_snr = 10 * np.log10(enhanced_power / enhanced_noise_power)
    
    scale_factor = np.max(np.abs(enhanced)) / np.max(np.abs(filtered))
    
    print(f"   ✓ 增强后最大幅度: {np.max(np.abs(enhanced)):.4f}")
    print(f"   ✓ 增强后RMS: {enhanced_rms:.6f}")
    print(f"   ✓ 增强后SNR: {enhanced_snr:.2f} dB")
    print(f"   ✓ 放大倍数: {scale_factor:.2f}x")
    print(f"   ✓ 功率放大: {scale_factor**2:.2f}x")
    
    # 总结
    print("\n" + "="*80)
    print("【第三部分：增强效果总结】")
    print("-"*80)
    
    print("\n📊 关键指标对比:")
    print(f"   {'指标':<20} {'原始':<15} {'滤波后':<15} {'增强后':<15}")
    print(f"   {'-'*65}")
    print(f"   {'最大幅度':<20} {np.max(np.abs(original)):<15.4f} {np.max(np.abs(filtered)):<15.4f} {np.max(np.abs(enhanced)):<15.4f}")
    print(f"   {'RMS能量':<20} {original_rms:<15.6f} {filtered_rms:<15.6f} {enhanced_rms:<15.6f}")
    print(f"   {'噪声功率':<20} {original_noise_power:<15.8f} {filtered_noise_power:<15.8f} {enhanced_noise_power:<15.8f}")
    print(f"   {'SNR (dB)':<20} {original_snr:<15.2f} {filtered_snr:<15.2f} {enhanced_snr:<15.2f}")
    
    print("\n💡 增强原理解释:")
    print("   1. 滤波器的作用 (300-3400 Hz带通)")
    print(f"      • 去除低频噪声 (<300 Hz): 环境嗡嗡声、空调声")
    print(f"      • 保留语音频段 (300-3400 Hz): 人声基频和谐波")
    print(f"      • 去除高频噪声 (>3400 Hz): 电子噪声、嘶嘶声")
    print(f"      • SNR提升: {filtered_snr - original_snr:+.2f} dB ✓")
    
    print("\n   2. 归一化的作用")
    print(f"      • 幅度放大: {scale_factor:.2f}倍")
    print(f"      • 功率放大: {scale_factor**2:.2f}倍")
    print(f"      • SNR变化: {enhanced_snr - filtered_snr:+.2f} dB (理论上应为0)")
    print(f"      • 作用: 标准化音量，防止削波")
    
    print("\n   3. 总体效果")
    total_snr_improvement = enhanced_snr - original_snr
    print(f"      • SNR总改善: {total_snr_improvement:+.2f} dB")
    print(f"      • 噪声功率降低: {(1 - enhanced_noise_power/original_noise_power)*100:.1f}%")
    print(f"      • 音量提升: {20*np.log10(scale_factor):.1f} dB")
    
    if total_snr_improvement > 0:
        print(f"      ✓ 增强有效！信噪比得到改善")
    else:
        print(f"      ⚠ 增强效果有限，可能需要调整参数")
    
    # ==================== 第四部分：可视化 ====================
    print("\n" + "="*80)
    print("【第四部分：生成可视化图表】")
    print("-"*80)
    
    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    
    # 时域波形对比（前0.5秒）
    plot_length = int(0.5 * processor.sample_rate)
    t = np.arange(plot_length) / processor.sample_rate
    
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(t, original[:plot_length], 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_title('原始音频波形', fontsize=11, fontweight='bold')
    ax1.set_xlabel('时间 (秒)')
    ax1.set_ylabel('幅度')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.95, f'Max: {np.max(np.abs(original)):.4f}\nRMS: {original_rms:.6f}',
             transform=ax1.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(t, filtered[:plot_length], 'g-', linewidth=0.5, alpha=0.7)
    ax2.set_title('滤波后波形', fontsize=11, fontweight='bold')
    ax2.set_xlabel('时间 (秒)')
    ax2.set_ylabel('幅度')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.95, f'Max: {np.max(np.abs(filtered)):.4f}\nRMS: {filtered_rms:.6f}',
             transform=ax2.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(t, enhanced[:plot_length], 'r-', linewidth=0.5, alpha=0.7)
    ax3.set_title('增强后波形', fontsize=11, fontweight='bold')
    ax3.set_xlabel('时间 (秒)')
    ax3.set_ylabel('幅度')
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.95, f'Max: {np.max(np.abs(enhanced)):.4f}\nRMS: {enhanced_rms:.6f}',
             transform=ax3.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # 频谱对比
    from scipy.fft import rfft, rfftfreq
    
    freqs = rfftfreq(len(original), 1/processor.sample_rate)
    original_fft = np.abs(rfft(original))
    filtered_fft = np.abs(rfft(filtered))
    enhanced_fft = np.abs(rfft(enhanced))
    
    # 只显示0-5000 Hz
    freq_mask = freqs <= 5000
    
    ax4 = plt.subplot(3, 3, 4)
    ax4.semilogy(freqs[freq_mask], original_fft[freq_mask], 'b-', linewidth=1, alpha=0.7)
    ax4.axvline(x=300, color='r', linestyle='--', alpha=0.5, label='300 Hz')
    ax4.axvline(x=3400, color='r', linestyle='--', alpha=0.5, label='3400 Hz')
    ax4.set_title('原始音频频谱', fontsize=11, fontweight='bold')
    ax4.set_xlabel('频率 (Hz)')
    ax4.set_ylabel('幅度 (对数)')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.legend(fontsize=8)
    
    ax5 = plt.subplot(3, 3, 5)
    ax5.semilogy(freqs[freq_mask], filtered_fft[freq_mask], 'g-', linewidth=1, alpha=0.7)
    ax5.axvline(x=300, color='r', linestyle='--', alpha=0.5, label='300 Hz')
    ax5.axvline(x=3400, color='r', linestyle='--', alpha=0.5, label='3400 Hz')
    ax5.set_title('滤波后频谱', fontsize=11, fontweight='bold')
    ax5.set_xlabel('频率 (Hz)')
    ax5.set_ylabel('幅度 (对数)')
    ax5.grid(True, alpha=0.3, which='both')
    ax5.legend(fontsize=8)
    
    ax6 = plt.subplot(3, 3, 6)
    ax6.semilogy(freqs[freq_mask], enhanced_fft[freq_mask], 'r-', linewidth=1, alpha=0.7)
    ax6.axvline(x=300, color='k', linestyle='--', alpha=0.5, label='300 Hz')
    ax6.axvline(x=3400, color='k', linestyle='--', alpha=0.5, label='3400 Hz')
    ax6.set_title('增强后频谱', fontsize=11, fontweight='bold')
    ax6.set_xlabel('频率 (Hz)')
    ax6.set_ylabel('幅度 (对数)')
    ax6.grid(True, alpha=0.3, which='both')
    ax6.legend(fontsize=8)
    
    # 时频谱图对比
    window = get_window('hann', 2048)
    
    f_orig, t_orig, Zxx_orig = stft(original, fs=processor.sample_rate, window=window,
                                     nperseg=2048, noverlap=1024)
    f_filt, t_filt, Zxx_filt = stft(filtered, fs=processor.sample_rate, window=window,
                                     nperseg=2048, noverlap=1024)
    f_enh, t_enh, Zxx_enh = stft(enhanced, fs=processor.sample_rate, window=window,
                                  nperseg=2048, noverlap=1024)
    
    freq_limit = 5000
    freq_idx = np.where(f_orig <= freq_limit)[0]
    
    ax7 = plt.subplot(3, 3, 7)
    pcm7 = ax7.pcolormesh(t_orig, f_orig[freq_idx], 
                          20*np.log10(np.abs(Zxx_orig[freq_idx, :]) + 1e-10),
                          shading='gouraud', cmap='viridis', vmin=-60, vmax=0)
    ax7.set_title('原始时频谱', fontsize=11, fontweight='bold')
    ax7.set_xlabel('时间 (秒)')
    ax7.set_ylabel('频率 (Hz)')
    plt.colorbar(pcm7, ax=ax7, label='dB')
    
    ax8 = plt.subplot(3, 3, 8)
    pcm8 = ax8.pcolormesh(t_filt, f_filt[freq_idx],
                          20*np.log10(np.abs(Zxx_filt[freq_idx, :]) + 1e-10),
                          shading='gouraud', cmap='viridis', vmin=-60, vmax=0)
    ax8.set_title('滤波后时频谱', fontsize=11, fontweight='bold')
    ax8.set_xlabel('时间 (秒)')
    ax8.set_ylabel('频率 (Hz)')
    plt.colorbar(pcm8, ax=ax8, label='dB')
    
    ax9 = plt.subplot(3, 3, 9)
    pcm9 = ax9.pcolormesh(t_enh, f_enh[freq_idx],
                          20*np.log10(np.abs(Zxx_enh[freq_idx, :]) + 1e-10),
                          shading='gouraud', cmap='viridis', vmin=-60, vmax=0)
    ax9.set_title('增强后时频谱', fontsize=11, fontweight='bold')
    ax9.set_xlabel('时间 (秒)')
    ax9.set_ylabel('频率 (Hz)')
    plt.colorbar(pcm9, ax=ax9, label='dB')
    
    plt.tight_layout()
    output_path = 'data/output/audio_enhancement_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 图表已保存: {output_path}")
    plt.show()
    
    # ==================== 第五部分：技术细节 ====================
    print("\n" + "="*80)
    print("【第五部分：技术细节与原理解析】")
    print("-"*80)
    
    print("\n1. 为什么选择 300-3400 Hz 带通滤波？")
    print("   • 人声基频范围: 男性 85-180 Hz, 女性 165-255 Hz")
    print("   • 语音能量集中: 300-3400 Hz (电话语音标准)")
    print("   • 谐波分布: 基频的整数倍，主要在此范围内")
    print("   • 噪声特性: 低频(<300Hz)环境噪声，高频(>3400Hz)电子噪声")
    
    print("\n2. 归一化增强的数学原理")
    print("   设原信号 x(t)，最大值 x_max")
    print("   归一化: y(t) = x(t) * (0.9 / x_max)")
    print("   ")
    print("   效果:")
    print(f"   • 线性缩放: 所有采样点同比例放大")
    print(f"   • 保持波形: 不改变信号形状")
    print(f"   • SNR不变: 信号噪声同比例，比值不变")
    print(f"   • 防削波: 留10%余量避免超过±1.0")
    
    print("\n3. 为什么归一化不改善SNR？")
    print("   SNR = 10*log10(P_signal / P_noise)")
    print("   如果 y(t) = k * x(t)")
    print("   则 P_y = k² * P_x")
    print("   SNR_y = 10*log10(k²*P_signal / k²*P_noise)")
    print("        = 10*log10(P_signal / P_noise)")
    print("        = SNR_x")
    print("   结论: 归一化是线性操作，不改变信噪比")
    
    print("\n4. 真正改善SNR的是滤波器")
    print(f"   • 原理: 去除非语音频段的噪声")
    print(f"   • 保留: 300-3400 Hz语音能量")
    print(f"   • 去除: 其他频段的噪声能量")
    print(f"   • 结果: 噪声功率↓, 信号功率基本不变, SNR↑")
    print(f"   • 本例: SNR提升 {filtered_snr - original_snr:.2f} dB")
    
    print("\n5. 完整增强流程的协同效果")
    print("   滤波器 + 归一化 = 噪声抑制 + 音量标准化")
    print("   • 质量改善: 由滤波器实现 (SNR提升)")
    print("   • 音量统一: 由归一化实现 (标准化)")
    print("   • 两者结合: 既清晰又响亮")
    
    print("\n" + "="*80)
    print("分析完成！".center(80))
    print("="*80)
    
    return {
        'original_snr': original_snr,
        'filtered_snr': filtered_snr,
        'enhanced_snr': enhanced_snr,
        'snr_improvement': total_snr_improvement,
        'scale_factor': scale_factor
    }

if __name__ == "__main__":
    input_file = r".\data\input\conversation_human.wav"
    results = analyze_audio_enhancement(input_file)
    
    print(f"\n最终结果:")
    print(f"  原始SNR: {results['original_snr']:.2f} dB")
    print(f"  增强后SNR: {results['enhanced_snr']:.2f} dB")
    print(f"  总改善: {results['snr_improvement']:+.2f} dB")
    print(f"  音量放大: {results['scale_factor']:.2f}倍")
