#!/usr/bin/env python
"""
对比处理后SNR的两种计算方法
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from audio_processor import AudioProcessor
from utils import estimate_noise
from analysis import FrequencyAnalysis

def compare_snr_calculation_methods(input_file):
    """对比处理后SNR的不同计算方法"""
    
    print("="*70)
    print("处理后SNR计算方法对比")
    print("="*70)
    
    # 创建处理器
    processor = AudioProcessor(sample_rate=44100)
    
    # 加载音频
    print("\n[1/3] 加载音频并估计噪声...")
    processor.load_audio(input_file)
    print(f"✅ 音频加载完成")
    
    # 应用滤波器
    print("\n[2/3] 应用带通滤波器...")
    processor.apply_filter('fir_bandpass', lowcut_freq=300, highcut_freq=3400, numtaps=101)
    processor.enhance_signal('normalize', target_max=0.9)
    print("✅ 滤波处理完成")
    
    # 计算SNR
    print("\n[3/3] 计算SNR...")
    freq_analysis = FrequencyAnalysis(processor.sample_rate)
    
    # 方法1: 原始噪声估计（处理前估计的噪声）
    original_snr = freq_analysis.calculate_snr(processor.original_data, processor.noise_estimate)
    
    # 方法2: 残差法（简单减法）
    residual_noise = processor.processed_data - processor.original_data
    residual_snr = freq_analysis.calculate_snr(processor.processed_data, residual_noise)
    
    # 方法3: 重新估计处理后信号的噪声（推荐）
    processed_noise_estimate = estimate_noise(
        processor.processed_data,
        processor.sample_rate,
        method='spectral_floor',
        percentile=10.0
    )
    estimated_snr = freq_analysis.calculate_snr(processor.processed_data, processed_noise_estimate)
    
    # 显示结果
    print("\n" + "="*70)
    print("📊 处理后SNR计算结果对比")
    print("="*70)
    
    print(f"\n原始信号SNR (spectral_floor法): {original_snr:.2f} dB")
    print("-" * 70)
    
    print(f"\n方法1: 残差法 (processed - original)")
    print(f"  假设: 处理前后的差异即为噪声")
    print(f"  结果: {residual_snr:.2f} dB")
    print(f"  问题: ❌ 包含了被滤除的有用信号成分")
    print(f"        会高估噪声，导致SNR偏低")
    
    print(f"\n方法2: 重新估计法 (spectral_floor)")
    print(f"  假设: 处理后信号仍然包含噪声，重新估计")
    print(f"  结果: {estimated_snr:.2f} dB")
    print(f"  优势: ✅ 准确估计处理后残留的实际噪声")
    print(f"        不受滤波器影响的信号成分干扰")
    
    print("\n" + "-"*70)
    print("📈 SNR改善量对比:")
    print("-" * 70)
    
    residual_improvement = residual_snr - original_snr
    estimated_improvement = estimated_snr - original_snr
    
    print(f"  残差法: {residual_improvement:+.2f} dB")
    if residual_improvement < 0:
        print(f"    ⚠️ 负值！说明该方法不准确")
    
    print(f"  重新估计法: {estimated_improvement:+.2f} dB")
    if estimated_improvement > 0:
        print(f"    ✅ 正值！滤波器确实改善了SNR")
    elif estimated_improvement > -3:
        print(f"    ✓ 接近0，滤波器保持了SNR")
    else:
        print(f"    ⚠️ 负值较大，滤波器可能去除了过多信号")
    
    # 详细解释
    print("\n" + "="*70)
    print("💡 为什么两种方法结果不同？")
    print("="*70)
    print("""
1. 残差法的问题:
   残差 = 处理后 - 原始
   这个差值包含：
   • 被去除的噪声 ✅
   • 被滤波器衰减的有用信号 ❌ (问题所在!)
   
   例如：带通滤波器(300-3400Hz)会去除：
   - 300Hz以下的低频成分（可能是有用的语音基频）
   - 3400Hz以上的高频成分（可能是语音的谐波）
   
   这些被去除的信号成分被错误地当作"噪声"，
   导致计算出的"噪声功率"偏大，SNR偏低。

2. 重新估计法的优势:
   对处理后的信号重新进行噪声估计：
   • 只估计真正的背景噪声 ✅
   • 不受滤波器影响 ✅
   • 能准确反映滤波后的实际信噪比 ✅
   
   这才是处理后信号的真实SNR！

3. 实际意义:
   如果残差法SNR < 原始SNR：
   → 不能说明滤波失败！可能只是去除了有用信号频率
   
   如果重新估计法SNR > 原始SNR：
   → 说明滤波器真正改善了信噪比！
    """)
    
    print("\n" + "="*70)
    print("🎯 推荐使用: 重新估计法")
    print("="*70)
    print(f"  原始SNR:   {original_snr:.2f} dB")
    print(f"  处理后SNR: {estimated_snr:.2f} dB")
    print(f"  改善量:    {estimated_improvement:+.2f} dB")
    
    if estimated_improvement > 3:
        print(f"\n  ✅ 优秀！滤波器显著改善了信噪比")
    elif estimated_improvement > 0:
        print(f"\n  ✓ 良好！滤波器改善了信噪比")
    elif estimated_improvement > -3:
        print(f"\n  → 中性。滤波器保持了信号质量，去除了部分频率")
    else:
        print(f"\n  ⚠️ 建议调整滤波器参数，可能去除了过多有用信号")
    
    print("="*70)

if __name__ == "__main__":
    input_file = r".\data\input\conversation_human.wav"
    compare_snr_calculation_methods(input_file)
