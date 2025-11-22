#!/usr/bin/env python
"""
解释归一化增强到底增强了什么
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

# 创建一个模拟信号
np.random.seed(42)
t = np.linspace(0, 1, 1000)
signal = 0.08 * np.sin(2 * np.pi * 5 * t)  # 低振幅信号 (max=0.08)
noise = 0.005 * np.random.randn(1000)       # 噪声
noisy_signal = signal + noise

# 归一化后
max_val = np.max(np.abs(noisy_signal))
normalized = noisy_signal * (0.9 / max_val)
normalized_signal = signal * (0.9 / max_val)
normalized_noise = noise * (0.9 / max_val)

# 计算统计量
original_signal_power = np.mean(signal ** 2)
original_noise_power = np.mean(noise ** 2)
original_snr = 10 * np.log10(original_signal_power / original_noise_power)

normalized_signal_power = np.mean(normalized_signal ** 2)
normalized_noise_power = np.mean(normalized_noise ** 2)
normalized_snr = 10 * np.log10(normalized_signal_power / normalized_noise_power)

scale_factor = 0.9 / max_val

print("="*70)
print("归一化增强到底增强了什么？")
print("="*70)

print("\n📊 数值对比:")
print("-"*70)
print(f"{'指标':<20} {'归一化前':<15} {'归一化后':<15} {'变化':<15}")
print("-"*70)
print(f"{'最大振幅':<20} {max_val:<15.4f} {0.9:<15.4f} {scale_factor:<15.2f}x")
print(f"{'信号功率':<20} {original_signal_power:<15.6f} {normalized_signal_power:<15.6f} {normalized_signal_power/original_signal_power:<15.2f}x")
print(f"{'噪声功率':<20} {original_noise_power:<15.8f} {normalized_noise_power:<15.6f} {normalized_noise_power/original_noise_power:<15.2f}x")
print(f"{'信噪比 (dB)':<20} {original_snr:<15.2f} {normalized_snr:<15.2f} {normalized_snr-original_snr:<+15.2f} dB")
print("-"*70)

print("\n💡 核心结论:")
print("-"*70)
print(f"1. 振幅放大: {scale_factor:.2f}倍")
print(f"   - 信号从 ±{max_val:.4f} 放大到 ±0.9000")
print(f"   - 使音频更响亮（音量提升）")
print(f"   - 充分利用 [-1, +1] 的数字音频范围")

print(f"\n2. 功率放大: {(scale_factor**2):.2f}倍")
print(f"   - 功率 = 振幅²")
print(f"   - 信号功率: {original_signal_power:.6f} → {normalized_signal_power:.4f}")
print(f"   - 噪声功率: {original_noise_power:.8f} → {normalized_noise_power:.6f}")

print(f"\n3. 信噪比不变: {normalized_snr-original_snr:+.2f} dB")
print(f"   - 信号和噪声同比例放大")
print(f"   - SNR = 10*log10(信号功率/噪声功率)")
print(f"   - 放大k倍后: SNR' = 10*log10(k²·信号/k²·噪声) = SNR")

print("\n" + "="*70)
print("🎯 归一化增强的实际意义")
print("="*70)

print("\n增强的是:")
print("  ✅ 音量（振幅）- 从很小的信号放大到接近最大值")
print("  ✅ 能量（功率）- 功率提升约100倍")
print("  ✅ 动态范围利用率 - 从9%提升到90%")
print("  ✅ 播放响度 - 听起来更响亮清晰")

print("\n没有增强的是:")
print("  ❌ 信噪比 - 保持不变（噪声也同比例放大）")
print("  ❌ 频率特性 - 频谱形状不变")
print("  ❌ 相对质量 - 信号与噪声的相对比例不变")

print("\n为什么需要归一化？")
print("  1. 统一音量 - 不同来源的音频处理后音量一致")
print("  2. 防止削波 - 留0.1安全边际避免 >1.0 的削波失真")
print("  3. 标准化输出 - 便于后续处理和对比")
print("  4. 改善听感 - 微弱信号被放大到合适的播放音量")

print("\n" + "="*70)

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上：时域波形对比
ax = axes[0, 0]
ax.plot(t[:200], noisy_signal[:200], 'b-', alpha=0.7, linewidth=1, label='归一化前')
ax.plot(t[:200], normalized[:200], 'r-', alpha=0.7, linewidth=1, label='归一化后')
ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='目标最大值 (0.9)')
ax.axhline(y=-0.9, color='r', linestyle='--', alpha=0.5)
ax.axhline(y=max_val, color='b', linestyle='--', alpha=0.5, label=f'原始最大值 ({max_val:.3f})')
ax.axhline(y=-max_val, color='b', linestyle='--', alpha=0.5)
ax.set_xlabel('时间 (秒)', fontsize=11)
ax.set_ylabel('振幅', fontsize=11)
ax.set_title('时域波形对比 - 振幅放大', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 右上：振幅分布直方图
ax = axes[0, 1]
ax.hist(noisy_signal, bins=50, alpha=0.6, color='blue', label='归一化前', density=True)
ax.hist(normalized, bins=50, alpha=0.6, color='red', label='归一化后', density=True)
ax.axvline(x=max_val, color='b', linestyle='--', linewidth=2, label=f'原始max={max_val:.3f}')
ax.axvline(x=0.9, color='r', linestyle='--', linewidth=2, label='目标max=0.9')
ax.set_xlabel('振幅', fontsize=11)
ax.set_ylabel('概率密度', fontsize=11)
ax.set_title('振幅分布 - 整体缩放', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 左下：功率谱对比
ax = axes[1, 0]
freqs_orig = np.fft.rfftfreq(len(noisy_signal), 1/1000)
psd_orig = np.abs(np.fft.rfft(noisy_signal)) ** 2
psd_norm = np.abs(np.fft.rfft(normalized)) ** 2

ax.semilogy(freqs_orig, psd_orig, 'b-', alpha=0.7, linewidth=1.5, label='归一化前')
ax.semilogy(freqs_orig, psd_norm, 'r-', alpha=0.7, linewidth=1.5, label='归一化后')
ax.set_xlabel('频率 (Hz)', fontsize=11)
ax.set_ylabel('功率谱密度 (对数刻度)', fontsize=11)
ax.set_title(f'功率谱对比 - 功率提升{scale_factor**2:.1f}倍', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim([0, 50])

# 右下：信息面板
ax = axes[1, 1]
ax.axis('off')

info_text = f"""
归一化增强详解
{'='*45}

原理：
  signal_out = signal_in × (0.9 / max_input)

效果：
  🔊 振幅放大：{scale_factor:.2f}x
  ⚡ 功率放大：{scale_factor**2:.1f}x
  📊 SNR变化：{normalized_snr-original_snr:+.2f} dB (不变)

数值示例：
  最大振幅：{max_val:.4f} → 0.9000
  信号RMS：{np.sqrt(original_signal_power):.4f} → {np.sqrt(normalized_signal_power):.4f}
  噪声RMS：{np.sqrt(original_noise_power):.5f} → {np.sqrt(normalized_noise_power):.4f}
  
关键特性：
  ✓ 线性操作 - 不改变频率特性
  ✓ 信号噪声同比例 - SNR保持不变
  ✓ 可逆操作 - 可还原原始振幅
  ✓ 防止削波 - 留10%安全边际

实际应用：
  • 音频后处理标准化
  • 统一不同来源的音量
  • 优化播放响度
  • 防止数字削波失真
"""

ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('data/output/normalize_enhancement_explanation.png', dpi=150, bbox_inches='tight')
print(f"\n📈 图表已保存到: data/output/normalize_enhancement_explanation.png")
plt.show()

print("\n" + "="*70)
print("总结：归一化增强提升的是'音量/振幅'，而不是'质量/信噪比'")
print("="*70)
