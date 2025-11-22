#!/usr/bin/env python
"""
详细解释 Spectral Floor (频谱底噪法) 的原理
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft, get_window
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

print("="*70)
print("Spectral Floor (频谱底噪法) 原理详解")
print("="*70)

# 创建示例信号：语音 + 噪声
np.random.seed(42)
sample_rate = 44100
duration = 2.0
t = np.linspace(0, duration, int(sample_rate * duration))

# 模拟语音：间歇性的正弦波（代表说话）
speech = np.zeros_like(t)
# 三段语音，中间有静音
speech[0:int(0.4*len(t))] = 0.5 * np.sin(2*np.pi*500*t[0:int(0.4*len(t))])
speech[int(0.6*len(t)):int(0.9*len(t))] = 0.3 * np.sin(2*np.pi*700*t[int(0.6*len(t)):int(0.9*len(t))])

# 背景噪声：持续的白噪声
noise = 0.05 * np.random.randn(len(t))

# 含噪信号
noisy_signal = speech + noise

print("\n【1. 核心思想】")
print("-"*70)
print("假设：")
print("  • 语音信号是间歇的（有说话、有停顿）")
print("  • 背景噪声是持续存在的")
print("  • 在频域中，每个频率点的能量随时间波动")
print("")
print("原理：")
print("  • 对于每个频率，语音出现时能量高，静音时能量低")
print("  • 取该频率在所有时间的低百分位数（如10%）")
print("  • 这个低百分位数代表了该频率的噪声水平")
print("")
print("优势：")
print("  ✓ 不需要显式检测静音段")
print("  ✓ 利用全部音频数据")
print("  ✓ 每个频率独立估计（频率分辨率高）")

print("\n【2. 算法步骤】")
print("-"*70)

# 步骤1：STFT
frame_length = 2048
window = get_window('hann', frame_length)
f, t_stft, Zxx = stft(noisy_signal, fs=sample_rate, window=window,
                      nperseg=frame_length, noverlap=frame_length//2)

magnitude = np.abs(Zxx)
print(f"步骤1：短时傅里叶变换 (STFT)")
print(f"  • 帧长：{frame_length} 样本 ({frame_length/sample_rate*1000:.1f} ms)")
print(f"  • 重叠：50%")
print(f"  • 频率分辨率：{f[1]-f[0]:.2f} Hz")
print(f"  • 结果维度：{magnitude.shape[0]} 频点 × {magnitude.shape[1]} 时间帧")

# 步骤2：计算百分位数
percentile = 10.0
noise_magnitude = np.percentile(magnitude, percentile, axis=1)

print(f"\n步骤2：对每个频率，在时间轴上取 {percentile}% 百分位数")
print(f"  • 输入：{magnitude.shape[0]} 个频率，每个有 {magnitude.shape[1]} 个时间点")
print(f"  • 输出：{len(noise_magnitude)} 个频率的噪声幅度估计")
print(f"  • 含义：每个频率最低 {percentile}% 的能量水平")

# 步骤3：重建噪声
noise_magnitude_2d = np.tile(noise_magnitude[:, np.newaxis], (1, magnitude.shape[1]))
phase = np.random.uniform(0, 2*np.pi, noise_magnitude_2d.shape)
noise_stft = noise_magnitude_2d * np.exp(1j * phase)
_, noise_estimate = istft(noise_stft, fs=sample_rate, window=window,
                         nperseg=frame_length, noverlap=frame_length//2)

if len(noise_estimate) > len(noisy_signal):
    noise_estimate = noise_estimate[:len(noisy_signal)]

print(f"\n步骤3：重建时域噪声信号")
print(f"  • 扩展噪声幅度到所有时间帧")
print(f"  • 使用随机相位（因为噪声相位不重要）")
print(f"  • 逆STFT变换回时域")
print(f"  • 最终噪声估计长度：{len(noise_estimate)} 样本")

print("\n【3. 为什么有效？】")
print("-"*70)
print("统计学角度：")
print("  • 假设信号占用时间比例 > 噪声独占时间")
print("  • 取10%百分位 → 捕获大部分静音/低能量段")
print("  • 这些段主要是噪声")
print("")
print("频域优势：")
print("  • 不同频率的噪声水平可能不同（频谱着色噪声）")
print("  • 逐频率估计可以捕获这种差异")
print("  • 例如：低频噪声可能比高频噪声强")

print("\n【4. 数值示例】")
print("-"*70)

# 选择几个典型频率进行说明
freq_indices = [10, 50, 100, 200]  # 对应不同频率
for idx in freq_indices:
    if idx < len(f):
        freq = f[idx]
        time_series = magnitude[idx, :]
        percentile_val = noise_magnitude[idx]
        
        print(f"\n频率 {freq:.1f} Hz:")
        print(f"  • 能量范围：{time_series.min():.6f} ~ {time_series.max():.6f}")
        print(f"  • 平均能量：{time_series.mean():.6f}")
        print(f"  • {percentile}%百分位：{percentile_val:.6f}")
        print(f"  • 解释：该频率在 {percentile}% 的时间里能量低于 {percentile_val:.6f}")

# 计算SNR改善
signal_power = np.mean(speech ** 2)
noise_power = np.mean(noise ** 2)
true_snr = 10 * np.log10(signal_power / noise_power)

estimated_noise_power = np.mean(noise_estimate ** 2)
estimated_snr = 10 * np.log10(signal_power / estimated_noise_power)

print("\n【5. 性能评估】")
print("-"*70)
print(f"真实噪声功率：{noise_power:.8f}")
print(f"估计噪声功率：{estimated_noise_power:.8f}")
print(f"误差：{abs(noise_power - estimated_noise_power) / noise_power * 100:.2f}%")
print(f"")
print(f"真实SNR：{true_snr:.2f} dB")
print(f"估计SNR：{estimated_snr:.2f} dB")
print(f"差异：{abs(true_snr - estimated_snr):.2f} dB")

print("\n【6. 参数选择】")
print("-"*70)
print("百分位数 (percentile):")
print("  • 太低 (如 5%)：可能低估噪声，容易受极小值影响")
print("  • 太高 (如 30%)：可能包含语音能量，高估噪声")
print("  • 推荐：10% - 20%")
print("")
print("帧长 (frame_length):")
print("  • 太短：频率分辨率低，估计不准确")
print("  • 太长：时间分辨率低，跟踪噪声变化困难")
print("  • 推荐：2048 样本 (46ms @ 44.1kHz)")

print("\n【7. 与其他方法对比】")
print("-"*70)
print("vs. VAD方法：")
print("  ✓ 不需要能量阈值，更鲁棒")
print("  ✓ 利用全部数据，不会丢弃信息")
print("  ✓ 适用于连续语音（VAD可能提取不到足够静音段）")
print("")
print("vs. 最小统计法：")
print("  ✓ 计算更简单，速度更快")
print("  ✓ 不需要滑动窗口跟踪")
print("  ~ 假设噪声较平稳（最小统计法可以跟踪缓慢变化）")
print("")
print("vs. 中值滤波：")
print("  ✓ 频域操作，频率分辨率更高")
print("  ✓ 可以处理频谱着色噪声")

print("\n【8. 适用场景】")
print("-"*70)
print("✓ 语音信号处理（有间歇）")
print("✓ 背景噪声相对平稳")
print("✓ 需要快速准确的噪声估计")
print("✓ 对频率特异性噪声的场景")
print("")
print("⚠️ 不适合：")
print("  • 持续说话无停顿（百分位数会包含语音）")
print("  • 噪声快速变化（需要自适应方法）")
print("  • 瞬态冲击噪声（会被百分位数平滑掉）")

# 绘图
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# 1. 时域信号对比
ax = axes[0, 0]
t_plot = t[:int(0.5*sample_rate)]  # 前0.5秒
ax.plot(t_plot, noisy_signal[:len(t_plot)], 'b-', alpha=0.7, linewidth=0.5, label='含噪信号')
ax.plot(t_plot, noise[:len(t_plot)], 'r-', alpha=0.7, linewidth=0.5, label='真实噪声')
ax.plot(t_plot, noise_estimate[:len(t_plot)], 'g--', alpha=0.7, linewidth=0.8, label='估计噪声')
ax.set_xlabel('时间 (秒)', fontsize=10)
ax.set_ylabel('幅度', fontsize=10)
ax.set_title('时域对比', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2. 频谱图 (Spectrogram)
ax = axes[0, 1]
pcm = ax.pcolormesh(t_stft, f[:300], 20*np.log10(magnitude[:300, :] + 1e-10), 
                    shading='gouraud', cmap='viridis')
ax.set_xlabel('时间 (秒)', fontsize=10)
ax.set_ylabel('频率 (Hz)', fontsize=10)
ax.set_title('时频谱图 - 观察能量分布', fontsize=11, fontweight='bold')
plt.colorbar(pcm, ax=ax, label='能量 (dB)')

# 3. 单个频率的时间演化
ax = axes[1, 0]
freq_idx = 50  # 选择一个中等频率
ax.plot(t_stft, magnitude[freq_idx, :], 'b-', linewidth=1.5, label=f'频率 {f[freq_idx]:.1f} Hz 的能量')
ax.axhline(y=noise_magnitude[freq_idx], color='r', linestyle='--', linewidth=2, 
           label=f'{percentile}% 百分位 = {noise_magnitude[freq_idx]:.4f}')
# 标记百分位位置
sorted_vals = np.sort(magnitude[freq_idx, :])
percentile_idx = int(len(sorted_vals) * percentile / 100)
ax.fill_between(t_stft, 0, np.max(magnitude[freq_idx, :]), where=(magnitude[freq_idx, :] <= sorted_vals[percentile_idx]),
                alpha=0.3, color='red', label='最低10%能量')
ax.set_xlabel('时间 (秒)', fontsize=10)
ax.set_ylabel('幅度', fontsize=10)
ax.set_title(f'频率 {f[freq_idx]:.1f} Hz 能量随时间变化', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 4. 不同频率的百分位数
ax = axes[1, 1]
freq_plot = f[:500]  # 只绘制0-5kHz
noise_mag_plot = noise_magnitude[:500]
mean_mag_plot = np.mean(magnitude[:500, :], axis=1)
ax.semilogy(freq_plot, mean_mag_plot, 'b-', linewidth=1.5, label='平均能量')
ax.semilogy(freq_plot, noise_mag_plot, 'r-', linewidth=2, label=f'{percentile}% 百分位（噪声估计）')
ax.set_xlabel('频率 (Hz)', fontsize=10)
ax.set_ylabel('幅度 (对数刻度)', fontsize=10)
ax.set_title('频率-能量关系', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

# 5. 直方图：某频率的能量分布
ax = axes[2, 0]
freq_idx = 50
energy_values = magnitude[freq_idx, :]
ax.hist(energy_values, bins=50, alpha=0.7, color='blue', edgecolor='black', density=True)
ax.axvline(x=noise_magnitude[freq_idx], color='r', linestyle='--', linewidth=2, 
           label=f'{percentile}% 百分位')
ax.axvline(x=np.median(energy_values), color='g', linestyle='--', linewidth=2, label='中位数 (50%)')
ax.set_xlabel('能量', fontsize=10)
ax.set_ylabel('概率密度', fontsize=10)
ax.set_title(f'频率 {f[freq_idx]:.1f} Hz 能量分布直方图', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 6. 算法流程图
ax = axes[2, 1]
ax.axis('off')

flowchart = """
╔═══════════════════════════════════════╗
║   Spectral Floor 算法流程            ║
╚═══════════════════════════════════════╝

输入：含噪声信号 x(t)
     ↓
[1] 短时傅里叶变换 (STFT)
     X(f, t) = STFT(x(t))
     ↓
[2] 计算幅度谱
     |X(f, t)|
     ↓
[3] 对每个频率 f，在时间轴上
    计算百分位数 (10%)
     N(f) = percentile(|X(f, :)|, 10)
     ↓
[4] 扩展到所有时间帧
     N(f, t) = N(f)  (所有 t)
     ↓
[5] 添加随机相位
     Ñ(f, t) = N(f, t) × exp(jφ)
     ↓
[6] 逆STFT变换
     n(t) = ISTFT(Ñ(f, t))
     ↓
输出：噪声估计 n(t)

关键点：
• 步骤3是核心：假设最低10%能量
  代表噪声水平
• 相位随机因为噪声相位无规律
• 结果是平稳噪声的估计
"""

ax.text(0.05, 0.95, flowchart, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('data/output/spectral_floor_explanation.png', dpi=150, bbox_inches='tight')
print(f"\n📊 图表已保存到: data/output/spectral_floor_explanation.png")
plt.show()

print("\n" + "="*70)
print("总结：Spectral Floor 通过频域统计分析，利用信号的间歇性")
print("      在每个频率独立估计噪声水平，简单高效且鲁棒")
print("="*70)
