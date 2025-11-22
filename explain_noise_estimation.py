#!/usr/bin/env python
"""
噪声估计方法详细演示

展示VAD（语音活动检测）方法如何估计噪声
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import estimate_noise_vad, frame_signal

def visualize_noise_estimation(audio_file=None):
    """可视化展示噪声估计过程"""
    
    print("="*70)
    print("噪声估计方法详细说明 - 基于VAD（语音活动检测）")
    print("="*70)
    
    # 如果提供了音频文件，使用实际音频；否则创建演示信号
    if audio_file:
        from utils import load_audio
        signal, sample_rate = load_audio(audio_file)
        print(f"\n使用实际音频: {audio_file}")
        print(f"  长度: {len(signal)} 样本 ({len(signal)/sample_rate:.2f} 秒)")
    else:
        # 创建演示信号：纯音 + 静音段 + 噪声
        sample_rate = 44100
        duration = 2
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # 创建有说话和静音交替的信号
        signal = np.zeros_like(t)
        # 0-0.5秒: 静音（只有噪声）
        signal[0:int(0.5*sample_rate)] = 0.05 * np.random.randn(int(0.5*sample_rate))
        # 0.5-1.5秒: 说话（440Hz音调 + 噪声）
        speech_start = int(0.5*sample_rate)
        speech_end = int(1.5*sample_rate)
        signal[speech_start:speech_end] = (
            0.5 * np.sin(2 * np.pi * 440 * t[speech_start:speech_end]) + 
            0.05 * np.random.randn(speech_end - speech_start)
        )
        # 1.5-2秒: 静音（只有噪声）
        signal[int(1.5*sample_rate):] = 0.05 * np.random.randn(len(signal) - int(1.5*sample_rate))
        
        print("\n使用演示信号（模拟对话场景）:")
        print("  0.0-0.5秒: 静音段（只有噪声）")
        print("  0.5-1.5秒: 说话段（语音+噪声）")
        print("  1.5-2.0秒: 静音段（只有噪声）")
    
    # 参数设置
    frame_length = 2048
    hop_length = 512
    energy_threshold_percentile = 20.0
    
    print(f"\n" + "="*70)
    print("步骤1: 分帧分析")
    print("="*70)
    print(f"  帧长度: {frame_length} 样本 ({frame_length/sample_rate*1000:.1f} ms)")
    print(f"  帧移: {hop_length} 样本 ({hop_length/sample_rate*1000:.1f} ms)")
    
    # 分帧
    frames = frame_signal(signal, frame_length, hop_length)
    print(f"  总帧数: {len(frames)}")
    
    print(f"\n" + "="*70)
    print("步骤2: 计算每帧能量")
    print("="*70)
    
    # 计算每帧能量
    frame_energy = np.sum(frames ** 2, axis=1)
    print(f"  能量公式: E = Σ(x²)")
    print(f"  能量范围: {np.min(frame_energy):.2e} ~ {np.max(frame_energy):.2e}")
    print(f"  能量均值: {np.mean(frame_energy):.2e}")
    
    print(f"\n" + "="*70)
    print("步骤3: 确定能量阈值")
    print("="*70)
    
    # 使用百分位数确定阈值
    energy_threshold = np.percentile(frame_energy, energy_threshold_percentile)
    print(f"  方法: 使用第 {energy_threshold_percentile} 百分位数")
    print(f"  含义: 能量最低的 {energy_threshold_percentile}% 的帧被认为是\"静音段\"")
    print(f"  阈值: {energy_threshold:.2e}")
    
    # 统计低能量帧
    silence_mask = frame_energy <= energy_threshold
    num_silence_frames = np.sum(silence_mask)
    print(f"  检测到的静音帧数: {num_silence_frames} / {len(frames)} ({num_silence_frames/len(frames)*100:.1f}%)")
    
    print(f"\n" + "="*70)
    print("步骤4: 提取静音段作为噪声样本")
    print("="*70)
    
    silence_frames = frames[silence_mask]
    print(f"  提取的静音帧数: {len(silence_frames)}")
    print(f"  静音段总样本数: {len(silence_frames) * frame_length}")
    
    print(f"\n" + "="*70)
    print("步骤5: 生成噪声估计信号")
    print("="*70)
    
    # 拼接噪声估计
    noise_estimate = silence_frames.flatten()[:len(signal)]
    
    if len(noise_estimate) < len(signal):
        print(f"  ⚠️ 静音段不足，需要扩展")
        noise_mean = np.mean(silence_frames)
        noise_std = np.std(silence_frames)
        additional_length = len(signal) - len(noise_estimate)
        print(f"     使用高斯分布生成额外 {additional_length} 个样本")
        print(f"     均值: {noise_mean:.4f}, 标准差: {noise_std:.4f}")
        additional_noise = np.random.normal(noise_mean, noise_std, additional_length)
        noise_estimate = np.concatenate([noise_estimate, additional_noise])
    
    print(f"  最终噪声估计长度: {len(noise_estimate)} 样本")
    
    print(f"\n" + "="*70)
    print("步骤6: 计算信噪比")
    print("="*70)
    
    # 计算SNR
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise_estimate ** 2)
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    print(f"  信号功率: {signal_power:.6f}")
    print(f"  噪声功率: {noise_power:.6f}")
    print(f"  信噪比 (SNR): {snr_db:.2f} dB")
    print(f"  ")
    print(f"  公式: SNR = 10 × log₁₀(P_signal / P_noise)")
    
    print(f"\n" + "="*70)
    print("💡 关键理解")
    print("="*70)
    print("""
1. 噪声估计的假设:
   - 信号中存在"静音段"（低能量段）
   - 静音段主要由噪声组成
   - 噪声在整段音频中相对平稳

2. VAD方法的优势:
   ✅ 不需要预先知道噪声特性
   ✅ 自动适应不同的音频
   ✅ 计算简单高效

3. VAD方法的局限:
   ⚠️ 如果音频中没有静音段，估计会不准确
   ⚠️ 对非平稳噪声（如突发噪声）效果较差
   ⚠️ 能量阈值的选择影响估计质量

4. 能量百分位数的影响:
   - 20% (默认): 取能量最低的20%帧作为噪声
   - 值越小: 越保守，只取最安静的部分
   - 值越大: 越激进，可能包含部分语音

5. 实际应用:
   - 对于对话/语音: 效果较好（有自然停顿）
   - 对于音乐: 效果中等（取决于是否有静音段）
   - 对于持续信号: 效果较差（无明显静音段）
    """)
    
    # 可视化（如果是演示信号）
    if not audio_file:
        print(f"\n" + "="*70)
        print("生成可视化图表...")
        print("="*70)
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 1. 原始信号
        time_axis = np.arange(len(signal)) / sample_rate
        axes[0].plot(time_axis, signal, linewidth=0.5, alpha=0.7)
        axes[0].set_title('原始信号（语音+噪声）', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('时间 (秒)')
        axes[0].set_ylabel('幅度')
        axes[0].grid(True, alpha=0.3)
        axes[0].axvspan(0, 0.5, alpha=0.2, color='green', label='静音段')
        axes[0].axvspan(0.5, 1.5, alpha=0.2, color='red', label='说话段')
        axes[0].axvspan(1.5, 2.0, alpha=0.2, color='green')
        axes[0].legend()
        
        # 2. 帧能量
        frame_times = np.arange(len(frame_energy)) * hop_length / sample_rate
        axes[1].plot(frame_times, frame_energy, marker='o', markersize=3, linewidth=1)
        axes[1].axhline(y=energy_threshold, color='r', linestyle='--', linewidth=2, 
                       label=f'能量阈值 (第{energy_threshold_percentile}百分位)')
        axes[1].fill_between(frame_times, 0, energy_threshold, alpha=0.3, color='green', 
                            label='低能量段（噪声）')
        axes[1].set_title('每帧能量分布', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('时间 (秒)')
        axes[1].set_ylabel('能量')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # 3. 噪声估计对比
        axes[2].plot(time_axis, signal, linewidth=0.5, alpha=0.5, label='原始信号')
        axes[2].plot(time_axis, noise_estimate, linewidth=0.5, alpha=0.7, 
                    label='估计的噪声', color='orange')
        axes[2].set_title('噪声估计结果', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('时间 (秒)')
        axes[2].set_ylabel('幅度')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        
        # 保存图表
        output_path = 'results/figures/noise_estimation_explained.png'
        from utils import ensure_dir
        ensure_dir('results/figures')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ 图表已保存: {output_path}")
        plt.close()
    
    print(f"\n" + "="*70)
    print("✅ 噪声估计过程演示完成！")
    print("="*70)

if __name__ == "__main__":
    # 使用演示信号
    print("\n【模式1: 演示信号】\n")
    visualize_noise_estimation()
    
    # 使用实际音频文件
    print("\n\n【模式2: 实际音频文件】\n")
    actual_file = r".\data\input\conversation_human.wav"
    import os
    if os.path.exists(actual_file):
        visualize_noise_estimation(actual_file)
    else:
        print(f"跳过实际音频测试（文件不存在: {actual_file}）")
