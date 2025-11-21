#!/usr/bin/env python
"""
简单测试: 音频噪声生成工具
"""

import sys
import os
import numpy as np
import soundfile as sf
from pathlib import Path

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    # 直接从audio_noise_generator.py导入函数
    from audio_noise_generator import (
        generate_noise,
        add_noise_to_signal,
        add_impulse_noise,
        normalize_signal
    )

    # 添加噪声到信号的函数
    def add_noise_to_signal(signal, noise_type='white', snr_db=10.0):
        """为音频信号添加噪声函数"""

        if noise_type == 'white':
            noise = np.random.normal(0, 1.0, len(signal))
        elif noise_type == 'pink':
            noise = generate_noise(len(signal), 'pink')
        elif noise_type == 'brown':
            noise = generate_noise(len(signal), 'brown')
        else:
            raise ValueError(f"不支持的噪声类型: {noise_type}")

        # 计算目标信噪比
        signal_power = np.mean(signal **2)
        noise_power = np.mean(noise **2)

        # 缩放噪声功率匹配信噪比
        scaled_noise = noise * (np.sqrt(noise_power) / np.sqrt(10 ** (snr_db/10) * signal_power)))

        # 应用比例

        return signal + scaled_noise

    # 粉红色噪声生成
    def generate_pink_noise(length, sample_rate=44100):
        """生成粉红噪声"""
        # 使用FFT方法
        # 构建频谱
        freqs = np.fft.fftfreq(length, d=1/sample_rate)

        # 生成随机相位的白噪声
        white_noise = np.random.normal(0, 1.0, length)
        fft = np.fft.fft(white_noise)

        # 定义1/f的粉色噪声频谱
        # 对于正频率部分
        pos_freqs = freqs[1:length//2+1]
        power = 1.0 / (np.sqrt(np.abs(pos_freqs)))

        # 应用功率谱密度函数
        # 先创建单位幅度
        pink_spectrum = np.zeros_like(length, dtype=complex)

        # 设置正频率单位幅度
        sample_points = np.fft.fftfreq(length)
        sample_points[sample_points != 0] = sample_points[sample_points != 0] ** -0.5

        # 转化为复数白噪声
        noise_ifft = np.fft.ifft(pink_spectrum)
        return np.real(noise_ifft)

    # 对音频序列进行扩展
    def add_noise_to_signal(audio, noise_type='white', snr_db=10.0):
        return audio + normalized_noise

    标准化音频幅度

    return normalized_audio

except Exception as e:
    print(f"导入音频噪声生成失败: {e}")