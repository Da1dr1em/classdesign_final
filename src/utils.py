"""
工具函数模块

提供音频处理相关的通用工具函数，包括文件操作、信号处理工具、可视化工具和性能评估。
"""

import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, List, Union


def ensure_dir(directory: str) -> None:
    """
    确保目录存在，如果不存在则创建

    Args:
        directory: 目录路径
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def list_audio_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    列出指定目录中的所有音频文件

    Args:
        directory: 目录路径
        extensions: 音频文件扩展名列表，默认为 ['.wav', '.mp3', '.flac', '.ogg']

    Returns:
        音频文件路径列表
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.ogg']

    audio_files = []
    for ext in extensions:
        audio_files.extend(Path(directory).glob(f'*{ext}'))

    return [str(f) for f in audio_files]


def load_audio(file_path: str, target_sr: int = 44100) -> Tuple[np.ndarray, int]:
    """
    加载音频文件并进行标准化处理

    Args:
        file_path: 音频文件路径
        target_sr: 目标采样率，默认为44.1kHz

    Returns:
        (audio_data, sample_rate): 音频数据和采样率

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 音频文件格式不支持
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"音频文件不存在: {file_path}")

    try:
        # 使用soundfile读取音频文件
        audio_data, sample_rate = sf.read(file_path)

        # 如果是立体声，转换为单声道
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # 重采样到目标采样率
        if sample_rate != target_sr:
            from scipy import signal
            # 计算重采样比例
            ratio = target_sr / sample_rate
            # 重采样
            audio_data = signal.resample(audio_data, int(len(audio_data) * ratio))
            sample_rate = target_sr

        return audio_data, sample_rate

    except Exception as e:
        raise ValueError(f"无法加载音频文件 {file_path}: {str(e)}")


def normalize_signal(signal: np.ndarray, target_max: float = 1.0) -> np.ndarray:
    """
    信号归一化处理

    Args:
        signal: 输入信号
        target_max: 目标最大值，默认为1.0

    Returns:
        归一化后的信号
    """
    if len(signal) == 0:
        return signal

    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal

    return signal * (target_max / max_val)


def pad_signal(signal: np.ndarray, target_length: int, mode: str = 'constant') -> np.ndarray:
    """
    信号补零处理

    Args:
        signal: 输入信号
        target_length: 目标长度
        mode: 补零模式，'constant'或'reflect'

    Returns:
        补零后的信号
    """
    current_length = len(signal)

    if current_length >= target_length:
        return signal[:target_length]

    padding = target_length - current_length

    if mode == 'constant':
        return np.pad(signal, (0, padding), mode='constant')
    elif mode == 'reflect':
        return np.pad(signal, (0, padding), mode='reflect')
    else:
        raise ValueError(f"不支持的补零模式: {mode}")


def frame_signal(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """
    信号分帧处理

    Args:
        signal: 输入信号
        frame_length: 帧长度
        hop_length: 帧移长度

    Returns:
        分帧后的信号矩阵，形状为 (帧数, 帧长度)
    """
    signal_length = len(signal)

    # 计算帧数
    num_frames = 1 + (signal_length - frame_length) // hop_length

    # 创建帧矩阵
    frames = np.zeros((num_frames, frame_length))

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        if end <= signal_length:
            frames[i] = signal[start:end]
        else:
            # 最后一帧可能不够长，需要补零
            frames[i, :signal_length-start] = signal[start:]

    return frames


def estimate_noise_vad(signal: np.ndarray, sample_rate: int = 44100, 
                      frame_length: int = 2048, hop_length: int = 512,
                      energy_threshold_percentile: float = 20.0) -> np.ndarray:
    """
    使用语音活动检测(VAD)估计噪声
    
    通过检测信号中的低能量段(静音段)来估计背景噪声特性
    
    Args:
        signal: 输入信号
        sample_rate: 采样率
        frame_length: 帧长度
        hop_length: 帧移
        energy_threshold_percentile: 能量阈值百分位数(0-100)，默认20表示取最低20%能量的帧
    
    Returns:
        估计的噪声信号
    """
    # 分帧
    frames = frame_signal(signal, frame_length, hop_length)
    
    # 计算每帧的能量
    frame_energy = np.sum(frames ** 2, axis=1)
    
    # 使用百分位数确定能量阈值
    energy_threshold = np.percentile(frame_energy, energy_threshold_percentile)
    
    # 找出低能量帧(静音段)
    silence_frames = frames[frame_energy <= energy_threshold]
    
    if len(silence_frames) == 0:
        # 如果没有检测到静音段，使用最低能量的10%帧
        num_frames = max(1, int(len(frames) * 0.1))
        silence_indices = np.argsort(frame_energy)[:num_frames]
        silence_frames = frames[silence_indices]
    
    # 将静音段拼接成噪声估计
    noise_estimate = silence_frames.flatten()[:len(signal)]
    
    # 如果噪声估计长度不足，用均值填充
    if len(noise_estimate) < len(signal):
        noise_mean = np.mean(silence_frames)
        noise_std = np.std(silence_frames)
        additional_noise = np.random.normal(noise_mean, noise_std, len(signal) - len(noise_estimate))
        noise_estimate = np.concatenate([noise_estimate, additional_noise])
    
    return noise_estimate


def add_noise(signal: np.ndarray, noise_type: str = 'white', snr_db: float = 10.0) -> np.ndarray:
    """
    向信号添加噪声

    Args:
        signal: 原始信号
        noise_type: 噪声类型，'white'（白噪声）或'pink'（粉红噪声）
        snr_db: 信噪比（dB）

    Returns:
        添加噪声后的信号
    """
    signal_power = np.mean(signal ** 2)

    # 计算噪声功率
    noise_power = signal_power / (10 ** (snr_db / 10))

    # 生成噪声
    if noise_type == 'white':
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    elif noise_type == 'pink':
        # 粉红噪声生成（简化版本）
        white_noise = np.random.normal(0, 1, len(signal))
        # 使用一阶差分滤波器近似粉红噪声
        pink_noise = np.cumsum(white_noise)
        pink_noise = pink_noise - np.mean(pink_noise)
        # 调整功率
        current_power = np.mean(pink_noise ** 2)
        noise = pink_noise * np.sqrt(noise_power / current_power)
    else:
        raise ValueError(f"不支持的噪声类型: {noise_type}")

    return signal + noise


def save_figure(fig: plt.Figure, file_path: str, dpi: int = 300,
                bbox_inches: str = 'tight') -> None:
    """
    保存图表

    Args:
        fig: matplotlib图表对象
        file_path: 保存路径
        dpi: 分辨率
        bbox_inches: 边界框设置
    """
    ensure_dir(os.path.dirname(file_path))
    fig.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)


def setup_plot_style() -> None:
    import matplotlib.pyplot as plt

    # 设置图表的中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimSun', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = [10, 6]

# 直接初始化
setup_plot_style()


def evaluate_noise_reduction(original_signal: np.ndarray,
                           noisy_signal: np.ndarray,
                           denoised_signal: np.ndarray) -> dict:
    """
    评估降噪效果

    Args:
        original_signal: 原始干净信号
        noisy_signal: 含噪信号
        denoised_signal: 降噪后信号

    Returns:
        包含各种评估指标的字典
    """
    metrics = {}

    # 计算信噪比(SNR)
    def calculate_snr(signal, noise):
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)

    # 原始信噪比
    noise = noisy_signal - original_signal
    metrics['original_snr_db'] = calculate_snr(original_signal, noise)

    # 降噪后信噪比
    residual_noise = denoised_signal - original_signal
    metrics['denoised_snr_db'] = calculate_snr(original_signal, residual_noise)

    # 信噪比改善量
    metrics['snr_improvement_db'] = metrics['denoised_snr_db'] - metrics['original_snr_db']

    # 均方根误差(RMSE)
    metrics['rmse'] = np.sqrt(np.mean((denoised_signal - original_signal) ** 2))

    # 峰值信噪比(PSNR)
    max_signal = np.max(np.abs(original_signal))
    if metrics['rmse'] > 0:
        metrics['psnr_db'] = 20 * np.log10(max_signal / metrics['rmse'])
    else:
        metrics['psnr_db'] = float('inf')

    # 相关系数
    correlation = np.corrcoef(original_signal, denoised_signal)[0, 1]
    metrics['correlation'] = correlation

    return metrics


def calculate_metrics(signal: np.ndarray) -> dict:
    """
    计算信号的基本统计特性

    Args:
        signal: 输入信号

    Returns:
        包含统计指标的字典
    """
    if len(signal) == 0:
        return {}

    metrics = {}

    # 基本统计量
    metrics['mean'] = np.mean(signal)
    metrics['std'] = np.std(signal)
    metrics['rms'] = np.sqrt(np.mean(signal ** 2))
    metrics['max'] = np.max(signal)
    metrics['min'] = np.min(signal)
    metrics['peak_to_peak'] = metrics['max'] - metrics['min']

    # 峰值因子和波形因子
    if metrics['rms'] > 0:
        metrics['crest_factor'] = metrics['max'] / metrics['rms']
        metrics['form_factor'] = metrics['rms'] / np.mean(np.abs(signal))
    else:
        metrics['crest_factor'] = 0
        metrics['form_factor'] = 0

    # 偏度和峰度
    metrics['skewness'] = np.mean(((signal - metrics['mean']) / metrics['std']) ** 3)
    metrics['kurtosis'] = np.mean(((signal - metrics['mean']) / metrics['std']) ** 4) - 3

    return metrics


def save_audio(audio_data: np.ndarray, sample_rate: int, file_path: str,
               format: str = 'WAV', subtype: str = 'PCM_16') -> None:
    """
    保存音频文件

    Args:
        audio_data: 音频数据
        sample_rate: 采样率
        file_path: 保存路径
        format: 音频格式
        subtype: 子类型
    """
    ensure_dir(os.path.dirname(file_path))
    sf.write(file_path, audio_data, sample_rate, format=format, subtype=subtype)


if __name__ == "__main__":
    # 测试工具函数
    print("utils.py 模块测试完成")