"""
时频域分析模块

实现信号的时域和频域分析，提供可视化和量化分析工具。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Tuple, Dict, Optional, List
import warnings

from utils import save_figure, setup_plot_style


class SignalAnalysis:
    """
    时域分析类

    提供信号的时域分析功能，包括波形绘制、统计特性计算、包络分析和自相关分析。
    """

    def __init__(self, sample_rate: int = 44100):
        """
        初始化时域分析器

        Args:
            sample_rate: 采样率，默认为44.1kHz
        """
        self.sample_rate = sample_rate
        setup_plot_style()

    def plot_time_domain(self, signal_data: np.ndarray, title: str = "时域波形",
                        time_axis: Optional[np.ndarray] = None,
                        show: bool = False, save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制时域波形

        Args:
            signal_data: 信号数据
            title: 图表标题
            time_axis: 时间轴，如果为None则自动生成
            show: 是否显示图表
            save_path: 保存路径

        Returns:
            matplotlib图表对象
        """
        if time_axis is None:
            time_axis = np.arange(len(signal_data)) / self.sample_rate

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time_axis, signal_data, linewidth=1)
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('幅度')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if save_path:
            save_figure(fig, save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def calculate_statistics(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        计算信号的统计特性

        Args:
            signal_data: 信号数据

        Returns:
            包含统计指标的字典
        """
        if len(signal_data) == 0:
            return {}

        stats = {}

        # 基本统计量
        stats['mean'] = np.mean(signal_data)
        stats['std'] = np.std(signal_data)
        stats['variance'] = np.var(signal_data)
        stats['rms'] = np.sqrt(np.mean(signal_data ** 2))
        stats['max'] = np.max(signal_data)
        stats['min'] = np.min(signal_data)
        stats['peak_to_peak'] = stats['max'] - stats['min']

        # 峰值因子和波形因子
        if stats['rms'] > 0:
            stats['crest_factor'] = stats['max'] / stats['rms']
            stats['form_factor'] = stats['rms'] / np.mean(np.abs(signal_data))
        else:
            stats['crest_factor'] = 0
            stats['form_factor'] = 0

        # 偏度和峰度
        if stats['std'] > 0:
            stats['skewness'] = np.mean(((signal_data - stats['mean']) / stats['std']) ** 3)
            stats['kurtosis'] = np.mean(((signal_data - stats['mean']) / stats['std']) ** 4) - 3
        else:
            stats['skewness'] = 0
            stats['kurtosis'] = 0

        return stats

    def plot_envelope(self, signal_data: np.ndarray, title: str = "信号包络",
                     show: bool = False, save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制信号包络

        Args:
            signal_data: 信号数据
            title: 图表标题
            show: 是否显示图表
            save_path: 保存路径

        Returns:
            matplotlib图表对象
        """
        # 使用希尔伯特变换计算包络
        analytic_signal = signal.hilbert(signal_data)
        amplitude_envelope = np.abs(analytic_signal)

        time_axis = np.arange(len(signal_data)) / self.sample_rate

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time_axis, signal_data, alpha=0.7, label='原始信号', linewidth=1)
        ax.plot(time_axis, amplitude_envelope, 'r-', label='包络', linewidth=2)
        ax.plot(time_axis, -amplitude_envelope, 'r-', linewidth=2)
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('幅度')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            save_figure(fig, save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def calculate_correlation(self, signal_data: np.ndarray, max_lag: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算自相关函数

        Args:
            signal_data: 信号数据
            max_lag: 最大延迟点数

        Returns:
            (lags, correlation): 延迟数组和自相关函数值
        """
        n = len(signal_data)
        max_lag = min(max_lag, n - 1)

        lags = np.arange(-max_lag, max_lag + 1)
        correlation = np.zeros(len(lags))

        for i, lag in enumerate(lags):
            if lag >= 0:
                correlation[i] = np.corrcoef(signal_data[:n-lag], signal_data[lag:])[0, 1]
            else:
                correlation[i] = np.corrcoef(signal_data[-lag:], signal_data[:n+lag])[0, 1]

        return lags / self.sample_rate, correlation


class FrequencyAnalysis:
    """
    频域分析类

    提供信号的频域分析功能，包括FFT频谱、功率谱密度、频谱图、信噪比和谐波失真分析。
    """

    def __init__(self, sample_rate: int = 44100):
        """
        初始化频域分析器

        Args:
            sample_rate: 采样率，默认为44.1kHz
        """
        self.sample_rate = sample_rate
        setup_plot_style()

    def plot_fft_spectrum(self, signal_data: np.ndarray, title: str = "FFT频谱",
                         window: str = 'none', show: bool = False,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制FFT频谱

        Args:
            signal_data: 信号数据
            title: 图表标题
            window: 窗函数类型
            show: 是否显示图表
            save_path: 保存路径

        Returns:
            matplotlib图表对象
        """
        n = len(signal_data)

        # 应用窗函数
        if window == 'hann':
            window_func = np.hanning(n)
        elif window == 'hamming':
            window_func = np.hamming(n)
        elif window == 'blackman':
            window_func = np.blackman(n)
        else:
            window_func = np.ones(n)

        windowed_signal = signal_data * window_func

        # 计算FFT
        fft_result = fft(windowed_signal)
        frequencies = fftfreq(n, 1 / self.sample_rate)

        # 只取正频率部分
        positive_freq_idx = frequencies >= 0
        frequencies = frequencies[positive_freq_idx]
        magnitude = np.abs(fft_result[positive_freq_idx])

        # 转换为dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(frequencies, magnitude_db, linewidth=1)
        ax.set_xlabel('频率 (Hz)')
        ax.set_ylabel('幅度 (dB)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, self.sample_rate / 2)

        if save_path:
            save_figure(fig, save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_power_spectral_density(self, signal_data: np.ndarray,
                                   title: str = "功率谱密度(PSD)",
                                   show: bool = False,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制功率谱密度

        Args:
            signal_data: 信号数据
            title: 图表标题
            show: 是否显示图表
            save_path: 保存路径

        Returns:
            matplotlib图表对象
        """
        frequencies, psd = signal.welch(signal_data, self.sample_rate,
                                       nperseg=1024, scaling='density')

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.semilogy(frequencies, psd, linewidth=1)
        ax.set_xlabel('频率 (Hz)')
        ax.set_ylabel('功率谱密度 (V²/Hz)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if save_path:
            save_figure(fig, save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_spectrogram(self, signal_data: np.ndarray, title: str = "频谱图",
                        nfft: int = 1024, hop_length: int = 256,
                        show: bool = False, save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制频谱图（时频分析）

        Args:
            signal_data: 信号数据
            title: 图表标题
            nfft: FFT点数
            hop_length: 帧移长度
            show: 是否显示图表
            save_path: 保存路径

        Returns:
            matplotlib图表对象
        """
        frequencies, times, spectrogram = signal.spectrogram(
            signal_data, self.sample_rate, nperseg=nfft, noverlap=nfft-hop_length
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.pcolormesh(times, frequencies, 10 * np.log10(spectrogram + 1e-10),
                          shading='gouraud', cmap='viridis')
        ax.set_ylabel('频率 (Hz)')
        ax.set_xlabel('时间 (秒)')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='强度 (dB)')

        if save_path:
            save_figure(fig, save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def calculate_snr(self, signal_data: np.ndarray, noise_data: np.ndarray) -> float:
        """
        计算信噪比(SNR)

        Args:
            signal_data: 信号数据
            noise_data: 噪声数据

        Returns:
            信噪比(dB)
        """
        signal_power = np.mean(signal_data ** 2)
        noise_power = np.mean(noise_data ** 2)

        if noise_power == 0:
            return float('inf')

        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def calculate_thd(self, signal_data: np.ndarray, fundamental_freq: float,
                     max_harmonics: int = 10) -> float:
        """
        计算总谐波失真(THD)

        Args:
            signal_data: 信号数据
            fundamental_freq: 基频(Hz)
            max_harmonics: 最大谐波次数

        Returns:
            总谐波失真百分比
        """
        n = len(signal_data)
        fft_result = fft(signal_data)
        frequencies = fftfreq(n, 1 / self.sample_rate)

        # 找到基频及其谐波
        fundamental_idx = np.argmin(np.abs(frequencies - fundamental_freq))
        fundamental_mag = np.abs(fft_result[fundamental_idx])

        harmonic_power = 0
        for harmonic in range(2, max_harmonics + 1):
            harmonic_freq = fundamental_freq * harmonic
            if harmonic_freq > self.sample_rate / 2:
                break

            harmonic_idx = np.argmin(np.abs(frequencies - harmonic_freq))
            harmonic_mag = np.abs(fft_result[harmonic_idx])
            harmonic_power += harmonic_mag ** 2

        if fundamental_mag == 0:
            return float('inf')

        thd_percent = np.sqrt(harmonic_power) / fundamental_mag * 100
        return thd_percent

    def compare_signals(self, original_signal: np.ndarray, processed_signal: np.ndarray,
                       title: str = "信号对比", show: bool = False,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        对比两个信号的时域和频域特性

        Args:
            original_signal: 原始信号
            processed_signal: 处理后的信号
            title: 图表标题
            show: 是否显示图表
            save_path: 保存路径

        Returns:
            matplotlib图表对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 时域对比
        time_axis = np.arange(len(original_signal)) / self.sample_rate
        axes[0, 0].plot(time_axis, original_signal, 'b-', alpha=0.7, label='原始信号', linewidth=1)
        axes[0, 0].plot(time_axis, processed_signal, 'r-', alpha=0.7, label='处理后信号', linewidth=1)
        axes[0, 0].set_xlabel('时间 (秒)')
        axes[0, 0].set_ylabel('幅度')
        axes[0, 0].set_title('时域波形对比')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 频域对比
        n = len(original_signal)
        freq_axis = fftfreq(n, 1 / self.sample_rate)[:n//2]

        orig_fft = np.abs(fft(original_signal))[:n//2]
        proc_fft = np.abs(fft(processed_signal))[:n//2]

        axes[0, 1].semilogy(freq_axis, orig_fft, 'b-', alpha=0.7, label='原始信号', linewidth=1)
        axes[0, 1].semilogy(freq_axis, proc_fft, 'r-', alpha=0.7, label='处理后信号', linewidth=1)
        axes[0, 1].set_xlabel('频率 (Hz)')
        axes[0, 1].set_ylabel('幅度')
        axes[0, 1].set_title('频域对比')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 差异信号
        difference = processed_signal - original_signal
        axes[1, 0].plot(time_axis, difference, 'g-', linewidth=1)
        axes[1, 0].set_xlabel('时间 (秒)')
        axes[1, 0].set_ylabel('幅度')
        axes[1, 0].set_title('差异信号')
        axes[1, 0].grid(True, alpha=0.3)

        # 频谱图对比
        axes[1, 1].plot(freq_axis, 20*np.log10(orig_fft+1e-10), 'b-', alpha=0.7, label='原始信号', linewidth=1)
        axes[1, 1].plot(freq_axis, 20*np.log10(proc_fft+1e-10), 'r-', alpha=0.7, label='处理后信号', linewidth=1)
        axes[1, 1].set_xlabel('频率 (Hz)')
        axes[1, 1].set_ylabel('幅度 (dB)')
        axes[1, 1].set_title('频谱对比 (dB)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.suptitle(title, fontsize=16)
        fig.subplots_adjust(top=0.93)

        if save_path:
            save_figure(fig, save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig


if __name__ == "__main__":
    # 测试分析模块
    sample_rate = 44100
    t = np.linspace(0, 1, sample_rate)
    test_signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)

    # 时域分析测试
    time_analyzer = SignalAnalysis(sample_rate)
    stats = time_analyzer.calculate_statistics(test_signal)
    print(f"时域统计特性: {stats}")

    # 频域分析测试
    freq_analyzer = FrequencyAnalysis(sample_rate)
    snr = freq_analyzer.calculate_snr(test_signal, 0.1 * np.random.randn(len(test_signal)))
    print(f"信噪比: {snr:.2f} dB")

    print("analysis.py 模块测试完成")