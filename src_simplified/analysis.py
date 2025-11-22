"""
信号分析 - 精简版
时域和频域信号分析
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq, rfft, rfftfreq

class SignalAnalysis:
    """时域信号分析类"""
    
    def __init__(self, sample_rate=44100):
        """
        初始化
        Args:
            sample_rate: 采样率(Hz)
        """
        self.sample_rate = sample_rate
    
    def calculate_statistics(self, signal):
        """
        计算信号统计特性
        Args:
            signal: 输入信号
        Returns:
            统计特性字典
        """
        return {
            'mean': np.mean(signal),
            'std': np.std(signal),
            'rms': np.sqrt(np.mean(signal ** 2)),
            'max': np.max(np.abs(signal)),
            'energy': np.sum(signal ** 2)
        }
    
    def plot_time_domain(self, signal, title='时域波形', save_path=None):
        """
        绘制时域波形
        Args:
            signal: 输入信号
            title: 图表标题
            save_path: 保存路径
        """
        t = np.arange(len(signal)) / self.sample_rate
        
        plt.figure(figsize=(10, 4))
        plt.plot(t, signal, linewidth=0.5)
        plt.xlabel('时间 (秒)')
        plt.ylabel('幅度')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_envelope(self, signal, title='信号包络', save_path=None):
        """
        绘制信号包络
        Args:
            signal: 输入信号
            title: 图表标题
            save_path: 保存路径
        """
        from scipy.signal import hilbert
        
        # 计算解析信号和包络
        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)
        
        t = np.arange(len(signal)) / self.sample_rate
        
        plt.figure(figsize=(10, 4))
        plt.plot(t, signal, 'b-', alpha=0.5, linewidth=0.5, label='原始信号')
        plt.plot(t, envelope, 'r-', linewidth=1.5, label='包络')
        plt.plot(t, -envelope, 'r-', linewidth=1.5)
        plt.xlabel('时间 (秒)')
        plt.ylabel('幅度')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class FrequencyAnalysis:
    """频域信号分析类"""
    
    def __init__(self, sample_rate=44100):
        """
        初始化
        Args:
            sample_rate: 采样率(Hz)
        """
        self.sample_rate = sample_rate
    
    def compute_fft_spectrum(self, signal):
        """
        计算FFT频谱
        Args:
            signal: 输入信号
        Returns:
            频谱分析结果字典
        """
        # 计算单边频谱
        spectrum = rfft(signal)
        freqs = rfftfreq(len(signal), 1/self.sample_rate)
        magnitude = np.abs(spectrum)
        
        # 找到主频率
        dominant_idx = np.argmax(magnitude[1:]) + 1  # 跳过DC分量
        dominant_freq = freqs[dominant_idx]
        
        return {
            'frequencies': freqs,
            'magnitude': magnitude,
            'phase': np.angle(spectrum),
            'dominant_frequency': dominant_freq
        }
    
    def calculate_snr(self, signal, noise):
        """
        计算信噪比
        Args:
            signal: 信号
            noise: 噪声
        Returns:
            SNR (dB)
        """
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def plot_fft_spectrum(self, signal, title='FFT频谱', save_path=None):
        """
        绘制FFT频谱
        Args:
            signal: 输入信号
            title: 图表标题
            save_path: 保存路径
        """
        result = self.compute_fft_spectrum(signal)
        freqs = result['frequencies']
        magnitude = result['magnitude']
        
        # 只显示0-5000 Hz
        mask = freqs <= 5000
        
        plt.figure(figsize=(10, 4))
        plt.plot(freqs[mask], magnitude[mask])
        plt.xlabel('频率 (Hz)')
        plt.ylabel('幅度')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_power_spectral_density(self, signal, title='功率谱密度', save_path=None):
        """
        绘制功率谱密度
        Args:
            signal: 输入信号
            title: 图表标题
            save_path: 保存路径
        """
        freqs, psd = scipy_signal.welch(signal, self.sample_rate, nperseg=2048)
        
        # 只显示0-5000 Hz
        mask = freqs <= 5000
        
        plt.figure(figsize=(10, 4))
        plt.semilogy(freqs[mask], psd[mask])
        plt.xlabel('频率 (Hz)')
        plt.ylabel('功率谱密度 (V²/Hz)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_spectrogram(self, signal, title='时频谱图', save_path=None):
        """
        绘制时频谱图
        Args:
            signal: 输入信号
            title: 图表标题
            save_path: 保存路径
        """
        f, t, Sxx = scipy_signal.spectrogram(signal, self.sample_rate, nperseg=2048)
        
        # 只显示0-5000 Hz
        freq_mask = f <= 5000
        
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t, f[freq_mask], 10*np.log10(Sxx[freq_mask, :] + 1e-10),
                      shading='gouraud', cmap='viridis')
        plt.ylabel('频率 (Hz)')
        plt.xlabel('时间 (秒)')
        plt.title(title)
        plt.colorbar(label='功率 (dB)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
