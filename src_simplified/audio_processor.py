"""
音频处理器 - 精简版
核心音频处理类
"""
import numpy as np
from utils import load_audio, save_audio, normalize_signal, estimate_noise
from filters import FilterDesign
from analysis import SignalAnalysis, FrequencyAnalysis

class AudioProcessor:
    """音频处理器核心类"""
    
    def __init__(self, sample_rate=44100, enable_plots=True):
        """
        初始化处理器
        Args:
            sample_rate: 采样率(Hz)
            enable_plots: 是否生成图表
        """
        self.sample_rate = sample_rate
        self.enable_plots = enable_plots
        
        # 核心模块
        self.filter_design = FilterDesign(sample_rate)
        self.signal_analysis = SignalAnalysis(sample_rate)
        self.frequency_analysis = FrequencyAnalysis(sample_rate)
        
        # 数据存储
        self.audio_data = None          # 当前音频数据
        self.original_data = None        # 原始音频数据
        self.processed_data = None       # 处理后的数据
        self.noise_estimate = None       # 噪声估计
        self.input_file = None           # 输入文件路径
    
    def load_audio(self, file_path):
        """
        加载音频文件
        Args:
            file_path: 音频文件路径
        """
        print(f"  加载音频: {file_path}")
        self.audio_data, _ = load_audio(file_path, self.sample_rate)
        self.original_data = self.audio_data.copy()
        self.input_file = file_path
        
        # 自动估计噪声(使用Spectral Floor方法)
        self.noise_estimate = estimate_noise(
            self.audio_data, 
            self.sample_rate,
            method='spectral_floor'
        )
        
        # 计算原始SNR
        snr = self.frequency_analysis.calculate_snr(
            self.audio_data, 
            self.noise_estimate
        )
        print(f"  音频长度: {len(self.audio_data)} 采样点")
        print(f"  原始SNR: {snr:.2f} dB")
    
    def analyze_time_domain(self):
        """时域分析"""
        print("  计算时域统计特性...")
        stats = self.signal_analysis.calculate_statistics(self.audio_data)
        print(f"  均值: {stats['mean']:.6f}")
        print(f"  RMS: {stats['rms']:.6f}")
        
        if self.enable_plots:
            self.signal_analysis.plot_time_domain(
                self.audio_data,
                title="原始信号时域波形",
                save_path="results/figures/time_domain_original.png"
            )
            self.signal_analysis.plot_envelope(
                self.audio_data,
                title="信号包络",
                save_path="results/figures/envelope_original.png"
            )
    
    def analyze_frequency_domain(self):
        """频域分析"""
        print("  计算频域特性...")
        spectrum = self.frequency_analysis.compute_fft_spectrum(self.audio_data)
        print(f"  主频率: {spectrum['dominant_frequency']:.2f} Hz")
        
        if self.enable_plots:
            self.frequency_analysis.plot_fft_spectrum(
                self.audio_data,
                title="FFT频谱",
                save_path="results/figures/fft_spectrum_original.png"
            )
            self.frequency_analysis.plot_power_spectral_density(
                self.audio_data,
                title="功率谱密度",
                save_path="results/figures/psd_original.png"
            )
            self.frequency_analysis.plot_spectrogram(
                self.audio_data,
                title="时频谱图",
                save_path="results/figures/spectrogram_original.png"
            )
    
    def apply_filter(self, filter_type, **kwargs):
        """
        应用滤波器
        Args:
            filter_type: 滤波器类型
            **kwargs: 滤波器参数(cutoff_freq, lowcut_freq, highcut_freq等)
        """
        print(f"  设计{filter_type}滤波器...")
        
        # 设计滤波器
        if filter_type == 'fir_lowpass':
            b = self.filter_design.design_fir_lowpass(**kwargs)
            self.processed_data = self.filter_design.apply_fir_filter(
                self.audio_data, b
            )
        elif filter_type == 'fir_highpass':
            b = self.filter_design.design_fir_highpass(**kwargs)
            self.processed_data = self.filter_design.apply_fir_filter(
                self.audio_data, b
            )
        elif filter_type == 'fir_bandpass':
            b = self.filter_design.design_fir_bandpass(**kwargs)
            self.processed_data = self.filter_design.apply_fir_filter(
                self.audio_data, b
            )
        elif filter_type == 'iir_lowpass':
            b, a = self.filter_design.design_iir_lowpass(**kwargs)
            self.processed_data = self.filter_design.apply_iir_filter(
                self.audio_data, b, a
            )
        elif filter_type == 'lms':
            self.processed_data = self.filter_design.apply_lms_filter(
                self.audio_data, 
                self.noise_estimate, 
                **kwargs
            )
        elif filter_type == 'nlms':
            self.processed_data = self.filter_design.apply_nlms_filter(
                self.audio_data, 
                self.noise_estimate, 
                **kwargs
            )
        elif filter_type == 'wiener':
            self.processed_data = self.filter_design.apply_wiener_filter(
                self.audio_data, 
                self.noise_estimate, 
                **kwargs
            )
        
        print(f"  滤波完成")
    
    def enhance_signal(self, method='normalize', **kwargs):
        """
        信号增强
        Args:
            method: 增强方法('normalize'或'dynamic_range')
            **kwargs: 方法参数
        """
        if method == 'normalize':
            target_max = kwargs.get('target_max', 0.9)
            self.processed_data = normalize_signal(
                self.processed_data, 
                target_max
            )
            print(f"  归一化到 {target_max}")
        elif method == 'dynamic_range':
            # 动态范围压缩
            threshold = kwargs.get('threshold', 0.5)
            ratio = kwargs.get('ratio', 2.0)
            compressed = self.processed_data.copy()
            mask = np.abs(compressed) > threshold
            compressed[mask] = threshold + (compressed[mask] - threshold) / ratio
            self.processed_data = compressed
            print(f"  动态压缩: 阈值={threshold}, 比率={ratio}")
    
    def analyze_processed_signal(self):
        """分析处理后的信号"""
        print("  计算性能指标...")
        
        # 重新估计处理后信号的噪声
        processed_noise = estimate_noise(
            self.processed_data,
            self.sample_rate,
            method='spectral_floor'
        )
        
        # 计算SNR
        original_snr = self.frequency_analysis.calculate_snr(
            self.original_data, 
            self.noise_estimate
        )
        processed_snr = self.frequency_analysis.calculate_snr(
            self.processed_data, 
            processed_noise
        )
        
        # 计算其他指标
        correlation = np.corrcoef(
            self.original_data, 
            self.processed_data
        )[0, 1]
        
        # 显示结果
        print("\n  " + "="*50)
        print("  性能评估结果:")
        print("  " + "="*50)
        print(f"  原始SNR:      {original_snr:.2f} dB")
        print(f"  处理后SNR:    {processed_snr:.2f} dB")
        print(f"  SNR改善:      {processed_snr - original_snr:+.2f} dB")
        print(f"  信号相关性:   {correlation:.3f}")
        print("  " + "="*50)
        
        if self.enable_plots:
            self._plot_comparison()
    
    def _plot_comparison(self):
        """绘制对比图"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('处理前后对比', fontsize=14, fontweight='bold')
        
        t = np.arange(len(self.original_data)) / self.sample_rate
        plot_len = min(len(t), int(0.5 * self.sample_rate))
        
        # 时域波形
        axes[0, 0].plot(t[:plot_len], self.original_data[:plot_len], 'b-', linewidth=0.5)
        axes[0, 0].set_title('原始信号')
        axes[0, 0].set_xlabel('时间(秒)')
        axes[0, 0].set_ylabel('幅度')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(t[:plot_len], self.processed_data[:plot_len], 'r-', linewidth=0.5)
        axes[0, 1].set_title('处理后信号')
        axes[0, 1].set_xlabel('时间(秒)')
        axes[0, 1].set_ylabel('幅度')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 频谱
        from scipy.fft import rfft, rfftfreq
        freqs = rfftfreq(len(self.original_data), 1/self.sample_rate)
        orig_fft = np.abs(rfft(self.original_data))
        proc_fft = np.abs(rfft(self.processed_data))
        
        freq_mask = freqs <= 5000
        axes[1, 0].semilogy(freqs[freq_mask], orig_fft[freq_mask], 'b-', linewidth=1)
        axes[1, 0].set_title('原始频谱')
        axes[1, 0].set_xlabel('频率(Hz)')
        axes[1, 0].set_ylabel('幅度(对数)')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].semilogy(freqs[freq_mask], proc_fft[freq_mask], 'r-', linewidth=1)
        axes[1, 1].set_title('处理后频谱')
        axes[1, 1].set_xlabel('频率(Hz)')
        axes[1, 1].set_ylabel('幅度(对数)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/figures/comparison_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_output(self, save_difference=False):
        """
        保存处理结果
        Args:
            save_difference: 是否保存差分信号(噪声)
        """
        import os
        
        # 生成输出文件名
        basename = os.path.splitext(os.path.basename(self.input_file))[0]
        output_file = f"data/output/{basename}_denoised.wav"
        
        # 保存处理后的音频
        save_audio(output_file, self.processed_data, self.sample_rate)
        print(f"\n  已保存: {output_file}")
        
        # 保存差分信号(噪声)
        if save_difference:
            difference = self.processed_data - self.original_data
            diff_file = f"data/output/{basename}_removed_noise.wav"
            save_audio(diff_file, difference, self.sample_rate)
            print(f"  已保存: {diff_file}")
