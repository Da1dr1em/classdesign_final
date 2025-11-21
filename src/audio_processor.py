"""
音频处理核心类

封装音频处理的核心功能，提供简洁的API供主程序调用，
管理音频文件的读取、处理和保存。
"""

import numpy as np
import os
from typing import Tuple, Dict, Optional, List, Any
import logging

from utils import load_audio, save_audio, normalize_signal, ensure_dir
from filters import FilterDesign
from analysis import SignalAnalysis, FrequencyAnalysis


class AudioProcessor:
    """
    音频处理器核心类

    提供完整的音频降噪处理流程，包括信号分析、滤波器设计、信号增强和结果评估。
    """

    def __init__(self, input_file: str = None, sample_rate: int = 44100):
        """
        初始化音频处理器

        Args:
            input_file: 输入音频文件路径
            sample_rate: 目标采样率，默认为44.1kHz
        """
        self.sample_rate = sample_rate
        self.input_file = input_file
        self.audio_data = None
        self.original_data = None
        self.filter_design = FilterDesign(sample_rate)
        self.signal_analysis = SignalAnalysis(sample_rate)
        self.frequency_analysis = FrequencyAnalysis(sample_rate)

        # 处理结果存储
        self.processed_data = None
        self.analysis_results = {}
        self.filter_coefficients = {}

        # 设置日志
        self._setup_logging()

        if input_file:
            self.load_audio(input_file)

    def _setup_logging(self):
        """设置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_audio(self, file_path: str) -> bool:
        """
        加载音频文件

        Args:
            file_path: 音频文件路径

        Returns:
            加载是否成功
        """
        try:
            self.logger.info(f"正在加载音频文件: {file_path}")
            self.audio_data, actual_sr = load_audio(file_path, self.sample_rate)
            self.original_data = self.audio_data.copy()
            self.input_file = file_path

            self.logger.info(f"音频加载成功: {len(self.audio_data)} 个采样点, {actual_sr} Hz")
            return True

        except Exception as e:
            self.logger.error(f"音频加载失败: {str(e)}")
            return False

    def analyze_time_domain(self) -> Dict[str, Any]:
        """
        时域分析

        Returns:
            时域分析结果
        """
        if self.audio_data is None:
            raise ValueError("请先加载音频数据")

        self.logger.info("正在进行时域分析")

        # 计算统计特性
        stats = self.signal_analysis.calculate_statistics(self.audio_data)

        # 绘制时域波形
        time_fig = self.signal_analysis.plot_time_domain(
            self.audio_data,
            title="原始信号时域波形",
            save_path="results/figures/time_domain_original.png"
        )

        # 绘制信号包络
        envelope_fig = self.signal_analysis.plot_envelope(
            self.audio_data,
            title="原始信号包络",
            save_path="results/figures/envelope_original.png"
        )

        # 计算自相关
        lags, correlation = self.signal_analysis.calculate_correlation(self.audio_data)

        self.analysis_results['time_domain'] = {
            'statistics': stats,
            'correlation': {
                'lags': lags,
                'values': correlation
            }
        }

        self.logger.info("时域分析完成")
        return self.analysis_results['time_domain']

    def analyze_frequency_domain(self) -> Dict[str, Any]:
        """
        频域分析

        Returns:
            频域分析结果
        """
        if self.audio_data is None:
            raise ValueError("请先加载音频数据")

        self.logger.info("正在进行频域分析")

        # 绘制FFT频谱
        fft_fig = self.frequency_analysis.plot_fft_spectrum(
            self.audio_data,
            title="原始信号FFT频谱",
            save_path="results/figures/fft_spectrum_original.png"
        )

        # 绘制功率谱密度
        psd_fig = self.frequency_analysis.plot_power_spectral_density(
            self.audio_data,
            title="原始信号功率谱密度",
            save_path="results/figures/psd_original.png"
        )

        # 绘制频谱图
        spectrogram_fig = self.frequency_analysis.plot_spectrogram(
            self.audio_data,
            title="原始信号频谱图",
            save_path="results/figures/spectrogram_original.png"
        )

        # 计算信噪比（如果有噪声估计）
        snr = None
        if hasattr(self, 'noise_estimate') and self.noise_estimate is not None:
            snr = self.frequency_analysis.calculate_snr(self.audio_data, self.noise_estimate)

        self.analysis_results['frequency_domain'] = {
            'snr': snr
        }

        self.logger.info("频域分析完成")
        return self.analysis_results['frequency_domain']

    def apply_filter(self, filter_type: str = 'fir_lowpass', **filter_params) -> np.ndarray:
        """
        应用滤波器

        Args:
            filter_type: 滤波器类型
                - 'fir_lowpass': FIR低通滤波器
                - 'fir_highpass': FIR高通滤波器
                - 'fir_bandpass': FIR带通滤波器
                - 'fir_bandstop': FIR带阻滤波器
                - 'iir_butterworth': IIR巴特沃斯滤波器
                - 'iir_chebyshev_i': IIR切比雪夫I型
                - 'iir_chebyshev_ii': IIR切比雪夫II型
                - 'iir_elliptic': IIR椭圆滤波器
                - 'adaptive_lms': LMS自适应滤波器
                - 'adaptive_nlms': 归一化LMS滤波器
                - 'wiener': 维纳滤波器
            **filter_params: 滤波器参数

        Returns:
            滤波后的信号
        """
        if self.audio_data is None:
            raise ValueError("请先加载音频数据")

        self.logger.info(f"正在应用滤波器: {filter_type}")

        filtered_signal = None

        try:
            if filter_type.startswith('fir_'):
                # FIR滤波器
                if filter_type == 'fir_lowpass':
                    cutoff = filter_params.get('cutoff_freq', 1000)
                    numtaps = filter_params.get('numtaps', 101)
                    coeffs = self.filter_design.design_fir_lowpass(cutoff, numtaps)
                    filtered_signal = self.filter_design.apply_fir_filter(self.audio_data, coeffs)

                elif filter_type == 'fir_highpass':
                    cutoff = filter_params.get('cutoff_freq', 100)
                    numtaps = filter_params.get('numtaps', 101)
                    coeffs = self.filter_design.design_fir_highpass(cutoff, numtaps)
                    filtered_signal = self.filter_design.apply_fir_filter(self.audio_data, coeffs)

                elif filter_type == 'fir_bandpass':
                    lowcut = filter_params.get('lowcut_freq', 300)
                    highcut = filter_params.get('highcut_freq', 3000)
                    numtaps = filter_params.get('numtaps', 101)
                    coeffs = self.filter_design.design_fir_bandpass(lowcut, highcut, numtaps)
                    filtered_signal = self.filter_design.apply_fir_filter(self.audio_data, coeffs)

                elif filter_type == 'fir_bandstop':
                    lowcut = filter_params.get('lowcut_freq', 1000)
                    highcut = filter_params.get('highcut_freq', 2000)
                    numtaps = filter_params.get('numtaps', 101)
                    coeffs = self.filter_design.design_fir_bandstop(lowcut, highcut, numtaps)
                    filtered_signal = self.filter_design.apply_fir_filter(self.audio_data, coeffs)

            elif filter_type.startswith('iir_'):
                # IIR滤波器
                if filter_type == 'iir_butterworth':
                    cutoff = filter_params.get('cutoff_freq', 1000)
                    order = filter_params.get('order', 4)
                    ftype = filter_params.get('filter_type', 'lowpass')
                    b, a = self.filter_design.design_iir_butterworth(cutoff, ftype, order)
                    filtered_signal = self.filter_design.apply_iir_filter(self.audio_data, b, a)

                elif filter_type == 'iir_chebyshev_i':
                    cutoff = filter_params.get('cutoff_freq', 1000)
                    order = filter_params.get('order', 4)
                    ftype = filter_params.get('filter_type', 'lowpass')
                    ripple = filter_params.get('ripple', 1.0)
                    b, a = self.filter_design.design_iir_chebyshev_i(cutoff, ftype, order, ripple)
                    filtered_signal = self.filter_design.apply_iir_filter(self.audio_data, b, a)

                elif filter_type == 'iir_chebyshev_ii':
                    cutoff = filter_params.get('cutoff_freq', 1000)
                    order = filter_params.get('order', 4)
                    ftype = filter_params.get('filter_type', 'lowpass')
                    attenuation = filter_params.get('attenuation', 40.0)
                    b, a = self.filter_design.design_iir_chebyshev_ii(cutoff, ftype, order, attenuation)
                    filtered_signal = self.filter_design.apply_iir_filter(self.audio_data, b, a)

                elif filter_type == 'iir_elliptic':
                    cutoff = filter_params.get('cutoff_freq', 1000)
                    order = filter_params.get('order', 4)
                    ftype = filter_params.get('filter_type', 'lowpass')
                    ripple = filter_params.get('ripple', 1.0)
                    attenuation = filter_params.get('attenuation', 40.0)
                    b, a = self.filter_design.design_iir_elliptic(cutoff, ftype, order, ripple, attenuation)
                    filtered_signal = self.filter_design.apply_iir_filter(self.audio_data, b, a)

            elif filter_type.startswith('adaptive_'):
                # 自适应滤波器
                if hasattr(self, 'noise_estimate') and self.noise_estimate is not None:
                    if filter_type == 'adaptive_lms':
                        order = filter_params.get('filter_order', 32)
                        step = filter_params.get('step_size', 0.01)
                        filtered_signal = self.filter_design.adaptive_lms_filter(
                            self.noise_estimate, self.audio_data, order, step
                        )

                    elif filter_type == 'adaptive_nlms':
                        order = filter_params.get('filter_order', 32)
                        step = filter_params.get('step_size', 0.01)
                        epsilon = filter_params.get('epsilon', 1e-6)
                        filtered_signal = self.filter_design.adaptive_nlms_filter(
                            self.noise_estimate, self.audio_data, order, step, epsilon
                        )
                else:
                    self.logger.warning("自适应滤波器需要噪声估计，将使用FIR低通滤波器替代")
                    coeffs = self.filter_design.design_fir_lowpass(1000, 101)
                    filtered_signal = self.filter_design.apply_fir_filter(self.audio_data, coeffs)

            elif filter_type == 'wiener':
                # 维纳滤波器
                if hasattr(self, 'noise_estimate') and self.noise_estimate is not None:
                    filter_length = filter_params.get('filter_length', 64)
                    filtered_signal = self.filter_design.wiener_filter(
                        self.audio_data, self.noise_estimate, filter_length
                    )
                else:
                    self.logger.warning("维纳滤波器需要噪声估计，将使用FIR低通滤波器替代")
                    coeffs = self.filter_design.design_fir_lowpass(1000, 101)
                    filtered_signal = self.filter_design.apply_fir_filter(self.audio_data, coeffs)

            else:
                raise ValueError(f"不支持的滤波器类型: {filter_type}")

            if filtered_signal is not None:
                self.processed_data = filtered_signal
                self.logger.info(f"滤波器应用成功: {filter_type}")

        except Exception as e:
            self.logger.error(f"滤波器应用失败: {str(e)}")
            # 使用默认滤波器
            coeffs = self.filter_design.design_fir_lowpass(1000, 101)
            self.processed_data = self.filter_design.apply_fir_filter(self.audio_data, coeffs)

        return self.processed_data

    def enhance_signal(self, enhancement_type: str = 'normalize', **params) -> np.ndarray:
        """
        信号增强

        Args:
            enhancement_type: 增强类型
                - 'normalize': 归一化
                - 'dynamic_range': 动态范围压缩
            **params: 增强参数

        Returns:
            增强后的信号
        """
        if self.processed_data is None:
            self.processed_data = self.audio_data.copy()

        self.logger.info(f"正在进行信号增强: {enhancement_type}")

        if enhancement_type == 'normalize':
            target_max = params.get('target_max', 0.9)
            self.processed_data = normalize_signal(self.processed_data, target_max)

        elif enhancement_type == 'dynamic_range':
            # 简单的动态范围压缩
            threshold = params.get('threshold', 0.5)
            ratio = params.get('ratio', 2.0)

            # 应用压缩
            compressed = self.processed_data.copy()
            above_threshold = np.abs(compressed) > threshold
            compressed[above_threshold] = np.sign(compressed[above_threshold]) * (
                threshold + (np.abs(compressed[above_threshold]) - threshold) / ratio
            )
            self.processed_data = compressed

        self.logger.info("信号增强完成")
        return self.processed_data

    def compress_signal(self, compression_type: str = 'mu_law', **params) -> np.ndarray:
        """
        信号压缩

        Args:
            compression_type: 压缩类型
                - 'mu_law': μ律压缩
            **params: 压缩参数

        Returns:
            压缩后的信号
        """
        if self.processed_data is None:
            self.processed_data = self.audio_data.copy()

        self.logger.info(f"正在进行信号压缩: {compression_type}")

        if compression_type == 'mu_law':
            mu = params.get('mu', 255)
            # μ律压缩
            compressed = np.sign(self.processed_data) * np.log1p(mu * np.abs(self.processed_data)) / np.log1p(mu)
            self.processed_data = compressed

        self.logger.info("信号压缩完成")
        return self.processed_data

    def analyze_processed_signal(self) -> Dict[str, Any]:
        """
        分析处理后的信号

        Returns:
            处理后信号的分析结果
        """
        if self.processed_data is None:
            raise ValueError("请先进行信号处理")

        self.logger.info("正在分析处理后的信号")

        # 对比分析
        comparison_fig = self.frequency_analysis.compare_signals(
            self.original_data, self.processed_data,
            title="降噪前后信号对比",
            save_path="results/figures/comparison_analysis.png"
        )

        # 计算性能指标
        from utils import evaluate_noise_reduction
        metrics = evaluate_noise_reduction(
            self.original_data,
            self.audio_data if hasattr(self, 'noise_estimate') else self.original_data,
            self.processed_data
        )

        self.analysis_results['processed'] = {
            'metrics': metrics,
            'comparison_figure': comparison_fig
        }

        self.logger.info("处理后信号分析完成")
        return self.analysis_results['processed']

    def save_output(self, output_file: str = None, save_difference: bool = True) -> bool:
        """
        保存处理结果

        Args:
            output_file: 输出文件路径
            save_difference: 是否同时保存差异信号（被滤除的成分）

        Returns:
            保存是否成功
        """
        if self.processed_data is None:
            self.logger.warning("没有处理后的数据，将保存原始数据")
            data_to_save = self.audio_data
        else:
            data_to_save = self.processed_data

        if output_file is None:
            if self.input_file:
                filename = os.path.basename(self.input_file)
                name, ext = os.path.splitext(filename)
                output_file = f"data/output/{name}_denoised{ext}"

                # 如果要保存差异信号，计算并保存
                if save_difference and self.processed_data is not None:
                    # 差异信号 = 原始信号 - 处理后信号
                    difference_signal = self.audio_data - self.processed_data
                    difference_file = f"data/output/{name}_removed_noise{ext}"
                    save_audio(difference_signal, self.sample_rate, difference_file)
                    self.logger.info(f"差异信号（被去除的成分）已保存: {difference_file}")
            else:
                output_file = "data/output/processed_audio.wav"

        try:
            if output_file:
                ensure_dir(os.path.dirname(output_file))
                save_audio(data_to_save, self.sample_rate, output_file)
                self.logger.info(f"处理后音频已保存: {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"音频保存失败: {str(e)}")
            return False

    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        运行完整的分析流程

        Returns:
            完整的分析结果
        """
        self.logger.info("开始完整分析流程")

        # 1. 时域分析
        time_results = self.analyze_time_domain()

        # 2. 频域分析
        freq_results = self.analyze_frequency_domain()

        # 3. 应用滤波器（使用默认FIR低通）
        self.apply_filter('fir_lowpass', cutoff_freq=1000, numtaps=101)

        # 4. 信号增强
        self.enhance_signal('normalize', target_max=0.9)

        # 5. 分析处理结果
        processed_results = self.analyze_processed_signal()

        # 6. 保存结果
        self.save_output()

        self.logger.info("完整分析流程完成")

        return {
            'time_domain': time_results,
            'frequency_domain': freq_results,
            'processed': processed_results
        }


if __name__ == "__main__":
    # 测试音频处理器
    processor = AudioProcessor(sample_rate=44100)

    # 创建测试信号
    t = np.linspace(0, 1, 44100)
    test_signal = np.sin(2 * np.pi * 440 * t) + 0.3 * np.random.randn(len(t))

    # 保存测试信号
    ensure_dir("data/input")
    save_audio(test_signal, 44100, "data/input/test_signal.wav")

    # 加载并处理
    processor.load_audio("data/input/test_signal.wav")
    results = processor.run_complete_analysis()

    print("audio_processor.py 模块测试完成")