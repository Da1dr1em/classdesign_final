"""
数字滤波器设计模块

实现各种数字滤波器的设计和应用，包括FIR滤波器、IIR滤波器、自适应滤波器和维纳滤波器。
这是DSP课程设计的核心模块之一。
"""

import numpy as np
from scipy import signal
from scipy.signal import firwin, iirfilter, lfilter, filtfilt
from typing import Tuple, Optional, List, Callable


class FilterDesign:
    """
    数字滤波器设计类

    提供FIR、IIR、自适应和维纳滤波器的设计和应用功能。
    """

    def __init__(self, sample_rate: int = 44100):
        """
        初始化滤波器设计器

        Args:
            sample_rate: 采样率，默认为44.1kHz
        """
        self.sample_rate = sample_rate

    # =========================================================================
    # FIR滤波器设计
    # =========================================================================

    def design_fir_lowpass(self, cutoff_freq: float, numtaps: int = 101,
                          window: str = 'hamming') -> np.ndarray:
        """
        设计FIR低通滤波器

        Args:
            cutoff_freq: 截止频率(Hz)
            numtaps: 滤波器阶数，必须是奇数
            window: 窗函数类型

        Returns:
            滤波器系数
        """
        # 归一化截止频率
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist

        # 设计FIR滤波器
        coefficients = firwin(numtaps, normalized_cutoff, window=window)
        return coefficients

    def design_fir_highpass(self, cutoff_freq: float, numtaps: int = 101,
                           window: str = 'hamming') -> np.ndarray:
        """
        设计FIR高通滤波器

        Args:
            cutoff_freq: 截止频率(Hz)
            numtaps: 滤波器阶数，必须是奇数
            window: 窗函数类型

        Returns:
            滤波器系数
        """
        # 归一化截止频率
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist

        # 设计FIR高通滤波器
        coefficients = firwin(numtaps, normalized_cutoff, window=window, pass_zero=False)
        return coefficients

    def design_fir_bandpass(self, lowcut_freq: float, highcut_freq: float,
                           numtaps: int = 101, window: str = 'hamming') -> np.ndarray:
        """
        设计FIR带通滤波器

        Args:
            lowcut_freq: 低频截止频率(Hz)
            highcut_freq: 高频截止频率(Hz)
            numtaps: 滤波器阶数，必须是奇数
            window: 窗函数类型

        Returns:
            滤波器系数
        """
        # 归一化截止频率
        nyquist = self.sample_rate / 2
        normalized_lowcut = lowcut_freq / nyquist
        normalized_highcut = highcut_freq / nyquist

        # 设计FIR带通滤波器
        coefficients = firwin(numtaps, [normalized_lowcut, normalized_highcut],
                            window=window, pass_zero=False)
        return coefficients

    def design_fir_bandstop(self, lowcut_freq: float, highcut_freq: float,
                           numtaps: int = 101, window: str = 'hamming') -> np.ndarray:
        """
        设计FIR带阻滤波器

        Args:
            lowcut_freq: 低频截止频率(Hz)
            highcut_freq: 高频截止频率(Hz)
            numtaps: 滤波器阶数，必须是奇数
            window: 窗函数类型

        Returns:
            滤波器系数
        """
        # 归一化截止频率
        nyquist = self.sample_rate / 2
        normalized_lowcut = lowcut_freq / nyquist
        normalized_highcut = highcut_freq / nyquist

        # 设计FIR带阻滤波器
        coefficients = firwin(numtaps, [normalized_lowcut, normalized_highcut],
                            window=window)
        return coefficients

    # =========================================================================
    # IIR滤波器设计
    # =========================================================================

    def design_iir_butterworth(self, cutoff_freq: float, filter_type: str = 'lowpass',
                              order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        设计巴特沃斯IIR滤波器

        Args:
            cutoff_freq: 截止频率(Hz)，对于带通/带阻滤波器是列表 [lowcut, highcut]
            filter_type: 滤波器类型 'lowpass', 'highpass', 'bandpass', 'bandstop'
            order: 滤波器阶数

        Returns:
            (b, a): 滤波器分子和分母系数
        """
        # 归一化截止频率
        nyquist = self.sample_rate / 2

        if isinstance(cutoff_freq, (list, tuple)):
            normalized_cutoff = [f / nyquist for f in cutoff_freq]
        else:
            normalized_cutoff = cutoff_freq / nyquist

        # 设计巴特沃斯滤波器
        b, a = iirfilter(order, normalized_cutoff, btype=filter_type, ftype='butter')
        return b, a

    def design_iir_chebyshev_i(self, cutoff_freq: float, filter_type: str = 'lowpass',
                              order: int = 4, ripple: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        设计切比雪夫I型IIR滤波器

        Args:
            cutoff_freq: 截止频率(Hz)
            filter_type: 滤波器类型
            order: 滤波器阶数
            ripple: 通带纹波(dB)

        Returns:
            (b, a): 滤波器分子和分母系数
        """
        # 归一化截止频率
        nyquist = self.sample_rate / 2

        if isinstance(cutoff_freq, (list, tuple)):
            normalized_cutoff = [f / nyquist for f in cutoff_freq]
        else:
            normalized_cutoff = cutoff_freq / nyquist

        # 设计切比雪夫I型滤波器
        b, a = iirfilter(order, normalized_cutoff, btype=filter_type,
                        ftype='cheby1', rp=ripple)
        return b, a

    def design_iir_chebyshev_ii(self, cutoff_freq: float, filter_type: str = 'lowpass',
                               order: int = 4, attenuation: float = 40.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        设计切比雪夫II型IIR滤波器

        Args:
            cutoff_freq: 截止频率(Hz)
            filter_type: 滤波器类型
            order: 滤波器阶数
            attenuation: 阻带衰减(dB)

        Returns:
            (b, a): 滤波器分子和分母系数
        """
        # 归一化截止频率
        nyquist = self.sample_rate / 2

        if isinstance(cutoff_freq, (list, tuple)):
            normalized_cutoff = [f / nyquist for f in cutoff_freq]
        else:
            normalized_cutoff = cutoff_freq / nyquist

        # 设计切比雪夫II型滤波器
        b, a = iirfilter(order, normalized_cutoff, btype=filter_type,
                        ftype='cheby2', rs=attenuation)
        return b, a

    def design_iir_elliptic(self, cutoff_freq: float, filter_type: str = 'lowpass',
                           order: int = 4, ripple: float = 1.0,
                           attenuation: float = 40.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        设计椭圆IIR滤波器

        Args:
            cutoff_freq: 截止频率(Hz)
            filter_type: 滤波器类型
            order: 滤波器阶数
            ripple: 通带纹波(dB)
            attenuation: 阻带衰减(dB)

        Returns:
            (b, a): 滤波器分子和分母系数
        """
        # 归一化截止频率
        nyquist = self.sample_rate / 2

        if isinstance(cutoff_freq, (list, tuple)):
            normalized_cutoff = [f / nyquist for f in cutoff_freq]
        else:
            normalized_cutoff = cutoff_freq / nyquist

        # 设计椭圆滤波器
        b, a = iirfilter(order, normalized_cutoff, btype=filter_type,
                        ftype='ellip', rp=ripple, rs=attenuation)
        return b, a

    # =========================================================================
    # 自适应滤波器
    # =========================================================================

    def adaptive_lms_filter(self, input_signal: np.ndarray, desired_signal: np.ndarray,
                           filter_order: int = 32, step_size: float = 0.01) -> np.ndarray:
        """
        LMS自适应滤波器

        Args:
            input_signal: 输入信号（参考噪声）
            desired_signal: 期望信号（含噪信号）
            filter_order: 滤波器阶数
            step_size: 步长参数

        Returns:
            滤波后的信号
        """
        n = len(input_signal)
        w = np.zeros(filter_order)  # 滤波器系数
        output_signal = np.zeros(n)

        for i in range(filter_order, n):
            # 获取当前输入向量
            x = input_signal[i-filter_order:i][::-1]

            # 计算滤波器输出
            y = np.dot(w, x)

            # 计算误差
            error = desired_signal[i] - y

            # 更新滤波器系数
            w = w + step_size * error * x

            # 保存输出
            output_signal[i] = y

        return output_signal

    def adaptive_nlms_filter(self, input_signal: np.ndarray, desired_signal: np.ndarray,
                            filter_order: int = 32, step_size: float = 0.01,
                            epsilon: float = 1e-6) -> np.ndarray:
        """
        归一化LMS自适应滤波器

        Args:
            input_signal: 输入信号
            desired_signal: 期望信号
            filter_order: 滤波器阶数
            step_size: 步长参数
            epsilon: 正则化参数

        Returns:
            滤波后的信号
        """
        n = len(input_signal)
        w = np.zeros(filter_order)  # 滤波器系数
        output_signal = np.zeros(n)

        for i in range(filter_order, n):
            # 获取当前输入向量
            x = input_signal[i-filter_order:i][::-1]

            # 计算滤波器输出
            y = np.dot(w, x)

            # 计算误差
            error = desired_signal[i] - y

            # 归一化步长
            normalized_step = step_size / (np.dot(x, x) + epsilon)

            # 更新滤波器系数
            w = w + normalized_step * error * x

            # 保存输出
            output_signal[i] = y

        return output_signal

    # =========================================================================
    # 维纳滤波器
    # =========================================================================

    def wiener_filter(self, noisy_signal: np.ndarray, noise_estimate: np.ndarray,
                     filter_length: int = 64) -> np.ndarray:
        """
        维纳滤波器实现

        Args:
            noisy_signal: 含噪信号
            noise_estimate: 噪声估计
            filter_length: 滤波器长度

        Returns:
            滤波后的信号
        """
        n = len(noisy_signal)

        # 计算信号和噪声的自相关
        signal_autocorr = self._autocorrelation(noisy_signal, filter_length)
        noise_autocorr = self._autocorrelation(noise_estimate, filter_length)

        # 计算信号和噪声的互相关
        cross_corr = self._crosscorrelation(noisy_signal, noise_estimate, filter_length)

        # 构建自相关矩阵
        R = self._toeplitz_matrix(signal_autocorr)

        # 构建互相关向量
        p = cross_corr

        # 求解维纳-霍普夫方程
        try:
            w = np.linalg.solve(R, p)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            w = np.linalg.pinv(R) @ p

        # 应用滤波器
        filtered_signal = np.convolve(noisy_signal, w, mode='same')

        return filtered_signal

    def _autocorrelation(self, x: np.ndarray, max_lag: int) -> np.ndarray:
        """计算自相关函数"""
        n = len(x)
        autocorr = np.zeros(max_lag)

        for lag in range(max_lag):
            if lag < n:
                autocorr[lag] = np.sum(x[:n-lag] * x[lag:]) / (n - lag)

        return autocorr

    def _crosscorrelation(self, x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
        """计算互相关函数"""
        n = len(x)
        crosscorr = np.zeros(max_lag)

        for lag in range(max_lag):
            if lag < n:
                crosscorr[lag] = np.sum(x[:n-lag] * y[lag:]) / (n - lag)

        return crosscorr

    def _toeplitz_matrix(self, first_column: np.ndarray) -> np.ndarray:
        """构建托普利兹矩阵"""
        n = len(first_column)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i >= j:
                    matrix[i, j] = first_column[i - j]
                else:
                    matrix[i, j] = first_column[j - i]

        return matrix

    # =========================================================================
    # 通用滤波器应用方法
    # =========================================================================

    def apply_fir_filter(self, signal: np.ndarray, coefficients: np.ndarray,
                        zero_phase: bool = True) -> np.ndarray:
        """
        应用FIR滤波器

        Args:
            signal: 输入信号
            coefficients: FIR滤波器系数
            zero_phase: 是否使用零相位滤波

        Returns:
            滤波后的信号
        """
        if zero_phase:
            # 零相位滤波（双向滤波）
            filtered_signal = filtfilt(coefficients, [1.0], signal)
        else:
            # 常规滤波
            filtered_signal = lfilter(coefficients, [1.0], signal)

        return filtered_signal

    def apply_iir_filter(self, signal: np.ndarray, b: np.ndarray, a: np.ndarray,
                        zero_phase: bool = True) -> np.ndarray:
        """
        应用IIR滤波器

        Args:
            signal: 输入信号
            b: 分子系数
            a: 分母系数
            zero_phase: 是否使用零相位滤波

        Returns:
            滤波后的信号
        """
        if zero_phase:
            # 零相位滤波（双向滤波）
            filtered_signal = filtfilt(b, a, signal)
        else:
            # 常规滤波
            filtered_signal = lfilter(b, a, signal)

        return filtered_signal

    def get_frequency_response(self, coefficients: np.ndarray, worN: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取滤波器频率响应

        Args:
            coefficients: 滤波器系数
            worN: 频率点数

        Returns:
            (frequencies, magnitude_response): 频率数组和幅度响应
        """
        w, h = signal.freqz(coefficients, worN=worN)
        frequencies = w * self.sample_rate / (2 * np.pi)
        magnitude_response = 20 * np.log10(np.abs(h) + 1e-10)

        return frequencies, magnitude_response


if __name__ == "__main__":
    # 测试滤波器设计
    filter_design = FilterDesign(sample_rate=44100)

    # 测试FIR低通滤波器
    fir_coeffs = filter_design.design_fir_lowpass(1000, numtaps=51)
    print(f"FIR低通滤波器系数长度: {len(fir_coeffs)}")

    # 测试IIR巴特沃斯滤波器
    b, a = filter_design.design_iir_butterworth(1000, order=4)
    print(f"IIR巴特沃斯滤波器: b={len(b)}, a={len(a)}")

    print("filters.py 模块测试完成")