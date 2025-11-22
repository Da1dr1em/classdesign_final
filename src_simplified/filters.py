"""
滤波器设计 - 精简版
FIR/IIR/自适应/维纳滤波器实现
"""
import numpy as np
from scipy.signal import firwin, iirfilter, lfilter, wiener

class FilterDesign:
    """滤波器设计类"""
    
    def __init__(self, sample_rate=44100):
        """
        初始化
        Args:
            sample_rate: 采样率(Hz)
        """
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2
    
    # ===== FIR滤波器设计 =====
    
    def design_fir_lowpass(self, cutoff_freq, numtaps=101, window='hamming'):
        """设计FIR低通滤波器"""
        normalized_cutoff = cutoff_freq / self.nyquist
        return firwin(numtaps, normalized_cutoff, window=window)
    
    def design_fir_highpass(self, cutoff_freq, numtaps=101, window='hamming'):
        """设计FIR高通滤波器"""
        normalized_cutoff = cutoff_freq / self.nyquist
        return firwin(numtaps, normalized_cutoff, window=window, pass_zero=False)
    
    def design_fir_bandpass(self, lowcut_freq, highcut_freq, numtaps=101, window='hamming'):
        """设计FIR带通滤波器"""
        normalized_lowcut = lowcut_freq / self.nyquist
        normalized_highcut = highcut_freq / self.nyquist
        return firwin(numtaps, [normalized_lowcut, normalized_highcut],
                     window=window, pass_zero=False)
    
    def apply_fir_filter(self, signal, b):
        """
        应用FIR滤波器
        Args:
            signal: 输入信号
            b: FIR滤波器系数
        Returns:
            滤波后的信号
        """
        return lfilter(b, 1, signal)
    
    # ===== IIR滤波器设计 =====
    
    def design_iir_lowpass(self, cutoff_freq, order=4):
        """设计IIR低通滤波器(巴特沃斯)"""
        normalized_cutoff = cutoff_freq / self.nyquist
        b, a = iirfilter(order, normalized_cutoff, btype='lowpass', ftype='butter')
        return b, a
    
    def design_iir_bandpass(self, lowcut_freq, highcut_freq, order=4):
        """设计IIR带通滤波器(巴特沃斯)"""
        normalized_cutoff = [lowcut_freq / self.nyquist, highcut_freq / self.nyquist]
        b, a = iirfilter(order, normalized_cutoff, btype='bandpass', ftype='butter')
        return b, a
    
    def apply_iir_filter(self, signal, b, a):
        """
        应用IIR滤波器
        Args:
            signal: 输入信号
            b, a: IIR滤波器系数
        Returns:
            滤波后的信号
        """
        return lfilter(b, a, signal)
    
    # ===== 自适应滤波器 =====
    
    def apply_lms_filter(self, signal, noise_reference, filter_length=32, mu=0.01):
        """
        LMS自适应滤波器
        Args:
            signal: 含噪声的信号
            noise_reference: 噪声参考信号
            filter_length: 滤波器长度
            mu: 步长参数
        Returns:
            滤波后的信号
        """
        N = len(signal)
        w = np.zeros(filter_length)  # 滤波器权重
        output = np.zeros(N)
        
        # 确保噪声参考信号长度匹配
        if len(noise_reference) < N:
            noise_reference = np.pad(noise_reference, (0, N - len(noise_reference)))
        
        for n in range(filter_length, N):
            # 提取输入向量
            x = noise_reference[n-filter_length:n][::-1]
            
            # 滤波器输出
            y = np.dot(w, x)
            
            # 误差信号
            e = signal[n] - y
            output[n] = e
            
            # 更新权重
            w = w + 2 * mu * e * x
        
        # 前面的样本直接复制
        output[:filter_length] = signal[:filter_length]
        
        return output
    
    def apply_nlms_filter(self, signal, noise_reference, filter_length=32, mu=0.1):
        """
        NLMS归一化LMS自适应滤波器
        Args:
            signal: 含噪声的信号
            noise_reference: 噪声参考信号
            filter_length: 滤波器长度
            mu: 步长参数
        Returns:
            滤波后的信号
        """
        N = len(signal)
        w = np.zeros(filter_length)
        output = np.zeros(N)
        epsilon = 1e-10  # 防止除零
        
        if len(noise_reference) < N:
            noise_reference = np.pad(noise_reference, (0, N - len(noise_reference)))
        
        for n in range(filter_length, N):
            x = noise_reference[n-filter_length:n][::-1]
            y = np.dot(w, x)
            e = signal[n] - y
            output[n] = e
            
            # 归一化步长
            norm_x = np.dot(x, x) + epsilon
            w = w + (mu / norm_x) * e * x
        
        output[:filter_length] = signal[:filter_length]
        
        return output
    
    # ===== 维纳滤波器 =====
    
    def apply_wiener_filter(self, signal, noise_estimate, mysize=None):
        """
        维纳滤波器
        Args:
            signal: 含噪声的信号
            noise_estimate: 噪声估计
            mysize: 滤波器窗口大小
        Returns:
            滤波后的信号
        """
        if mysize is None:
            mysize = min(5, len(signal) // 10)
        
        # 估计噪声方差
        noise_var = np.var(noise_estimate)
        
        # 应用维纳滤波
        filtered = wiener(signal, mysize=mysize, noise=noise_var)
        
        return filtered
