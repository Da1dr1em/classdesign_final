"""
工具函数 - 精简版
音频I/O、噪声估计、信号处理等工具函数
"""
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft, get_window

def load_audio(file_path, target_sr=44100):
    """
    加载音频文件
    Args:
        file_path: 音频文件路径
        target_sr: 目标采样率
    Returns:
        (audio_data, sample_rate): 音频数据和采样率
    """
    import librosa
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr

def save_audio(file_path, audio_data, sample_rate):
    """
    保存音频文件
    Args:
        file_path: 输出文件路径
        audio_data: 音频数据
        sample_rate: 采样率
    """
    sf.write(file_path, audio_data, sample_rate)

def normalize_signal(signal, target_max=0.9):
    """
    归一化信号
    Args:
        signal: 输入信号
        target_max: 目标最大值(0-1之间)
    Returns:
        归一化后的信号
    """
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal * (target_max / max_val)
    return signal

def estimate_noise(signal, sample_rate=44100, method='spectral_floor', **kwargs):
    """
    估计信号中的噪声
    Args:
        signal: 输入信号
        sample_rate: 采样率
        method: 估计方法
            - 'spectral_floor': 频谱底噪法(推荐)
            - 'vad': 语音活动检测法
            - 'minimum_statistics': 最小统计法
            - 'median_filter': 中值滤波法
        **kwargs: 各方法的参数
    Returns:
        噪声估计信号
    """
    if method == 'spectral_floor':
        return estimate_noise_spectral_floor(signal, sample_rate, **kwargs)
    elif method == 'vad':
        return estimate_noise_vad(signal, sample_rate, **kwargs)
    elif method == 'minimum_statistics':
        return estimate_noise_minimum_statistics(signal, sample_rate, **kwargs)
    elif method == 'median_filter':
        return estimate_noise_median_filter(signal, sample_rate, **kwargs)
    else:
        raise ValueError(f"未知的噪声估计方法: {method}")

def estimate_noise_spectral_floor(signal, sample_rate=44100, 
                                  frame_length=2048, percentile=10.0):
    """
    频谱底噪法估计噪声
    原理: 对每个频率点，取时间轴上的低百分位数作为噪声水平
    
    Args:
        signal: 输入信号
        sample_rate: 采样率
        frame_length: 帧长
        percentile: 百分位数(0-100)
    Returns:
        噪声估计信号
    """
    # STFT变换
    window = get_window('hann', frame_length)
    f, t, Zxx = stft(signal, fs=sample_rate, window=window,
                     nperseg=frame_length, noverlap=frame_length//2)
    
    # 计算幅度谱
    magnitude = np.abs(Zxx)
    
    # 对每个频率，取时间轴上的低百分位数
    noise_magnitude = np.percentile(magnitude, percentile, axis=1, keepdims=True)
    
    # 扩展到所有时间帧
    noise_magnitude = np.tile(noise_magnitude, (1, magnitude.shape[1]))
    
    # 随机相位(噪声相位无规律)
    phase = np.random.uniform(0, 2*np.pi, noise_magnitude.shape)
    noise_stft = noise_magnitude * np.exp(1j * phase)
    
    # 逆STFT变换回时域
    _, noise_estimate = istft(noise_stft, fs=sample_rate, window=window,
                              nperseg=frame_length, noverlap=frame_length//2)
    
    # 调整长度
    if len(noise_estimate) > len(signal):
        noise_estimate = noise_estimate[:len(signal)]
    elif len(noise_estimate) < len(signal):
        pad_length = len(signal) - len(noise_estimate)
        noise_estimate = np.pad(noise_estimate, (0, pad_length), mode='wrap')
    
    return noise_estimate

def estimate_noise_vad(signal, sample_rate=44100, frame_length=2048,
                      hop_length=512, energy_threshold_percentile=20.0):
    """
    VAD(语音活动检测)法估计噪声
    原理: 检测低能量帧作为静音段，提取静音段特征作为噪声
    
    Args:
        signal: 输入信号
        sample_rate: 采样率
        frame_length: 帧长
        hop_length: 帧移
        energy_threshold_percentile: 能量阈值百分位数
    Returns:
        噪声估计信号
    """
    # 分帧
    frames = frame_signal(signal, frame_length, hop_length)
    
    # 计算每帧能量
    frame_energy = np.sum(frames ** 2, axis=1)
    
    # 使用百分位数确定阈值
    energy_threshold = np.percentile(frame_energy, energy_threshold_percentile)
    
    # 提取低能量帧(静音段)
    silent_frames = frames[frame_energy < energy_threshold]
    
    if len(silent_frames) == 0:
        # 如果没有检测到静音帧，返回低能量噪声
        return np.random.randn(len(signal)) * 0.001
    
    # 计算静音帧的均值作为噪声模板
    noise_template = np.mean(silent_frames, axis=0)
    
    # 重复噪声模板以匹配信号长度
    num_repeats = int(np.ceil(len(signal) / len(noise_template)))
    noise_estimate = np.tile(noise_template, num_repeats)[:len(signal)]
    
    return noise_estimate

def estimate_noise_minimum_statistics(signal, sample_rate=44100,
                                     frame_length=2048, hop_length=512,
                                     window_size=10):
    """
    最小统计法估计噪声
    原理: 在时频域追踪局部最小能量
    
    Args:
        signal: 输入信号
        sample_rate: 采样率
        frame_length: 帧长
        hop_length: 帧移
        window_size: 滑动窗口大小(帧数)
    Returns:
        噪声估计信号
    """
    # 分帧
    frames = frame_signal(signal, frame_length, hop_length)
    num_frames = len(frames)
    
    # 对每帧应用窗函数并计算频谱
    window = get_window('hann', frame_length)
    noise_spectrum = np.zeros((num_frames, frame_length // 2 + 1))
    
    for i in range(num_frames):
        windowed_frame = frames[i] * window
        spectrum = np.abs(np.fft.rfft(windowed_frame))
        
        # 在滑动窗口内寻找最小值
        start_idx = max(0, i - window_size // 2)
        end_idx = min(num_frames, i + window_size // 2 + 1)
        
        if start_idx == 0:
            noise_spectrum[i] = spectrum
        else:
            # 计算窗口内的最小频谱
            window_spectra = []
            for j in range(start_idx, min(end_idx, i + 1)):
                windowed = frames[j] * window
                window_spectra.append(np.abs(np.fft.rfft(windowed)))
            noise_spectrum[i] = np.min(window_spectra, axis=0)
    
    # 将频谱转换回时域
    noise_frames = np.zeros_like(frames)
    for i in range(num_frames):
        # 使用逆FFT重建信号(随机相位)
        phase = np.random.uniform(0, 2*np.pi, len(noise_spectrum[i]))
        complex_spectrum = noise_spectrum[i] * np.exp(1j * phase)
        noise_frames[i] = np.fft.irfft(complex_spectrum, n=frame_length)
    
    # 重叠相加重建信号
    noise_estimate = overlap_add(noise_frames, hop_length, len(signal))
    
    return noise_estimate

def estimate_noise_median_filter(signal, sample_rate=44100,
                                frame_length=2048, hop_length=512):
    """
    中值滤波法估计噪声
    原理: 对能量序列应用中值滤波，去除短时能量峰值
    
    Args:
        signal: 输入信号
        sample_rate: 采样率
        frame_length: 帧长
        hop_length: 帧移
    Returns:
        噪声估计信号
    """
    from scipy.ndimage import median_filter
    
    # 分帧
    frames = frame_signal(signal, frame_length, hop_length)
    
    # 计算每帧RMS能量
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    
    # 应用中值滤波平滑能量曲线
    smoothed_rms = median_filter(frame_rms, size=21)
    
    # 重建噪声信号
    noise_frames = np.zeros_like(frames)
    for i in range(len(frames)):
        # 根据平滑后的能量缩放原始帧
        if frame_rms[i] > 0:
            scale = smoothed_rms[i] / frame_rms[i]
            noise_frames[i] = frames[i] * scale
        else:
            noise_frames[i] = frames[i]
    
    # 重叠相加重建信号
    noise_estimate = overlap_add(noise_frames, hop_length, len(signal))
    
    return noise_estimate

def frame_signal(signal, frame_length, hop_length):
    """
    信号分帧
    Args:
        signal: 输入信号
        frame_length: 帧长度
        hop_length: 帧移长度
    Returns:
        分帧后的信号矩阵 (帧数 × 帧长度)
    """
    signal_length = len(signal)
    num_frames = 1 + (signal_length - frame_length) // hop_length
    
    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        if end <= signal_length:
            frames[i] = signal[start:end]
        else:
            frames[i, :signal_length-start] = signal[start:]
    
    return frames

def overlap_add(frames, hop_length, target_length):
    """
    重叠相加法重建信号
    Args:
        frames: 分帧矩阵 (帧数 × 帧长度)
        hop_length: 帧移
        target_length: 目标信号长度
    Returns:
        重建的信号
    """
    num_frames, frame_length = frames.shape
    signal = np.zeros(target_length)
    overlap_count = np.zeros(target_length)
    
    for i in range(num_frames):
        start = i * hop_length
        end = min(start + frame_length, target_length)
        signal[start:end] += frames[i][:end-start]
        overlap_count[start:end] += 1
    
    # 归一化(考虑重叠)
    overlap_count[overlap_count == 0] = 1
    signal = signal / overlap_count
    
    return signal
