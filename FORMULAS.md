# 音频降噪系统关键公式

## 1. 信号基础

### 1.1 信号功率
$$P_{signal} = \frac{1}{N} \sum_{n=0}^{N-1} x^2[n]$$

其中：
- $x[n]$ 为离散信号
- $N$ 为信号长度

### 1.2 信号均方根(RMS)
$$RMS = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x^2[n]}$$

### 1.3 信号归一化
$$x_{norm}[n] = x[n] \cdot \frac{A_{target}}{max(|x[n]|)}$$

其中 $A_{target}$ 为目标最大幅度（通常取0.9）

## 2. 信噪比(SNR)计算

### 2.1 基本定义
$$SNR_{dB} = 10 \log_{10} \left( \frac{P_{signal}}{P_{noise}} \right)$$

### 2.2 含噪信号SNR
对于含噪信号 $y[n] = s[n] + d[n]$：
$$SNR_{dB} = 10 \log_{10} \left( \frac{\sum_{n=0}^{N-1} s^2[n]}{\sum_{n=0}^{N-1} d^2[n]} \right)$$

### 2.3 处理后SNR（重估计法）
$$SNR_{processed} = 10 \log_{10} \left( \frac{\sum_{n=0}^{N-1} \hat{y}^2[n]}{\sum_{n=0}^{N-1} \hat{d}^2[n]} \right)$$

其中：
- $\hat{y}[n]$ 为处理后的信号
- $\hat{d}[n]$ 为重新估计的噪声

## 3. 噪声估计

### 3.1 语音活动检测(VAD)法

帧能量：
$$E_i = \sum_{n=0}^{L-1} x^2[iH + n]$$

能量阈值（百分位数法）：
$$E_{th} = \text{percentile}(\{E_i\}, p)$$

其中 $p$ 为百分位数（如20%），$H$ 为帧移，$L$ 为帧长

### 3.2 频谱底噪法(Spectral Floor)

短时傅里叶变换(STFT)：
$$X(k, m) = \sum_{n=0}^{L-1} x[n + mH] \cdot w[n] \cdot e^{-j2\pi kn/L}$$

噪声幅度估计：
$$|\hat{D}(k)| = \text{percentile}_m(|X(k, m)|, p)$$

其中：
- $k$ 为频率索引
- $m$ 为时间帧索引
- $w[n]$ 为窗函数（Hann窗）
- $p$ 为百分位数（推荐10%）

### 3.3 最小统计法

滑动窗口内最小值：
$$\hat{D}(k, m) = \min_{i \in [m-W/2, m+W/2]} |X(k, i)|$$

其中 $W$ 为窗口大小（帧数）

## 4. 滤波器设计

### 4.1 FIR滤波器（窗函数法）

理想频率响应：
$$H_d(e^{j\omega}) = \begin{cases} 
1, & \omega \in [\omega_1, \omega_2] \text{ (带通)} \\
0, & \text{其他}
\end{cases}$$

脉冲响应（带通）：
$$h[n] = \frac{\sin(\omega_2(n-\alpha))}{\pi(n-\alpha)} - \frac{\sin(\omega_1(n-\alpha))}{\pi(n-\alpha)}$$

其中 $\alpha = \frac{N-1}{2}$ 为延迟

加窗：
$$h_w[n] = h[n] \cdot w[n]$$

Hann窗函数：
$$w[n] = 0.5 \left(1 - \cos\left(\frac{2\pi n}{N-1}\right)\right), \quad 0 \leq n \leq N-1$$

### 4.2 IIR滤波器（Butterworth）

传递函数：
$$H(s) = \frac{1}{\sqrt{1 + \left(\frac{s}{\omega_c}\right)^{2N}}}$$

双线性变换：
$$s = \frac{2}{T} \cdot \frac{1-z^{-1}}{1+z^{-1}}$$

其中：
- $N$ 为滤波器阶数
- $\omega_c$ 为截止频率
- $T$ 为采样周期

### 4.3 LMS自适应滤波器

权重更新公式：
$$\mathbf{w}(n+1) = \mathbf{w}(n) + \mu \cdot e(n) \cdot \mathbf{x}(n)$$

误差信号：
$$e(n) = d(n) - y(n) = d(n) - \mathbf{w}^T(n)\mathbf{x}(n)$$

其中：
- $\mu$ 为步长参数（收敛因子）
- $\mathbf{w}(n)$ 为权重向量
- $\mathbf{x}(n)$ 为输入向量
- $d(n)$ 为期望信号

### 4.4 NLMS自适应滤波器

归一化更新：
$$\mathbf{w}(n+1) = \mathbf{w}(n) + \frac{\mu}{\epsilon + ||\mathbf{x}(n)||^2} \cdot e(n) \cdot \mathbf{x}(n)$$

其中 $\epsilon$ 为小常数，防止除零

### 4.5 维纳滤波器

最优传递函数：
$$H_{opt}(k) = \frac{|S(k)|^2}{|S(k)|^2 + |D(k)|^2}$$

等价于：
$$H_{opt}(k) = \frac{SNR(k)}{1 + SNR(k)}$$

其中：
- $S(k)$ 为信号频谱
- $D(k)$ 为噪声频谱
- $SNR(k)$ 为频域信噪比

## 5. 频域分析

### 5.1 离散傅里叶变换(DFT)
$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}, \quad k = 0, 1, ..., N-1$$

### 5.2 快速傅里叶变换(FFT)
复杂度：$O(N \log N)$

### 5.3 功率谱密度(PSD)

Welch方法：
$$P_{xx}(f) = \frac{1}{K} \sum_{i=0}^{K-1} \left| \text{FFT}(x_i[n] \cdot w[n]) \right|^2$$

其中：
- $K$ 为分段数
- $x_i[n]$ 为第 $i$ 段信号
- $w[n]$ 为窗函数

### 5.4 频谱图(Spectrogram)
$$S(k, m) = \left| \sum_{n=0}^{L-1} x[n + mH] \cdot w[n] \cdot e^{-j2\pi kn/L} \right|^2$$

## 6. 性能评估

### 6.1 均方误差(MSE)
$$MSE = \frac{1}{N} \sum_{n=0}^{N-1} (x[n] - \hat{x}[n])^2$$

### 6.2 均方根误差(RMSE)
$$RMSE = \sqrt{MSE}$$

### 6.3 信噪比改善量
$$\Delta SNR = SNR_{processed} - SNR_{original}$$

### 6.4 峰值信噪比(PSNR)
$$PSNR_{dB} = 20 \log_{10} \left( \frac{\max(|x[n]|)}{RMSE} \right)$$

### 6.5 相关系数
$$\rho = \frac{\sum_{n=0}^{N-1}(x[n] - \bar{x})(\hat{x}[n] - \bar{\hat{x}})}{\sqrt{\sum_{n=0}^{N-1}(x[n] - \bar{x})^2} \sqrt{\sum_{n=0}^{N-1}(\hat{x}[n] - \bar{\hat{x}})^2}}$$

## 7. 实际应用参数

### 7.1 采样定理
$$f_s \geq 2f_{max}$$

本系统采用 $f_s = 44100$ Hz

### 7.2 电话语音带宽
$$f \in [300, 3400] \text{ Hz}$$

### 7.3 STFT参数选择

帧长：
$$L = 2048 \text{ samples} \approx 46.4 \text{ ms}$$

帧移：
$$H = 512 \text{ samples} \approx 11.6 \text{ ms}$$

重叠率：
$$\text{Overlap} = \frac{L - H}{L} = 75\%$$

### 7.4 频率分辨率
$$\Delta f = \frac{f_s}{L} = \frac{44100}{2048} \approx 21.5 \text{ Hz}$$

## 8. 实验结果示例

基于 `conversation_human.wav` 的处理结果：

- **原始信号SNR**: 21.08 dB
- **处理后SNR**: 22.62 dB
- **SNR改善**: +1.53 dB
- **信号放大倍数**: 10.06×
- **处理方法**: Spectral Floor + FIR带通滤波 + 归一化

噪声估计精度对比：
- VAD方法: 6.28 dB（误差 -14.80 dB）
- **Spectral Floor方法: 21.12 dB（误差 +0.04 dB）** ✓ 推荐
