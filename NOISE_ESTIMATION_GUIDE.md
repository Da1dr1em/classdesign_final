# 噪声估计和信噪比输出功能说明

## 功能概述

已成功添加噪声估计功能并在滤波器性能分析时输出信噪比。

## 新增功能

### 1. 噪声估计功能 (`src/utils.py`)

新增 `estimate_noise_vad()` 函数，使用**语音活动检测(VAD)**方法估计背景噪声：

**工作原理：**
- 将信号分帧（默认2048样本/帧，512样本步长）
- 计算每帧的能量
- 识别低能量帧（默认为最低20%能量的帧）作为"静音段"
- 从静音段中提取噪声特性
- 如果检测不到静音段，使用能量最低的10%帧

**参数：**
```python
estimate_noise_vad(
    signal,                          # 输入信号
    sample_rate=44100,               # 采样率
    frame_length=2048,               # 帧长度
    hop_length=512,                  # 帧移
    energy_threshold_percentile=20.0 # 能量阈值百分位
)
```

### 2. 自动噪声估计 (`src/audio_processor.py`)

**AudioProcessor类改进：**
- 添加 `noise_estimate` 属性用于存储噪声估计结果
- 添加 `_estimate_noise()` 方法执行噪声估计
- 在 `load_audio()` 后自动调用噪声估计
- 在日志中输出估计的信噪比

**示例输出：**
```
2025-11-21 19:32:47,000 - audio_processor - INFO - 正在估计噪声...
2025-11-21 19:32:47,002 - audio_processor - INFO - 噪声估计完成，估计信噪比: 0.12 dB
```

### 3. 信噪比输出 (`src/audio_processor.py`, `src/main.py`)

**在性能分析中输出详细的SNR信息：**

#### 日志输出：
```
==================================================
信噪比分析结果:
  原始信号SNR: 0.12 dB
  处理后SNR: 7.37 dB
  SNR改善: 7.26 dB
==================================================
```

#### 控制台输出（main.py）：
```
==================================================
性能指标分析:
==================================================

📊 基于噪声估计的信噪比:
  - 原始信号SNR: 0.12 dB
  - 处理后SNR: 7.37 dB
  - SNR改善: 7.26 dB

📈 降噪质量评估:
  - 相关系数: 0.909
  - RMSE: 0.1934
==================================================
```

## 计算方法

### 信噪比计算公式

$$\text{SNR} = 10 \log_{10}\left(\frac{P_{\text{signal}}}{P_{\text{noise}}}\right)$$

其中：
- $P_{\text{signal}} = \frac{1}{N}\sum_{i=1}^{N} x_i^2$ （信号功率）
- $P_{\text{noise}} = \frac{1}{N}\sum_{i=1}^{N} n_i^2$ （噪声功率）

### 三种SNR指标

1. **原始信号SNR**：使用估计的噪声计算原始信号的信噪比
2. **处理后SNR**：滤波后信号的信噪比（噪声定义为：处理后信号 - 原始信号）
3. **SNR改善**：处理后SNR - 原始SNR（正值表示改善，负值表示退化）

## 使用方法

### 命令行使用

```bash
# 基本用法（自动进行噪声估计）
python src/main.py --input data/input/audio.wav --filter fir_lowpass --cutoff 1000 --enhance

# 使用带通滤波器
python src/main.py --input data/input/conversation.wav --filter fir_bandpass --cutoff 300 --highcut 3400 --enhance
```

### Python代码使用

```python
from audio_processor import AudioProcessor

# 创建处理器
processor = AudioProcessor(sample_rate=44100)

# 加载音频（自动进行噪声估计）
processor.load_audio("data/input/audio.wav")

# 查看噪声估计结果
if processor.noise_estimate is not None:
    print("噪声估计成功")
    # 计算SNR
    snr = processor.frequency_analysis.calculate_snr(
        processor.audio_data, 
        processor.noise_estimate
    )
    print(f"估计SNR: {snr:.2f} dB")

# 应用滤波器
processor.apply_filter('fir_lowpass', cutoff_freq=1500, numtaps=101)

# 分析处理后信号（自动输出SNR）
results = processor.analyze_processed_signal()

# 获取SNR指标
metrics = results['metrics']
print(f"原始SNR: {metrics['original_snr_estimated']:.2f} dB")
print(f"处理后SNR: {metrics['processed_snr_estimated']:.2f} dB")
print(f"SNR改善: {metrics['snr_improvement_estimated']:.2f} dB")
```

## 测试验证

已创建两个测试脚本验证功能：

### 1. 噪声估计测试
```bash
python test_noise_estimation.py
```
- 测试VAD噪声估计算法
- 验证AudioProcessor集成
- 对比真实SNR与估计SNR

### 2. 完整流程测试
```bash
python test_snr_output.py
```
- 测试完整的处理流程
- 验证SNR计算和输出
- 显示性能改善

## 技术说明

### 噪声估计的局限性

1. **适用场景**：
   - ✅ 有明显静音段的语音信号
   - ✅ 音乐信号（有停顿）
   - ✅ 间歇性噪声

2. **不适用场景**：
   - ❌ 持续无静音的信号
   - ❌ 非平稳噪声
   - ❌ 信噪比极低的信号

### 改进方向

如需更高级的噪声估计，可考虑：
- 最小统计法（Minimum Statistics）
- 基于谱减的噪声估计
- 自适应噪声跟踪
- 深度学习方法

## 依赖的模块

- `src/utils.py`：噪声估计函数
- `src/audio_processor.py`：集成噪声估计和SNR计算
- `src/analysis.py`：SNR计算方法
- `src/main.py`：用户界面输出

## 更新日期

2025年11月21日
