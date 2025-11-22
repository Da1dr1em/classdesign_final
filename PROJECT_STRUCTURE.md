# 项目目录结构

## 📁 工程目录结构说明

```
classdesign_final/
│
├── src/                                    # 源代码目录
│   ├── __init__.py                        # Python包初始化文件
│   ├── main.py                            # 主程序入口（命令行接口）
│   ├── audio_processor.py                 # 音频处理核心类
│   ├── filters.py                         # 滤波器实现（FIR/IIR/自适应/维纳）
│   ├── analysis.py                        # 信号分析模块（频域、时域、SNR）
│   ├── utils.py                           # 工具函数（噪声估计、归一化等）
│   └── __pycache__/                       # Python字节码缓存
│
├── data/                                   # 数据目录
│   ├── input/                             # 输入音频文件
│   │   ├── conversation_human.wav         # 人声对话录音（主要测试文件）
│   │   ├── human_record_with_noise.wav    # 带噪声的人声录音
│   │   ├── sine_500hz_clean.wav           # 500Hz纯正弦波（干净）
│   │   ├── sine_500hz_15db_noisy.wav      # 500Hz正弦波+15dB噪声
│   │   ├── noisy_500hz.wav                # 500Hz含噪声信号
│   │   ├── clean_500hz.wav                # 500Hz纯净信号
│   │   └── test_snr.wav                   # SNR测试音频
│   │
│   └── output/                            # 输出音频文件和图表
│       ├── conversation_human_denoised.wav              # 降噪后的音频
│       ├── conversation_human_removed_noise.wav         # 提取的噪声信号
│       ├── human_record_with_noise_denoised.wav         # 处理后音频
│       ├── sine_500hz_15db_noisy_denoised.wav           # 正弦波降噪结果
│       ├── sine_500hz_15db_noisy_removed_noise.wav      # 提取的噪声
│       ├── audio_enhancement_analysis.png               # 音频增强分析图表
│       ├── normalize_enhancement_explanation.png        # 归一化原理图
│       └── spectral_floor_explanation.png               # Spectral Floor原理图
│
└── results/                                # 结果和可视化目录
    └── figures/                           # 图表文件
        ├── comparison_analysis.png        # 对比分析图
        ├── envelope_original.png          # 包络分析图
        ├── fft_spectrum_original.png      # FFT频谱图
        ├── psd_original.png               # 功率谱密度图
        ├── spectrogram_original.png       # 时频谱图
        ├── time_domain_original.png       # 时域波形图
        ├── noise_estimation_explained.png # 噪声估计说明图
        │
        ├── bandpass/                      # 带通滤波器结果
        │   ├── comparison_analysis.png
        │   ├── conversation_human_denoised.wav
        │   ├── conversation_human_removed_noise.wav
        │   ├── envelope_original.png
        │   ├── fft_spectrum_original.png
        │   ├── psd_original.png
        │   ├── spectrogram_original.png
        │   └── time_domain_original.png
        │
        ├── bandpass__enhance/             # 带通+增强结果
        │   ├── comparison_analysis.png
        │   ├── conversation_human_denoised.wav
        │   ├── conversation_human_removed_noise.wav
        │   ├── envelope_original.png
        │   ├── fft_spectrum_original.png
        │   ├── psd_original.png
        │   ├── spectrogram_original.png
        │   └── time_domain_original.png
        │
        ├── lowpass/                       # 低通滤波器结果
        │   └── (类似结构)
        │
        └── test1_wav_output/              # 测试输出
            └── (测试结果文件)
```

## 📝 目录说明

### 1. **src/ - 源代码目录**
存放所有核心功能模块的源代码

| 文件 | 功能 | 主要类/函数 |
|------|------|------------|
| `main.py` | 命令行入口程序 | 参数解析、流程控制 |
| `audio_processor.py` | 音频处理核心 | `AudioProcessor` 类 |
| `filters.py` | 滤波器实现 | FIR, IIR, LMS, NLMS, Wiener |
| `analysis.py` | 信号分析 | `FrequencyAnalysis` 类, SNR计算 |
| `utils.py` | 工具函数 | 噪声估计, 归一化, 信号处理 |

**关键功能模块：**
- **噪声估计**: 4种方法（VAD, 最小统计, Spectral Floor, 中值滤波）
- **滤波器**: 7种类型（FIR低通/高通/带通, IIR, LMS, NLMS, Wiener）
- **信号分析**: 时域、频域、时频域分析
- **信号增强**: 归一化、动态范围压缩

### 2. **data/ - 数据目录**

#### 2.1 **data/input/ - 输入音频**
存放原始测试音频文件

| 文件 | 类型 | 用途 |
|------|------|------|
| `conversation_human.wav` | 人声对话 | 主要测试文件（18秒，44.1kHz） |
| `human_record_with_noise.wav` | 含噪人声 | 噪声环境测试 |
| `sine_500hz_clean.wav` | 纯正弦波 | 算法验证基准 |
| `sine_500hz_15db_noisy.wav` | 含噪正弦波 | SNR性能测试 |

#### 2.2 **data/output/ - 处理结果**
存放处理后的音频文件和分析图表

**音频文件命名规则：**
- `*_denoised.wav`: 降噪后的音频
- `*_removed_noise.wav`: 提取出的噪声信号
- `*.png`: 分析图表

**重要图表：**
- `audio_enhancement_analysis.png`: 完整增强流程可视化
- `spectral_floor_explanation.png`: Spectral Floor算法原理
- `normalize_enhancement_explanation.png`: 归一化原理说明

### 3. **results/ - 结果目录**

#### 3.1 **results/figures/ - 图表文件**
按滤波器类型组织的实验结果

**子目录结构：**
- `bandpass/`: 带通滤波器（300-3400 Hz）结果
- `bandpass__enhance/`: 带通+归一化增强结果
- `lowpass/`: 低通滤波器结果
- `test1_wav_output/`: 测试输出

**每个子目录包含：**
- ✓ 对比分析图 (`comparison_analysis.png`)
- ✓ 时域波形 (`time_domain_original.png`)
- ✓ FFT频谱 (`fft_spectrum_original.png`)
- ✓ 功率谱密度 (`psd_original.png`)
- ✓ 时频谱图 (`spectrogram_original.png`)
- ✓ 包络分析 (`envelope_original.png`)
- ✓ 处理后音频和噪声信号

## 🔧 使用方式

### 基本命令
```bash
# 1. 带通滤波 (300-3400 Hz)
python src/main.py --input data/input/conversation_human.wav --filter fir_bandpass --cutoff 300 --highcut 3400

# 2. 带通滤波 + 增强
python src/main.py --input data/input/conversation_human.wav --filter fir_bandpass --cutoff 300 --highcut 3400 --enhance

# 3. 跳过图表生成（快速模式）
python src/main.py --input data/input/conversation_human.wav --filter fir_bandpass --cutoff 300 --highcut 3400 --enhance --no-plots
```

### 输出位置
- **音频文件**: `data/output/`
- **图表文件**: `results/figures/<filter_type>/`

## 📊 文件统计

```
总计文件数：
├── 源代码：5个Python模块
├── 输入音频：7个WAV文件
├── 输出音频：6个WAV文件
├── 分析图表：20+ PNG文件
└── 总大小：约50+ MB
```

## 🎯 核心工作流程

```
输入音频 (data/input/)
    ↓
音频处理 (src/)
    ├─ 加载音频
    ├─ 噪声估计 (Spectral Floor)
    ├─ 滤波处理 (300-3400 Hz)
    └─ 信号增强 (归一化)
    ↓
生成结果
    ├─ 处理后音频 → data/output/
    └─ 分析图表 → results/figures/
```

## 📈 关键性能指标

基于 `conversation_human.wav` 的测试结果：
- **原始SNR**: 21.08 dB
- **处理后SNR**: 22.62 dB
- **SNR改善**: +1.53 dB
- **音量放大**: 10.06倍
- **噪声功率降低**: 显著

---

**生成时间**: 2025-11-22  
**项目**: DSP音频降噪系统课程设计
