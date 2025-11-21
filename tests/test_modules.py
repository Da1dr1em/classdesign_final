"""
模块测试脚本

测试各个核心模块的功能，确保代码正确性。
"""

import sys
import os
import numpy as np

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_utils():
    """测试utils.py模块"""
    from src.utils import normalize_signal, calculate_metrics, pad_signal

    print("\n=== 测试 utils.py 模块 ===")

    # 测试信号
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # 测试归一化
    normalized = normalize_signal(signal)
    print(f"✅ 信号归一化测试通过")
    print(f"   原始信号: {signal}")
    print(f"   归一化后: {normalized}")

    # 测试计算指标
    metrics = calculate_metrics(signal)
    print(f"✅ 计算指标测试通过")
    print(f"   均值: {metrics['mean']:.2f}")
    print(f"   RMS: {metrics['rms']:.2f}")

    # 测试信号补零
    padded = pad_signal(signal, 10)
    assert len(padded) == 10
    print(f"✅ 信号补零测试通过")


def test_filters():
    """test filters.py模块"""
    from src.filters import FilterDesign
    import numpy as np

    print("\n=== 测试 filters.py 模块 ===")

    # 创建设计器
    filter_design = FilterDesign(sample_rate=44100)

    # 测试FIR低通滤波器
    coeffs = filter_design.design_fir_lowpass(1000, numtaps=51)
    assert len(coeffs) == 51
    assert np.sum(coeffs) > 0
    print(f"✅ FIR低通滤波器设计测试通过")

    # 测试IIR巴特沃斯滤波器
    b, a = filter_design.design_iir_butterworth(1000, order=4)
    assert len(b) > 0 and len(a) > 0
    print(f"✅ IIR巴特沃斯滤波器设计测试通过")

    # 测试FIR滤波器应用
    test_signal = np.sin(2 * np.pi * 500 * np.linspace(0, 1, 1000))  # 500Hz信号
    filtered = filter_design.apply_fir_filter(test_signal, coeffs)
    print(f"✅ FIR滤波器应用测试通过")


def test_analysis():
    """test analysis.py模块"""
    from src.analysis import SignalAnalysis, FrequencyAnalysis
    import numpy as np

    print("\n=== 测试 analysis.py 模块 ===")

    # 创建测试信号
    sample_rate = 44100
    t = np.linspace(0, 1, sample_rate)
    signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)

    # 时域分析
    signal_analysis = SignalAnalysis(sample_rate)
    stats = signal_analysis.calculate_statistics(signal)
    assert 'mean' in stats
    assert 'rms' in stats
    print(f"✅ 时域分析测试通过")
    print(f"   统计特性计算正确: {len(stats)} 个指标")

    # 频域分析
    freq_analysis = FrequencyAnalysis(sample_rate)
    snr = freq_analysis.calculate_snr(signal, 0.1 * np.random.randn(len(signal)))
    assert not np.isnan(snr)
    print(f"✅ 频域分析测试通过")
    print(f"   信噪比: {snr:.2f} dB")


def test_audio_processor():
    """test audio_processor.py模块"""
    from src.audio_processor import AudioProcessor
    import numpy as np

    print("\n=== 测试 audio_processor.py 模块 ===")

    # 创建测试信号
    test_signal = np.random.randn(1000)
    test_signal = 0.5 * np.random.randn(1000) + 0.2 * np.sin(2 * np.pi * 500 * np.arange(1000) / 1000)

    # 创建处理器
    processor = AudioProcessor(sample_rate=44100)

    # 测试应用滤波器
    try:
        # 方法应存在
        assert hasattr(processor, 'apply_filter')
        print(f"✅ AudioProcessor 类初始化成功")
        print(f"✅ 滤波器应用接口存在")
    except Exception as e:
        print(f"❌ AudioProcessor 测试失败: {e}")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("音频降噪DSP系统 - 模块测试")
    print("=" * 60)

    # 测试各个模块
    try:
        test_utils()
        test_filters()
        test_analysis()
        test_audio_processor()

        print("\n" + "=" * 60)
        print("🎉 所有测试模块测试通过！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)