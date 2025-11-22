#!/usr/bin/env python
"""
对比不同的噪声估计方法
"""
import sys
import numpy as np
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import estimate_noise, load_audio
from analysis import FrequencyAnalysis

def compare_noise_estimation_methods(audio_file):
    """对比不同噪声估计方法的效果"""
    
    print("="*70)
    print("噪声估计方法对比测试")
    print("="*70)
    
    # 加载音频
    print(f"\n加载音频: {audio_file}")
    signal, sample_rate = load_audio(audio_file)
    print(f"  长度: {len(signal)} 样本 ({len(signal)/sample_rate:.2f} 秒)")
    print(f"  采样率: {sample_rate} Hz")
    
    freq_analysis = FrequencyAnalysis(sample_rate)
    
    # 测试的方法列表
    methods = [
        {
            'name': 'VAD (语音活动检测)',
            'method': 'vad',
            'params': {'energy_threshold_percentile': 20.0},
            'description': '检测静音段提取噪声'
        },
        {
            'name': 'Minimum Statistics (最小统计法)',
            'method': 'minimum_statistics',
            'params': {'window_size': 10},
            'description': '追踪局部最小能量'
        },
        {
            'name': 'Spectral Floor (频谱底噪法)',
            'method': 'spectral_floor',
            'params': {'percentile': 10.0},
            'description': '取每个频率的低百分位数'
        },
        {
            'name': 'Median Filter (中值滤波法)',
            'method': 'median_filter',
            'params': {},
            'description': '中值滤波平滑能量曲线'
        }
    ]
    
    results = []
    
    print("\n" + "="*70)
    print("开始测试各种方法...")
    print("="*70)
    
    for i, method_info in enumerate(methods, 1):
        print(f"\n[{i}/{len(methods)}] 测试: {method_info['name']}")
        print(f"    说明: {method_info['description']}")
        
        try:
            # 计时
            start_time = time.time()
            
            # 估计噪声
            noise_estimate = estimate_noise(
                signal, 
                sample_rate,
                method=method_info['method'],
                **method_info['params']
            )
            
            elapsed_time = time.time() - start_time
            
            # 计算SNR
            snr = freq_analysis.calculate_snr(signal, noise_estimate)
            
            # 计算噪声统计
            noise_power = np.mean(noise_estimate ** 2)
            noise_std = np.std(noise_estimate)
            
            result = {
                'name': method_info['name'],
                'method': method_info['method'],
                'snr': snr,
                'noise_power': noise_power,
                'noise_std': noise_std,
                'time': elapsed_time,
                'success': True
            }
            
            print(f"    ✅ 成功")
            print(f"       SNR: {snr:.2f} dB")
            print(f"       噪声功率: {noise_power:.6f}")
            print(f"       噪声标准差: {noise_std:.4f}")
            print(f"       耗时: {elapsed_time:.3f} 秒")
            
        except Exception as e:
            result = {
                'name': method_info['name'],
                'method': method_info['method'],
                'success': False,
                'error': str(e)
            }
            print(f"    ❌ 失败: {str(e)}")
        
        results.append(result)
    
    # 汇总结果
    print("\n" + "="*70)
    print("📊 结果汇总")
    print("="*70)
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if successful_results:
        print(f"\n{'方法':<30} {'SNR (dB)':<12} {'耗时 (秒)':<12} {'推荐度'}")
        print("-" * 70)
        
        for result in successful_results:
            snr_str = f"{result['snr']:.2f}"
            time_str = f"{result['time']:.3f}"
            
            # 根据SNR和速度给出推荐度
            if result['snr'] > 5 and result['time'] < 1:
                recommendation = "⭐⭐⭐⭐⭐"
            elif result['snr'] > 5:
                recommendation = "⭐⭐⭐⭐"
            elif result['time'] < 1:
                recommendation = "⭐⭐⭐"
            else:
                recommendation = "⭐⭐"
            
            print(f"{result['name']:<30} {snr_str:<12} {time_str:<12} {recommendation}")
    
    print("\n" + "="*70)
    print("💡 选择建议")
    print("="*70)
    print("""
1. VAD (语音活动检测法)
   优点: 速度最快
   缺点: 需要明显的静音段，提取帧数可能很少
   适用: 有停顿的对话、间歇性噪声
   
2. Minimum Statistics (最小统计法)
   优点: 适合连续信号，不需要静音段
   缺点: 计算量较大，可能过于保守
   适用: 持续的语音或音乐
   
3. Spectral Floor (频谱底噪法) ⭐推荐⭐
   优点: 准确度高，在频域分析每个频率成分
   缺点: 计算量中等
   适用: 大多数场景，特别是背景噪声相对平稳时
   
4. Median Filter (中值滤波法)
   优点: 折中方案，速度和准确度平衡
   缺点: 对突发噪声不敏感
   适用: 一般场景
    """)
    
    # 找出最佳方法
    if successful_results:
        best_snr = max(successful_results, key=lambda x: x['snr'])
        fastest = min(successful_results, key=lambda x: x['time'])
        
        print("\n🏆 最佳选择:")
        print(f"  最高SNR: {best_snr['name']} ({best_snr['snr']:.2f} dB)")
        print(f"  最快速度: {fastest['name']} ({fastest['time']:.3f} 秒)")
        
        # 综合推荐
        balanced = sorted(successful_results, 
                         key=lambda x: x['snr'] / (x['time'] + 0.1),
                         reverse=True)[0]
        print(f"  综合推荐: {balanced['name']} (SNR={balanced['snr']:.2f} dB, 耗时={balanced['time']:.3f}秒)")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    audio_file = r".\data\input\conversation_human.wav"
    compare_noise_estimation_methods(audio_file)
