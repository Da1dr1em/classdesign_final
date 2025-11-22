"""
主程序 - 精简版
音频降噪处理系统入口
"""
import argparse
from audio_processor import AudioProcessor

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='音频降噪处理系统')
    parser.add_argument('--input', required=True, help='输入音频文件路径')
    parser.add_argument('--filter', default='fir_bandpass', 
                       choices=['fir_lowpass', 'fir_highpass', 'fir_bandpass', 
                               'iir_lowpass', 'lms', 'nlms', 'wiener'],
                       help='滤波器类型')
    parser.add_argument('--cutoff', type=float, default=300, help='截止频率(Hz)')
    parser.add_argument('--highcut', type=float, help='高截止频率(Hz)，带通滤波器使用')
    parser.add_argument('--enhance', action='store_true', help='是否进行信号增强')
    parser.add_argument('--no-plots', action='store_true', help='跳过图表生成')
    args = parser.parse_args()
    
    print("="*60)
    print("音频降噪数字信号处理系统")
    print("="*60)
    
    # 创建处理器
    processor = AudioProcessor(sample_rate=44100, 
                              enable_plots=not args.no_plots)
    
    # 步骤1: 加载音频
    print(f"\n[步骤1] 加载音频: {args.input}")
    processor.load_audio(args.input)
    
    # 步骤2: 时域分析
    print("\n[步骤2] 时域分析")
    processor.analyze_time_domain()
    
    # 步骤3: 频域分析
    print("\n[步骤3] 频域分析")
    processor.analyze_frequency_domain()
    
    # 步骤4: 滤波处理
    print(f"\n[步骤4] 应用{args.filter}滤波器")
    if args.filter == 'fir_bandpass' and args.highcut:
        processor.apply_filter(args.filter, 
                             lowcut_freq=args.cutoff,
                             highcut_freq=args.highcut)
    else:
        processor.apply_filter(args.filter, cutoff_freq=args.cutoff)
    
    # 步骤5: 信号增强（可选）
    if args.enhance:
        print("\n[步骤5] 信号增强")
        processor.enhance_signal('normalize', target_max=0.9)
    
    # 步骤6: 性能评估与保存
    print("\n[步骤6] 性能评估与保存结果")
    processor.analyze_processed_signal()
    processor.save_output(save_difference=True)
    
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)

if __name__ == "__main__":
    main()
