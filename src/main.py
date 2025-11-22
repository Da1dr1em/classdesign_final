"""
主程序控制器

项目的入口程序，协调各个模块的工作流程，
实现完整的6个步骤的数字信号处理流程。
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Optional, List

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from audio_processor import AudioProcessor
from utils import list_audio_files, ensure_dir


def main():
    """主函数，控制整个处理流程"""
    parser = create_argument_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("音频降噪数字信号处理系统")
    print("=" * 60)

    # 确保输出目录存在
    ensure_dir("data/output")
    ensure_dir("results/figures")

    if args.input:
        # 处理单个文件
        process_single_file(args.input, args)
    elif args.batch:
        # 批量处理
        process_batch_files(args.batch, args)
    else:
        # 交互式模式
        interactive_mode(args)


def create_argument_parser() -> argparse.ArgumentParser:
    """
    创建命令行参数解析器

    Returns:
        参数解析器
    """
    parser = argparse.ArgumentParser(
        description="音频降噪数字信号处理系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python src/main.py --input data/input/noisy_audio.wav
  python src/main.py --batch data/input/
  python src/main.py --interactive
        """
    )

    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--input', type=str, help='输入音频文件路径')
    input_group.add_argument('--batch', type=str, help='批量处理目录路径')
    input_group.add_argument('--interactive', action='store_true', help='交互式模式')

    # 滤波器选项
    parser.add_argument('--filter', type=str, default='fir_lowpass',
                       choices=['fir_lowpass', 'fir_highpass', 'fir_bandpass', 'fir_bandstop',
                               'iir_butterworth', 'iir_chebyshev_i', 'iir_chebyshev_ii', 'iir_elliptic',
                               'adaptive_lms', 'adaptive_nlms', 'wiener'],
                       help='滤波器类型')

    # 滤波器参数
    parser.add_argument('--cutoff', type=float, default=1000.0,
                       help='截止频率(Hz)，对于带通/带阻滤波器使用低频截止')
    parser.add_argument('--highcut', type=float, default=3000.0,
                       help='高频截止频率(Hz)，用于带通/带阻滤波器')
    parser.add_argument('--order', type=int, default=4,
                       help='滤波器阶数(IIR滤波器)')
    parser.add_argument('--numtaps', type=int, default=101,
                       help='FIR滤波器阶数')

    # 处理选项
    parser.add_argument('--enhance', action='store_true',
                       help='启用信号增强')
    parser.add_argument('--compress', action='store_true',
                       help='启用信号压缩')
    parser.add_argument('--sample-rate', type=int, default=44100,
                       help='目标采样率')

    # 输出选项
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存处理结果')
    parser.add_argument('--no-plots', action='store_true',
                       help='不生成图表（加快处理速度）')

    return parser


def process_single_file(input_file: str, args) -> bool:
    """
    处理单个音频文件

    Args:
        input_file: 输入文件路径
        args: 命令行参数

    Returns:
        处理是否成功
    """
    print(f"\n正在处理文件: {input_file}")
    print("-" * 40)

    try:
        # 创建音频处理器
        enable_plots = not args.no_plots
        processor = AudioProcessor(sample_rate=args.sample_rate, enable_plots=enable_plots)

        # 1. 加载音频文件
        print("步骤 1/6: 加载音频文件...")
        if not processor.load_audio(input_file):
            print("❌ 音频文件加载失败")
            return False
        print("✅ 音频文件加载成功")

        # 2. 时域分析
        print("步骤 2/6: 时域分析...")
        time_results = processor.analyze_time_domain()
        print("✅ 时域分析完成")

        # 3. 频域分析
        print("步骤 3/6: 频域分析...")
        freq_results = processor.analyze_frequency_domain()
        print("✅ 频域分析完成")

        # 4. 设计并应用滤波器
        print("步骤 4/6: 应用滤波器...")
        filter_params = {
            'cutoff_freq': args.cutoff,
            'numtaps': args.numtaps,
            'order': args.order,
            'lowcut_freq': args.cutoff,
            'highcut_freq': args.highcut
        }
        processor.apply_filter(args.filter, **filter_params)
        print(f"✅ {args.filter} 滤波器应用完成")

        # 5. 信号增强和压缩
        print("步骤 5/6: 信号增强和压缩...")
        if args.enhance:
            processor.enhance_signal('normalize', target_max=0.9)
            print("✅ 信号增强完成")

        if args.compress:
            processor.compress_signal('mu_law', mu=255)
            print("✅ 信号压缩完成")

        # 6. 分析处理后信号
        print("步骤 6/6: 分析处理后信号...")
        processed_results = processor.analyze_processed_signal()
        print("✅ 处理后信号分析完成")

        # 显示性能指标
        if 'metrics' in processed_results:
            metrics = processed_results['metrics']
            print("\n" + "="*50)
            print("性能指标分析:")
            print("="*50)
            
            # 显示基于噪声估计的信噪比
            if 'original_snr_estimated' in metrics:
                print("\n📊 基于噪声估计法的信噪比 (Spectral Floor):")
                print(f"  - 原始信号SNR: {metrics['original_snr_estimated']:.2f} dB")
                print(f"  - 处理后SNR: {metrics['processed_snr_estimated']:.2f} dB")
                print(f"  - SNR改善: {metrics['snr_improvement_estimated']:+.2f} dB")
                
                if 'residual_snr' in metrics:
                    print(f"\n📉 基于残差法的信噪比 (仅供参考):")
                    print(f"  - 处理后SNR: {metrics['residual_snr']:.2f} dB")
                    print(f"  - 说明: 残差法假设处理前后差异即为噪声")
            
            # 显示其他性能指标
            print("\n📈 降噪质量评估:")
            if 'correlation' in metrics:
                print(f"  - 相关系数: {metrics['correlation']:.3f}")
            if 'rmse' in metrics:
                print(f"  - RMSE: {metrics['rmse']:.4f}")
            if 'original_snr_db' in metrics:
                print(f"  - 原始信噪比(评估): {metrics['original_snr_db']:.2f} dB")
            if 'denoised_snr_db' in metrics:
                print(f"  - 降噪后信噪比(评估): {metrics['denoised_snr_db']:.2f} dB")
            print("="*50)

        # 保存结果
        if not args.no_save:
            output_file = args.output if args.output else None
            if processor.save_output(output_file):
                print(f"\n✅ 处理结果已保存")

        print("\n🎉 音频处理完成!")
        print(f"📊 分析图表保存在: results/figures/")
        print(f"🎵 处理后的音频保存在: data/output/")

        return True

    except Exception as e:
        print(f"❌ 处理过程中发生错误: {str(e)}")
        return False


def process_batch_files(directory: str, args) -> None:
    """
    批量处理音频文件

    Args:
        directory: 目录路径
        args: 命令行参数
    """
    print(f"\n正在批量处理目录: {directory}")
    print("=" * 50)

    # 获取音频文件列表
    audio_files = list_audio_files(directory)

    if not audio_files:
        print("❌ 目录中没有找到音频文件")
        return

    print(f"找到 {len(audio_files)} 个音频文件:")
    for file in audio_files:
        print(f"  - {file}")

    # 处理每个文件
    success_count = 0
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] 处理: {os.path.basename(audio_file)}")

        # 修改输出路径以避免覆盖
        original_args = args
        if not args.output:
            # 创建临时参数对象
            class TempArgs:
                def __init__(self, original_args, output_suffix):
                    for attr in dir(original_args):
                        if not attr.startswith('_'):
                            setattr(self, attr, getattr(original_args, attr))
                    self.output = f"data/output/{os.path.basename(audio_file).rsplit('.', 1)[0]}_denoised.wav"

            args = TempArgs(original_args, f"_{i}")

        if process_single_file(audio_file, args):
            success_count += 1

        # 恢复原始参数
        args = original_args

    print(f"\n批量处理完成!")
    print(f"成功处理: {success_count}/{len(audio_files)} 个文件")


def interactive_mode(args) -> None:
    """
    交互式模式

    Args:
        args: 命令行参数
    """
    print("\n🎵 交互式音频降噪处理模式")
    print("=" * 40)

    while True:
        print("\n请选择操作:")
        print("1. 处理单个音频文件")
        print("2. 批量处理音频文件")
        print("3. 查看帮助")
        print("4. 退出")

        choice = input("\n请输入选择 (1-4): ").strip()

        if choice == '1':
            # 处理单个文件
            input_file = input("请输入音频文件路径: ").strip()
            if os.path.exists(input_file):
                process_single_file(input_file, args)
            else:
                print("❌ 文件不存在，请检查路径")

        elif choice == '2':
            # 批量处理
            directory = input("请输入音频文件目录路径: ").strip()
            if os.path.exists(directory) and os.path.isdir(directory):
                process_batch_files(directory, args)
            else:
                print("❌ 目录不存在，请检查路径")

        elif choice == '3':
            # 显示帮助
            print("\n帮助信息:")
            print("- 支持的音频格式: WAV, MP3, FLAC, OGG")
            print("- 处理流程: 加载 → 时域分析 → 频域分析 → 滤波 → 增强 → 分析")
            print("- 输出文件保存在: data/output/")
            print("- 分析图表保存在: results/figures/")

        elif choice == '4':
            print("👋 感谢使用音频降噪系统!")
            break

        else:
            print("❌ 无效选择，请重新输入")


def run_complete_pipeline(input_file: str, output_file: str = None,
                         filter_type: str = 'fir_lowpass',
                         sample_rate: int = 44100) -> dict:
    """
    执行完整的分析处理流程

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        filter_type: 滤波器类型
        sample_rate: 采样率

    Returns:
        完整的处理结果
    """
    print("🚀 开始完整音频处理流程")
    start_time = time.time()

    # 创建处理器
    processor = AudioProcessor(sample_rate=sample_rate)

    # 加载音频
    if not processor.load_audio(input_file):
        raise ValueError("音频文件加载失败")

    # 运行完整分析
    results = processor.run_complete_analysis()

    # 保存结果
    if output_file:
        processor.save_output(output_file)

    elapsed_time = time.time() - start_time
    print(f"✅ 完整处理流程完成，耗时: {elapsed_time:.2f} 秒")

    return results


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，程序退出")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {str(e)}")
        sys.exit(1)