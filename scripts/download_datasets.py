#!/usr/bin/env python3
"""
下载常用的 benchmarks 数据集到本地 data/ 目录
支持离线评测使用
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 常用 benchmarks 数据集配置
# 格式: (dataset_path, dataset_name, description)
COMMON_DATASETS = [
    # 数学和推理
    ("gsm8k", "main", "GSM8K - 数学问题求解"),
    # MATH 数据集有多个子任务，需要分别下载
    ("EleutherAI/hendrycks_math", "algebra", "MATH Algebra - 代数"),
    ("EleutherAI/hendrycks_math", "counting_and_probability", "MATH Counting & Probability - 计数与概率"),
    ("EleutherAI/hendrycks_math", "geometry", "MATH Geometry - 几何"),
    ("EleutherAI/hendrycks_math", "intermediate_algebra", "MATH Intermediate Algebra - 中级代数"),
    ("EleutherAI/hendrycks_math", "number_theory", "MATH Number Theory - 数论"),
    ("EleutherAI/hendrycks_math", "prealgebra", "MATH Prealgebra - 预代数"),
    ("EleutherAI/hendrycks_math", "precalculus", "MATH Precalculus - 微积分预备"),
    
    # 常识推理
    ("Rowan/hellaswag", None, "HellaSwag - 常识推理"),
    ("winogrande", "winogrande_xl", "WinoGrande - 常识推理"),
    ("baber/piqa", None, "PIQA - 物理常识问答"),
    
    # 问答
    ("allenai/ai2_arc", "ARC-Easy", "ARC Easy - 科学问答"),
    ("allenai/ai2_arc", "ARC-Challenge", "ARC Challenge - 科学问答"),
    ("super_glue", "boolq", "BoolQ - 布尔问答"),
    ("trivia_qa", "rc.nocontext", "TriviaQA - 问答"),
    
    # 语言建模
    ("EleutherAI/lambada_openai", "default", "LAMBADA OpenAI - 语言建模"),
    ("wikitext", "wikitext-2-raw-v1", "WikiText-2 - 语言建模"),
    
    # 多任务理解
    ("cais/mmlu", "all", "MMLU - 多任务语言理解"),
    
    # 真实性评估
    ("truthful_qa", "generation", "TruthfulQA - 真实性评估"),
    
    # 其他
    ("sciq", None, "SciQ - 科学问答"),
    ("openbookqa", "main", "OpenBookQA - 开放书问答"),
    # 注意: social_i_qa 使用旧脚本格式，可能无法直接下载
    # ("social_i_qa", None, "Social IQA - 社交推理"),
]

# 可选的大型数据集（下载时间较长）
LARGE_DATASETS = [
    ("pile", None, "The Pile - 大规模语言建模数据集"),
]

def download_dataset(dataset_path: str, dataset_name: str | None, save_dir: Path, skip_existing: bool = True) -> bool:
    """下载单个数据集"""
    try:
        # 创建保存目录名
        save_path = save_dir / dataset_path.replace("/", "_")
        if dataset_name:
            save_path = save_path / dataset_name
        
        # 检查是否已存在
        if skip_existing and save_path.exists():
            print(f"⏭  跳过 (已存在): {dataset_path}" + (f" ({dataset_name})" if dataset_name else ""))
            return True
        
        print(f"\n正在下载: {dataset_path}" + (f" ({dataset_name})" if dataset_name else ""))
        
        # 加载数据集
        # 注意: 新版本的 datasets 库不再支持 trust_remote_code
        try:
            if dataset_name:
                dataset = load_dataset(dataset_path, dataset_name)
            else:
                dataset = load_dataset(dataset_path)
        except Exception as e:
            error_msg = str(e)
            # 检查是否需要指定配置名称
            if "Config name is missing" in error_msg or "Please pick one among" in error_msg:
                print(f"  ⚠️  此数据集需要指定配置名称")
                print(f"  💡 提示: 请检查数据集文档，指定正确的 dataset_name")
                raise ValueError(f"数据集 {dataset_path} 需要指定配置名称") from e
            # 检查是否是旧脚本格式
            elif "Dataset scripts are no longer supported" in error_msg:
                print(f"  ⚠️  此数据集使用旧脚本格式，无法直接下载")
                print(f"  💡 提示: 该数据集可能已迁移或需要手动下载")
                raise RuntimeError(f"数据集 {dataset_path} 使用旧脚本格式，无法下载") from e
            else:
                raise
        
        # 创建保存目录
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存数据集
        dataset.save_to_disk(str(save_path))
        
        # 打印数据集信息
        print(f"✓ 已保存到: {save_path}")
        for split_name, split_data in dataset.items():
            print(f"  - {split_name}: {len(split_data)} 个样本")
        
        return True
        
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="下载 benchmarks 数据集到本地")
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="包含大型数据集（下载时间较长）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载已存在的数据集"
    )
    args = parser.parse_args()
    
    # 创建 data 目录
    DATA_DIR.mkdir(exist_ok=True)
    print(f"数据集将保存到: {DATA_DIR.absolute()}")
    
    # 选择要下载的数据集
    datasets_to_download = COMMON_DATASETS.copy()
    if args.include_large:
        datasets_to_download.extend(LARGE_DATASETS)
        print("\n⚠️  已包含大型数据集，下载可能需要较长时间")
    
    # 统计
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    # 下载所有数据集
    for dataset_path, dataset_name, description in datasets_to_download:
        print(f"\n{'='*60}")
        print(f"数据集: {description}")
        print(f"路径: {dataset_path}" + (f" | 配置: {dataset_name}" if dataset_name else ""))
        
        result = download_dataset(
            dataset_path, 
            dataset_name, 
            DATA_DIR, 
            skip_existing=not args.force
        )
        
        if result:
            # 检查是否是跳过
            save_path = DATA_DIR / dataset_path.replace("/", "_")
            if dataset_name:
                save_path = save_path / dataset_name
            if save_path.exists() and not args.force:
                # 可能是跳过的
                skip_count += 1
            success_count += 1
        else:
            fail_count += 1
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"下载完成!")
    print(f"成功: {success_count} 个")
    if skip_count > 0:
        print(f"跳过 (已存在): {skip_count} 个")
    print(f"失败: {fail_count} 个")
    print(f"\n数据集保存在: {DATA_DIR.absolute()}")
    print("\n使用方法:")
    print("数据集会自动缓存在 HuggingFace 缓存目录，可以直接使用。")
    print("如需指定本地路径，在任务 YAML 配置中使用:")
    print("  dataset_kwargs:")
    print("    data_dir: ./data/dataset_name/")

if __name__ == "__main__":
    main()

