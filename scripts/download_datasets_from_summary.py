#!/usr/bin/env python3
"""
根据 EVALUATION_DATASETS_SUMMARY.md 文档中提到的数据集，
从对应的 lm_eval/tasks 配置文件中提取 dataset_path 和 dataset_name，
并下载到 /data 目录
"""

import sys
from pathlib import Path
from typing import Dict, List, Set, Optional
from datasets import load_dataset

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TASKS_DIR = PROJECT_ROOT / "lm_eval" / "tasks"

# 添加项目根目录到路径
sys.path.insert(0, str(PROJECT_ROOT))


# 任务名称映射（文档中的名称 -> 实际任务名称）
TASK_NAME_MAPPING = {
    "BBH": "bbh",
    "BigBenchHard": "bbh",
    "MMLU-Pro": "mmlu_pro",
    "Math-hard": "leaderboard_math_hard",
    "IFEval": "ifeval",
    "Musr": "leaderboard_musr",
    "GSM8K": "gsm8k",
    "MATH": "hendrycks_math",
    "Hendrycks Math": "hendrycks_math",
    "HellaSwag": "hellaswag",
    "WinoGrande": "winogrande",
    "PIQA": "piqa",
    "ARC": "arc",
    "AI2 ARC": "arc",
    "MMLU": "mmlu",
    "TriviaQA": "triviaqa",
    "BoolQ": "super_glue",
    "OpenBookQA": "openbookqa",
    "LAMBADA": "lambada",
    "WikiText": "wikitext",
    "TruthfulQA": "truthfulqa",
    "HumanEval": "humaneval",
    "SuperGLUE": "super_glue",
    "GLUE": "glue",
    "SciQ": "sciq",
    "C-Eval": "ceval",
    "AGIEval": "agieval",
    "LogiQA": "logiqa",
    "LogiQA 2.0": "logiqa2",
    "XQuAD": "xquad",
    "LongBench": "longbench",
}


def get_task_names_from_mapping() -> List[str]:
    """从 TASK_NAME_MAPPING 中获取所有任务名称（按长度降序排序）
    
    返回列表而不是集合，确保更长的任务名称优先匹配
    例如：mmlu_pro 应该在 mmlu 之前匹配，避免 mmlu_prox 被 mmlu 误匹配
    """
    # 返回映射中的所有值（去重），按长度降序排序
    task_names = list(set(TASK_NAME_MAPPING.values()))
    # 按长度降序排序，确保更长的任务名称优先匹配
    task_names.sort(key=len, reverse=True)
    return task_names


def find_yaml_files_for_tasks(task_names: List[str]) -> List[Dict]:
    """直接查找任务目录下的 YAML 文件并提取配置"""
    try:
        from lm_eval import utils
        
        configs = []
        found_tasks = set()
        
        # 遍历任务目录
        for task_dir in TASKS_DIR.iterdir():
            if not task_dir.is_dir() or task_dir.name.startswith("_"):
                continue
            
            task_dir_name = task_dir.name
            
            # 严格匹配：只匹配完全相等或精确前缀匹配
            # 只匹配：1) 完全相等 2) 以 task_name + "_" 开头（确保是子任务，不是变体）
            # 排除所有已知的变体目录
            excluded_dirs = {
                "mmlu": ["mmlu_prox", "mmlu-redux", "mmlu-redux-spanish", "mmlu-pro-plus", 
                        "mmlusr", "afrimmlu", "arabicmmlu", "darijammlu", "egymmlu", 
                        "global_mmlu", "kmmlu", "turkishmmlu", "cmmlu", "tmmluplus"],
                "mmlu_pro": ["mmlu_prox"],
                "glue": ["basqueglue", "code_x_glue"],
                "hellaswag": ["darijahellaswag", "egyhellaswag"],
                "winogrande": ["icelandic_winogrande"],
                "mgsm": ["afrimgsm"],
                "piqa": ["global_piqa"],
                "truthfulqa": ["truthfulqa-multi"],
                "longbench": ["longbench2"],
                "logiqa": [],  # logiqa2 是独立的，不会被 logiqa 匹配到（因为 logiqa2 不以 logiqa_ 开头）
                "arc": ["arc_mt"],  # arc_mt 是多语言版本，不是核心 ARC
                "gsm8k": ["gsm8k_platinum"],  # gsm8k_platinum 是变体
                "humaneval": ["humaneval_infilling"],  # humaneval_infilling 是变体
                "lambada": ["lambada_cloze", "lambada_multilingual", "lambada_multilingual_stablelm"],  # 只保留核心 lambada
            }
            
            matched = False
            for task_name in task_names:
                # 首先检查是否是被排除的目录
                if task_name in excluded_dirs:
                    if task_dir_name in excluded_dirs[task_name]:
                        continue
                
                # 严格匹配：完全相等或精确前缀匹配（task_name + "_"）
                if task_dir_name == task_name:
                    matched = True
                    break
                elif task_dir_name.startswith(task_name + "_"):
                    # 确保不是被排除的变体
                    # 例如：mmlu_prox 不应该被 mmlu 匹配（虽然它不以 mmlu_ 开头，但为了安全）
                    matched = True
                    break
            
            if not matched:
                continue
            
            # 查找该目录下的所有 YAML 文件
            yaml_files = list(task_dir.glob("*.yaml"))
            # 也查找子目录中的 YAML 文件
            for subdir in task_dir.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("_"):
                    yaml_files.extend(subdir.glob("*.yaml"))
            
            for yaml_file in yaml_files:
                # 跳过以 _ 开头的文件（通常是模板文件）
                if yaml_file.name.startswith("_"):
                    continue
                
                try:
                    config = utils.load_yaml_config(str(yaml_file), mode="simple")
                    
                    # 检查是否是组配置文件（有 group 字段）
                    if "group" in config:
                        # 这是组配置文件，跳过（组内的子任务会单独处理）
                        continue
                    
                    task_name_in_config = config.get("task")
                    dataset_path = config.get("dataset_path")
                    
                    if not dataset_path:
                        continue
                    
                    # 计算相对于 TASKS_DIR 的路径，用于确定保存目录结构
                    relative_path = yaml_file.relative_to(TASKS_DIR)
                    # 获取任务目录名（第一级目录）
                    task_folder = relative_path.parts[0]
                    
                    # 严格匹配任务名称：只匹配完全相等或精确前缀匹配
                    # 支持两种前缀格式：task_name + "_" 或 task_name + "-"（如 ceval-valid_*）
                    # 排除所有已知的变体任务
                    excluded_variants = {
                        "mmlu": ["mmlu_prox", "mmlu_prox_lite", "mmlu_prox_", "mmlu_prox_lite_",
                                "mmlu_redux", "mmlu_redux_", "mmlu_llama", "mmlu_cot_llama"],
                        "mmlu_pro": ["mmlu_prox", "mmlu_prox_lite", "mmlu_prox_", "mmlu_prox_lite_"],
                        "arc": ["arc_mt", "arc_challenge_mt"],  # arc_mt 是多语言版本
                        "gsm8k": ["gsm8k_platinum"],  # gsm8k_platinum 是变体
                        "humaneval": ["humaneval_multi_line_infilling", "humaneval_single_line_infilling", 
                                     "humaneval_random_span_infilling", "humaneval_infilling"],  # infilling 变体
                        "lambada": ["lambada_openai_cloze", "lambada_multilingual", "lambada_multilingual_stablelm"],  # 只保留核心 lambada
                    }
                    
                    if task_name_in_config:
                        for task_name in task_names:
                            # 检查是否是被排除的变体
                            excluded = False
                            if task_name in excluded_variants:
                                for variant in excluded_variants[task_name]:
                                    if task_name_in_config.startswith(variant):
                                        excluded = True
                                        break
                            
                            if excluded:
                                continue
                            
                            # 严格匹配：完全相等或精确前缀匹配
                            # 支持 task_name + "_" 或 task_name + "-" 前缀
                            if (task_name_in_config == task_name or 
                                task_name_in_config.startswith(task_name + "_") or
                                task_name_in_config.startswith(task_name + "-")):
                                key = (task_folder, dataset_path, config.get("dataset_name"))
                                if key not in found_tasks:
                                    configs.append({
                                        "task_name": task_name_in_config,
                                        "dataset_path": dataset_path,
                                        "dataset_name": config.get("dataset_name"),
                                        "task_folder": task_folder,  # 保存任务目录名
                                        "yaml_path": str(yaml_file)
                                    })
                                    found_tasks.add(key)
                                    break
                    else:
                        # 没有任务名称，但目录名匹配，也添加
                        key = (task_folder, dataset_path, config.get("dataset_name"))
                        if key not in found_tasks:
                            configs.append({
                                "task_name": task_dir_name,
                                "dataset_path": dataset_path,
                                "dataset_name": config.get("dataset_name"),
                                "task_folder": task_folder,  # 保存任务目录名
                                "yaml_path": str(yaml_file)
                            })
                            found_tasks.add(key)
                            
                except Exception as e:
                    # 忽略加载失败的文件
                    pass
        
        return configs
        
    except Exception as e:
        print(f"查找 YAML 文件失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def deduplicate_configs(configs: List[Dict]) -> List[Dict]:
    """去重配置（相同的 task_folder, dataset_path 和 dataset_name 只保留一个）"""
    seen = set()
    unique_configs = []
    
    for config in configs:
        # 使用 task_folder, dataset_path 和 dataset_name 作为唯一键
        key = (config.get("task_folder"), config["dataset_path"], config.get("dataset_name"))
        if key not in seen:
            seen.add(key)
            unique_configs.append(config)
    
    return unique_configs


def download_dataset(dataset_path: str, dataset_name: Optional[str], task_folder: str, save_dir: Path, skip_existing: bool = True) -> bool:
    """下载单个数据集
    
    Args:
        dataset_path: HuggingFace 数据集路径（如 allenai/ai2_arc）
        dataset_name: 数据集配置名称（如 ARC-Easy）
        task_folder: 任务目录名（如 arc），用于保持目录结构
        save_dir: 保存根目录
        skip_existing: 是否跳过已存在的数据集
    """
    try:
        # 使用任务目录结构，保持与 lm_eval/tasks 相同的文件夹结构
        # 例如：lm_eval/tasks/arc/ -> data/arc/
        save_path = save_dir / task_folder
        if dataset_name:
            save_path = save_path / dataset_name
        
        # 检查是否已存在
        if skip_existing and save_path.exists():
            print(f"⏭  跳过 (已存在): {dataset_path}" + (f" ({dataset_name})" if dataset_name else ""))
            return True
        
        print(f"\n正在下载: {dataset_path}" + (f" ({dataset_name})" if dataset_name else ""))
        
        # 加载数据集
        try:
            # 注意: 新版本的 datasets 库不再支持 trust_remote_code
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
                return False
            # 检查是否是旧脚本格式
            elif "Dataset scripts are no longer supported" in error_msg or "isn't based on a loading script" in error_msg:
                print(f"  ⚠️  此数据集使用旧脚本格式，无法直接下载")
                print(f"  💡 提示: 该数据集可能已迁移或需要手动下载")
                return False
            # 检查代理相关错误
            elif "SOCKS proxy" in error_msg or "socksio" in error_msg:
                print(f"  ⚠️  代理配置问题: {error_msg}")
                print(f"  💡 提示: 如果需要使用 SOCKS 代理，请安装: pip install httpx[socks]")
                print(f"  💡 或者禁用代理环境变量后重试")
                return False
            else:
                raise
        
        # 只保留 test 和 validation splits，移除 train
        splits_to_keep = []
        splits_to_remove = []
        for split_name in dataset.keys():
            split_lower = split_name.lower()
            # 保留 test 和 validation（包括各种变体）
            if any(x in split_lower for x in ['test', 'val', 'dev', 'validation']):
                splits_to_keep.append(split_name)
            else:
                splits_to_remove.append(split_name)
        
        # 创建只包含 test 和 validation 的数据集
        filtered_dataset = {}
        for split_name in splits_to_keep:
            filtered_dataset[split_name] = dataset[split_name]
        
        if not filtered_dataset:
            print(f"  ⚠️  数据集没有 test 或 validation split，跳过")
            return False
        
        # 确认并创建保存目录
        if not save_path.exists():
            print(f"  创建目录: {save_path}")
            save_path.mkdir(parents=True, exist_ok=True)
        elif not save_path.is_dir():
            print(f"  ⚠️  路径已存在但不是目录: {save_path}")
            return False
        else:
            print(f"  目录已存在: {save_path}")
        
        # 保存过滤后的数据集
        from datasets import DatasetDict
        filtered_dataset_dict = DatasetDict(filtered_dataset)
        filtered_dataset_dict.save_to_disk(str(save_path))
        
        # 打印数据集信息
        print(f"✓ 已保存到: {save_path}")
        for split_name, split_data in filtered_dataset.items():
            print(f"  - {split_name}: {len(split_data)} 个样本")
        if splits_to_remove:
            print(f"  ⏭  已跳过 train splits: {', '.join(splits_to_remove)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="根据汇总文档下载评测数据集")
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载已存在的数据集"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["all", "general", "chinese"],
        default="all",
        help="选择要下载的数据集类别: all(全部), general(通用), chinese(中文)"
    )
    args = parser.parse_args()
    
    # 创建 data 目录
    DATA_DIR.mkdir(exist_ok=True)
    print(f"数据集将保存到: {DATA_DIR.absolute()}")
    print("=" * 60)
    
    # 从 TASK_NAME_MAPPING 获取任务名称
    print("\n从 TASK_NAME_MAPPING 获取任务名称...")
    task_names = get_task_names_from_mapping()
    print(f"找到 {len(task_names)} 个任务名称")
    
    # 根据类别过滤
    if args.category == "general":
        # 通用数据集（排除中文特定的）
        chinese_tasks = {"ceval", "cmmlu", "tmlu", "tmmluplus", "aclue", "agieval", 
                        "logiqa", "logiqa2", "zhoblimp", "xquad", "mlqa", 
                        "mgsm", "longbench"}
        task_names = [t for t in task_names if t not in chinese_tasks]
    elif args.category == "chinese":
        # 只保留中文数据集
        chinese_tasks = {"ceval", "cmmlu", "tmlu", "tmmluplus", "aclue", "agieval", 
                        "logiqa", "logiqa2", "zhoblimp"}
        task_names = [t for t in task_names if t in chinese_tasks or any(ct in t for ct in chinese_tasks)]
    
    # 重新排序（按长度降序）
    task_names.sort(key=len, reverse=True)
    
    print(f"过滤后剩余 {len(task_names)} 个任务")
    print(f"任务列表: {', '.join(sorted(task_names))}")
    
    # 查找 YAML 文件并提取配置
    print("\n查找任务配置文件...")
    configs = find_yaml_files_for_tasks(task_names)
    print(f"找到 {len(configs)} 个任务配置")
    
    # 去重
    configs = deduplicate_configs(configs)
    print(f"去重后剩余 {len(configs)} 个唯一数据集配置")
    
    # 统计
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    # 下载所有数据集
    print("\n" + "=" * 60)
    print("开始下载数据集...")
    print("=" * 60)
    
    for config in configs:
        dataset_path = config["dataset_path"]
        dataset_name = config.get("dataset_name")
        task_name = config.get("task_name", "unknown")
        task_folder = config.get("task_folder", "unknown")
        
        print(f"\n任务: {task_name}")
        print(f"数据集: {dataset_path}" + (f" | 配置: {dataset_name}" if dataset_name else ""))
        print(f"保存到: data/{task_folder}/" + (f"{dataset_name}/" if dataset_name else ""))
        
        result = download_dataset(
            dataset_path, 
            dataset_name,
            task_folder,
            DATA_DIR, 
            skip_existing=not args.force
        )
        
        if result:
            # 检查是否是跳过
            save_path = DATA_DIR / task_folder
            if dataset_name:
                save_path = save_path / dataset_name
            if save_path.exists() and not args.force:
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


if __name__ == "__main__":
    main()

