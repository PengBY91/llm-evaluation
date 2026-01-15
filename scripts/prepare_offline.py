#!/usr/bin/env python3
"""
离线评测资源准备脚本

功能：
1. 扫描 /data 目录获取所有数据集列表
2. 确保这些数据集已下载到 HuggingFace 缓存
3. 下载常用评测指标
4. 设置离线模式环境变量

使用方法：
    python scripts/prepare_offline.py

选项：
    --download-metrics   下载评测指标到本地
    --verify-only        仅验证数据集是否完整，不下载
    --force              强制重新下载所有资源
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import argparse
import shutil

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "assets" / "data"
METRICS_CACHE_DIR = PROJECT_ROOT / "outputs" / "cache" / "metrics"

# 常用评测指标列表 (HuggingFace evaluate 库)
COMMON_METRICS = [
    "exact_match",
    "accuracy",
    "f1",
    "precision",
    "recall",
    "bleu",
    "rouge",
    "sacrebleu",
    "bertscore",
    "perplexity",
]


def get_local_datasets() -> List[Dict[str, Any]]:
    """扫描 /data 目录获取所有本地数据集信息"""
    datasets = []
    
    if not DATA_DIR.exists():
        print(f"[警告] 数据目录不存在: {DATA_DIR}")
        return []
    
    # 遍历顶层目录
    for dir_path in DATA_DIR.iterdir():
        if not dir_path.is_dir():
            continue
        if dir_path.name.startswith('.') or dir_path.name in ['datasets_metadata', 'tasks', 'tokenizers']:
            continue
        
        dataset_name = dir_path.name
        
        # 检查是否是直接数据集
        is_direct_dataset = (
            (dir_path / 'dataset_dict.json').exists() or 
            (dir_path / 'dataset_info.json').exists()
        )
        
        if is_direct_dataset:
            datasets.append({
                "name": dataset_name,
                "path": str(dir_path),
                "is_group": False,
                "subtasks": []
            })
        else:
            # 检查子目录
            subtasks = []
            for sub_dir in dir_path.iterdir():
                if sub_dir.is_dir():
                    if ((sub_dir / 'dataset_dict.json').exists() or 
                        (sub_dir / 'dataset_info.json').exists()):
                        subtasks.append({
                            "name": sub_dir.name,
                            "path": str(sub_dir)
                        })
            
            if subtasks:
                datasets.append({
                    "name": dataset_name,
                    "path": str(dir_path),
                    "is_group": True,
                    "subtasks": subtasks
                })
    
    return datasets


def verify_dataset(dataset_path: str) -> Tuple[bool, str]:
    """验证数据集是否完整可用"""
    try:
        from datasets import load_from_disk
        
        path = Path(dataset_path)
        if not path.exists():
            return False, f"路径不存在: {dataset_path}"
        
        # 尝试加载数据集
        dataset = load_from_disk(dataset_path)
        
        # 检查是否有 splits
        if not dataset:
            return False, "数据集为空"
        
        splits = list(dataset.keys())
        if not splits:
            return False, "没有找到任何 splits"
        
        # 检查每个 split 是否有数据
        for split_name in splits:
            split_data = dataset[split_name]
            if len(split_data) == 0:
                return False, f"Split '{split_name}' 为空"
        
        return True, f"有效 (splits: {', '.join(splits)})"
        
    except Exception as e:
        return False, f"加载失败: {str(e)}"


def download_metrics(metrics: List[str] = None, force: bool = False):
    """下载评测指标到本地缓存"""
    if metrics is None:
        metrics = COMMON_METRICS
    
    try:
        import evaluate
    except ImportError:
        print("[警告] evaluate 库未安装，跳过指标下载")
        print("  提示: 运行 pip install evaluate 安装")
        return
    
    METRICS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"下载评测指标到本地缓存")
    print(f"缓存目录: {METRICS_CACHE_DIR}")
    print(f"{'='*60}")
    
    success_count = 0
    fail_count = 0
    
    for metric_name in metrics:
        print(f"\n正在处理: {metric_name}")
        try:
            # 尝试加载指标（这会触发下载）
            metric = evaluate.load(metric_name)
            print(f"  ✓ 指标 '{metric_name}' 已缓存")
            success_count += 1
        except Exception as e:
            print(f"  ✗ 下载失败: {e}")
            fail_count += 1
    
    print(f"\n指标下载完成: 成功 {success_count} 个, 失败 {fail_count} 个")


def create_offline_config():
    """创建离线模式配置文件"""
    config_path = PROJECT_ROOT / "configs" / "offline_config.json"
    cache_dir = PROJECT_ROOT / "outputs" / "cache"
    hf_cache_dir = cache_dir / "huggingface"
    
    config = {
        "offline_mode": True,
        "data_dir": str(DATA_DIR),
        "cache_dir": str(cache_dir),
        "hf_cache_dir": str(hf_cache_dir),
        "environment_variables": {
            "HF_HOME": str(hf_cache_dir),
            "HF_DATASETS_CACHE": str(hf_cache_dir / "datasets"),
            "TRANSFORMERS_CACHE": str(hf_cache_dir / "transformers"),
            "HF_EVALUATE_CACHE": str(hf_cache_dir / "evaluate"),
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1"
        }
    }
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\n离线配置已保存到: {config_path}")
    return config_path


def print_offline_instructions():
    """打印离线使用说明"""
    print(f"""
{'='*60}
离线评测使用说明
{'='*60}

1. 设置环境变量（在运行评测前执行）:

   Linux/Mac:
   export HF_DATASETS_OFFLINE=1
   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1

   Windows PowerShell:
   $env:HF_DATASETS_OFFLINE="1"
   $env:HF_HUB_OFFLINE="1"
   $env:TRANSFORMERS_OFFLINE="1"

   Windows CMD:
   set HF_DATASETS_OFFLINE=1
   set HF_HUB_OFFLINE=1
   set TRANSFORMERS_OFFLINE=1

2. 启动后端服务:
   python web_backend/app.py

3. 数据集位置: {DATA_DIR}

注意: 已修改的 lm_eval 代码会自动检测离线模式并使用本地数据。
{'='*60}
""")


def main():
    parser = argparse.ArgumentParser(description="离线评测资源准备脚本")
    parser.add_argument(
        "--download-metrics",
        action="store_true",
        help="下载评测指标到本地"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="仅验证数据集是否完整，不下载"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载所有资源"
    )
    args = parser.parse_args()
    
    # 设置缓存目录到项目根目录
    cache_dir = PROJECT_ROOT / "outputs" / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    hf_cache_dir = cache_dir / "huggingface"
    hf_cache_dir.mkdir(exist_ok=True)
    
    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache_dir / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache_dir / "transformers")
    os.environ["HF_EVALUATE_CACHE"] = str(hf_cache_dir / "evaluate")
    
    print(f"""
{'='*60}
离线评测资源准备工具
{'='*60}
项目目录: {PROJECT_ROOT}
数据目录: {DATA_DIR}
缓存目录: {hf_cache_dir}
{'='*60}
""")
    
    # 1. 扫描本地数据集
    print("步骤 1: 扫描本地数据集")
    print("-" * 40)
    
    datasets = get_local_datasets()
    
    if not datasets:
        print("[警告] 未找到任何数据集!")
        print(f"请先将数据集放入 {DATA_DIR} 目录")
        sys.exit(1)
    
    print(f"找到 {len(datasets)} 个数据集/数据集组:")
    
    total_subtasks = 0
    for ds in datasets:
        if ds["is_group"]:
            subtask_count = len(ds["subtasks"])
            total_subtasks += subtask_count
            print(f"  📁 {ds['name']} (组, {subtask_count} 个子任务)")
        else:
            print(f"  📄 {ds['name']}")
    
    print(f"\n总计: {len(datasets)} 个数据集组, {total_subtasks} 个子任务")
    
    # 2. 验证数据集完整性
    print(f"\n步骤 2: 验证数据集完整性")
    print("-" * 40)
    
    valid_count = 0
    invalid_count = 0
    
    for ds in datasets:
        if ds["is_group"]:
            # 验证子任务
            for subtask in ds["subtasks"]:
                is_valid, msg = verify_dataset(subtask["path"])
                status = "✓" if is_valid else "✗"
                print(f"  {status} {ds['name']}/{subtask['name']}: {msg}")
                if is_valid:
                    valid_count += 1
                else:
                    invalid_count += 1
        else:
            is_valid, msg = verify_dataset(ds["path"])
            status = "✓" if is_valid else "✗"
            print(f"  {status} {ds['name']}: {msg}")
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
    
    print(f"\n验证完成: {valid_count} 个有效, {invalid_count} 个无效")
    
    if args.verify_only:
        print("\n[仅验证模式] 跳过后续步骤")
        sys.exit(0 if invalid_count == 0 else 1)
    
    # 3. 下载评测指标（可选）
    if args.download_metrics:
        download_metrics(force=args.force)
    
    # 4. 创建离线配置
    print(f"\n步骤 3: 创建离线配置")
    print("-" * 40)
    create_offline_config()
    
    # 5. 打印使用说明
    print_offline_instructions()
    
    if invalid_count > 0:
        print(f"[警告] 有 {invalid_count} 个数据集验证失败，可能需要重新下载")
        sys.exit(1)


if __name__ == "__main__":
    main()
