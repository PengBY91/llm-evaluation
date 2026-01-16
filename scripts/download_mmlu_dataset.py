#!/usr/bin/env python3
"""
预下载 MMLU 数据集到本地缓存

运行此脚本后，lm-eval 评测将不再需要网络连接。
数据集会缓存到 HuggingFace 默认缓存目录 (~/.cache/huggingface/datasets)
或 HF_DATASETS_CACHE 环境变量指定的目录。

Usage:
    python scripts/download_mmlu_dataset.py
"""

import os
import sys
from pathlib import Path

# 设置缓存目录到项目目录（可选）
project_root = Path(__file__).parent.parent
cache_dir = project_root / "outputs" / "cache" / "huggingface"
cache_dir.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(cache_dir)
os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")

from datasets import load_dataset

# MMLU 所有子任务列表
MMLU_SUBTASKS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions"
]


def download_mmlu():
    """下载 MMLU 数据集的所有子任务"""
    print(f"缓存目录: {cache_dir / 'datasets'}")
    print(f"开始下载 MMLU 数据集 ({len(MMLU_SUBTASKS)} 个子任务)...\n")
    
    success_count = 0
    failed_tasks = []
    
    for i, subtask in enumerate(MMLU_SUBTASKS, 1):
        print(f"[{i}/{len(MMLU_SUBTASKS)}] 下载 {subtask}...", end=" ", flush=True)
        try:
            # 下载所有 splits: test, validation, dev
            dataset = load_dataset("cais/mmlu", subtask, trust_remote_code=True)
            print(f"✓ (test: {len(dataset['test'])}, dev: {len(dataset['dev'])})")
            success_count += 1
        except Exception as e:
            print(f"✗ 失败: {e}")
            failed_tasks.append(subtask)
    
    print(f"\n{'='*50}")
    print(f"下载完成: {success_count}/{len(MMLU_SUBTASKS)} 个子任务成功")
    
    if failed_tasks:
        print(f"失败的任务: {', '.join(failed_tasks)}")
        return 1
    
    print(f"\n数据集已缓存到: {cache_dir / 'datasets'}")
    print("现在可以离线运行 mmlu_generative 评测任务了！")
    return 0


if __name__ == "__main__":
    sys.exit(download_mmlu())
