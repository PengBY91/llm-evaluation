#!/usr/bin/env python3
"""
预下载所有支持的评测数据集到本地缓存

运行此脚本后，lm-eval 评测将不再需要网络连接。
数据集会缓存到项目的 outputs/cache/huggingface/datasets 目录。

Usage:
    python scripts/download_all_datasets.py [--dataset DATASET_NAME]
    
Examples:
    python scripts/download_all_datasets.py              # 下载所有数据集
    python scripts/download_all_datasets.py --dataset mmlu  # 只下载 MMLU
    python scripts/download_all_datasets.py --dataset arc   # 只下载 ARC
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# 设置缓存目录到项目目录
project_root = Path(__file__).parent.parent
cache_dir = project_root / "outputs" / "cache" / "huggingface"
cache_dir.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(cache_dir)
os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")

from datasets import load_dataset

# ============================================================
# 数据集配置：HuggingFace 数据集路径和子任务
# ============================================================

DATASETS_CONFIG: Dict[str, Dict] = {
    # MMLU - 57 个子任务
    "mmlu": {
        "hf_path": "cais/mmlu",
        "subtasks": [
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
    },
    
    # ARC (AI2 Reasoning Challenge)
    "arc": {
        "hf_path": "allenai/ai2_arc",
        "subtasks": ["ARC-Challenge", "ARC-Easy"]
    },
    
    # HellaSwag
    "hellaswag": {
        "hf_path": "Rowan/hellaswag",
        "subtasks": None  # 无子任务
    },
    
    # WinoGrande
    "winogrande": {
        "hf_path": "allenai/winogrande",
        "subtasks": ["winogrande_xl"]
    },
    
    # PIQA
    "piqa": {
        "hf_path": "ybisk/piqa",
        "subtasks": None
    },
    
    # OpenBookQA
    "openbookqa": {
        "hf_path": "allenai/openbookqa",
        "subtasks": ["main"]
    },
    
    # TruthfulQA
    "truthfulqa": {
        "hf_path": "truthfulqa/truthful_qa",
        "subtasks": ["generation", "multiple_choice"]
    },
    
    # GSM8K
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "subtasks": ["main"]
    },
    
    # LAMBADA
    "lambada": {
        "hf_path": "EleutherAI/lambada_openai",
        "subtasks": None
    },
    
    # SciQ
    "sciq": {
        "hf_path": "allenai/sciq",
        "subtasks": None
    },
    
    # TriviaQA
    "triviaqa": {
        "hf_path": "mandarjoshi/trivia_qa",
        "subtasks": ["rc.nocontext"]
    },
    
    # WikiText
    "wikitext": {
        "hf_path": "Salesforce/wikitext",
        "subtasks": ["wikitext-2-raw-v1", "wikitext-103-raw-v1"]
    },
    
    # SuperGLUE 子任务
    "super_glue": {
        "hf_path": "aps/super_glue",
        "subtasks": ["boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc"]
    },
    
    # C-Eval (中文评测)
    "ceval": {
        "hf_path": "ceval/ceval-exam",
        "subtasks": [
            "computer_network", "operating_system", "computer_architecture",
            "college_programming", "college_physics", "college_chemistry",
            "advanced_mathematics", "probability_and_statistics", "discrete_mathematics",
            "electrical_engineer", "metrology_engineer", "high_school_mathematics",
            "high_school_physics", "high_school_chemistry", "high_school_biology",
            "middle_school_mathematics", "middle_school_biology", "middle_school_physics",
            "middle_school_chemistry", "veterinary_medicine", "college_economics",
            "business_administration", "marxism", "mao_zedong_thought", "education_science",
            "teacher_qualification", "high_school_politics", "high_school_geography",
            "middle_school_politics", "middle_school_geography", "modern_chinese_history",
            "ideological_and_moral_cultivation", "logic", "law", "chinese_language_and_literature",
            "art_studies", "professional_tour_guide", "legal_professional", "high_school_chinese",
            "high_school_history", "middle_school_history", "civil_servant",
            "sports_science", "plant_protection", "basic_medicine", "clinical_medicine",
            "urban_and_rural_planner", "accountant", "fire_engineer", "environmental_impact_assessment_engineer",
            "tax_accountant", "physician"
        ]
    },
    
    # HENDRYCKS MATH
    "hendrycks_math": {
        "hf_path": "lighteval/MATH",
        "subtasks": None
    },
    
    # BBH (Big-Bench Hard)
    "bbh": {
        "hf_path": "lukaemon/bbh",
        "subtasks": [
            "boolean_expressions", "causal_judgement", "date_understanding",
            "disambiguation_qa", "dyck_languages", "formal_fallacies",
            "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
            "logical_deduction_seven_objects", "logical_deduction_three_objects",
            "movie_recommendation", "multistep_arithmetic_two", "navigate",
            "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
            "ruin_names", "salient_translation_error_detection", "snarks",
            "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects",
            "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects",
            "web_of_lies", "word_sorting"
        ]
    },
}


def download_dataset(name: str, config: Dict) -> Tuple[int, int, List[str]]:
    """
    下载单个数据集
    
    返回: (成功数, 总数, 失败列表)
    """
    hf_path = config["hf_path"]
    subtasks = config.get("subtasks")
    
    success = 0
    failed = []
    
    if subtasks:
        total = len(subtasks)
        for i, subtask in enumerate(subtasks, 1):
            print(f"  [{i}/{total}] {subtask}...", end=" ", flush=True)
            try:
                load_dataset(hf_path, subtask)
                print("✓")
                success += 1
            except Exception as e:
                print(f"✗ ({str(e)[:50]}...)")
                failed.append(f"{name}/{subtask}")
    else:
        total = 1
        print(f"  下载中...", end=" ", flush=True)
        try:
            load_dataset(hf_path)
            print("✓")
            success = 1
        except Exception as e:
            print(f"✗ ({str(e)[:50]}...)")
            failed.append(name)
    
    return success, total, failed


def main():
    parser = argparse.ArgumentParser(description="预下载评测数据集")
    parser.add_argument("--dataset", "-d", type=str, help="指定下载的数据集名称")
    parser.add_argument("--list", "-l", action="store_true", help="列出所有可下载的数据集")
    args = parser.parse_args()
    
    if args.list:
        print("可下载的数据集:")
        for name, config in DATASETS_CONFIG.items():
            subtasks = config.get("subtasks")
            count = len(subtasks) if subtasks else 1
            print(f"  {name}: {count} 个子任务 ({config['hf_path']})")
        return 0
    
    print(f"缓存目录: {cache_dir / 'datasets'}\n")
    
    datasets_to_download = {}
    if args.dataset:
        if args.dataset not in DATASETS_CONFIG:
            print(f"错误: 未知数据集 '{args.dataset}'")
            print(f"可用数据集: {', '.join(DATASETS_CONFIG.keys())}")
            return 1
        datasets_to_download = {args.dataset: DATASETS_CONFIG[args.dataset]}
    else:
        datasets_to_download = DATASETS_CONFIG
    
    total_success = 0
    total_count = 0
    all_failed = []
    
    for name, config in datasets_to_download.items():
        subtasks = config.get("subtasks")
        count = len(subtasks) if subtasks else 1
        print(f"\n{'='*50}")
        print(f"下载 {name} ({count} 个子任务)")
        print(f"HuggingFace 路径: {config['hf_path']}")
        print(f"{'='*50}")
        
        success, total, failed = download_dataset(name, config)
        total_success += success
        total_count += total
        all_failed.extend(failed)
    
    print(f"\n{'='*60}")
    print(f"下载完成: {total_success}/{total_count} 个数据集/子任务成功")
    
    if all_failed:
        print(f"\n失败的数据集:")
        for f in all_failed:
            print(f"  - {f}")
    else:
        print("\n所有数据集下载成功！")
    
    print(f"\n数据集已缓存到: {cache_dir / 'datasets'}")
    print("现在可以离线运行评测任务了！")
    
    return 1 if all_failed else 0


if __name__ == "__main__":
    sys.exit(main())
