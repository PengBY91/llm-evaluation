#!/usr/bin/env python3
"""
预缓存 TaskManager 数据

将 lm_eval 的任务配置预先解析并保存为 JSON 缓存文件，
避免每次启动时重新扫描和解析数百个 YAML 文件。

使用方法：
    python scripts/cache_task_manager.py

生成的缓存文件：
    cache/task_manager_cache.json
"""

import os
import sys
import json
import time
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_FILE = CACHE_DIR / "task_manager_cache.json"


def build_task_cache():
    """构建任务缓存"""
    print("=" * 60)
    print("TaskManager 缓存构建工具")
    print("=" * 60)
    
    CACHE_DIR.mkdir(exist_ok=True)
    
    print("\n正在初始化 TaskManager（这需要一些时间）...")
    start_time = time.time()
    
    from lm_eval.tasks import TaskManager
    tm = TaskManager()
    
    elapsed = time.time() - start_time
    print(f"TaskManager 初始化完成，耗时: {elapsed:.2f} 秒")
    
    # 提取需要缓存的数据
    cache_data = {
        "all_subtasks": list(tm.all_subtasks) if hasattr(tm, 'all_subtasks') else [],
        "all_groups": list(tm.all_groups) if hasattr(tm, 'all_groups') else [],
        "all_tags": list(tm.all_tags) if hasattr(tm, 'all_tags') else [],
        "all_tasks": list(tm.all_tasks) if hasattr(tm, 'all_tasks') else [],
        "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cache_version": "1.0"
    }
    
    # 保存到 JSON 文件
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n缓存已保存到: {CACHE_FILE}")
    print(f"  - subtasks: {len(cache_data['all_subtasks'])} 个")
    print(f"  - groups: {len(cache_data['all_groups'])} 个")
    print(f"  - tags: {len(cache_data['all_tags'])} 个")
    print(f"  - all_tasks: {len(cache_data['all_tasks'])} 个")
    
    print("\n" + "=" * 60)
    print("缓存构建完成！现在 TaskManager 加载将从缓存读取，速度大幅提升。")
    print("=" * 60)
    
    return cache_data


def main():
    build_task_cache()


if __name__ == "__main__":
    main()
