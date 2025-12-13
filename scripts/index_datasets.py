#!/usr/bin/env python3
"""
离线索引数据集脚本
扫描 /data 目录，收集数据集信息，并保存到 data/datasets_index.json
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目根目录到 sys.path，以便导入 web_backend.api.datasets
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../web_backend')))

# 尝试导入 datasets 相关的函数
# 由于 web_backend 结构问题，我们可能需要复制部分逻辑或者直接修改 web_backend/api/datasets.py 使其更模块化
# 这里为了稳健，我们将直接调用 web_backend.api.datasets 中的 rebuild_dataset_index 函数
# 这需要在 web_backend/api/datasets.py 中先实现该函数
# 但由于我需要先创建脚本，所以我将把逻辑放在这里，或者先修改 api/datasets.py

def main():
    parser = argparse.ArgumentParser(description="索引本地数据集")
    parser.add_argument("--force", action="store_true", help="强制重新扫描，忽略现有索引")
    args = parser.parse_args()

    print("开始索引数据集...")
    
    try:
        from web_backend.api import datasets
        
        # 调用后端 API 中的重建索引逻辑
        # 我们将在 api/datasets.py 中添加一个 rebuild_dataset_index 函数
        if hasattr(datasets, 'rebuild_dataset_index'):
            datasets_list = datasets.rebuild_dataset_index()
            print(f"索引完成，共找到 {len(datasets_list)} 个数据集")
            print(f"索引文件已保存到: {datasets.INDEX_FILE}")
        else:
            print("错误: web_backend.api.datasets 模块中未找到 rebuild_dataset_index 函数")
            print("请先更新后端代码")
            sys.exit(1)
            
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保在项目根目录下运行此脚本，或者正确设置了 PYTHONPATH")
        sys.exit(1)
    except Exception as e:
        print(f"索引过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
