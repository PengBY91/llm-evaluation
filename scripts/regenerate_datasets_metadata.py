#!/usr/bin/env python3
"""
重新生成数据集元数据
从 /data 目录下的已有数据集提取元信息，并尝试匹配对应的 YAML 配置文件
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from datasets import load_from_disk, DatasetDict, Dataset

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
TASKS_DIR = PROJECT_ROOT / "lm_eval" / "tasks"
DATA_DIR = PROJECT_ROOT / "assets" / "data"
DATASETS_METADATA_DIR = PROJECT_ROOT / "assets" / "datasets_metadata"

# 添加项目根目录到路径，以便导入模块
sys.path.insert(0, str(PROJECT_ROOT))


def infer_category(dataset_path: str, task_name: Optional[str] = None, tags: Optional[List[str]] = None) -> str:
    """推断数据集类别"""
    # 从路径推断
    path_lower = dataset_path.lower()
    if any(x in path_lower for x in ['math', 'gsm8k', 'hendrycks']):
        return '数学推理'
    elif any(x in path_lower for x in ['arc', 'sciq', 'openbookqa']):
        return '科学问答'
    elif any(x in path_lower for x in ['hellaswag', 'winogrande', 'piqa']):
        return '常识推理'
    elif any(x in path_lower for x in ['mmlu', 'truthful']):
        return '多任务理解'
    elif any(x in path_lower for x in ['lambada', 'wikitext']):
        return '语言建模'
    elif any(x in path_lower for x in ['trivia', 'boolq']):
        return '问答'
    elif any(x in path_lower for x in ['ceval', 'cmmlu', 'tmlu', 'agieval', 'logiqa']):
        return '中文评测'
    
    # 从标签推断
    if tags:
        tag_str = ' '.join(tags).lower()
        if any(x in tag_str for x in ['math', 'reasoning']):
            return '数学推理'
        elif any(x in tag_str for x in ['science', 'knowledge']):
            return '科学问答'
        elif any(x in tag_str for x in ['commonsense']):
            return '常识推理'
        elif any(x in tag_str for x in ['language']):
            return '语言建模'
    
    return '其他'


def is_dataset_directory(dataset_dir: Path) -> bool:
    """检查目录是否是数据集目录（包含 dataset_dict.json 或 state.json）"""
    return (dataset_dir / "dataset_dict.json").exists() or (dataset_dir / "state.json").exists()


def count_arrow_files(dataset_dir: Path, split_name: str) -> Optional[int]:
    """通过统计 Arrow 文件数量来估算数据集大小"""
    try:
        split_dir = dataset_dir / split_name
        if not split_dir.exists():
            return None
        
        # 查找所有 .arrow 文件
        arrow_files = list(split_dir.glob("*.arrow"))
        if arrow_files:
            # 尝试从文件名推断（data-00000-of-00001.arrow 格式）
            # 或者直接读取文件大小来估算
            # 这里简化处理，返回 None 表示无法确定
            return None
        
        return None
    except:
        return None


def load_dataset_info(dataset_dir: Path) -> Optional[Dict[str, Any]]:
    """加载数据集信息（splits 和样本数量）"""
    splits = []
    num_examples = {}
    
    # 首先尝试从 dataset_dict.json 读取 splits
    dataset_dict_file = dataset_dir / "dataset_dict.json"
    if dataset_dict_file.exists():
        try:
            with open(dataset_dict_file, "r", encoding="utf-8") as f:
                dataset_dict = json.load(f)
                if "splits" in dataset_dict:
                    splits = dataset_dict["splits"]
        except:
            pass
    
    # 如果没有从 dataset_dict.json 获取到 splits，尝试从目录结构推断
    if not splits:
        # 检查是否有子目录（splits）
        for subdir in dataset_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("."):
                # 检查是否是 split 目录（包含 dataset_info.json 或 state.json）
                if (subdir / "dataset_info.json").exists() or (subdir / "state.json").exists():
                    splits.append(subdir.name)
    
    # 如果还是没有找到 splits，尝试加载数据集
    if not splits:
        try:
            dataset = load_from_disk(str(dataset_dir))
            
            # 处理 DatasetDict 或 Dataset
            if isinstance(dataset, DatasetDict):
                splits = list(dataset.keys())
                num_examples = {split: len(dataset[split]) for split in splits}
            elif isinstance(dataset, Dataset):
                splits = ["default"]
                num_examples = {"default": len(dataset)}
        except Exception as e:
            # 加载失败，继续使用从文件系统获取的信息
            pass
    
    # 如果还是没有 splits，使用默认值
    if not splits:
        splits = ["default"]
    
    # 尝试从 dataset_info.json 或 Arrow 文件获取样本数量
    for split_name in splits:
        if split_name not in num_examples:
            # 首先尝试从 dataset_info.json 读取
            info_file = dataset_dir / split_name / "dataset_info.json"
            if info_file.exists():
                try:
                    with open(info_file, "r", encoding="utf-8") as f:
                        info = json.load(f)
                        if "splits" in info and isinstance(info["splits"], dict):
                            # splits 是一个字典，包含每个 split 的信息
                            for split_key, split_info in info["splits"].items():
                                if split_key == split_name and "num_examples" in split_info:
                                    num_examples[split_name] = split_info["num_examples"]
                                    break
                except:
                    pass
            
            # 如果还没有获取到，尝试使用 pyarrow 读取 Arrow 文件
            if split_name not in num_examples:
                try:
                    import pyarrow as pa
                    split_dir = dataset_dir / split_name
                    arrow_files = list(split_dir.glob("data-*.arrow"))
                    if arrow_files:
                        # 读取第一个 Arrow 文件
                        arrow_file = arrow_files[0]
                        try:
                            with pa.memory_map(str(arrow_file), 'r') as source:
                                table = pa.ipc.open_file(source).read_all()
                                num_examples[split_name] = len(table)
                        except:
                            pass
                except ImportError:
                    # pyarrow 不可用，跳过
                    pass
    
    return {
        "splits": splits,
        "num_examples": num_examples if num_examples else {}
    }


def find_matching_yaml(dataset_dir: Path) -> Optional[Dict[str, Any]]:
    """尝试找到匹配的 YAML 配置文件
    
    Args:
        dataset_dir: 数据集目录路径（相对于 DATA_DIR）
    
    Returns:
        包含 YAML 配置信息的字典，如果找不到则返回 None
    """
    try:
        from lm_eval import utils
        
        # 获取相对于 DATA_DIR 的路径
        relative_path = dataset_dir.relative_to(DATA_DIR)
        parts = relative_path.parts
        
        # 第一级目录通常是任务目录名
        if len(parts) == 0:
            return None
        
        task_folder = parts[0]
        config_name = parts[1] if len(parts) > 1 else None
        
        # 在 TASKS_DIR 下查找对应的 YAML 文件
        task_dir = TASKS_DIR / task_folder
        if not task_dir.exists():
            return None
        
        # 查找所有 YAML 文件
        yaml_files = []
        for yaml_file in task_dir.rglob("*.yaml"):
            if not yaml_file.name.startswith("_"):
                yaml_files.append(yaml_file)
        
        # 尝试匹配 YAML 文件
        for yaml_file in yaml_files:
            try:
                config = utils.load_yaml_config(str(yaml_file), mode="simple")
                
                # 跳过组配置文件
                if "group" in config:
                    continue
                
                yaml_dataset_path = config.get("dataset_path")
                yaml_dataset_name = config.get("dataset_name")
                task_name = config.get("task")
                
                # 尝试匹配：根据 dataset_name 或 task_name
                if config_name:
                    # 如果有配置名称，尝试匹配
                    if yaml_dataset_name == config_name or (task_name and config_name in task_name):
                        return {
                            "yaml_path": str(yaml_file),
                            "task_name": task_name,
                            "dataset_path": yaml_dataset_path,
                            "dataset_name": yaml_dataset_name,
                            "config": config
                        }
                else:
                    # 没有配置名称，尝试匹配任务目录
                    if yaml_dataset_path and task_folder in yaml_dataset_path:
                        return {
                            "yaml_path": str(yaml_file),
                            "task_name": task_name,
                            "dataset_path": yaml_dataset_path,
                            "dataset_name": yaml_dataset_name,
                            "config": config
                        }
            except:
                continue
        
        return None
    except Exception as e:
        return None


def extract_tags_from_config(config: Dict[str, Any]) -> List[str]:
    """从配置中提取标签"""
    tags = []
    if "tag" in config:
        tag_value = config["tag"]
        if isinstance(tag_value, str):
            tags = [tag_value]
        elif isinstance(tag_value, list):
            tags = tag_value
    return tags


def get_metadata_file_path(dataset_dir: Path) -> Path:
    """根据数据集目录路径生成元数据文件路径
    
    保持与 /data 目录相同的结构到 /datasets_metadata
    """
    relative_path = dataset_dir.relative_to(DATA_DIR)
    # 将路径转换为 JSON 文件路径
    # 例如: data/arc/ARC-Easy -> datasets_metadata/arc/ARC-Easy.json
    metadata_file = DATASETS_METADATA_DIR / relative_path.with_suffix(".json")
    return metadata_file


def process_dataset_directory(dataset_dir: Path) -> Optional[Dict[str, Any]]:
    """处理一个数据集目录，生成元数据"""
    try:
        relative_path = dataset_dir.relative_to(DATA_DIR)
        print(f"\n处理数据集: {relative_path}")
        
        # 加载数据集信息
        dataset_info = load_dataset_info(dataset_dir)
        if dataset_info is None:
            print(f"  ⚠ 无法加载数据集信息")
            return None
        
        splits = dataset_info["splits"]
        num_examples = dataset_info["num_examples"]
        
        print(f"  ✓ Splits: {splits}")
        if num_examples:
            total = sum(num_examples.values())
            print(f"  ✓ 总计 {total} 个样本")
        
        # 尝试找到匹配的 YAML 文件
        yaml_info = find_matching_yaml(dataset_dir)
        
        # 构建元数据
        parts = relative_path.parts
        task_folder = parts[0] if len(parts) > 0 else None
        config_name = parts[1] if len(parts) > 1 else None
        
        metadata = {
            "local_path": str(dataset_dir),
            "is_local": True,
            "splits": splits,
            "num_examples": num_examples,
        }
        
        if yaml_info:
            metadata.update({
                "path": yaml_info["dataset_path"],
                "config_name": yaml_info["dataset_name"],
                "task_name": yaml_info["task_name"],
                "yaml_path": yaml_info["yaml_path"],
                "tags": extract_tags_from_config(yaml_info["config"]),
                "category": infer_category(
                    yaml_info["dataset_path"], 
                    yaml_info["task_name"],
                    extract_tags_from_config(yaml_info["config"])
                ),
            })
            
            if "metadata" in yaml_info["config"]:
                metadata["task_metadata"] = yaml_info["config"]["metadata"]
            
            print(f"  ✓ 找到匹配的 YAML: {yaml_info['task_name']}")
        else:
            # 没有找到 YAML，使用目录信息推断
            dataset_path = relative_path.as_posix().replace("/", "_")
            metadata.update({
                "path": dataset_path,
                "config_name": config_name,
                "task_name": None,
                "yaml_path": None,
                "tags": [],
                "category": infer_category(str(relative_path)),
            })
            print(f"  ⚠ 未找到匹配的 YAML 文件")
        
        return metadata
        
    except Exception as e:
        print(f"  ✗ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def scan_datasets(data_dir: Path) -> List[Path]:
    """扫描 /data 目录下的所有数据集目录"""
    datasets = []
    
    def scan_recursive(directory: Path):
        """递归扫描目录"""
        for item in directory.iterdir():
            if not item.is_dir() or item.name.startswith("."):
                continue
            
            if is_dataset_directory(item):
                # 这是一个数据集目录
                datasets.append(item)
            else:
                # 可能是包含多个数据集的目录，继续扫描子目录
                scan_recursive(item)
    
    scan_recursive(data_dir)
    return sorted(datasets)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="从 /data 目录重新生成数据集元数据")
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新生成所有元数据（覆盖已存在的）"
    )
    args = parser.parse_args()
    
    # 确保目录存在
    DATASETS_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"数据集目录: {DATA_DIR.absolute()}")
    print(f"元数据目录: {DATASETS_METADATA_DIR.absolute()}")
    print("=" * 60)
    
    if not DATA_DIR.exists():
        print(f"错误: 数据集目录不存在: {DATA_DIR}")
        return 1
    
    # 扫描所有数据集
    print("扫描 /data 目录下的数据集...")
    dataset_dirs = scan_datasets(DATA_DIR)
    print(f"找到 {len(dataset_dirs)} 个数据集目录")
    print("=" * 60)
    
    # 统计
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    # 处理每个数据集
    for dataset_dir in dataset_dirs:
        try:
            # 生成元数据文件路径
            metadata_file = get_metadata_file_path(dataset_dir)
            
            # 检查是否已存在
            if not args.force and metadata_file.exists():
                print(f"\n⏭ 跳过（已存在）: {metadata_file.relative_to(DATASETS_METADATA_DIR)}")
                skip_count += 1
                continue
            
            # 处理数据集
            metadata = process_dataset_directory(dataset_dir)
            
            if metadata is None:
                fail_count += 1
                continue
            
            # 保存元数据
            try:
                metadata_file.parent.mkdir(parents=True, exist_ok=True)
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                print(f"  ✓ 已保存: {metadata_file.relative_to(DATASETS_METADATA_DIR)}")
                success_count += 1
            except Exception as e:
                print(f"  ✗ 保存失败: {e}")
                fail_count += 1
                
        except Exception as e:
            print(f"\n✗ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
    
    # 打印总结
    print("\n" + "=" * 60)
    print("处理完成!")
    print(f"成功: {success_count} 个")
    if skip_count > 0:
        print(f"跳过 (已存在): {skip_count} 个")
    print(f"失败: {fail_count} 个")
    print(f"\n元数据保存在: {DATASETS_METADATA_DIR.absolute()}")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
