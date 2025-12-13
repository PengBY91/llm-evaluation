#!/usr/bin/env python3
"""
整理 /data 目录下所有评测任务的信息，包括 output_type 和是否支持 Ollama
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_output_type_from_task_name(task_name: str):
    """根据任务名称从 TaskManager 获取 output_type"""
    try:
        from lm_eval.tasks import TaskManager
        from lm_eval import utils
        from pathlib import Path
        
        task_manager = TaskManager()
        
        # 检查是否是任务
        if task_name in task_manager.all_subtasks:
            task_info = task_manager.task_index.get(task_name, {})
            yaml_path = task_info.get("yaml_path", -1)
            
            if yaml_path != -1:
                try:
                    config = utils.load_yaml_config(yaml_path, mode="simple")
                    output_type = config.get("output_type", "generate_until")
                    
                    # 如果配置中有 include，也需要检查被包含的配置
                    if "include" in config:
                        include_path = Path(yaml_path).parent / config["include"]
                        if include_path.exists():
                            include_config = utils.load_yaml_config(str(include_path), mode="simple")
                            output_type = include_config.get("output_type", output_type)
                    
                    return output_type
                except Exception as e:
                    print(f"Warning: 无法读取任务 {task_name} 的配置: {e}", file=sys.stderr)
                    pass
        
        # 检查是否是组
        if task_name in task_manager.all_groups:
            task_info = task_manager.task_index.get(task_name, {})
            yaml_path = task_info.get("yaml_path", -1)
            
            if yaml_path != -1:
                try:
                    config = utils.load_yaml_config(yaml_path, mode="simple")
                    output_type = config.get("output_type", "generate_until")
                    return output_type
                except Exception as e:
                    print(f"Warning: 无法读取组 {task_name} 的配置: {e}", file=sys.stderr)
                    pass
        
        return None
    except Exception as e:
        print(f"Error getting output_type for {task_name}: {e}", file=sys.stderr)
        return None


def scan_data_directory():
    """扫描 /data 目录下的所有数据集"""
    data_dir = project_root / "data"
    if not data_dir.exists():
        return []
    
    results = []
    
    # 首先从 TaskManager 获取所有任务信息（用于后续匹配）
    task_manager = None
    try:
        from lm_eval.tasks import TaskManager
        task_manager = TaskManager()
    except Exception as e:
        print(f"Warning: 无法初始化 TaskManager: {e}", file=sys.stderr)
    
    for dataset_dir in sorted(data_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
            continue
        
        # 跳过特殊目录
        if dataset_dir.name in ['tasks', 'datasets_metadata']:
            continue
        
        dataset_path = dataset_dir.name
        
        # 检查是否有子文件夹（配置）
        config_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        if config_dirs:
            # 有子文件夹，每个子文件夹是一个配置
            for config_dir in sorted(config_dirs):
                config_name = config_dir.name
                
                # 尝试从 TaskManager 匹配任务
                task_name = None
                output_type = None
                
                if task_manager:
                    # 尝试匹配任务
                    for t_name in task_manager.all_subtasks:
                        task_info = task_manager.task_index.get(t_name, {})
                        yaml_path = task_info.get("yaml_path", -1)
                        
                        if yaml_path != -1:
                            try:
                                from lm_eval import utils
                                config = utils.load_yaml_config(yaml_path, mode="simple")
                                yaml_dataset_path = config.get("dataset_path")
                                yaml_dataset_name = config.get("dataset_name")
                                
                                # 匹配路径
                                path_matches = (
                                    yaml_dataset_path == dataset_path or
                                    (yaml_dataset_path and yaml_dataset_path.replace("/", "_") == dataset_path) or
                                    (yaml_dataset_path and yaml_dataset_path == dataset_path.replace("_", "/"))
                                )
                                
                                # 匹配配置名称
                                config_matches = False
                                if config_name is None and yaml_dataset_name is None:
                                    config_matches = True
                                elif config_name is not None and yaml_dataset_name is not None:
                                    config_matches = (config_name == yaml_dataset_name)
                                elif config_name == "all":
                                    config_matches = True
                                
                                if path_matches and config_matches:
                                    task_name = t_name
                                    output_type = config.get("output_type", "generate_until")
                                    
                                    # 检查 include
                                    if "include" in config:
                                        include_path = Path(yaml_path).parent / config["include"]
                                        if include_path.exists():
                                            include_config = utils.load_yaml_config(str(include_path), mode="simple")
                                            output_type = include_config.get("output_type", output_type)
                                    break
                            except Exception:
                                continue
                    
                    # 如果没找到，尝试匹配组
                    if not task_name:
                        for group_name in task_manager.all_groups:
                            task_info = task_manager.task_index.get(group_name, {})
                            yaml_path = task_info.get("yaml_path", -1)
                            
                            if yaml_path != -1:
                                try:
                                    from lm_eval import utils
                                    config = utils.load_yaml_config(yaml_path, mode="simple")
                                    yaml_dataset_path = config.get("dataset_path")
                                    yaml_dataset_name = config.get("dataset_name")
                                    
                                    path_matches = (
                                        yaml_dataset_path == dataset_path or
                                        (yaml_dataset_path and yaml_dataset_path.replace("/", "_") == dataset_path) or
                                        (yaml_dataset_path and yaml_dataset_path == dataset_path.replace("_", "/"))
                                    )
                                    
                                    config_matches = (
                                        (config_name is None and yaml_dataset_name is None) or
                                        (config_name is not None and yaml_dataset_name is not None and config_name == yaml_dataset_name) or
                                        config_name == "all"
                                    )
                                    
                                    if path_matches and config_matches:
                                        task_name = group_name
                                        output_type = config.get("output_type", "generate_until")
                                        break
                                except Exception:
                                    continue
                
                # 如果找到了 task_name，但还没有 output_type，再次获取
                if task_name and not output_type:
                    output_type = get_output_type_from_task_name(task_name)
                
                # 默认值
                if not output_type:
                    output_type = "unknown"
                
                # 判断是否支持 Ollama
                ollama_supported = output_type == "generate_until"
                
                results.append({
                    "dataset_dir": dataset_path,
                    "config_name": config_name,
                    "task_name": task_name or f"{dataset_path}/{config_name}",
                    "output_type": output_type,
                    "ollama_supported": ollama_supported,
                    "display_name": f"{dataset_path}/{config_name}" if config_name else dataset_path
                })
        else:
            # 没有子文件夹，直接是数据集
            task_name = None
            output_type = None
            
            if task_manager:
                # 尝试匹配任务
                for t_name in task_manager.all_subtasks:
                    task_info = task_manager.task_index.get(t_name, {})
                    yaml_path = task_info.get("yaml_path", -1)
                    
                    if yaml_path != -1:
                        try:
                            from lm_eval import utils
                            config = utils.load_yaml_config(yaml_path, mode="simple")
                            yaml_dataset_path = config.get("dataset_path")
                            yaml_dataset_name = config.get("dataset_name")
                            
                            path_matches = (
                                yaml_dataset_path == dataset_path or
                                (yaml_dataset_path and yaml_dataset_path.replace("/", "_") == dataset_path) or
                                (yaml_dataset_path and yaml_dataset_path == dataset_path.replace("_", "/"))
                            )
                            
                            config_matches = (yaml_dataset_name is None)
                            
                            if path_matches and config_matches:
                                task_name = t_name
                                output_type = config.get("output_type", "generate_until")
                                
                                # 检查 include
                                if "include" in config:
                                    include_path = Path(yaml_path).parent / config["include"]
                                    if include_path.exists():
                                        include_config = utils.load_yaml_config(str(include_path), mode="simple")
                                        output_type = include_config.get("output_type", output_type)
                                break
                        except Exception:
                            continue
            
            # 如果找到了 task_name，但还没有 output_type，再次获取
            if task_name and not output_type:
                output_type = get_output_type_from_task_name(task_name)
            
            if not output_type:
                output_type = "unknown"
            
            ollama_supported = output_type == "generate_until"
            
            results.append({
                "dataset_dir": dataset_path,
                "config_name": None,
                "task_name": task_name or dataset_path,
                "output_type": output_type,
                "ollama_supported": ollama_supported,
                "display_name": dataset_path
            })
    
    return results


def generate_markdown(results):
    """生成 Markdown 文档"""
    md = []
    md.append("# /data 目录下评测任务信息汇总\n\n")
    md.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    md.append("## 说明\n\n")
    md.append("- **output_type**: 任务的输出类型，包括 `generate_until`、`multiple_choice`、`loglikelihood`、`loglikelihood_rolling`\n")
    md.append("- **Ollama 支持**: 只有 `output_type` 为 `generate_until` 的任务可以在 Ollama 等不支持 logprobs 的模型上运行\n")
    md.append("- **unknown**: 表示该数据集在 TaskManager 中未找到对应的任务定义\n\n")
    md.append("## 任务列表\n\n")
    
    # 按 output_type 分类
    by_output_type = {}
    for result in results:
        output_type = result["output_type"]
        if output_type not in by_output_type:
            by_output_type[output_type] = []
        by_output_type[output_type].append(result)
    
    # 统计信息
    total = len(results)
    ollama_supported_count = sum(1 for r in results if r["ollama_supported"])
    unknown_count = sum(1 for r in results if r["output_type"] == "unknown")
    
    md.append("### 统计信息\n\n")
    md.append(f"- **总任务数**: {total}\n")
    md.append(f"- **支持 Ollama** (generate_until): {ollama_supported_count} ({ollama_supported_count/total*100:.1f}%)\n")
    md.append(f"- **不支持 Ollama**: {total - ollama_supported_count} ({(total-ollama_supported_count)/total*100:.1f}%)\n")
    md.append(f"- **未知类型** (未匹配到任务): {unknown_count} ({unknown_count/total*100:.1f}%)\n\n")
    
    md.append("### 按输出类型分类\n\n")
    for output_type in sorted(by_output_type.keys()):
        count = len(by_output_type[output_type])
        ollama_support = "✅" if output_type == "generate_until" else "❌"
        md.append(f"#### {output_type} ({count} 个任务) {ollama_support}\n\n")
        md.append("| 数据集目录 | 配置名称 | 任务名称 | 输出类型 | Ollama 支持 |\n")
        md.append("|-----------|---------|---------|---------|------------|\n")
        
        for result in sorted(by_output_type[output_type], key=lambda x: x["display_name"]):
            config = result["config_name"] or "-"
            task = result["task_name"] or "-"
            ollama = "✅ 支持" if result["ollama_supported"] else "❌ 不支持"
            md.append(f"| `{result['dataset_dir']}` | `{config}` | `{task}` | `{result['output_type']}` | {ollama} |\n")
        md.append("\n")
    
    md.append("### 完整列表（按数据集目录排序）\n\n")
    md.append("| 数据集目录 | 配置名称 | 任务名称 | 输出类型 | Ollama 支持 |\n")
    md.append("|-----------|---------|---------|---------|------------|\n")
    
    for result in sorted(results, key=lambda x: x["display_name"]):
        config = result["config_name"] or "-"
        task = result["task_name"] or "-"
        ollama = "✅ 支持" if result["ollama_supported"] else "❌ 不支持"
        md.append(f"| `{result['dataset_dir']}` | `{config}` | `{task}` | `{result['output_type']}` | {ollama} |\n")
    
    return "".join(md)


def main():
    print("正在扫描 /data 目录...")
    results = scan_data_directory()
    print(f"找到 {len(results)} 个数据集")
    
    print("正在生成 Markdown 文档...")
    markdown = generate_markdown(results)
    
    output_file = project_root / "docs" / "data_tasks_complete_list.md"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown)
    
    print(f"文档已保存到: {output_file}")
    print(f"\n统计信息:")
    print(f"- 总任务数: {len(results)}")
    ollama_count = sum(1 for r in results if r['ollama_supported'])
    unknown_count = sum(1 for r in results if r['output_type'] == 'unknown')
    print(f"- 支持 Ollama: {ollama_count}")
    print(f"- 不支持 Ollama: {len(results) - ollama_count}")
    print(f"- 未知类型: {unknown_count}")


if __name__ == "__main__":
    main()
