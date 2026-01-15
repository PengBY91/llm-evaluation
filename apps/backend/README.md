# 大模型评测任务管理平台 - 后端

基于 FastAPI 的后端 API 服务。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行

```bash
python main.py
```

或者使用 uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8087 --reload
```

API 文档将在 http://localhost:8087/docs 可用。

## API 端点

### 评测任务管理 (`/api/tasks`)

- `POST /api/tasks/` - 创建新任务
- `GET /api/tasks/` - 获取任务列表
- `GET /api/tasks/{task_id}` - 获取任务详情
- `DELETE /api/tasks/{task_id}` - 删除任务
- `GET /api/tasks/{task_id}/results` - 获取任务结果
- `GET /api/tasks/{task_id}/download` - 下载任务结果
- `GET /api/tasks/{task_id}/progress` - 获取任务进度

### 数据集管理 (`/api/datasets`)

- `GET /api/datasets/` - 获取数据集列表
- `GET /api/datasets/{dataset_name}` - 获取数据集详情
- `POST /api/datasets/` - 添加数据集
- `DELETE /api/datasets/{dataset_name}` - 删除数据集
- `GET /api/datasets/{dataset_name}/samples` - 获取数据集样本

### 任务配置管理 (`/api/configs`)

- `POST /api/configs/` - 创建配置
- `GET /api/configs/` - 获取配置列表
- `GET /api/configs/{config_id}` - 获取配置详情
- `PUT /api/configs/{config_id}` - 更新配置
- `DELETE /api/configs/{config_id}` - 删除配置

## 目录结构

- `main.py` - FastAPI 应用入口
- `api/` - API 路由模块
  - `tasks.py` - 评测任务管理
  - `datasets.py` - 数据集管理
  - `configs.py` - 配置管理
- `results/` - 评测结果存储目录（自动创建）
- `configs/` - 配置存储目录（自动创建）

