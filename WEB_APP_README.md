# 大模型评测任务管理平台

基于 `lm-evaluation-harness` 的 Web 应用，提供可视化的评测任务管理界面。

## 项目结构

```
lm-evaluation-harness/
├── web_backend/          # 后端 API 服务（FastAPI）
│   ├── main.py          # 应用入口
│   ├── api/             # API 路由
│   │   ├── tasks.py     # 评测任务管理
│   │   ├── datasets.py  # 数据集管理
│   │   └── configs.py   # 配置管理
│   └── requirements.txt # Python 依赖
├── web_frontend/        # 前端应用（Vue3）
│   ├── src/
│   │   ├── views/       # 页面组件
│   │   ├── api/         # API 客户端
│   │   └── router/      # 路由配置
│   └── package.json     # Node.js 依赖
└── WEB_APP_README.md    # 本文档
```

## 快速开始

### 1. 启动后端服务

```bash
cd web_backend
pip install -r requirements.txt
python main.py
```

后端服务将在 http://localhost:8087 启动。

### 2. 启动前端应用

```bash
cd web_frontend
npm install
npm run dev
```

前端应用将在 http://localhost:3003 启动。

### 3. 访问应用

打开浏览器访问 http://localhost:3003

## 功能特性

### 1. 评测任务管理

- ✅ 创建新的评测任务
- ✅ 查看任务列表和状态
- ✅ 实时查看任务进度
- ✅ 下载评测结果
- ✅ 删除已完成或失败的任务

### 2. 任务配置管理

- ✅ 创建评测配置模板
- ✅ 编辑和更新配置
- ✅ 删除配置
- ✅ 在创建任务时选择已有配置

### 3. 数据集管理

- ✅ 浏览所有可用数据集
- ✅ 添加新数据集（从 HuggingFace Hub）
- ✅ 查看数据集详情和样本
- ✅ 删除本地数据集

## API 文档

后端启动后，可以访问以下地址查看 API 文档：

- Swagger UI: http://localhost:8087/docs
- ReDoc: http://localhost:8087/redoc

## 使用示例

### 创建评测任务

1. 在"评测任务管理"页面点击"新建任务"
2. 填写任务信息：
   - 任务名称
   - 模型类型（如 `openai-chat-completions`）
   - 模型参数（JSON 格式，如 `{"model": "gpt-3.5-turbo", "base_url": "https://api.example.com/v1"}`）
   - 选择评测任务（如 `gsm8k`, `hellaswag`）
   - 其他可选参数
3. 点击"创建"，任务将在后台运行

### 使用配置模板

1. 在"任务配置"页面创建常用配置
2. 创建任务时选择该配置，会自动填充相关参数

### 管理数据集

1. 在"数据集管理"页面查看所有数据集
2. 点击"添加数据集"从 HuggingFace Hub 下载数据集
3. 点击"详情"查看数据集信息和样本预览

## 注意事项

1. **任务运行**: 评测任务在后台异步运行，可以通过刷新页面查看最新状态
2. **结果存储**: 评测结果保存在 `results/` 目录下
3. **配置存储**: 任务配置保存在 `configs/` 目录下
4. **数据集存储**: 本地数据集保存在 `data/` 目录下（与项目的数据集目录共享）

## 开发说明

### 后端开发

- 使用 FastAPI 框架
- API 路由定义在 `api/` 目录下
- 支持 CORS，允许前端跨域访问

### 前端开发

- 使用 Vue3 Composition API
- UI 组件库：Element Plus
- HTTP 客户端：Axios
- 开发服务器支持热重载

## 故障排除

### 后端无法启动

- 检查 Python 版本（需要 Python 3.8+）
- 确认已安装所有依赖：`pip install -r requirements.txt`
- 检查端口 8000 是否被占用

### 前端无法启动

- 检查 Node.js 版本（需要 Node.js 16+）
- 确认已安装所有依赖：`npm install`
- 检查端口 3000 是否被占用

### API 请求失败

- 确认后端服务正在运行
- 检查浏览器控制台的错误信息
- 确认 CORS 配置正确

## 后续改进

- [ ] 添加用户认证和权限管理
- [ ] 使用数据库存储任务和配置（替代内存存储）
- [ ] 添加任务队列管理（Celery）
- [ ] 支持任务暂停和恢复
- [ ] 添加更多图表和可视化
- [ ] 支持批量任务创建
- [ ] 添加任务历史记录和统计

