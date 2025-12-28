<template>
  <div class="tasks-view">
    <div class="view-header">
      <div class="header-title">
        <h2>评测任务管理</h2>
        <span class="header-subtitle">管理和监控所有的 LLM 评测实验</span>
      </div>
      <div>
        <el-button 
          type="primary" 
          @click="handleCreateTaskClick"
          :loading="loadingAvailableTasks"
          class="create-btn"
        >
          <el-icon><Plus /></el-icon>
          新建评测任务
        </el-button>
      </div>
    </div>

    <!-- 统计信息 -->
    <div class="statistics-row">
      <el-row :gutter="20">
        <el-col :span="6">
          <el-card shadow="hover" class="stat-card">
            <template #footer>
              <div class="stat-label">总计任务</div>
            </template>
            <div class="stat-value">{{ tasks.length }}</div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card shadow="hover" class="stat-card running">
            <template #footer>
              <div class="stat-label">运行中</div>
            </template>
            <div class="stat-value">{{ tasks.filter(t => t.status === 'running').length }}</div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card shadow="hover" class="stat-card completed">
            <template #footer>
              <div class="stat-label">已完成</div>
            </template>
            <div class="stat-value">{{ tasks.filter(t => t.status === 'completed').length }}</div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card shadow="hover" class="stat-card failed">
            <template #footer>
              <div class="stat-label">失败</div>
            </template>
            <div class="stat-value">{{ tasks.filter(t => t.status === 'failed').length }}</div>
          </el-card>
        </el-col>
      </el-row>
    </div>

    <el-table :data="tasks" v-loading="loading" stripe class="main-table" header-cell-class-name="table-header">
      <el-table-column prop="name" label="任务详情" min-width="250">
        <template #default="{ row }">
          <div class="task-info">
            <div class="task-name">{{ row.name }}</div>
            <div class="task-meta">
              <el-icon><Monitor /></el-icon> {{ row.model_name || row.model }}
            </div>
          </div>
        </template>
      </el-table-column>
      <el-table-column prop="tasks" label="评测数据集" min-width="300">
        <template #default="{ row }">
          <div class="dataset-tags">
            <template v-if="row.datasets && row.datasets.length > 0">
              <el-tag 
                v-for="ds in row.datasets" 
                :key="ds.id" 
                size="small" 
                effect="plain"
                class="dataset-tag"
              >
                <span v-if="ds.config_name && !ds.name.includes(ds.config_name)">
                   {{ ds.name }} ({{ ds.config_name }})
                </span>
                <span v-else-if="ds.path && ds.path !== ds.name">
                   {{ ds.name }} ({{ ds.path }})
                </span>
                <span v-else>
                   {{ ds.name }}
                </span>
              </el-tag>
            </template>
            <template v-else>
               <el-tag v-for="task in row.tasks" :key="task" size="small" effect="plain" class="dataset-tag">
                 {{ task }}
               </el-tag>
            </template>
          </div>
        </template>
      </el-table-column>
      <el-table-column prop="status" label="当前状态" width="120">
        <template #default="{ row }">
          <el-tag :type="getStatusType(row.status)" effect="dark" class="status-tag">
            {{ getStatusText(row.status) }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="created_at" label="创建时间" width="180">
        <template #default="{ row }">
          <div class="time-cell">
            <el-icon><Clock /></el-icon>
            <span>{{ formatTime(row.created_at) }}</span>
          </div>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="280" fixed="right">
        <template #default="{ row }">
          <div class="action-buttons">
            <el-tooltip content="查看详情" placement="top">
              <el-button circle size="small" @click="viewTask(row.id)"><el-icon><View /></el-icon></el-button>
            </el-tooltip>
            
            <el-tooltip content="下载结果" placement="top">
              <el-button 
                circle 
                size="small" 
                type="success" 
                @click="downloadResults(row.id)"
                :disabled="row.status !== 'completed'"
              >
                <el-icon><Download /></el-icon>
              </el-button>
            </el-tooltip>

            <el-tooltip content="启动任务" placement="top">
              <el-button 
                circle 
                size="small" 
                type="primary" 
                @click="startTask(row.id)"
                :disabled="row.status === 'running' || row.status === 'pending'"
              >
                <el-icon><VideoPlay /></el-icon>
              </el-button>
            </el-tooltip>

            <el-tooltip content="终止任务" placement="top">
              <el-button 
                circle 
                size="small" 
                type="warning" 
                @click="stopTask(row.id)"
                :disabled="row.status !== 'running'"
              >
                <el-icon><VideoPause /></el-icon>
              </el-button>
            </el-tooltip>

            <el-tooltip content="删除任务" placement="top">
              <el-button 
                circle 
                size="small" 
                type="danger" 
                @click="deleteTask(row.id)"
                :disabled="row.status === 'running'"
              >
                <el-icon><Delete /></el-icon>
              </el-button>
            </el-tooltip>
          </div>
        </template>
      </el-table-column>
    </el-table>

    <!-- 创建任务对话框 -->
    <el-dialog 
      v-model="showCreateDialog" 
      title="新建评测任务" 
      width="850px"
      :close-on-click-modal="false"
      @opened="handleDialogOpened"
      class="custom-dialog"
    >
      <el-form :model="taskForm" label-position="top" class="task-form">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="任务名称" required>
              <el-input v-model="taskForm.name" placeholder="请输入任务名称" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="选择已有模型（可选）">
              <el-select 
                v-model="selectedModelId" 
                clearable 
                placeholder="从已注册模型中选择"
                @change="handleModelSelect"
                style="width: 100%"
              >
                <el-option 
                  v-for="model in models" 
                  :key="model.id" 
                  :label="model.name" 
                  :value="model.id" 
                >
                  <div class="model-option">
                    <span>{{ model.name }}</span>
                    <el-tag size="small" type="info">{{ getModelTypeLabel(model.model_type) }}</el-tag>
                  </div>
                </el-option>
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="模型接口类型" required>
              <el-select v-model="taskForm.model" placeholder="请选择模型类型" style="width: 100%">
                <el-option label="OpenAI Chat Completions" value="openai-chat-completions" />
                <el-option label="OpenAI Completions" value="openai-completions" />
                <el-option label="HuggingFace" value="hf" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="模型参数" required tooltip="模型名称、API地址等关键参数">
              <el-input 
                v-model="modelArgsStr" 
                type="textarea" 
                :rows="3"
                placeholder='例如: {"model": "gpt-3.5-turbo", "base_url": "https://api.example.com/v1"}'
              />
            </el-form-item>
          </el-col>
        </el-row>

        <el-form-item label="选择评测任务" required>
          <el-select 
            v-model="taskForm.tasks" 
            multiple 
            collapse-tags
            collapse-tags-tooltip
            placeholder="请选择评测任务"
            style="width: 100%"
            :loading="loadingAvailableTasks"
            filterable
            @visible-change="handleTaskSelectVisible"
            @focus="handleTaskSelectFocus"
            value-key="id"
          >
            <el-option 
              v-for="dataset in availableTasks.filter(d => d.task_name)"
              :key="dataset.id"
              :label="getDatasetLabel(dataset)" 
              :value="dataset"
              :disabled="!getDatasetCompatibilityInfo(dataset).compatible"
            >
              <el-tooltip 
                v-if="!getDatasetCompatibilityInfo(dataset).compatible" 
                :content="getDatasetCompatibilityInfo(dataset).reason" 
                placement="right"
              >
                <div class="dataset-option disabled">
                  <span class="dataset-name">{{ dataset.name }}</span>
                  <el-tag size="small" type="danger">不支持</el-tag>
                </div>
              </el-tooltip>
              <div v-else class="dataset-option">
                <span class="dataset-name">{{ dataset.name }}</span>
                <div class="dataset-extra">
                  <el-tag v-if="dataset.subtasks && dataset.subtasks.length > 0" size="small" type="info">
                    {{ dataset.subtasks.length }} 个子任务
                  </el-tag>
                  <el-icon v-if="dataset.tags && (dataset.tags.includes('lm_eval_group') || dataset.tags.includes('lm_eval_task'))" color="#67C23A"><CircleCheck /></el-icon>
                </div>
              </div>
            </el-option>
          </el-select>
          <div class="task-help">
            <template v-if="loadingAvailableTasks">
              <el-icon class="is-loading"><Loading /></el-icon> 正在从 /data 目录加载数据集...
            </template>
            <template v-else-if="availableTasks.length > 0">
              共 {{ availableTasks.filter(d => d.task_name).length }} 个可用评测任务 
              <span class="tip">💡 包含子任务的评测会自动测试所有子任务并汇总结果</span>
            </template>
            <template v-else>
              点击下拉框加载 /data 目录下的数据集
            </template>
          </div>
        </el-form-item>

        <el-divider content-position="left">运行配置</el-divider>

        <el-row :gutter="20">
          <el-col :span="8">
            <el-form-item label="Few-shot 数量">
              <el-input-number v-model="taskForm.num_fewshot" :min="0" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="Batch Size">
              <el-input-number v-model="taskForm.batch_size" :min="1" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="样本限制">
              <el-input-number v-model="taskForm.limit" :min="1" style="width: 100%" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="应用 Chat 模板">
              <el-switch v-model="taskForm.apply_chat_template" inline-prompt active-text="是" inactive-text="否" />
              <span class="switch-tip">如果是对话模型，建议开启</span>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="记录模型样本">
              <el-switch v-model="taskForm.log_samples" inline-prompt active-text="是" inactive-text="否" />
              <span class="switch-tip">开启后可下载详细评测样本</span>
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="showCreateDialog = false">取消</el-button>
          <el-button type="primary" @click="createTask">创建任务</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus, Refresh, Monitor, Clock, View, Download, VideoPlay, VideoPause, Delete, CircleCheck, Loading } from '@element-plus/icons-vue'
import { tasksApi } from '../api/tasks'
import { modelsApi } from '../api/models'
import { datasetsApi } from '../api/datasets'

const router = useRouter()
const tasks = ref([])
const models = ref([])
const availableTasks = ref([])  // 存储完整的数据集信息对象
const availableDatasets = ref([])  // 存储所有数据集信息，用于查找
const loading = ref(false)
const showCreateDialog = ref(false)
const currentTask = ref(null)
const modelArgsStr = ref('{}')
const loadingAvailableTasks = ref(false)
const selectedModelId = ref(null)

const taskForm = ref({
  name: '',
  model: 'openai-chat-completions',
  model_id: null,  // 模型 ID（如果提供，后端会从本地文件自动构建 model_args）
  model_args: {},
  tasks: [],
  num_fewshot: null,
  batch_size: null,
  limit: null,
  log_samples: true,
  apply_chat_template: false,
  gen_kwargs: null,
  config_id: null
})

const loadTasks = async () => {
  if (loading.value) {
    // 防止重复加载
    return
  }
  
  loading.value = true
  try {
    tasks.value = await tasksApi.getTasks()
  } catch (error) {
    console.error('加载任务列表失败:', error)
    ElMessage.error('加载任务列表失败: ' + (error.message || '未知错误'))
  } finally {
    loading.value = false
  }
}

const loadModels = async () => {
  try {
    models.value = await modelsApi.getModels()
  } catch (error) {
    console.error('加载模型列表失败:', error)
    // 不显示错误消息，因为这不是关键操作
  }
}

const getModelTypeLabel = (type) => {
  const typeMap = {
    'openai-chat-completions': 'OpenAI Chat',
    'openai-completions': 'OpenAI Completions',
    'hf': 'HuggingFace'
  }
  return typeMap[type] || type
}

const getDatasetLabel = (dataset) => {
  // 后端已经统一格式化为 "Task (Config)"，直接使用即可
  // 如果后端没有格式化（旧数据），这里做一个兜底
  if (dataset.config_name && !dataset.name.includes(dataset.config_name)) {
    return `${dataset.name} (${dataset.config_name})`
  }
  return dataset.name
}

const getDatasetCompatibilityInfo = (dataset) => {
  if (!taskForm.value.model) return { compatible: true }
  
  // 检查模型类型是否为 chat completions
  const isChatInterface = taskForm.value.model === 'openai-chat-completions'

  const taskName = (dataset.task_name || dataset.name).toLowerCase()
  const outputType = dataset.output_type ? dataset.output_type.toLowerCase() : null
  
  if (isChatInterface) {
    let reason = '此任务通常需要 logprobs，OpenAI Chat 模型不支持。建议使用 Completions 模型或选择该任务的 CoT/Generative 版本。'
    
    if (outputType === 'loglikelihood' || outputType === 'loglikelihood_rolling' || outputType === 'multiple_choice') {
        if (!taskName.includes('cot') && !taskName.includes('generative') && !taskName.includes('gen')) {
            return { compatible: false, reason }
        }
    }

    if (!outputType) {
        const loglikelihoodTasks = ['mmlu', 'hellaswag', 'arc', 'winogrande', 'piqa', 'lambada', 'sciq', 'boolq', 'triviaqa']
        const isLoglikelihoodTask = loglikelihoodTasks.some(t => taskName.includes(t))
        const isGenerative = taskName.includes('cot') || taskName.includes('generative') || taskName.includes('gen')
        
        if (isLoglikelihoodTask && !isGenerative) {
            return { compatible: false, reason }
        }
    }
  }
  
  return { compatible: true }
}

const handleCreateTaskClick = () => {
  // 重置表单
  selectedModelId.value = null
  taskForm.value = {
    name: '',
    model: 'openai-chat-completions',
    model_id: null,
    model_args: {},
    tasks: [],
    num_fewshot: null,
    batch_size: null,
    limit: null,
    log_samples: true,
    apply_chat_template: false,
    gen_kwargs: null,
    config_id: null
  }
  modelArgsStr.value = '{}'
  
  // 打开对话框
  showCreateDialog.value = true
}

const handleModelSelect = async (modelId) => {
  if (!modelId) {
    taskForm.value.model_id = null
    taskForm.value.model_args = {}
    modelArgsStr.value = '{}'
    return
  }
  
  try {
    // 从模型列表中找到选中的模型，获取其 model_type
    const selectedModel = models.value.find(m => m.id === modelId)
    if (selectedModel) {
      taskForm.value.model = selectedModel.model_type
      taskForm.value.model_id = modelId
      
      // 如果是 chat completions，自动启用 chat template
      if (selectedModel.model_type === 'openai-chat-completions') {
        taskForm.value.apply_chat_template = true
      }
      
      // 构建模型参数的预览（后端会从 model_id 自动构建，这里只是预览）
      const previewArgs = {
        model: selectedModel.model_name || '',
        base_url: selectedModel.base_url || '',
        api_key: selectedModel.api_key === '***' ? '(已保存，后端会自动使用)' : '(未设置)',
        num_concurrent: selectedModel.max_concurrent || 1
      }
      // 移除空值
      Object.keys(previewArgs).forEach(key => {
        if (previewArgs[key] === '' || previewArgs[key] === null || previewArgs[key] === undefined) {
          delete previewArgs[key]
        }
      })
      modelArgsStr.value = JSON.stringify(previewArgs, null, 2)
      taskForm.value.model_args = previewArgs
    } else {
      ElMessage.warning('未找到选中的模型')
    }
  } catch (error) {
    console.error('加载模型配置失败:', error)
    ElMessage.error('加载模型配置失败: ' + (error.message || '未知错误'))
  }
}

const createTask = async () => {
  try {
    // 如果提供了 model_id，不需要解析 model_args（后端会自动构建）
    // 如果没有提供 model_id，需要解析 model_args
    if (!taskForm.value.model_id) {
      try {
        taskForm.value.model_args = JSON.parse(modelArgsStr.value || '{}')
      } catch (e) {
        ElMessage.error('模型参数格式错误，请输入有效的JSON，或选择已有模型')
        return
      }
    }
    
    // 将选择的数据集对象转换为任务创建请求格式
    // 如果 tasks 是数据集对象数组，优先使用 task_name（如果存在），否则使用 name
    const taskNames = taskForm.value.tasks.map(task => {
      if (typeof task === 'object' && task !== null) {
        // 数据集对象，优先使用 task_name（从 TaskManager 获取的正确任务名称）
        // 如果没有 task_name，提示用户
        if (!task.task_name) {
          ElMessage.warning(`数据集 "${task.name}" 没有对应的任务名称（task_name），可能无法创建任务。请确保该数据集在 TaskManager 中有对应的任务定义。`)
        }
        return task.task_name || task.name
      } else if (typeof task === 'string') {
        // 字符串（兼容旧代码）
        return task
      } else {
        throw new Error('无效的任务格式')
      }
    })
    
    // 构建请求数据，包含数据集信息
    // 注意：lm-eval 会自动处理 Group 下的子任务，不需要前端展开
    const requestData = {
      ...taskForm.value,
      tasks: taskNames,  // 使用正确的任务名称（优先 task_name）
      datasets: taskForm.value.tasks
        .filter(task => typeof task === 'object' && task !== null)
        .map(task => ({
          id: task.id,
          name: task.name,  // 数据集显示名称
          task_name: task.task_name,  // 正确的任务名称
          path: task.path,
          config_name: task.config_name
        }))
    }
    
    // 如果提供了 model_id，清空 model_args（让后端自动构建）
    if (requestData.model_id) {
      requestData.model_args = undefined
    }
    
    await tasksApi.createTask(requestData)
    ElMessage.success('任务创建成功')
    showCreateDialog.value = false
    loadTasks()
    
    // 重置表单
    selectedModelId.value = null
    taskForm.value = {
      name: '',
      model: 'openai-chat-completions',
      model_id: null,
      model_args: {},
      tasks: [],
      num_fewshot: null,
      batch_size: null,
      device: null,
      limit: null,
      log_samples: true,
      apply_chat_template: false,
      gen_kwargs: null,
      config_id: null
    }
    modelArgsStr.value = '{}'
  } catch (error) {
    ElMessage.error('创建任务失败: ' + error.message)
  }
}

const viewTask = (taskId) => {
  // 在新标签页打开详情页
  const routeUrl = router.resolve({
    name: 'TaskDetail',
    params: { id: taskId }
  })
  window.open(routeUrl.href, '_blank')
}

const deleteTask = async (taskId) => {
  try {
    await ElMessageBox.confirm('确定要删除这个任务吗？', '提示', {
      type: 'warning'
    })
    await tasksApi.deleteTask(taskId)
    ElMessage.success('任务已删除')
    loadTasks()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除任务失败: ' + error.message)
    }
  }
}

const downloadResults = async (taskId) => {
  try {
    const blob = await tasksApi.downloadTaskResults(taskId)
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `task_${taskId}_results.json`
    a.click()
    window.URL.revokeObjectURL(url)
    ElMessage.success('下载成功')
  } catch (error) {
    ElMessage.error('下载失败: ' + error.message)
  }
}

const startTask = async (taskId) => {
  try {
    await tasksApi.startTask(taskId)
    ElMessage.success('任务已启动')
    loadTasks()
  } catch (error) {
    ElMessage.error('启动任务失败: ' + error.message)
  }
}

const stopTask = async (taskId) => {
  try {
    await tasksApi.stopTask(taskId)
    ElMessage.success('任务已终止')
    loadTasks()
  } catch (error) {
    ElMessage.error('终止任务失败: ' + error.message)
  }
}

const getStatusType = (status) => {
  const map = {
    pending: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'danger'
  }
  return map[status] || 'info'
}

const getStatusText = (status) => {
  const map = {
    pending: '等待中',
    running: '运行中',
    completed: '已完成',
    failed: '失败'
  }
  return map[status] || status
}

const formatTime = (timeStr) => {
  if (!timeStr) return ''
  return new Date(timeStr).toLocaleString('zh-CN')
}

const formatErrorMessage = (errorMessage) => {
  if (!errorMessage) return ''
  
  // 尝试识别并转换常见的错误任务名称格式
  // 例如：'cais/mmlu_all' -> 'mmlu'
  // 这个函数作为后端转换的补充，处理一些边缘情况
  
  let formatted = errorMessage
  
  // 常见的错误格式模式：path/config_name 或 path_config_name
  // 尝试提取可能的任务名称并转换
  // 注意：这里只是简单的模式匹配，主要转换应该在后端完成
  
  // 匹配 'xxx/yyy_zzz' 或 'xxx_yyy_zzz' 格式
  const patterns = [
    // 匹配 'cais/mmlu_all' 格式
    /'([^']+\/[^']+_[^']+)'/g,
    // 匹配 "cais/mmlu_all" 格式
    /"([^"]+\/[^"]+_[^"]+)"/g,
    // 匹配不带引号的格式（在错误信息中）
    /\b([a-zA-Z0-9_]+\/[a-zA-Z0-9_]+_[a-zA-Z0-9_]+)\b/g,
    // 匹配下划线格式
    /\b([a-zA-Z0-9_]+_[a-zA-Z0-9_]+_[a-zA-Z0-9_]+)\b/g
  ]
  
  // 尝试提取可能的任务名称
  // 由于前端没有完整的映射表，这里主要处理明显的路径格式
  // 例如：cais/mmlu_all -> mmlu (提取最后的主要部分)
  patterns.forEach(pattern => {
    formatted = formatted.replace(pattern, (match, taskName) => {
      // 如果是 path/config_name 格式，尝试提取主要任务名称
      if (taskName.includes('/')) {
        const parts = taskName.split('/')
        const lastPart = parts[parts.length - 1]
        // 如果最后一部分包含下划线，可能是 config_name
        if (lastPart.includes('_')) {
          // 尝试提取主要部分（去掉可能的 config_name）
          // 例如：mmlu_all -> mmlu
          const mainPart = lastPart.split('_')[0]
          // 如果看起来像是一个有效的任务名称（简短，没有特殊字符）
          if (mainPart.length > 0 && mainPart.length < 20 && /^[a-zA-Z0-9_]+$/.test(mainPart)) {
            return match.replace(taskName, mainPart)
          }
        }
      } else if (taskName.includes('_')) {
        // 处理 path_config_name 格式
        // 例如：cais_mmlu_all -> mmlu
        const parts = taskName.split('_')
        if (parts.length >= 2) {
          // 尝试提取中间或最后的主要部分
          // 通常任务名称在中间或最后
          const possibleName = parts[parts.length - 2] || parts[parts.length - 1]
          if (possibleName.length > 0 && possibleName.length < 20 && /^[a-zA-Z0-9_]+$/.test(possibleName)) {
            return match.replace(taskName, possibleName)
          }
        }
      }
      return match
    })
  })
  
  return formatted
}

const loadAvailableTasks = async () => {
  // 防止重复加载
  if (loadingAvailableTasks.value) {
    return
  }
  
  // 如果已经加载过，直接返回
  if (availableTasks.value.length > 0) {
    return
  }
  
  loadingAvailableTasks.value = true
  try {
    // 从 datasets API 获取 /data 目录下的本地数据集
    // 由于后端限制 page_size 最大为 100，需要分页加载所有数据集
    let allDatasets = []
    let page = 1
    const pageSize = 100  // 后端限制最大为 100
    let hasMore = true
    
    while (hasMore) {
      const response = await datasetsApi.getDatasets({
        is_local: true,  // 只获取本地数据集
        groups_only: true,  // 只获取 Group 级别的数据集
        page: page,
        page_size: pageSize
      })
      
      // 检查响应数据结构
      if (!response || !response.datasets) {
        console.warn('数据集 API 返回数据格式异常:', response)
        break
      }
      
      allDatasets = allDatasets.concat(response.datasets)
      
      // 判断是否还有更多数据
      const total = response.total || 0
      const currentCount = page * pageSize
      hasMore = currentCount < total
      page++
    }
    
    if (allDatasets.length === 0) {
      availableTasks.value = []
      availableDatasets.value = []
      ElMessage.warning('/data 目录下没有找到数据集')
      return
    }
    
    // 过滤并确保所有数据集都有正确的 name 字段
    const validDatasets = allDatasets
      .filter(dataset => dataset && (dataset.name || dataset.path))  // 过滤无效数据
      .map(dataset => {
        // 确保 name 字段存在（应该从 TaskManager 获取，但如果没有则构造）
        if (!dataset.name) {
          // 如果没有 name，则根据路径构造（兼容旧数据）
          let taskName = dataset.path.replace(/\//g, '_')  // 将路径中的 "/" 替换为 "_"
          if (dataset.config_name) {
            taskName = `${taskName}_${dataset.config_name}`
          }
          dataset.name = taskName
        }
        return dataset
      })
    
    // 去重（基于 id），保留第一个
    const uniqueDatasets = []
    const seenIds = new Set()
    for (const dataset of validDatasets) {
      if (!seenIds.has(dataset.id)) {
        seenIds.add(dataset.id)
        uniqueDatasets.push(dataset)
      }
    }
    
    // 按名称排序
    uniqueDatasets.sort((a, b) => a.name.localeCompare(b.name))
    
    availableTasks.value = uniqueDatasets
    availableDatasets.value = uniqueDatasets
    
    if (availableTasks.value.length === 0) {
      ElMessage.warning('/data 目录下没有找到数据集，请先下载数据集')
    }
  } catch (error) {
    console.error('加载数据集列表失败:', error)
    // 改进错误信息显示
    let errorMessage = '加载数据集列表失败'
    if (error) {
      if (error instanceof Error) {
        errorMessage += ': ' + error.message
      } else if (typeof error === 'string') {
        errorMessage += ': ' + error
      } else if (error.message) {
        errorMessage += ': ' + error.message
      } else if (error.detail) {
        errorMessage += ': ' + error.detail
      } else if (error.response && error.response.data) {
        const data = error.response.data
        errorMessage += ': ' + (data.detail || data.message || '未知错误')
      } else {
        errorMessage += ': 未知错误'
      }
    }
    ElMessage.error(errorMessage)
    availableTasks.value = []
  } finally {
    loadingAvailableTasks.value = false
  }
}

const handleTaskSelectVisible = (visible) => {
  // 当下拉框打开时，如果还没有加载数据，则加载
  if (visible && availableTasks.value.length === 0 && !loadingAvailableTasks.value) {
    loadAvailableTasks()
  }
}

const handleTaskSelectFocus = () => {
  // 当获得焦点时，如果还没有加载数据，则加载
  if (availableTasks.value.length === 0 && !loadingAvailableTasks.value) {
    loadAvailableTasks()
  }
}

const handleDialogOpened = () => {
  // 对话框打开时，只加载模型列表（如果需要），不加载任务列表
  if (models.value.length === 0) {
    loadModels()
  }
}

onMounted(() => {
  loadTasks()
  loadModels()
  // 不再在页面加载时加载任务列表，改为懒加载
})
</script>

<style scoped>
.tasks-view {
  background: transparent;
  padding: 0;
}

.view-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 24px;
  background: white;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 2px 12px 0 rgba(0,0,0,0.05);
}

.header-title h2 {
  margin: 0;
  font-size: 24px;
  color: #303133;
  font-weight: 600;
}

.header-subtitle {
  font-size: 14px;
  color: #909399;
  margin-top: 4px;
  display: block;
}

.create-btn {
  padding: 12px 20px;
  font-weight: 500;
  border-radius: 8px;
}

/* 统计卡片 */
.statistics-row {
  margin-bottom: 24px;
}

.stat-card {
  text-align: center;
  border-radius: 12px;
  border: none;
  transition: transform 0.3s;
}

.stat-card:hover {
  transform: translateY(-4px);
}

.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #303133;
  margin-bottom: 8px;
}

.stat-label {
  font-size: 13px;
  color: #909399;
}

.stat-card.running .stat-value { color: #e6a23c; }
.stat-card.completed .stat-value { color: #67c23a; }
.stat-card.failed .stat-value { color: #f56c6c; }

/* 表格样式 */
.main-table {
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 2px 12px 0 rgba(0,0,0,0.05);
}

:deep(.table-header) {
  background-color: #f5f7fa !important;
  color: #606266;
  font-weight: 600;
}

.task-info .task-name {
  font-weight: 600;
  color: #303133;
  margin-bottom: 4px;
}

.task-meta {
  font-size: 12px;
  color: #909399;
  display: flex;
  align-items: center;
  gap: 4px;
}

.dataset-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.dataset-tag {
  border-radius: 4px;
}

.status-tag {
  min-width: 80px;
  text-align: center;
  font-weight: 500;
}

.time-cell {
  display: flex;
  align-items: center;
  gap: 6px;
  color: #606266;
  font-size: 13px;
}

.action-buttons {
  display: flex;
  gap: 8px;
}

/* 对话框与表单 */
.custom-dialog :deep(.el-dialog) {
  border-radius: 16px;
}

.task-form {
  padding: 10px 0;
}

.model-option {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.dataset-option {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.dataset-option.disabled {
  opacity: 0.6;
}

.dataset-name {
  font-weight: 500;
}

.dataset-extra {
  display: flex;
  align-items: center;
  gap: 8px;
}

.task-help {
  font-size: 12px;
  color: #909399;
  margin-top: 8px;
  display: flex;
  align-items: center;
  gap: 4px;
}

.task-help .tip {
  color: #409eff;
}

.switch-tip {
  font-size: 12px;
  color: #909399;
  margin-left: 12px;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}
</style>

