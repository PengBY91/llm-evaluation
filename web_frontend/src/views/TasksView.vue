<template>
  <div class="tasks-view">
    <div class="view-header">
      <h2>评测任务管理</h2>
      <div>
        <el-button 
          type="warning" 
          @click="handleFixTaskNames"
          :loading="fixingTaskNames"
          style="margin-right: 10px;"
        >
          修复任务名称
        </el-button>
        <el-button 
          type="primary" 
          @click="handleCreateTaskClick"
          :loading="loadingAvailableTasks"
        >
          <el-icon><Plus /></el-icon>
          新建任务
        </el-button>
      </div>
    </div>

    <el-table :data="tasks" v-loading="loading" stripe>
      <el-table-column prop="name" label="任务名称" width="200" />
      <el-table-column prop="model" label="模型" width="150">
        <template #default="{ row }">
          {{ row.model_name || row.model }}
        </template>
      </el-table-column>
      <el-table-column prop="tasks" label="评测任务" width="200">
        <template #default="{ row }">
          <el-tag v-for="task in row.tasks" :key="task" size="small" style="margin-right: 5px;">
            {{ task }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="status" label="状态" width="100">
        <template #default="{ row }">
          <el-tag :type="getStatusType(row.status)">
            {{ getStatusText(row.status) }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="created_at" label="创建时间" width="180">
        <template #default="{ row }">
          {{ formatTime(row.created_at) }}
        </template>
      </el-table-column>
      <el-table-column label="操作" width="250" fixed="right">
        <template #default="{ row }">
          <el-button size="small" @click="viewTask(row.id)">查看</el-button>
          <el-button 
            size="small" 
            type="success" 
            @click="downloadResults(row.id)"
            :disabled="row.status !== 'completed'"
          >
            下载结果
          </el-button>
          <el-button 
            size="small" 
            type="primary" 
            @click="startTask(row.id)"
            :disabled="row.status === 'running' || row.status === 'pending'"
          >
            启动
          </el-button>
          <el-button 
            size="small" 
            type="warning" 
            @click="stopTask(row.id)"
            :disabled="row.status !== 'running'"
          >
            终止
          </el-button>
          <el-button 
            size="small" 
            type="danger" 
            @click="deleteTask(row.id)"
            :disabled="row.status === 'running'"
          >
            删除
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- 创建任务对话框 -->
    <el-dialog 
      v-model="showCreateDialog" 
      title="新建评测任务" 
      width="800px"
      :close-on-click-modal="false"
      @opened="handleDialogOpened"
    >
      <el-form :model="taskForm" label-width="120px">
        <el-form-item label="任务名称" required>
          <el-input v-model="taskForm.name" placeholder="请输入任务名称" />
        </el-form-item>
        <el-form-item label="选择模型">
          <el-select 
            v-model="selectedModelId" 
            clearable 
            placeholder="选择已有模型（可选，选择后会自动填充模型参数）"
            @change="handleModelSelect"
          >
            <el-option 
              v-for="model in models" 
              :key="model.id" 
              :label="model.name" 
              :value="model.id" 
            >
              <div>
                <div>{{ model.name }}</div>
                <div style="font-size: 12px; color: #999;">{{ getModelTypeLabel(model.model_type) }}</div>
              </div>
            </el-option>
          </el-select>
        </el-form-item>
        <el-form-item label="模型参数" required>
          <el-input 
            v-model="modelArgsStr" 
            type="textarea" 
            :rows="3"
            placeholder='例如: {"model": "gpt-3.5-turbo", "base_url": "https://api.example.com/v1"}'
          />
        </el-form-item>
        <el-form-item label="模型类型" required>
          <el-select v-model="taskForm.model" placeholder="请选择模型类型">
            <el-option label="OpenAI Chat Completions" value="openai-chat-completions" />
            <el-option label="OpenAI Completions" value="openai-completions" />
            <el-option label="HuggingFace" value="hf" />
          </el-select>
        </el-form-item>
        <el-form-item label="评测任务" required>
          <el-select 
            v-model="taskForm.tasks" 
            multiple 
            placeholder="请选择评测任务（点击下拉框加载 /data 目录下的数据集）"
            style="width: 100%"
            :loading="loadingAvailableTasks"
            filterable
            :filter-method="filterTasks"
            @visible-change="handleTaskSelectVisible"
            @focus="handleTaskSelectFocus"
            value-key="uniqueKey"
          >
            <el-option 
              v-for="dataset in filteredAvailableTasks" 
              :key="dataset.uniqueKey" 
              :label="getDatasetLabel(dataset)" 
              :value="dataset"
            >
              <div>
                <div>{{ getDatasetLabel(dataset) }}</div>
                <div style="font-size: 12px; color: #999;" v-if="dataset.path">
                  {{ dataset.path }}
                </div>
              </div>
            </el-option>
          </el-select>
          <div v-if="loadingAvailableTasks" style="font-size: 12px; color: #999; margin-top: 5px;">
            正在从 /data 目录加载数据集...
          </div>
          <div v-else-if="availableTasks.length === 0" style="font-size: 12px; color: #999; margin-top: 5px;">
            点击下拉框加载 /data 目录下的数据集
          </div>
        </el-form-item>
        <el-form-item label="Few-shot数量">
          <el-input-number v-model="taskForm.num_fewshot" :min="0" />
        </el-form-item>
        <el-form-item label="Batch Size">
          <el-input-number v-model="taskForm.batch_size" :min="1" />
        </el-form-item>
        <el-form-item label="限制样本数">
          <el-input-number v-model="taskForm.limit" :min="1" />
        </el-form-item>
        <el-form-item label="应用Chat模板">
          <el-switch v-model="taskForm.apply_chat_template" />
        </el-form-item>
        <el-form-item label="记录样本">
          <el-switch v-model="taskForm.log_samples" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showCreateDialog = false">取消</el-button>
        <el-button type="primary" @click="createTask">创建</el-button>
      </template>
    </el-dialog>

    <!-- 任务详情对话框 -->
    <el-dialog v-model="showDetailDialog" title="任务详情" width="900px">
      <div v-if="currentTask">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="任务ID">{{ currentTask.id }}</el-descriptions-item>
          <el-descriptions-item label="任务名称">{{ currentTask.name }}</el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="getStatusType(currentTask.status)">
              {{ getStatusText(currentTask.status) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="模型">
            {{ currentTask.model_name || currentTask.model }}
          </el-descriptions-item>
          <el-descriptions-item label="创建时间">{{ formatTime(currentTask.created_at) }}</el-descriptions-item>
          <el-descriptions-item label="更新时间">{{ formatTime(currentTask.updated_at) }}</el-descriptions-item>
        </el-descriptions>
        
        <el-divider />
        
        <h3>评测任务</h3>
        <el-tag v-for="task in currentTask.tasks" :key="task" style="margin-right: 5px;">
          {{ task }}
        </el-tag>
        
        <el-divider />
        
        <h3>进度信息</h3>
        <el-alert 
          v-if="currentTask.error_message" 
          :title="formatErrorMessage(currentTask.error_message)" 
          type="error" 
          style="margin-bottom: 10px"
        />
        <div v-if="currentTask.progress">
          <pre>{{ JSON.stringify(currentTask.progress, null, 2) }}</pre>
        </div>
        <div v-else>暂无进度信息</div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'
import { tasksApi } from '../api/tasks'
import { modelsApi } from '../api/models'
import { datasetsApi } from '../api/datasets'

const tasks = ref([])
const models = ref([])
const availableTasks = ref([])  // 存储完整的数据集信息对象
const filteredAvailableTasks = ref([])  // 过滤后的数据集列表
const availableDatasets = ref([])  // 存储所有数据集信息，用于查找
const loading = ref(false)
const showCreateDialog = ref(false)
const showDetailDialog = ref(false)
const currentTask = ref(null)
const modelArgsStr = ref('{}')
const loadingAvailableTasks = ref(false)
const fixingTaskNames = ref(false)
const taskFilterKeyword = ref('')  // 用于存储过滤关键词

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
  loading.value = true
  try {
    tasks.value = await tasksApi.getTasks()
  } catch (error) {
    ElMessage.error('加载任务列表失败: ' + error.message)
  } finally {
    loading.value = false
  }
}

const loadModels = async () => {
  try {
    models.value = await modelsApi.getModels()
  } catch (error) {
    console.error('加载模型列表失败:', error)
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

// 获取数据集的显示标签（数据集名称 + 配置名称）
const getDatasetLabel = (dataset) => {
  if (!dataset) return ''
  const name = dataset.name || ''
  const configName = dataset.config_name
  if (configName) {
    return `${name} (${configName})`
  }
  return name
}

// 获取数据集的唯一键（用于去重和 value-key）
const getDatasetKey = (dataset) => {
  if (!dataset) return ''
  const name = dataset.name || ''
  const configName = dataset.config_name
  if (configName) {
    return `${name}__${configName}`
  }
  return name
}

// 过滤任务列表
const filterTasks = (query) => {
  taskFilterKeyword.value = query
  if (!query) {
    filteredAvailableTasks.value = availableTasks.value
    return
  }
  
  const queryLower = query.toLowerCase()
  filteredAvailableTasks.value = availableTasks.value.filter(dataset => {
    const label = getDatasetLabel(dataset).toLowerCase()
    const name = (dataset.name || '').toLowerCase()
    const configName = (dataset.config_name || '').toLowerCase()
    const path = (dataset.path || '').toLowerCase()
    
    // 同时搜索名称、配置名称和路径
    return label.includes(queryLower) || 
           name.includes(queryLower) || 
           configName.includes(queryLower) ||
           path.includes(queryLower)
  })
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
  
  // 重置过滤状态
  taskFilterKeyword.value = ''
  filteredAvailableTasks.value = availableTasks.value
  
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
    const requestData = {
      ...taskForm.value,
      tasks: taskNames,  // 使用正确的任务名称（优先 task_name）
      datasets: taskForm.value.tasks
        .filter(task => typeof task === 'object' && task !== null)
        .map(task => ({
          name: task.name,  // 数据集显示名称（来自 YAML 配置中的 task 字段）
          task_name: task.task_name,  // 正确的任务名称（从 TaskManager 获取，用于评测）
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

const viewTask = async (taskId) => {
  try {
    currentTask.value = await tasksApi.getTask(taskId)
    showDetailDialog.value = true
  } catch (error) {
    ElMessage.error('获取任务详情失败: ' + error.message)
  }
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

const handleFixTaskNames = async () => {
  try {
    await ElMessageBox.confirm(
      '此操作将修复所有任务中错误的任务名称（如 gsm8k_main -> gsm8k）。是否继续？',
      '确认修复',
      {
        type: 'warning',
        confirmButtonText: '确定',
        cancelButtonText: '取消'
      }
    )
    
    fixingTaskNames.value = true
    const response = await tasksApi.fixTaskNames()
    
    if (response.fixed_count > 0) {
      ElMessage.success(`成功修复 ${response.fixed_count} 个任务的任务名称`)
      // 重新加载任务列表
      loadTasks()
    } else {
      ElMessage.info('未发现需要修复的任务名称')
    }
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('修复任务名称失败: ' + (error.message || '未知错误'))
    }
  } finally {
    fixingTaskNames.value = false
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
    
    // 过滤并确保所有数据集都有正确的 name 字段和唯一键
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
        // 添加唯一键字段（用于 value-key 和去重）
        dataset.uniqueKey = getDatasetKey(dataset)
        return dataset
      })
    
    // 去重（基于数据集名称 + 配置名称），保留第一个
    const uniqueDatasets = []
    const seenKeys = new Set()
    for (const dataset of validDatasets) {
      const key = dataset.uniqueKey
      if (!seenKeys.has(key)) {
        seenKeys.add(key)
        uniqueDatasets.push(dataset)
      }
    }
    
    // 按显示标签排序
    uniqueDatasets.sort((a, b) => getDatasetLabel(a).localeCompare(getDatasetLabel(b)))
    
    availableTasks.value = uniqueDatasets
    filteredAvailableTasks.value = uniqueDatasets
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
  // 重置过滤状态
  taskFilterKeyword.value = ''
  filteredAvailableTasks.value = availableTasks.value
}

onMounted(() => {
  loadTasks()
  loadModels()
  // 不再在页面加载时加载任务列表，改为懒加载
})
</script>

<style scoped>
.tasks-view {
  background: white;
  padding: 20px;
  border-radius: 4px;
}

.view-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.view-header h2 {
  margin: 0;
}

pre {
  background: #f5f5f5;
  padding: 10px;
  border-radius: 4px;
  overflow-x: auto;
}
</style>

