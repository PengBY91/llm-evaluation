<template>
  <div class="tasks-view">
    <div class="view-header">
      <h2>评测任务管理</h2>
      <div>
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
      <el-table-column prop="tasks" label="评测任务" width="300">
        <template #default="{ row }">
          <div v-if="row.datasets && row.datasets.length > 0">
             <div v-for="ds in row.datasets" :key="ds.id" style="margin-bottom: 4px;">
                <el-tag size="small" style="margin-right: 5px;">
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
             </div>
          </div>
          <div v-else>
             <el-tag v-for="task in row.tasks" :key="task" size="small" style="margin-right: 5px; margin-bottom: 2px;">
               {{ task }}
             </el-tag>
          </div>
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
                <div style="opacity: 0.6; cursor: not-allowed;">
                  <div style="display: flex; align-items: center;">
                    <span style="font-weight: 500;">
                      {{ dataset.name }}
                      <span v-if="dataset.config_name && dataset.name !== dataset.config_name" style="font-weight: normal; font-size: 13px;">
                        ({{ dataset.config_name }})
                      </span>
                    </span>
                    <el-tag size="small" type="danger" style="margin-left: 5px;">不支持</el-tag>
                  </div>
                  <div style="font-size: 12px; color: #999;" v-if="dataset.path && dataset.path !== dataset.name">
                    Dataset: {{ dataset.path }}{{ dataset.config_name ? ` / ${dataset.config_name}` : '' }}
                  </div>
                </div>
              </el-tooltip>
              <div v-else>
                <div style="font-weight: 500;">
                  {{ dataset.name }}
                  <span v-if="dataset.config_name && dataset.name !== dataset.config_name" style="font-weight: normal; color: #666; font-size: 13px;">
                    ({{ dataset.config_name }})
                  </span>
                </div>
                <div style="font-size: 12px; color: #999;" v-if="dataset.path && dataset.path !== dataset.name">
                  Dataset: {{ dataset.path }}{{ dataset.config_name ? ` / ${dataset.config_name}` : '' }}
                </div>
              </div>
            </el-option>
          </el-select>
          <div v-if="loadingAvailableTasks" style="font-size: 12px; color: #999; margin-top: 5px;">
            正在从 /data 目录加载数据集...
          </div>
          <div v-else-if="availableTasks.length > 0" style="font-size: 12px; color: #999; margin-top: 5px;">
            共 {{ availableTasks.length }} 个数据集，其中 {{ availableTasks.filter(d => d.task_name).length }} 个可用于创建评测任务 
            <span v-if="availableTasks.filter(d => !d.task_name).length > 0">
              （{{ availableTasks.filter(d => !d.task_name).length }} 个未匹配到任务，已禁用）
            </span>
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
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'
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

const getDatasetLabel = (dataset) => {
  // 后端已经统一格式化为 "Task (Config)"，直接使用即可
  // 如果后端没有格式化（旧数据），这里做一个兜底
  if (dataset.config_name && !dataset.name.includes(dataset.config_name)) {
    return `${dataset.name} (${dataset.config_name})`
  }
  return dataset.name
}

const isDatasetCompatible = (dataset) => {
  // 如果没有选择模型类型，默认都兼容
  if (!taskForm.value.model) return true
  
  // 检查模型类型是否为 chat completions
  const isChatInterface = taskForm.value.model === 'openai-chat-completions'
  
  const taskName = (dataset.task_name || dataset.name).toLowerCase()
  const outputType = dataset.output_type ? dataset.output_type.toLowerCase() : null
  
  // 只有在使用 Chat Completions 接口（不支持 logprobs）时才进行严格检查
  // 如果使用的是 Completions 接口（默认），则支持所有任务
  if (isChatInterface) {
    // 1. 如果 output_type 明确为 loglikelihood，则不兼容
    // 除非它是 generative 任务（虽然通常 generative 对应 generate_until）
    if (outputType === 'loglikelihood' || outputType === 'loglikelihood_rolling') {
       // 再检查一下任务名称是否包含 cot/gen，因为有时候 output_type 可能是 loglikelihood 但通过特殊方式支持（较少见）
       if (taskName.includes('cot') || taskName.includes('generative') || taskName.includes('gen')) {
           return true
       }
       // 既然我们现在主要用 Completions 接口，其实是支持 loglikelihood 的。
       // 但是这里 isChatModel 判断的是 model.includes('chat')，
       // 而我们的默认模型类型虽然叫 openai-completions，但如果用户在创建时选择了 chat...
       // 等等，我们已经在 ModelsView 强制把类型改成了 openai-completions。
       // 只要用户使用的是 Completions 接口，就支持 logprobs。
       // 只有当用户显式选择 Chat Completions 接口（不支持 logprobs）时才需要警告。
       return false
    }
    
    // 2. 如果 output_type 明确为 multiple_choice (通常也是 loglikelihood)
    if (outputType === 'multiple_choice') {
        if (taskName.includes('cot') || taskName.includes('generative') || taskName.includes('gen')) {
           return true
       }
       return false
    }

    // 3. 如果 output_type 是 generate_until，则兼容
    if (outputType === 'generate_until') {
        return true
    }

    // 4. 如果 output_type 未知，回退到基于名称的启发式判断
    // 检查是否是已知需要 loglikelihood 的任务
    const loglikelihoodTasks = ['mmlu', 'hellaswag', 'arc', 'winogrande', 'piqa', 'lambada', 'sciq', 'boolq', 'triviaqa']
    const isLoglikelihoodTask = loglikelihoodTasks.some(t => taskName.includes(t))
    const isGenerative = taskName.includes('cot') || taskName.includes('generative') || taskName.includes('gen')
    
    if (isLoglikelihoodTask && !isGenerative) {
       return false
    }
  }
  
  return true
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
    const requestData = {
      ...taskForm.value,
      tasks: taskNames,  // 使用正确的任务名称（优先 task_name）
      datasets: taskForm.value.tasks
        .filter(task => typeof task === 'object' && task !== null)
        .map(task => ({
          id: task.id,
          name: task.name,  // 数据集显示名称（文件夹名称）
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

