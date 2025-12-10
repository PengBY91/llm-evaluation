<template>
  <div class="tasks-view">
    <div class="view-header">
      <h2>评测任务管理</h2>
      <el-button type="primary" @click="showCreateDialog = true">
        <el-icon><Plus /></el-icon>
        新建任务
      </el-button>
    </div>

    <el-table :data="tasks" v-loading="loading" stripe>
      <el-table-column prop="name" label="任务名称" width="200" />
      <el-table-column prop="model" label="模型" width="150" />
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
    <el-dialog v-model="showCreateDialog" title="新建评测任务" width="800px">
      <el-form :model="taskForm" label-width="120px">
        <el-form-item label="任务名称" required>
          <el-input v-model="taskForm.name" placeholder="请输入任务名称" />
        </el-form-item>
        <el-form-item label="模型类型" required>
          <el-select v-model="taskForm.model" placeholder="请选择模型类型">
            <el-option label="OpenAI Chat Completions" value="openai-chat-completions" />
            <el-option label="OpenAI Completions" value="openai-completions" />
            <el-option label="HuggingFace" value="hf" />
            <el-option label="vLLM" value="vllm" />
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
        <el-form-item label="评测任务" required>
          <el-select 
            v-model="taskForm.tasks" 
            multiple 
            placeholder="请选择评测任务"
            style="width: 100%"
          >
            <el-option 
              v-for="task in availableTasks" 
              :key="task" 
              :label="task" 
              :value="task" 
            />
          </el-select>
        </el-form-item>
        <el-form-item label="Few-shot数量">
          <el-input-number v-model="taskForm.num_fewshot" :min="0" />
        </el-form-item>
        <el-form-item label="Batch Size">
          <el-input-number v-model="taskForm.batch_size" :min="1" />
        </el-form-item>
        <el-form-item label="设备">
          <el-input v-model="taskForm.device" placeholder="例如: cuda:0, cpu" />
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
          <el-descriptions-item label="模型">{{ currentTask.model }}</el-descriptions-item>
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
          :title="currentTask.error_message" 
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

const tasks = ref([])
const models = ref([])
const availableTasks = ref([])
const loading = ref(false)
const showCreateDialog = ref(false)
const showDetailDialog = ref(false)
const currentTask = ref(null)
const modelArgsStr = ref('{}')

const selectedModelId = ref(null)

const taskForm = ref({
  name: '',
  model: 'openai-chat-completions',
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
    'hf': 'HuggingFace',
    'vllm': 'vLLM',
    'local-completions': 'Local Completions'
  }
  return typeMap[type] || type
}

const handleModelSelect = async (modelId) => {
  if (!modelId) {
    return
  }
  
  try {
    const modelArgsData = await modelsApi.getModelArgs(modelId)
    taskForm.value.model = modelArgsData.model_type
    taskForm.value.model_args = modelArgsData.model_args
    modelArgsStr.value = JSON.stringify(modelArgsData.model_args, null, 2)
    
    // 如果是 chat completions，自动启用 chat template
    if (modelArgsData.model_type === 'openai-chat-completions') {
      taskForm.value.apply_chat_template = true
    }
  } catch (error) {
    ElMessage.error('加载模型配置失败: ' + error.message)
  }
}

const createTask = async () => {
  try {
    // 解析模型参数
    try {
      taskForm.value.model_args = JSON.parse(modelArgsStr.value || '{}')
    } catch (e) {
      ElMessage.error('模型参数格式错误，请输入有效的JSON')
      return
    }
    
    await tasksApi.createTask(taskForm.value)
    ElMessage.success('任务创建成功')
    showCreateDialog.value = false
    loadTasks()
    
    // 重置表单
    selectedModelId.value = null
    taskForm.value = {
      name: '',
      model: 'openai-chat-completions',
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

const loadAvailableTasks = async () => {
  try {
    const data = await tasksApi.getAvailableTasks()
    availableTasks.value = data.subtasks || []
  } catch (error) {
    console.error('加载可用任务列表失败:', error)
    // 使用默认任务列表
    availableTasks.value = ['gsm8k', 'hellaswag', 'arc_easy', 'arc_challenge', 'mmlu']
  }
}

onMounted(() => {
  loadTasks()
  loadModels()
  loadAvailableTasks()
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

