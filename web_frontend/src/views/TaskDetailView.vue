<template>
  <div class="task-detail-view" v-loading="loading">
    <div class="view-header">
      <div class="header-left">
        <el-button @click="goBack" icon="ArrowLeft" circle style="margin-right: 15px;" />
        <h2>{{ currentTask?.name || '加载中...' }}</h2>
      </div>
      <div class="header-actions">
        <el-button type="primary" @click="refreshTask">刷新</el-button>
        <el-button 
          type="success" 
          @click="downloadResults"
          :disabled="!currentTask || currentTask.status !== 'completed'"
        >
          下载结果
        </el-button>
        <el-button 
          type="primary" 
          @click="startTask"
          :disabled="!currentTask || currentTask.status === 'running' || currentTask.status === 'pending'"
        >
          启动
        </el-button>
        <el-button 
          type="warning" 
          @click="stopTask"
          :disabled="!currentTask || currentTask.status !== 'running'"
        >
          终止
        </el-button>
        <el-button 
          type="danger" 
          @click="deleteTask"
          :disabled="!currentTask || currentTask.status === 'running'"
        >
          删除
        </el-button>
      </div>
    </div>

    <div v-if="currentTask" class="task-content">
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
      <div v-if="currentTask.datasets && currentTask.datasets.length > 0">
         <el-tag v-for="ds in currentTask.datasets" :key="ds.id" style="margin-right: 5px; margin-bottom: 5px;">
            {{ getDatasetLabel(ds) }}
         </el-tag>
      </div>
      <div v-else>
        <el-tag v-for="task in currentTask.tasks" :key="task" style="margin-right: 5px;">
          {{ task }}
        </el-tag>
      </div>
      
      <el-divider />
      
      <h3>进度信息</h3>
      <el-alert 
        v-if="currentTask.error_message" 
        :title="formatErrorMessage(currentTask.error_message)" 
        type="error" 
        style="margin-bottom: 10px"
      />
      <div v-if="currentTask.progress" class="progress-container">
        <pre>{{ JSON.stringify(currentTask.progress, null, 2) }}</pre>
      </div>
      <div v-else class="no-progress">暂无进度信息</div>
    </div>
    <div v-else-if="!loading" class="not-found">
      <el-empty description="未找到任务信息" />
      <el-button type="primary" @click="goBack">返回列表</el-button>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { ArrowLeft } from '@element-plus/icons-vue'
import { tasksApi } from '../api/tasks'

const route = useRoute()
const router = useRouter()

const taskId = route.params.id
const loading = ref(false)
const currentTask = ref(null)

const loadTaskDetail = async () => {
  if (!taskId) return
  
  loading.value = true
  try {
    currentTask.value = await tasksApi.getTask(taskId)
  } catch (error) {
    ElMessage.error('获取任务详情失败: ' + error.message)
  } finally {
    loading.value = false
  }
}

const refreshTask = () => {
  loadTaskDetail()
}

const downloadResults = async () => {
  if (!currentTask.value) return
  try {
    const blob = await tasksApi.downloadTaskResults(currentTask.value.id)
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `task_${currentTask.value.id}_results.json`
    a.click()
    window.URL.revokeObjectURL(url)
    ElMessage.success('下载成功')
  } catch (error) {
    ElMessage.error('下载失败: ' + error.message)
  }
}

const startTask = async () => {
  if (!currentTask.value) return
  try {
    await tasksApi.startTask(currentTask.value.id)
    ElMessage.success('任务已启动')
    loadTaskDetail()
  } catch (error) {
    ElMessage.error('启动任务失败: ' + error.message)
  }
}

const stopTask = async () => {
  if (!currentTask.value) return
  try {
    await tasksApi.stopTask(currentTask.value.id)
    ElMessage.success('任务已终止')
    loadTaskDetail()
  } catch (error) {
    ElMessage.error('终止任务失败: ' + error.message)
  }
}

const deleteTask = async () => {
  if (!currentTask.value) return
  try {
    await ElMessageBox.confirm('确定要删除这个任务吗？', '提示', {
      type: 'warning'
    })
    await tasksApi.deleteTask(currentTask.value.id)
    ElMessage.success('任务已删除')
    router.push('/tasks')
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除任务失败: ' + error.message)
    }
  }
}

const goBack = () => {
  router.push('/tasks')
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

const getDatasetLabel = (dataset) => {
  if (dataset.config_name && !dataset.name.includes(dataset.config_name)) {
    return `${dataset.name} (${dataset.config_name})`
  }
  return dataset.name
}

const formatErrorMessage = (errorMessage) => {
  if (!errorMessage) return ''
  
  let formatted = errorMessage
  
  const patterns = [
    /'([^']+\/[^']+_[^']+)'/g,
    /"([^"]+\/[^"]+_[^"]+)"/g,
    /\b([a-zA-Z0-9_]+\/[a-zA-Z0-9_]+_[a-zA-Z0-9_]+)\b/g,
    /\b([a-zA-Z0-9_]+_[a-zA-Z0-9_]+_[a-zA-Z0-9_]+)\b/g
  ]
  
  patterns.forEach(pattern => {
    formatted = formatted.replace(pattern, (match, taskName) => {
      if (taskName.includes('/')) {
        const parts = taskName.split('/')
        const lastPart = parts[parts.length - 1]
        if (lastPart.includes('_')) {
          const mainPart = lastPart.split('_')[0]
          if (mainPart.length > 0 && mainPart.length < 20 && /^[a-zA-Z0-9_]+$/.test(mainPart)) {
            return match.replace(taskName, mainPart)
          }
        }
      } else if (taskName.includes('_')) {
        const parts = taskName.split('_')
        if (parts.length >= 2) {
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

onMounted(() => {
  loadTaskDetail()
})
</script>

<style scoped>
.task-detail-view {
  background: white;
  padding: 20px;
  border-radius: 4px;
  min-height: calc(100vh - 100px);
}

.view-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  border-bottom: 1px solid #ebeef5;
  padding-bottom: 15px;
}

.header-left {
  display: flex;
  align-items: center;
}

.view-header h2 {
  margin: 0;
}

.header-actions {
  display: flex;
  gap: 10px;
}

.task-content {
  padding: 0 10px;
}

.progress-container {
  background: #f5f5f5;
  padding: 15px;
  border-radius: 4px;
  overflow-x: auto;
}

.progress-container pre {
  margin: 0;
}

.no-progress {
  color: #909399;
  font-style: italic;
  padding: 20px;
  background: #f9f9f9;
  border-radius: 4px;
  text-align: center;
}

.not-found {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 50px;
}
</style>
