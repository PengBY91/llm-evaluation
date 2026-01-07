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
        <el-descriptions-item label="模型类型">
          {{ currentTask.model || '-' }}
        </el-descriptions-item>
        <el-descriptions-item label="模型标识">
          {{ currentTask.model_name || currentTask.model_args?.model || '-' }}
        </el-descriptions-item>
        <el-descriptions-item label="API 端点" v-if="currentTask.model_args?.base_url">
          <el-link :href="currentTask.model_args.base_url" target="_blank" type="primary">{{ currentTask.model_args.base_url }}</el-link>
        </el-descriptions-item>
        <el-descriptions-item label="创建时间">{{ formatTime(currentTask.created_at) }}</el-descriptions-item>
        <el-descriptions-item label="更新时间">{{ formatTime(currentTask.updated_at) }}</el-descriptions-item>
        <el-descriptions-item label="执行时长" v-if="executionTime">
          <el-tag type="info">{{ executionTime }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="并发数" v-if="currentTask.model_args?.num_concurrent">
          {{ currentTask.model_args.num_concurrent }}
        </el-descriptions-item>
      </el-descriptions>
      
      <el-divider />
      
      <!-- 模型参数详情 -->
      <div v-if="currentTask.model_args && Object.keys(currentTask.model_args).length > 0" style="margin-bottom: 20px;">
        <h3>模型参数</h3>
        <el-descriptions border :column="3" size="small">
          <el-descriptions-item 
            v-for="(value, key) in filteredModelArgs" 
            :key="key" 
            :label="formatParamName(key)"
          >
            {{ formatParamValue(key, value) }}
          </el-descriptions-item>
        </el-descriptions>
      </div>
      
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
      
      <h3>评测结果详情</h3>
      <div v-if="currentTask.results && currentTask.results.results" class="results-detail">
        <!-- 总体结果摘要 -->
        <el-card class="box-card" style="margin-bottom: 20px;">
          <template #header>
            <div class="card-header">
              <span>总体摘要</span>
            </div>
          </template>
          <el-table :data="resultsSummaryTable" style="width: 100%" border stripe>
            <el-table-column prop="task" label="任务" width="200" show-overflow-tooltip fixed />
            <el-table-column 
              v-for="metric in summaryMetrics" 
              :key="metric" 
              :prop="metric" 
              :label="formatMetricName(metric)"
              width="180"
            >
              <template #default="{ row }">
                <div style="display: flex; align-items: center; gap: 8px;">
                  <span>{{ formatMetricValue(row[metric]) }}</span>
                  <el-tag v-if="isPercentageMetric(metric)" size="small" :type="getMetricTagType(row[metric])">
                    {{ formatPercentage(row[metric]) }}
                  </el-tag>
                </div>
              </template>
            </el-table-column>
            <el-table-column prop="samples" label="样本数" width="100" />
          </el-table>
        </el-card>

        <!-- 详细结果表格 -->
        <el-collapse v-model="activeResultNames">
          <el-collapse-item title="详细指标" name="1">
            <div v-for="(taskResult, taskName) in currentTask.results.results" :key="taskName" style="margin-bottom: 15px;">
              <h4 style="margin: 10px 0;">{{ taskName }}</h4>
              <el-descriptions border :column="3" size="small">
                <el-descriptions-item 
                  v-for="(value, key) in filterMetrics(taskResult)" 
                  :key="key" 
                  :label="key"
                >
                  {{ formatMetricValue(value) }}
                </el-descriptions-item>
              </el-descriptions>
            </div>
          </el-collapse-item>
          
          <el-collapse-item title="配置信息" name="2">
             <div class="config-info">
                <pre v-if="currentTask.results.config">{{ JSON.stringify(currentTask.results.config, null, 2) }}</pre>
                <div v-else>无配置信息</div>
             </div>
          </el-collapse-item>

          <el-collapse-item title="样本预览 (前5条)" name="3" v-if="currentTask.results.samples_preview">
             <div v-for="(samples, taskName) in currentTask.results.samples_preview" :key="taskName" style="margin-bottom: 20px;">
                <h4 style="margin: 10px 0; color: #409eff;">{{ taskName }}</h4>
                <div v-for="(sample, idx) in samples" :key="idx" class="sample-item">
                   <div style="font-weight: bold; margin-bottom: 5px;">Sample {{ idx + 1 }}</div>
                   <div style="display: grid; grid-template-columns: 100px 1fr; gap: 10px; font-size: 13px;">
                      <div style="font-weight: 500;">Input:</div>
                      <div style="white-space: pre-wrap; background: #f9f9f9; padding: 5px;">{{ sample.doc || sample.arguments?.[0]?.[0] || '-' }}</div>
                      
                      <div style="font-weight: 500;">Target:</div>
                      <div style="white-space: pre-wrap; background: #f0f9eb; padding: 5px;">{{ sample.target || sample.arguments?.[0]?.[1] || '-' }}</div>
                      
                      <div style="font-weight: 500;">Output:</div>
                      <div style="white-space: pre-wrap; background: #fdf6ec; padding: 5px;">{{ sample.resps?.[0]?.[0] || sample.resps?.[0] || '-' }}</div>
                      
                      <div style="font-weight: 500;" v-if="sample.acc !== undefined">Accuracy:</div>
                      <div v-if="sample.acc !== undefined">{{ sample.acc }}</div>
                   </div>
                </div>
             </div>
          </el-collapse-item>
        </el-collapse>
      </div>
      <div v-else class="no-results">
        <el-empty description="暂无详细评测结果" :image-size="100">
           <template #description>
              <p v-if="currentTask.status === 'completed'">结果文件可能已丢失或格式不正确</p>
              <p v-else>任务尚未完成，请等待评测结束</p>
           </template>
        </el-empty>
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
import { ref, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { ArrowLeft } from '@element-plus/icons-vue'
import { tasksApi } from '../api/tasks'

const route = useRoute()
const router = useRouter()

const taskId = route.params.id
const loading = ref(false)
const currentTask = ref(null)
const activeResultNames = ref(['1'])

const resultsSummaryTable = computed(() => {
  if (!currentTask.value?.results?.results) return []
  
  const results = currentTask.value.results.results
  return Object.keys(results).map(taskName => {
    const taskResult = results[taskName]
    // 提取主要指标
    const row = { task: taskName }
    
    // 自动查找并添加指标（忽略 alias 和 stderr）
    Object.keys(taskResult).forEach(key => {
      if (!key.endsWith(',stderr') && !key.endsWith(',none') && key !== 'alias') {
        row[key] = taskResult[key]
      }
    })
    
    // 如果有样本数信息
    if (currentTask.value.results.n_shot) {
       row.samples = currentTask.value.results.n_shot[taskName] || '-'
    } else {
       row.samples = '-'
    }
    
    return row
  })
})

const summaryMetrics = computed(() => {
  if (resultsSummaryTable.value.length === 0) return []
  // 从第一行提取所有指标键（排除 task 和 samples）
  const firstRow = resultsSummaryTable.value[0]
  return Object.keys(firstRow).filter(key => key !== 'task' && key !== 'samples')
})

// 计算执行时长
const executionTime = computed(() => {
  if (!currentTask.value) return null
  const created = new Date(currentTask.value.created_at)
  const updated = new Date(currentTask.value.updated_at)
  const diffMs = updated - created
  
  if (diffMs < 1000) return `${diffMs}ms`
  if (diffMs < 60000) return `${(diffMs / 1000).toFixed(1)}秒`
  if (diffMs < 3600000) return `${(diffMs / 60000).toFixed(1)}分钟`
  return `${(diffMs / 3600000).toFixed(2)}小时`
})

// 过滤模型参数（排除敏感信息）
const filteredModelArgs = computed(() => {
  if (!currentTask.value?.model_args) return {}
  const filtered = {}
  const excludeKeys = ['api_key', 'tokenizer'] // 排除敏感或冗长的参数
  
  Object.keys(currentTask.value.model_args).forEach(key => {
    if (!excludeKeys.includes(key)) {
      filtered[key] = currentTask.value.model_args[key]
    }
  })
  return filtered
})

const formatParamName = (key) => {
  const nameMap = {
    'model': '模型名称',
    'base_url': 'API地址',
    'num_concurrent': '并发数',
    'max_length': '最大长度',
    'batch_size': '批处理大小',
    'temperature': '温度',
    'top_p': 'Top P',
    'aiohttp_client_timeout': '超时时间'
  }
  return nameMap[key] || key
}

const formatParamValue = (key, value) => {
  if (value === null || value === undefined) return '-'
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

const isPercentageMetric = (metric) => {
  // 判断是否是百分比类型的指标（通常是0-1之间的小数）
  const percentageMetrics = ['acc', 'accuracy', 'f1', 'exact_match', 'em']
  return percentageMetrics.some(pm => metric.toLowerCase().includes(pm))
}

const formatPercentage = (value) => {
  if (typeof value !== 'number') return ''
  return `${(value * 100).toFixed(2)}%`
}

const getMetricTagType = (value) => {
  if (typeof value !== 'number') return 'info'
  if (value >= 0.9) return 'success'
  if (value >= 0.7) return 'warning'
  return 'danger'
}

const formatMetricName = (key) => {
  // 简单格式化指标名称
  if (key.includes(',')) return key.split(',')[0]
  return key
}

const formatMetricValue = (value) => {
  if (typeof value === 'number') {
    // 如果是小数，保留4位小数；如果是整数或百分比，适当处理
    if (Math.abs(value) < 0.0001) return value.toExponential(2)
    return parseFloat(value.toFixed(4))
  }
  return value
}

const filterMetrics = (metrics) => {
  // 过滤掉不重要的指标展示
  const filtered = {}
  Object.keys(metrics).forEach(key => {
    if (key !== 'alias') {
       filtered[key] = metrics[key]
    }
  })
  return filtered
}

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

.results-detail {
  margin-top: 15px;
}

.config-info {
  background: #f5f7fa;
  padding: 15px;
  border-radius: 4px;
  max-height: 400px;
  overflow: auto;
}

.sample-item {
  background: white;
  border: 1px solid #ebeef5;
  border-radius: 4px;
  padding: 15px;
  margin-bottom: 10px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.no-results {
  padding: 20px 0;
}
</style>
