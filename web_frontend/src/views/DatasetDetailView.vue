<template>
  <div class="dataset-detail-view" v-loading="loading">
    <div class="view-header">
      <div class="header-left">
        <el-button @click="goBack" icon="ArrowLeft" circle style="margin-right: 15px;" />
        <h2>{{ currentDataset?.name || '加载中...' }}</h2>
      </div>
      <div class="header-actions">
        <el-button type="primary" @click="refreshDataset">刷新</el-button>
      </div>
    </div>

    <div v-if="currentDataset" class="dataset-content">
      <el-descriptions :column="2" border>
        <el-descriptions-item label="数据集名称">{{ currentDataset.name }}</el-descriptions-item>
        <el-descriptions-item label="路径">{{ currentDataset.path }}</el-descriptions-item>
        <el-descriptions-item label="配置名称">
          {{ currentDataset.config_name || '-' }}
        </el-descriptions-item>
        <el-descriptions-item label="类别">
          <el-tag v-if="currentDataset.category" size="small">{{ currentDataset.category }}</el-tag>
          <span v-else>-</span>
        </el-descriptions-item>
        <el-descriptions-item label="本地状态">
          <el-tag :type="currentDataset.is_local ? 'success' : 'info'">
            {{ currentDataset.is_local ? '已下载' : '未下载' }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="本地路径">
          {{ currentDataset.local_path || '-' }}
        </el-descriptions-item>
        <el-descriptions-item label="描述" :span="2">
          {{ currentDataset.description || '-' }}
        </el-descriptions-item>
      </el-descriptions>
      
      <!-- 子任务列表（如果有） -->
      <div v-if="currentDataset.subtasks && currentDataset.subtasks.length > 0">
        <el-divider />
        <h3>子任务 ({{ currentDataset.subtasks.length }})</h3>
        <p style="color: #666; font-size: 13px; margin-bottom: 15px;">
          选择此数据集进行评测时，会自动测试以下所有子任务并汇总结果
        </p>
        <el-table :data="subtasksTable" border style="width: 100%; max-width: 800px;">
          <el-table-column prop="name" label="子任务名称" min-width="150" />
          <el-table-column prop="task_name" label="评测任务名" min-width="150">
            <template #default="{ row }">
              <span>{{ row.task_name }}</span>
              <el-tag v-if="row.verified" size="small" type="success" style="margin-left: 5px;">✓</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="path" label="本地路径" min-width="200" show-overflow-tooltip />
        </el-table>
      </div>
      
      <el-divider />
      
      <h3>Splits</h3>
      <div class="splits-container">
        <el-tag v-for="split in currentDataset.splits" :key="split" style="margin-right: 5px;">
          {{ split }}
        </el-tag>
        <span v-if="!currentDataset.splits || currentDataset.splits.length === 0" style="color: #999;">
          暂无 Splits 信息
        </span>
      </div>
      
      <el-divider v-if="currentDataset.num_examples" />
      
      <h3 v-if="currentDataset.num_examples">样本数量</h3>
      <el-table v-if="currentDataset.num_examples" :data="numExamplesTable" border style="width: 100%; max-width: 600px;">
        <el-table-column prop="split" label="Split" />
        <el-table-column prop="count" label="数量" />
      </el-table>

      <el-divider />
      
      <h3>README</h3>
      <div v-loading="loadingReadme" class="readme-container">
         <div v-if="readmeContent" 
              v-html="formatMarkdown(readmeContent)" 
              class="readme-content">
         </div>
         <div v-else style="color: #999; font-style: italic; padding: 20px;">
            暂无 README 内容
         </div>
      </div>
      
      <el-divider v-if="currentDataset.splits && currentDataset.splits.length > 0" />
      
      <div v-if="currentDataset && currentDataset.splits && currentDataset.splits.length > 0">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
          <h3 style="margin: 0;">样本预览</h3>
        </div>
        
        <div class="samples-preview">
          <div class="preview-toolbar">
            <div class="toolbar-left">
              <span class="toolbar-label">数据划分:</span>
              <el-select v-model="selectedSplit" @change="loadSamples" size="default" style="width: 150px;">
                <el-option 
                  v-for="split in currentDataset.splits" 
                  :key="split" 
                  :label="split" 
                  :value="split" 
                />
              </el-select>
            </div>
            <div class="toolbar-right">
              <el-radio-group v-model="sampleDisplayFormat" size="small">
                <el-radio-button label="json">JSON</el-radio-button>
                <el-radio-button label="markdown">列表</el-radio-button>
              </el-radio-group>
            </div>
          </div>

          <div v-if="loadingSamples" class="loading-container">
            <el-icon class="is-loading"><Loading /></el-icon>
            <span>正在加载样本数据...</span>
          </div>
          
          <div v-else-if="samples.length === 0" class="empty-samples">
            <el-empty description="该 Split 下暂无样本数据" />
          </div>

          <div v-else class="samples-list">
            <el-card 
              v-for="(sample, index) in samples" 
              :key="index" 
              class="sample-card"
              shadow="never"
            >
              <template #header>
                <div class="sample-card-header">
                  <el-tag size="small" effect="dark" type="info">#{{ index + 1 }}</el-tag>
                  <span class="sample-title">数据样例</span>
                </div>
              </template>
              
              <div v-if="sampleDisplayFormat === 'json'" class="json-wrap">
                <pre class="sample-content-json">{{ formatSampleAsJson(sample) }}</pre>
              </div>
              
              <div v-else class="fields-wrap">
                <div v-for="(value, key) in sample" :key="key" class="field-item">
                  <div class="field-label">{{ key }}</div>
                  <div class="field-value">{{ formatValue(value) }}</div>
                </div>
              </div>
            </el-card>
          </div>
        </div>
      </div>
    </div>
    <div v-else-if="!loading" class="not-found">
      <el-empty description="未找到数据集信息" />
      <el-button type="primary" @click="goBack">返回列表</el-button>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { ArrowLeft, Loading } from '@element-plus/icons-vue'
import { datasetsApi } from '../api/datasets'

const route = useRoute()
const router = useRouter()

// ID 可能包含 /（如 arc/ARC-Challenge），需要解码
const datasetId = decodeURIComponent(route.params.id)
const loading = ref(false)
const currentDataset = ref(null)
const readmeContent = ref('')
const loadingReadme = ref(false)
const selectedSplit = ref('train')
const samples = ref([])
const loadingSamples = ref(false)
const sampleDisplayFormat = ref('json')

const numExamplesTable = computed(() => {
  if (!currentDataset.value?.num_examples) return []
  return Object.entries(currentDataset.value.num_examples).map(([split, count]) => ({
    split,
    count: count.toLocaleString()
  }))
})

// 子任务表格数据
const subtasksTable = computed(() => {
  if (!currentDataset.value?.subtasks) return []
  
  const basePath = currentDataset.value.local_path || ''
  const parentName = currentDataset.value.name || ''
  
  return currentDataset.value.subtasks.map(subtaskName => {
    // 生成 task_name（转换格式：ARC-Challenge -> arc_challenge）
    const taskName = subtaskName.toLowerCase().replace(/-/g, '_')
    
    return {
      name: subtaskName,
      task_name: taskName,
      path: basePath ? `${basePath}/${subtaskName}` : subtaskName,
      verified: true  // 假设已经验证过（后端会验证）
    }
  })
})

// 将样本格式化为 JSON 字符串
const formatSampleAsJson = (sample) => {
  try {
    return JSON.stringify(sample, null, 2)
  } catch (error) {
    return String(sample)
  }
}

// HTML 转义
const escapeHtml = (text) => {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

// 将样本格式化为 Markdown 格式
const formatSampleAsMarkdown = (sample) => {
  // 弃用的方法，逻辑已移至模板
  return ""
}

const formatValue = (val) => {
  if (val === null || val === undefined) return '-'
  if (typeof val === 'object') {
    return JSON.stringify(val, null, 2)
  }
  return String(val)
}

// 简单的 Markdown 格式化（用于 README）
const formatMarkdown = (text) => {
  if (!text) return ''
  let html = text
    .replace(/^# (.*$)/gm, '<h1 style="font-size: 24px; font-weight: bold; margin: 24px 0 16px; border-bottom: 1px solid #eaecef; padding-bottom: .3em;">$1</h1>')
    .replace(/^## (.*$)/gm, '<h2 style="font-size: 20px; font-weight: bold; margin: 24px 0 16px; border-bottom: 1px solid #eaecef; padding-bottom: .3em;">$1</h2>')
    .replace(/^### (.*$)/gm, '<h3 style="font-size: 18px; font-weight: bold; margin: 20px 0 16px;">$1</h3>')
    .replace(/```([\s\S]*?)```/g, '<pre style="background: #f6f8fa; padding: 16px; border-radius: 6px; overflow-x: auto; line-height: 1.45;"><code>$1</code></pre>')
    .replace(/`([^`]+)`/g, '<code style="background: rgba(175,184,193,0.2); padding: .2em .4em; border-radius: 6px; font-size: 85%;">$1</code>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" style="color: #0969da; text-decoration: none;">$1</a>')
    .replace(/^\s*[-*]\s+(.*)$/gm, '<li style="margin-left: 20px;">$1</li>')
    .replace(/\n/g, '<br>')
  
  return html
}

const loadDatasetDetail = async () => {
  if (!datasetId) return
  
  loading.value = true
  try {
    currentDataset.value = await datasetsApi.getDataset(datasetId)
    
    // 初始化选中的 split
    if (currentDataset.value.splits && currentDataset.value.splits.length > 0) {
      selectedSplit.value = currentDataset.value.splits[0]
      // 自动加载样本
      loadSamples()
    }
    
    // 加载 README
    loadReadme(currentDataset.value.id)
  } catch (error) {
    ElMessage.error('获取数据集详情失败: ' + error.message)
  } finally {
    loading.value = false
  }
}

const loadReadme = async (id) => {
  loadingReadme.value = true
  readmeContent.value = ''
  try {
    const result = await datasetsApi.getDatasetReadme(id)
    if (result && result.content) {
      readmeContent.value = result.content
    }
  } catch (error) {
    console.warn('获取 README 失败:', error)
  } finally {
    loadingReadme.value = false
  }
}

const loadSamples = async () => {
  if (!currentDataset.value || !selectedSplit.value) {
    samples.value = []
    return
  }
  loadingSamples.value = true
  samples.value = []
  try {
    const datasetName = currentDataset.value.name || currentDataset.value.path
    const result = await datasetsApi.getDatasetSamples(
      datasetName,
      selectedSplit.value,
      5  // 加载 5 条样例
    )
    samples.value = Array.isArray(result) ? result : []
  } catch (error) {
    console.error('加载样本失败:', error)
    const errorMsg = error.detail || error.message || '未知错误'
    ElMessage.error('加载样本失败: ' + errorMsg)
    samples.value = []
  } finally {
    loadingSamples.value = false
  }
}

const refreshDataset = async () => {
  loading.value = true
  try {
    await datasetsApi.refreshCache()
    await loadDatasetDetail()
    ElMessage.success('详情已刷新')
  } catch (error) {
    ElMessage.error('刷新失败')
    loadDatasetDetail()
  } finally {
    loading.value = false
  }
}

const goBack = () => {
  router.push('/datasets')
}

onMounted(() => {
  loadDatasetDetail()
})
</script>

<style scoped>
.dataset-detail-view {
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

.dataset-content {
  padding: 0 10px;
}

.splits-container {
  margin: 10px 0;
}

.readme-container {
  margin-top: 15px;
}

.readme-content {
  background: #ffffff;
  padding: 20px;
  border: 1px solid #e1e4e8;
  border-radius: 6px;
  overflow-x: auto;
  line-height: 1.6;
}

.samples-preview {
  margin-top: 15px;
  background: #fcfcfc;
  padding: 15px;
  border-radius: 8px;
  border: 1px solid #f0f0f0;
}

.preview-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 15px;
  border-bottom: 1px solid #ebeef5;
}

.toolbar-left, .toolbar-right {
  display: flex;
  align-items: center;
  gap: 10px;
}

.toolbar-label {
  font-size: 14px;
  color: #606266;
}

.loading-container {
  text-align: center;
  padding: 60px 0;
  color: #909399;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}

.loading-container .el-icon {
  font-size: 28px;
}

.samples-list {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.sample-card {
  border: 1px solid #ebeef5;
}

.sample-card-header {
  display: flex;
  align-items: center;
  gap: 10px;
}

.sample-title {
  font-weight: bold;
  font-size: 14px;
}

.json-wrap {
  background: #282c34;
  padding: 15px;
  border-radius: 4px;
}

.sample-content-json {
  margin: 0;
  font-family: 'Fira Code', 'Monaco', 'Menlo', monospace;
  font-size: 12px;
  color: #abb2bf;
  white-space: pre-wrap;
  word-break: break-all;
}

.fields-wrap {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.field-item {
  border-bottom: 1px dashed #f0f2f5;
  padding-bottom: 8px;
}

.field-item:last-child {
  border-bottom: none;
  padding-bottom: 0;
}

.field-label {
  font-size: 12px;
  color: #909399;
  font-weight: bold;
  margin-bottom: 4px;
  text-transform: uppercase;
}

.field-value {
  font-size: 14px;
  color: #303133;
  line-height: 1.6;
  white-space: pre-wrap;
}

.not-found {
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 50px;
}
</style>
