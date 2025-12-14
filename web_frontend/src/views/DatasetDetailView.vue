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
      
      <el-divider />
      
      <h3>Splits</h3>
      <div class="splits-container">
        <el-tag v-for="split in currentDataset.splits" :key="split" style="margin-right: 5px;">
          {{ split }}
        </el-tag>
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
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <el-select v-model="selectedSplit" @change="loadSamples" style="width: 200px;">
              <el-option 
                v-for="split in currentDataset.splits" 
                :key="split" 
                :label="split" 
                :value="split" 
              />
            </el-select>
            <el-radio-group v-model="sampleDisplayFormat" size="small">
              <el-radio-button label="json">JSON</el-radio-button>
              <el-radio-button label="markdown">Markdown</el-radio-button>
            </el-radio-group>
          </div>
          <div v-if="loadingSamples" style="text-align: center; padding: 40px;">
            <el-icon class="is-loading" style="font-size: 24px;"><Loading /></el-icon>
            <div style="margin-top: 10px;">正在加载样本...</div>
          </div>
          <div v-else-if="samples.length === 0" style="text-align: center; padding: 40px; color: #999; background: #f9f9f9; border-radius: 4px;">
            暂无样本数据
          </div>
          <div v-else>
            <div 
              v-for="(sample, index) in samples" 
              :key="index" 
              class="sample-item"
            >
              <div class="sample-header">
                样本 {{ index + 1 }}
              </div>
              <pre 
                v-if="sampleDisplayFormat === 'json'"
                class="sample-content-json"
              >{{ formatSampleAsJson(sample) }}</pre>
              <div 
                v-else
                class="sample-content-markdown"
                v-html="formatSampleAsMarkdown(sample)"
              ></div>
            </div>
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

const datasetId = route.params.id
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
  let markdown = ''
  for (const [key, value] of Object.entries(sample)) {
    markdown += `### ${key}\n\n`
    if (value === null || value === undefined) {
      markdown += `*null*\n\n`
    } else if (typeof value === 'object') {
      markdown += `\`\`\`json\n${JSON.stringify(value, null, 2)}\n\`\`\`\n\n`
    } else if (typeof value === 'string' && (value.includes('\n') || value.length > 100)) {
      markdown += `\`\`\`\n${value}\n\`\`\`\n\n`
    } else {
      markdown += `${value}\n\n`
    }
  }
  // 简单的 Markdown 转 HTML
  let html = markdown
    .replace(/```json\n([\s\S]*?)```/g, (match, code) => {
      return `<pre style="background: #f5f5f5; padding: 12px; border-radius: 4px; overflow-x: auto; margin: 8px 0; border-left: 3px solid #409eff;"><code style="font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace; font-size: 13px;">${escapeHtml(code.trim())}</code></pre>`
    })
    .replace(/```\n([\s\S]*?)```/g, (match, code) => {
      return `<pre style="background: #f5f5f5; padding: 12px; border-radius: 4px; overflow-x: auto; margin: 8px 0; border-left: 3px solid #67c23a;"><code style="font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace; font-size: 13px;">${escapeHtml(code.trim())}</code></pre>`
    })
    .replace(/### (.*?)\n\n/g, '<h4 style="margin: 16px 0 8px 0; color: #409eff; font-size: 16px; font-weight: 600;">$1</h4>')
    .replace(/\*(.*?)\*/g, '<em style="color: #909399;">$1</em>')
    .replace(/\n\n/g, '</p><p style="margin: 8px 0;">')
    .replace(/\n/g, '<br>')
  
  return `<p style="margin: 8px 0;">${html}</p>`
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
      2  // 只加载2条样例
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

const refreshDataset = () => {
  loadDatasetDetail()
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
}

.sample-item {
  margin-bottom: 20px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  padding: 15px;
  background: #f5f7fa;
}

.sample-header {
  font-weight: bold;
  margin-bottom: 10px;
  color: #409eff;
}

.sample-content-json {
  margin: 0;
  padding: 10px;
  background: white;
  border-radius: 4px;
  overflow-x: auto;
  font-size: 13px;
  line-height: 1.5;
}

.sample-content-markdown {
  padding: 10px;
  background: white;
  border-radius: 4px;
  overflow-x: auto;
  font-size: 13px;
  line-height: 1.5;
}

.not-found {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 50px;
}
</style>
