<template>
  <div class="datasets-view">
    <div class="view-header">
      <h2>数据集管理</h2>
      <div class="header-actions">
        <el-button @click="refreshCache" :loading="refreshing">
          <el-icon><Refresh /></el-icon>
          刷新缓存
        </el-button>
        <el-button type="primary" @click="showAddDialog = true">
          <el-icon><Plus /></el-icon>
          添加数据集
        </el-button>
      </div>
    </div>

    <!-- 过滤和搜索 -->
    <div class="filters">
      <el-input
        v-model="searchKeyword"
        placeholder="搜索数据集名称、路径或描述"
        clearable
        style="width: 300px; margin-right: 10px;"
        @clear="handleSearch"
        @keyup.enter="handleSearch"
      >
        <template #prefix>
          <el-icon><Search /></el-icon>
        </template>
      </el-input>
      
      <el-select
        v-model="selectedCategory"
        placeholder="选择类别"
        clearable
        style="width: 150px; margin-right: 10px;"
        @change="handleFilter"
      >
        <el-option label="全部" value="" />
        <el-option
          v-for="category in categories"
          :key="category"
          :label="category"
          :value="category"
        />
      </el-select>
      
      <el-select
        v-model="localFilter"
        placeholder="本地状态"
        clearable
        style="width: 120px; margin-right: 10px;"
        @change="handleFilter"
      >
        <el-option label="已下载" :value="true" />
      </el-select>
      
      <el-button @click="handleSearch">搜索</el-button>
    </div>

    <el-table :data="datasets" v-loading="loading" stripe>
      <el-table-column label="数据集名称" width="350" show-overflow-tooltip>
        <template #default="{ row }">
          <span>{{ row.path }}</span>
          <span v-if="row.config_name" style="color: #909399; margin-left: 8px;">({{ row.config_name }})</span>
        </template>
      </el-table-column>
      <el-table-column prop="category" label="类别" width="150">
        <template #default="{ row }">
          <el-tag v-if="row.category" size="small">{{ row.category }}</el-tag>
          <span v-else>-</span>
        </template>
      </el-table-column>
      <el-table-column prop="num_examples" label="样本数" width="250">
        <template #default="{ row }">
          <div v-if="row.num_examples">
            <div v-for="(count, split) in row.num_examples" :key="split" style="margin-bottom: 4px;">
              <el-tag size="small" style="margin-right: 5px;">{{ split }}</el-tag>
              <span>{{ count.toLocaleString() }}</span>
            </div>
          </div>
          <span v-else>-</span>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="200" fixed="right">
        <template #default="{ row }">
          <el-button 
            size="small" 
            type="primary"
            @click="viewDataset(row)"
          >
            数据集详情
          </el-button>
          <el-button 
            size="small" 
            type="danger" 
            @click="deleteDataset(row.name)"
            :disabled="!row.is_local"
          >
            删除
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- 分页 -->
    <div class="pagination">
      <el-pagination
        v-model:current-page="currentPage"
        v-model:page-size="pageSize"
        :page-sizes="[10, 20, 50, 100]"
        :total="total"
        layout="total, sizes, prev, pager, next, jumper"
        @size-change="handleSizeChange"
        @current-change="handlePageChange"
      />
    </div>

    <!-- 添加数据集对话框 -->
    <el-dialog v-model="showAddDialog" title="添加数据集" width="600px">
      <el-form :model="datasetForm" label-width="120px">
        <el-form-item label="数据集路径" required>
          <el-input 
            v-model="datasetForm.dataset_path" 
            placeholder="例如: gsm8k, allenai/ai2_arc"
          />
        </el-form-item>
        <el-form-item label="配置名称">
          <el-input 
            v-model="datasetForm.dataset_name" 
            placeholder="多配置数据集需要指定配置名称（可选）"
          />
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="datasetForm.description" type="textarea" :rows="2" />
        </el-form-item>
        <el-form-item label="保存到本地">
          <el-switch v-model="datasetForm.save_local" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showAddDialog = false">取消</el-button>
        <el-button type="primary" @click="addDataset" :loading="adding">添加</el-button>
      </template>
    </el-dialog>

    <!-- 数据集详情对话框 -->
    <el-dialog 
      v-model="showDetailDialog" 
      title="数据集详情" 
      width="900px"
      :close-on-click-modal="false"
    >
      <div v-if="currentDataset">
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
        <el-tag v-for="split in currentDataset.splits" :key="split" style="margin-right: 5px;">
          {{ split }}
        </el-tag>
        
        <el-divider v-if="currentDataset.num_examples" />
        
        <h3 v-if="currentDataset.num_examples">样本数量</h3>
        <el-table v-if="currentDataset.num_examples" :data="numExamplesTable" border>
          <el-table-column prop="split" label="Split" />
          <el-table-column prop="count" label="数量" />
        </el-table>
        
        <el-divider v-if="currentDataset.splits && currentDataset.splits.length > 0" />
        
        <div v-if="currentDataset && currentDataset.splits && currentDataset.splits.length > 0">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <h3 style="margin: 0;">样本预览</h3>
            <el-button 
              v-if="showSamplesPreview === false"
              type="primary" 
              size="small"
              @click="handleShowSamplesPreview"
            >
              查看样本
            </el-button>
          </div>
          
          <div v-if="showSamplesPreview === true">
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
            <div v-if="loadingSamples" style="text-align: center; padding: 20px;">
              加载中...
            </div>
            <div v-else-if="samples.length === 0" style="text-align: center; padding: 20px; color: #999;">
              暂无样本数据
            </div>
            <div v-else>
              <div 
                v-for="(sample, index) in samples" 
                :key="index" 
                style="margin-bottom: 20px; border: 1px solid #dcdfe6; border-radius: 4px; padding: 15px; background: #f5f7fa;"
              >
                <div style="font-weight: bold; margin-bottom: 10px; color: #409eff;">
                  样本 {{ index + 1 }}
                </div>
                <pre 
                  v-if="sampleDisplayFormat === 'json'"
                  style="margin: 0; padding: 10px; background: white; border-radius: 4px; overflow-x: auto; font-size: 13px; line-height: 1.5;"
                >{{ formatSampleAsJson(sample) }}</pre>
                <div 
                  v-else
                  style="padding: 10px; background: white; border-radius: 4px; overflow-x: auto; font-size: 13px; line-height: 1.5;"
                  v-html="formatSampleAsMarkdown(sample)"
                ></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus, Refresh, Search } from '@element-plus/icons-vue'
import { datasetsApi } from '../api/datasets'

const datasets = ref([])
const categories = ref([])
const total = ref(0)
const currentPage = ref(1)
const pageSize = ref(20)
const loading = ref(false)
const refreshing = ref(false)
const showAddDialog = ref(false)
const showDetailDialog = ref(false)
const currentDataset = ref(null)
const selectedSplit = ref('train')
const samples = ref([])
const loadingSamples = ref(false)
const showSamplesPreview = ref(false)
const sampleDisplayFormat = ref('json')
const adding = ref(false)
const searchKeyword = ref('')
const selectedCategory = ref('')
const localFilter = ref(true)  // 默认只显示本地数据集

const datasetForm = ref({
  dataset_path: '',
  dataset_name: '',
  description: '',
  save_local: true
})

const numExamplesTable = computed(() => {
  if (!currentDataset.value?.num_examples) return []
  return Object.entries(currentDataset.value.num_examples).map(([split, count]) => ({
    split,
    count: count.toLocaleString()
  }))
})

const sampleKeys = computed(() => {
  if (samples.value.length === 0) return []
  return Object.keys(samples.value[0])
})

// 将样本格式化为 JSON 字符串
const formatSampleAsJson = (sample) => {
  try {
    return JSON.stringify(sample, null, 2)
  } catch (error) {
    return String(sample)
  }
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
  // 简单的 Markdown 转 HTML（只处理基本格式）
  let html = markdown
    // 处理代码块（JSON）
    .replace(/```json\n([\s\S]*?)```/g, (match, code) => {
      return `<pre style="background: #f5f5f5; padding: 12px; border-radius: 4px; overflow-x: auto; margin: 8px 0; border-left: 3px solid #409eff;"><code style="font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace; font-size: 13px;">${escapeHtml(code.trim())}</code></pre>`
    })
    // 处理代码块（普通文本）
    .replace(/```\n([\s\S]*?)```/g, (match, code) => {
      return `<pre style="background: #f5f5f5; padding: 12px; border-radius: 4px; overflow-x: auto; margin: 8px 0; border-left: 3px solid #67c23a;"><code style="font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace; font-size: 13px;">${escapeHtml(code.trim())}</code></pre>`
    })
    // 处理标题
    .replace(/### (.*?)\n\n/g, '<h4 style="margin: 16px 0 8px 0; color: #409eff; font-size: 16px; font-weight: 600;">$1</h4>')
    // 处理斜体
    .replace(/\*(.*?)\*/g, '<em style="color: #909399;">$1</em>')
    // 处理换行
    .replace(/\n\n/g, '</p><p style="margin: 8px 0;">')
    .replace(/\n/g, '<br>')
  
  return `<p style="margin: 8px 0;">${html}</p>`
}

// HTML 转义
const escapeHtml = (text) => {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

const loadDatasets = async () => {
  loading.value = true
  try {
    const params = {
      page: currentPage.value,
      page_size: pageSize.value
    }
    
    if (selectedCategory.value) {
      params.category = selectedCategory.value
    }
    
    // 只显示 /data 目录下的本地数据集
    if (localFilter.value !== null && localFilter.value !== '') {
      params.is_local = localFilter.value === true || localFilter.value === 'true'
    } else {
      params.is_local = true  // 默认只显示本地数据集
    }
    
    if (searchKeyword.value) {
      params.search = searchKeyword.value
    }
    
    const response = await datasetsApi.getDatasets(params)
    datasets.value = response.datasets || []
    total.value = response.total || 0
    categories.value = response.categories || []
  } catch (error) {
    ElMessage.error('加载数据集列表失败: ' + error.message)
  } finally {
    loading.value = false
  }
}

const handleSearch = () => {
  currentPage.value = 1
  loadDatasets()
}

const handleFilter = () => {
  currentPage.value = 1
  loadDatasets()
}

const handlePageChange = (page) => {
  currentPage.value = page
  loadDatasets()
}

const handleSizeChange = (size) => {
  pageSize.value = size
  currentPage.value = 1
  loadDatasets()
}

const refreshCache = async () => {
  refreshing.value = true
  try {
    await datasetsApi.refreshCache()
    ElMessage.success('缓存已刷新')
    loadDatasets()
  } catch (error) {
    ElMessage.error('刷新缓存失败: ' + error.message)
  } finally {
    refreshing.value = false
  }
}

const addDataset = async () => {
  adding.value = true
  try {
    await datasetsApi.addDataset(datasetForm.value)
    ElMessage.success('数据集添加成功')
    showAddDialog.value = false
    datasetForm.value = {
      dataset_path: '',
      dataset_name: '',
      description: '',
      save_local: true
    }
    loadDatasets()
  } catch (error) {
    ElMessage.error('添加数据集失败: ' + error.message)
  } finally {
    adding.value = false
  }
}

const viewDataset = async (dataset) => {
  try {
    // 重置样本预览状态
    showSamplesPreview.value = false
    samples.value = []
    selectedSplit.value = 'train'
    
    // 如果数据集已经有完整信息，直接使用
    if (dataset.splits && dataset.local_path) {
      currentDataset.value = dataset
      if (dataset.splits && dataset.splits.length > 0) {
        selectedSplit.value = dataset.splits[0]
      }
      showDetailDialog.value = true
    } else {
      // 否则从 API 获取详情
      currentDataset.value = await datasetsApi.getDataset(dataset.name)
      if (currentDataset.value.splits && currentDataset.value.splits.length > 0) {
        selectedSplit.value = currentDataset.value.splits[0]
      }
      showDetailDialog.value = true
    }
  } catch (error) {
    ElMessage.error('获取数据集详情失败: ' + error.message)
  }
}

const handleShowSamplesPreview = async () => {
  showSamplesPreview.value = true
  // 显示预览时自动加载样本
  if (currentDataset.value && selectedSplit.value) {
    await loadSamples()
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
    // 确保使用正确的数据集名称
    // 优先使用 name，如果没有则使用 path
    const datasetName = currentDataset.value.name || currentDataset.value.path
    console.log('加载样本 - 数据集信息:', {
      name: currentDataset.value.name,
      path: currentDataset.value.path,
      config_name: currentDataset.value.config_name,
      used_name: datasetName,
      split: selectedSplit.value
    })
    
    const result = await datasetsApi.getDatasetSamples(
      datasetName,
      selectedSplit.value,
      2  // 只加载2条样例
    )
    console.log('加载样本成功:', result)
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

const deleteDataset = async (datasetName) => {
  try {
    await ElMessageBox.confirm('确定要删除这个本地数据集吗？', '提示', {
      type: 'warning'
    })
    await datasetsApi.deleteDataset(datasetName)
    ElMessage.success('数据集已删除')
    loadDatasets()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除数据集失败: ' + error.message)
    }
  }
}

onMounted(() => {
  loadDatasets()
})
</script>

<style scoped>
.datasets-view {
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

.header-actions {
  display: flex;
  gap: 10px;
}

.filters {
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
}

.pagination {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}

h3 {
  margin: 10px 0;
  font-size: 16px;
  font-weight: 500;
}
</style>

