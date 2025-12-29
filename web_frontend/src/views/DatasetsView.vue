<template>
  <div class="datasets-view">
    <div class="view-header">
      <div class="header-content">
        <div class="header-title">
          <h2>数据集管理</h2>
          <span class="header-subtitle">管理本地和云端评测数据集</span>
        </div>
        
        <div class="header-stats">
          <div class="header-stat-item">
            <div class="stat-item-value">{{ total }}</div>
            <div class="stat-item-label">总数据集</div>
          </div>
          <div class="stat-item-divider"></div>
          <div class="header-stat-item">
            <div class="stat-item-value category">{{ categories.length }}</div>
            <div class="stat-item-label">数据类别</div>
          </div>
          <div class="stat-item-divider"></div>
          <div class="header-stat-item">
            <div class="stat-item-value local">{{ datasets.filter(d => d.is_local).length }}</div>
            <div class="stat-item-label">本地已下载</div>
          </div>
        </div>
      </div>

      <div class="header-actions">
        <el-button @click="rebuildIndex" :loading="refreshing" class="action-btn">
          <el-icon><Refresh /></el-icon>
          重建索引
        </el-button>
        <el-button type="primary" @click="showAddDialog = true" class="action-btn">
          <el-icon><Plus /></el-icon>
          添加数据集
        </el-button>
      </div>
    </div>

    <!-- 过滤和搜索 -->
    <div class="filters-container">
      <div class="filters">
        <el-input
          v-model="searchKeyword"
          placeholder="搜索数据集名称、路径或描述"
          clearable
          class="search-input"
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
          class="filter-select"
          @change="handleFilter"
        >
          <el-option label="全部类别" value="" />
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
          class="status-select"
          @change="handleFilter"
        >
          <el-option label="全部状态" :value="null" />
          <el-option label="已下载" :value="true" />
        </el-select>
        
        <el-button type="primary" plain @click="handleSearch">搜索</el-button>
      </div>
    </div>

    <!-- 数据集表格 -->
    <el-table :data="datasets" v-loading="loading" stripe class="main-table" header-cell-class-name="table-header">
      <el-table-column label="数据集名称" min-width="250">
        <template #default="{ row }">
          <div class="dataset-info">
            <div class="dataset-name">{{ row.name }}</div>
            <div class="dataset-id" v-if="row.id && row.id !== row.name">{{ row.id }}</div>
          </div>
        </template>
      </el-table-column>
      <el-table-column prop="task_name" label="评测任务名" width="200">
        <template #default="{ row }">
          <div class="task-name-cell">
            <code class="task-code">{{ row.task_name || '-' }}</code>
            <el-tooltip v-if="row.tags && (row.tags.includes('lm_eval_group') || row.tags.includes('lm_eval_task'))" content="LM-Eval 原生支持">
              <el-icon class="verified-icon"><CircleCheck /></el-icon>
            </el-tooltip>
          </div>
        </template>
      </el-table-column>
      <el-table-column prop="category" label="类别" width="150" align="center">
        <template #default="{ row }">
          <el-tag v-if="row.category" size="small" effect="plain" class="category-tag">{{ row.category }}</el-tag>
          <span v-else class="empty-text">-</span>
        </template>
      </el-table-column>
      <el-table-column label="详情" width="120" align="center">
        <template #default="{ row }">
          <el-tag v-if="row.subtasks && row.subtasks.length > 0" size="small" type="info">
            {{ row.subtasks.length }} 个子任务
          </el-tag>
          <span v-else class="empty-text">单一任务</span>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="180" fixed="right">
        <template #default="{ row }">
          <div class="action-buttons">
            <el-button size="small" type="primary" plain @click="viewDataset(row)">详情</el-button>
            <el-button 
              size="small" 
              type="danger" 
              plain
              @click="deleteDataset(row.id)"
              :disabled="!row.is_local"
            >
              删除
            </el-button>
          </div>
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
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus, Refresh, Search, CircleCheck } from '@element-plus/icons-vue'
import { datasetsApi } from '../api/datasets'

const router = useRouter()
const datasets = ref([])
const categories = ref([])
const total = ref(0)
const currentPage = ref(1)
const pageSize = ref(20)
const loading = ref(false)
const refreshing = ref(false)
const showAddDialog = ref(false)
const adding = ref(false)
const searchKeyword = ref('')
const selectedCategory = ref('')
const localFilter = ref(true)

const datasetForm = ref({
  dataset_path: '',
  dataset_name: '',
  description: '',
  save_local: true
})

const loadDatasets = async () => {
  if (loading.value) return
  
  loading.value = true
  try {
    const params = {
      page: currentPage.value,
      page_size: pageSize.value,
      groups_only: true  // 只获取 Group 级别的数据集
    }
    
    if (selectedCategory.value) {
      params.category = selectedCategory.value
    }
    
    if (localFilter.value !== null && localFilter.value !== '') {
      params.is_local = localFilter.value === true || localFilter.value === 'true'
    } else {
      params.is_local = true
    }
    
    if (searchKeyword.value) {
      params.search = searchKeyword.value
    }
    
    const response = await datasetsApi.getDatasets(params)
    datasets.value = response.datasets || []
    total.value = response.total || 0
    categories.value = response.categories || []
  } catch (error) {
    console.error('加载数据集列表失败:', error)
    ElMessage.error('加载数据集列表失败: ' + (error.message || '未知错误'))
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

const rebuildIndex = async () => {
  if (refreshing.value) {
    ElMessage.warning('正在重建中，请稍候')
    return
  }
  
  refreshing.value = true
  try {
    const result = await datasetsApi.rebuildIndex()
    ElMessage.success(result.message || '索引重建任务已启动')
    
    // 因为是异步的，我们稍等一下再刷新列表，并再次调用 loadDatasets
    setTimeout(() => {
      loadDatasets()
    }, 2000)
    
    setTimeout(() => {
      loadDatasets()
    }, 5000)
  } catch (error) {
    ElMessage.error('重建索引失败: ' + (error.message || '未知错误'))
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

const viewDataset = (dataset) => {
  const encodedId = encodeURIComponent(dataset.id)
  const routeUrl = router.resolve({
    name: 'DatasetDetail',
    params: { id: encodedId }
  })
  window.open(routeUrl.href, '_blank')
}

const deleteDataset = async (datasetId) => {
  try {
    await ElMessageBox.confirm('确定要删除这个本地数据集吗？', '提示', {
      type: 'warning'
    })
    await datasetsApi.deleteDataset(datasetId)
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
  background: transparent;
  padding: 0;
}

.view-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  background: white;
  padding: 24px 32px;
  border-radius: 12px;
  box-shadow: 0 2px 12px 0 rgba(0,0,0,0.05);
}

.header-content {
  display: flex;
  align-items: center;
  gap: 48px;
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

.header-stats {
  display: flex;
  align-items: center;
  gap: 32px;
  padding: 4px 0;
}

.header-stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 80px;
}

.stat-item-value {
  font-size: 22px;
  font-weight: 700;
  color: #303133;
  line-height: 1.2;
}

.stat-item-value.category { color: #409eff; }
.stat-item-value.local { color: #67c23a; }

.stat-item-label {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
}

.stat-item-divider {
  width: 1px;
  height: 24px;
  background-color: #ebeef5;
}

.action-btn {
  border-radius: 8px;
  padding: 10px 16px;
}

/* 过滤器 */
.filters-container {
  background: white;
  padding: 20px;
  border-radius: 12px 12px 0 0;
  margin-bottom: 0;
  border-bottom: 1px solid #f0f2f5;
}

.filters {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
}

.search-input {
  width: 320px;
}

.filter-select {
  width: 160px;
}

.status-select {
  width: 130px;
}

/* 表格样式 */
.main-table {
  border-radius: 0 0 12px 12px;
  overflow: hidden;
  box-shadow: 0 2px 12px 0 rgba(0,0,0,0.05);
}

:deep(.table-header) {
  background-color: #f5f7fa !important;
  color: #606266;
  font-weight: 600;
}

.dataset-info .dataset-name {
  font-weight: 600;
  color: #303133;
  margin-bottom: 4px;
}

.dataset-id {
  font-size: 12px;
  color: #909399;
}

.task-name-cell {
  display: flex;
  align-items: center;
  gap: 8px;
}

.task-code {
  background: #f0f2f5;
  padding: 2px 6px;
  border-radius: 4px;
  font-family: monospace;
  font-size: 13px;
  color: #444;
}

.verified-icon {
  color: #67c23a;
  font-size: 16px;
}

.category-tag {
  border-radius: 12px;
  padding: 0 12px;
}

.empty-text {
  color: #c0c4cc;
  font-size: 13px;
}

.action-buttons {
  display: flex;
  gap: 8px;
}

.pagination {
  margin-top: 24px;
  display: flex;
  justify-content: flex-end;
  background: white;
  padding: 16px;
  border-radius: 12px;
  box-shadow: 0 2px 12px 0 rgba(0,0,0,0.05);
}
</style>
