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
        <el-option label="全部" value="" />
        <el-option label="已下载" :value="true" />
        <el-option label="未下载" :value="false" />
      </el-select>
      
      <el-button @click="handleSearch">搜索</el-button>
    </div>

    <el-table :data="datasets" v-loading="loading" stripe>
      <el-table-column prop="name" label="数据集名称" width="200" />
      <el-table-column prop="path" label="路径" width="200" />
      <el-table-column prop="category" label="类别" width="120">
        <template #default="{ row }">
          <el-tag v-if="row.category" size="small">{{ row.category }}</el-tag>
          <span v-else>-</span>
        </template>
      </el-table-column>
      <el-table-column prop="config_name" label="配置名称" width="150" />
      <el-table-column prop="description" label="描述" width="200" show-overflow-tooltip />
      <el-table-column prop="is_local" label="本地状态" width="100">
        <template #default="{ row }">
          <el-tag :type="row.is_local ? 'success' : 'info'">
            {{ row.is_local ? '已下载' : '未下载' }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="splits" label="Splits" width="150">
        <template #default="{ row }">
          <el-tag v-for="split in row.splits" :key="split" size="small" style="margin-right: 5px;">
            {{ split }}
          </el-tag>
          <span v-if="!row.splits || row.splits.length === 0">-</span>
        </template>
      </el-table-column>
      <el-table-column prop="num_examples" label="样本数" width="150">
        <template #default="{ row }">
          <div v-if="row.num_examples">
            <div v-for="(count, split) in row.num_examples" :key="split">
              {{ split }}: {{ count.toLocaleString() }}
            </div>
          </div>
          <span v-else>-</span>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="200" fixed="right">
        <template #default="{ row }">
          <el-button size="small" @click="viewDataset(row)">详情</el-button>
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
    <el-dialog v-model="showDetailDialog" title="数据集详情" width="800px">
      <div v-if="currentDataset">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="数据集名称">{{ currentDataset.name }}</el-descriptions-item>
          <el-descriptions-item label="路径">{{ currentDataset.path }}</el-descriptions-item>
          <el-descriptions-item label="配置名称">
            {{ currentDataset.config_name || '-' }}
          </el-descriptions-item>
          <el-descriptions-item label="本地状态">
            <el-tag :type="currentDataset.is_local ? 'success' : 'info'">
              {{ currentDataset.is_local ? '已下载' : '未下载' }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="本地路径" :span="2">
            {{ currentDataset.local_path || '-' }}
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
        
        <el-divider />
        
        <h3>样本预览</h3>
        <el-select v-model="selectedSplit" @change="loadSamples" style="margin-bottom: 10px;">
          <el-option 
            v-for="split in currentDataset.splits" 
            :key="split" 
            :label="split" 
            :value="split" 
          />
        </el-select>
        <el-table :data="samples" border max-height="400">
          <el-table-column 
            v-for="key in sampleKeys" 
            :key="key" 
            :prop="key" 
            :label="key"
            :show-overflow-tooltip="true"
          />
        </el-table>
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
const adding = ref(false)
const searchKeyword = ref('')
const selectedCategory = ref('')
const localFilter = ref('')

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
    
    if (localFilter.value !== '') {
      params.is_local = localFilter.value === true || localFilter.value === 'true'
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
    currentDataset.value = await datasetsApi.getDataset(dataset.name)
    if (currentDataset.value.splits && currentDataset.value.splits.length > 0) {
      selectedSplit.value = currentDataset.value.splits[0]
      await loadSamples()
    }
    showDetailDialog.value = true
  } catch (error) {
    ElMessage.error('获取数据集详情失败: ' + error.message)
  }
}

const loadSamples = async () => {
  if (!currentDataset.value || !selectedSplit.value) return
  try {
    samples.value = await datasetsApi.getDatasetSamples(
      currentDataset.value.name,
      selectedSplit.value,
      10
    )
  } catch (error) {
    ElMessage.error('加载样本失败: ' + error.message)
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

