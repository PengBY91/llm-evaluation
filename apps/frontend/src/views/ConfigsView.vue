<template>
  <div class="configs-view">
    <div class="view-header">
      <h2>任务配置管理</h2>
      <el-button type="primary" @click="showCreateDialog = true">
        <el-icon><Plus /></el-icon>
        新建配置
      </el-button>
    </div>

    <el-table :data="configs" v-loading="loading" stripe>
      <el-table-column prop="name" label="配置名称" width="200" />
      <el-table-column prop="description" label="描述" width="200" />
      <el-table-column prop="model" label="模型" width="150" />
      <el-table-column prop="tasks" label="评测任务" width="250">
        <template #default="{ row }">
          <el-tag v-for="task in row.tasks" :key="task" size="small" style="margin-right: 5px;">
            {{ task }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="created_at" label="创建时间" width="180">
        <template #default="{ row }">
          {{ formatTime(row.created_at) }}
        </template>
      </el-table-column>
      <el-table-column label="操作" width="200" fixed="right">
        <template #default="{ row }">
          <el-button size="small" @click="editConfig(row)">编辑</el-button>
          <el-button size="small" type="danger" @click="deleteConfig(row.id)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- 创建/编辑配置对话框 -->
    <el-dialog 
      v-model="showCreateDialog" 
      :title="editingConfig ? '编辑配置' : '新建配置'" 
      width="800px"
    >
      <el-form :model="configForm" label-width="120px">
        <el-form-item label="配置名称" required>
          <el-input v-model="configForm.name" placeholder="请输入配置名称" />
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="configForm.description" type="textarea" :rows="2" />
        </el-form-item>
        <el-form-item label="模型类型" required>
          <el-select v-model="configForm.model" placeholder="请选择模型类型">
            <el-option label="OpenAI Chat Completions" value="openai-chat-completions" />
            <el-option label="OpenAI Completions" value="openai-completions" />
            <el-option label="HuggingFace" value="hf" />
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
            v-model="configForm.tasks" 
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
          <el-input-number v-model="configForm.num_fewshot" :min="0" />
        </el-form-item>
        <el-form-item label="Batch Size">
          <el-input-number v-model="configForm.batch_size" :min="1" />
        </el-form-item>
        <el-form-item label="设备">
          <el-input v-model="configForm.device" placeholder="例如: cuda:0, cpu" />
        </el-form-item>
        <el-form-item label="限制样本数">
          <el-input-number v-model="configForm.limit" :min="1" />
        </el-form-item>
        <el-form-item label="应用Chat模板">
          <el-switch v-model="configForm.apply_chat_template" />
        </el-form-item>
        <el-form-item label="记录样本">
          <el-switch v-model="configForm.log_samples" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="cancelEdit">取消</el-button>
        <el-button type="primary" @click="saveConfig">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'
import { configsApi } from '../api/configs'
import { tasksApi } from '../api/tasks'
import { datasetsApi } from '../api/datasets'

const configs = ref([])
const availableTasks = ref([])
const loading = ref(false)
const showCreateDialog = ref(false)
const editingConfig = ref(null)
const modelArgsStr = ref('{}')

const configForm = ref({
  name: '',
  description: '',
  model: 'openai-chat-completions',
  model_args: {},
  tasks: [],
  num_fewshot: null,
  batch_size: null,
  device: null,
  limit: null,
  log_samples: true,
  apply_chat_template: false,
  gen_kwargs: null
})

const loadConfigs = async () => {
  loading.value = true
  try {
    configs.value = await configsApi.getConfigs()
  } catch (error) {
    ElMessage.error('加载配置列表失败: ' + error.message)
  } finally {
    loading.value = false
  }
}

const editConfig = (config) => {
  editingConfig.value = config
  configForm.value = { ...config }
  modelArgsStr.value = JSON.stringify(config.model_args || {}, null, 2)
  showCreateDialog.value = true
}

const cancelEdit = () => {
  editingConfig.value = null
  showCreateDialog.value = false
  resetForm()
}

const resetForm = () => {
  configForm.value = {
    name: '',
    description: '',
    model: 'openai-chat-completions',
    model_args: {},
    tasks: [],
    num_fewshot: null,
    batch_size: null,
    device: null,
    limit: null,
    log_samples: true,
    apply_chat_template: false,
    gen_kwargs: null
  }
  modelArgsStr.value = '{}'
}

const saveConfig = async () => {
  try {
    // 解析模型参数
    try {
      configForm.value.model_args = JSON.parse(modelArgsStr.value || '{}')
    } catch (e) {
      ElMessage.error('模型参数格式错误，请输入有效的JSON')
      return
    }
    
    if (editingConfig.value) {
      await configsApi.updateConfig(editingConfig.value.id, configForm.value)
      ElMessage.success('配置更新成功')
    } else {
      await configsApi.createConfig(configForm.value)
      ElMessage.success('配置创建成功')
    }
    
    showCreateDialog.value = false
    resetForm()
    editingConfig.value = null
    loadConfigs()
  } catch (error) {
    ElMessage.error('保存配置失败: ' + error.message)
  }
}

const deleteConfig = async (configId) => {
  try {
    await ElMessageBox.confirm('确定要删除这个配置吗？', '提示', {
      type: 'warning'
    })
    await configsApi.deleteConfig(configId)
    ElMessage.success('配置已删除')
    loadConfigs()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除配置失败: ' + error.message)
    }
  }
}

const formatTime = (timeStr) => {
  if (!timeStr) return ''
  return new Date(timeStr).toLocaleString('zh-CN')
}

const loadAvailableTasks = async () => {
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
      console.warn('/data 目录下没有找到数据集')
      return
    }
    
    // 将数据集转换为任务名称
    // 优先使用后端返回的 name 字段（这是从 TaskManager 获取的正确任务名称）
    // 如果没有 name 字段，才根据路径构造
    const taskNames = allDatasets
      .filter(dataset => dataset && (dataset.name || dataset.path))  // 过滤无效数据
      .map(dataset => {
        // 优先使用 name 字段（这是正确的任务名称，如 "gsm8k"）
        if (dataset.name) {
          return dataset.name
        }
        // 如果没有 name，则根据路径构造（兼容旧数据）
        let taskName = dataset.path.replace(/\//g, '_')  // 将路径中的 "/" 替换为 "_"
        if (dataset.config_name) {
          taskName = `${taskName}_${dataset.config_name}`
        }
        return taskName
      })
    
    // 去重并排序
    availableTasks.value = [...new Set(taskNames)].sort()
  } catch (error) {
    console.error('加载可用任务列表失败:', error)
    availableTasks.value = []
  }
}

onMounted(() => {
  loadConfigs()
  loadAvailableTasks()
})
</script>

<style scoped>
.configs-view {
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
</style>

