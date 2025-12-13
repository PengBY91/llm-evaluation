<template>
  <div class="models-view">
    <div class="view-header">
      <h2>模型管理</h2>
      <el-button type="primary" @click="handleAddModel">
        <el-icon><Plus /></el-icon>
        添加模型
      </el-button>
    </div>

    <el-table :data="models" v-loading="loading" stripe>
      <el-table-column prop="name" label="模型名称" width="200" />
      <el-table-column prop="model_type" label="模型类型" width="150">
        <template #default="{ row }">
          <el-tag size="small">{{ getModelTypeLabel(row.model_type) }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="model_name" label="模型标识" width="200">
        <template #default="{ row }">
          {{ row.model_name || '-' }}
        </template>
      </el-table-column>
      <el-table-column prop="base_url" label="API URL" min-width="200" show-overflow-tooltip />
      <el-table-column label="操作" width="220" fixed="right">
        <template #default="{ row }">
          <el-button 
            size="small" 
            type="success"
            @click="testModelConnection(row)"
            :loading="testingModels[row.id]"
          >
            测试
          </el-button>
          <el-button size="small" @click="editModel(row)">编辑</el-button>
          <el-button size="small" type="danger" @click="deleteModel(row.id)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>


    <!-- 创建/编辑模型对话框 -->
    <el-dialog 
      v-model="showCreateDialog" 
      :title="editingModel ? '编辑模型' : '添加模型'" 
      width="800px"
    >
      <el-form :model="modelForm" label-width="120px" :rules="rules" ref="modelFormRef">
        <el-form-item label="模型名称" prop="name" required>
          <el-input v-model="modelForm.name" placeholder="请输入模型名称" />
        </el-form-item>
        <el-form-item label="模型类型" prop="model_type" required>
          <el-select 
            v-model="modelForm.model_type" 
            placeholder="请选择模型类型"
            style="width: 100%"
          >
            <el-option 
              v-for="type in modelTypes" 
              :key="type.value" 
              :label="type.label" 
              :value="type.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="模型标识" prop="model_name" :required="needsModelName">
          <el-input 
            v-model="modelForm.model_name" 
            placeholder="例如: gpt-3.5-turbo, qwen3:8b"
          />
          <div class="form-tip">{{ needsModelName ? '（必需）' : '（可选）' }}</div>
        </el-form-item>
        <el-form-item label="API URL" :required="needsBaseUrl">
          <div style="display: flex; gap: 10px;">
            <el-input 
              v-model="apiUrl" 
              placeholder="例如: http://localhost:11434/v1"
            />
            <el-button 
              @click="testConnection" 
              :loading="testingConnection"
              type="primary"
            >
              测试
            </el-button>
          </div>
          <el-alert
            v-if="testResult"
            :title="testResult.message"
            :type="testResult.success ? 'success' : 'error'"
            :description="testResult.details"
            :closable="true"
            @close="testResult = null"
            style="margin-top: 8px;"
          />
        </el-form-item>
        <el-form-item label="API Key">
          <el-input 
            v-model="modelForm.api_key" 
            type="password" 
            show-password
            placeholder="可选，如果需要认证"
          />
        </el-form-item>
        
        <!-- 高级选项 -->
        <el-divider>
          <el-button text @click="showAdvancedOptions = !showAdvancedOptions">
            {{ showAdvancedOptions ? '收起' : '展开' }}高级选项
            <el-icon><ArrowDown v-if="!showAdvancedOptions" /><ArrowUp v-else /></el-icon>
          </el-button>
        </el-divider>
        
        <el-collapse-transition>
          <div v-show="showAdvancedOptions">
            <el-form-item label="最大并发数">
              <el-input-number 
                v-model="modelForm.max_concurrent" 
                :min="1" 
                :max="100"
                style="width: 100%"
              />
            </el-form-item>
            <el-form-item label="最长Token">
              <el-input-number 
                v-model="modelForm.max_tokens" 
                :min="1"
                style="width: 100%"
              />
            </el-form-item>
            <el-form-item label="描述">
              <el-input v-model="modelForm.description" type="textarea" :rows="2" />
            </el-form-item>
            <el-form-item label="其他配置">
              <el-input 
                v-model="otherConfigStr" 
                type="textarea" 
                :rows="3"
                placeholder='JSON 格式，例如: {"temperature": 0.7}'
              />
            </el-form-item>
          </div>
        </el-collapse-transition>
      </el-form>
      <template #footer>
        <el-button @click="cancelEdit">取消</el-button>
        <el-button type="primary" @click="saveModel">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus, ArrowDown, ArrowUp } from '@element-plus/icons-vue'
import { modelsApi } from '../api/models'

const models = ref([])
const modelTypes = ref([])
const loading = ref(false)
const showCreateDialog = ref(false)
const editingModel = ref(null)
const showAdvancedOptions = ref(false)
const modelFormRef = ref(null)
const otherConfigStr = ref('{}')
const apiUrl = ref('')  // 合并的 URL 字段
const testingConnection = ref(false)
const testResult = ref(null)
const testingModels = ref({})  // 跟踪每个模型的测试状态

const modelForm = ref({
  name: '',
  model_type: '',
  description: '',
  base_url: '',
  api_key: '',
  max_concurrent: null,
  max_tokens: null,
  model_name: '',
  other_config: {}
})

const rules = {
  name: [{ required: true, message: '请输入模型名称', trigger: 'blur' }],
  model_type: [{ required: true, message: '请选择模型类型', trigger: 'change' }]
}

const needsModelName = computed(() => {
  if (!modelForm.value.model_type) return false
  const type = modelTypes.value.find(t => t.value === modelForm.value.model_type)
  return type && type.requires && type.requires.includes('model_name')
})

const needsBaseUrl = computed(() => {
  if (!modelForm.value.model_type) return false
  const type = modelTypes.value.find(t => t.value === modelForm.value.model_type)
  return type && type.requires && type.requires.includes('base_url')
})

const needsPort = computed(() => {
  if (!modelForm.value.model_type) return false
  const type = modelTypes.value.find(t => t.value === modelForm.value.model_type)
  return type && type.optional && type.optional.includes('port')
})

const getModelTypeLabel = (type) => {
  const modelType = modelTypes.value.find(t => t.value === type)
  return modelType ? modelType.label : type
}

const loadModels = async () => {
  loading.value = true
  try {
    models.value = await modelsApi.getModels()
  } catch (error) {
    ElMessage.error('加载模型列表失败: ' + error.message)
  } finally {
    loading.value = false
  }
}

const loadModelTypes = async () => {
  try {
    const response = await modelsApi.getModelTypes()
    modelTypes.value = response.model_types || []
  } catch (error) {
    // 使用默认类型
    modelTypes.value = [
      { value: 'openai-chat-completions', label: 'OpenAI Chat Completions', description: 'OpenAI 兼容的聊天完成 API（支持 Ollama、vLLM 等）' },
      { value: 'openai-completions', label: 'OpenAI Completions', description: 'OpenAI 兼容的完成 API' },
      { value: 'hf', label: 'HuggingFace', description: 'HuggingFace Transformers 模型' }
    ]
  }
}

const handleAddModel = () => {
  editingModel.value = null
  resetForm()
  showAdvancedOptions.value = false
  showCreateDialog.value = true
}

const handleModelTypeChange = () => {
  // 根据模型类型提供默认值提示，但不强制清空字段
  // 用户可以保留之前的值或手动修改
}

const editModel = (model) => {
  editingModel.value = model
  modelForm.value = { ...model }
  // 如果 api_key 是隐藏值，清空让用户重新输入
  if (modelForm.value.api_key === '***') {
    modelForm.value.api_key = ''
  }
  // 直接使用 base_url（如果旧数据有 port，需要合并）
  if (model.port && model.base_url && !model.base_url.includes(':' + model.port)) {
    // 兼容旧数据：如果有单独的 port，合并到 URL 中
    try {
      const urlObj = new URL(model.base_url)
      urlObj.port = String(model.port)
      apiUrl.value = urlObj.toString()
    } catch (e) {
      apiUrl.value = model.base_url
    }
  } else {
    apiUrl.value = model.base_url || ''
  }
  otherConfigStr.value = JSON.stringify(model.other_config || {}, null, 2)
  // 如果有高级选项的值，自动展开
  showAdvancedOptions.value = !!(model.max_concurrent || model.max_tokens || model.description || (model.other_config && Object.keys(model.other_config).length > 0))
  showCreateDialog.value = true
}

const cancelEdit = () => {
  editingModel.value = null
  showCreateDialog.value = false
  resetForm()
}

const resetForm = () => {
  modelForm.value = {
    name: '',
    model_type: '',
    description: '',
    base_url: '',
    api_key: '',
    max_concurrent: null,
    max_tokens: null,
    model_name: '',
    other_config: {}
  }
  apiUrl.value = ''
  otherConfigStr.value = '{}'
  testResult.value = null
  showAdvancedOptions.value = false
  if (modelFormRef.value) {
    modelFormRef.value.resetFields()
  }
}

const saveModel = async () => {
  if (!modelFormRef.value) return
  
  try {
    await modelFormRef.value.validate()
    
    // 验证 API URL（如果需要）
    if (needsBaseUrl.value && (!apiUrl.value || !apiUrl.value.trim())) {
      ElMessage.error('该模型类型需要提供 API URL')
      return
    }
    
    // 直接使用完整的 URL 作为 base_url
    modelForm.value.base_url = apiUrl.value.trim() || ''
    
    // 解析其他配置
    try {
      if (otherConfigStr.value.trim()) {
        modelForm.value.other_config = JSON.parse(otherConfigStr.value)
      } else {
        modelForm.value.other_config = {}
      }
    } catch (e) {
      ElMessage.error('其他配置格式错误，请输入有效的JSON')
      return
    }
    
    // 清理数据，移除空字符串和 null 值（但保留 0 和 false）
    // 注意：api_key 即使是空字符串也应该保留，以便清除已保存的 api_key
    const cleanData = { ...modelForm.value }
    Object.keys(cleanData).forEach(key => {
      // api_key 特殊处理：空字符串表示清除，应该保留
      if (key === 'api_key') {
        // api_key 保留原值（包括空字符串），让后端处理
        return
      }
      if (cleanData[key] === '' || cleanData[key] === null) {
        delete cleanData[key]
      }
    })
    
    // 确保必需字段存在
    if (!cleanData.name || !cleanData.model_type) {
      ElMessage.error('模型名称和模型类型是必需的')
      return
    }
    
    if (editingModel.value) {
      await modelsApi.updateModel(editingModel.value.id, cleanData)
      ElMessage.success('模型更新成功')
    } else {
      await modelsApi.createModel(cleanData)
      ElMessage.success('模型创建成功')
    }
    
    showCreateDialog.value = false
    resetForm()
    editingModel.value = null
    loadModels()
  } catch (error) {
    if (error !== false) {
      let errorMessage = '保存模型失败'
      if (error) {
        if (error instanceof Error) {
          errorMessage += ': ' + (error.message || '未知错误')
        } else if (typeof error === 'string') {
          errorMessage += ': ' + error
        } else if (error.message) {
          errorMessage += ': ' + error.message
        } else if (error.detail) {
          if (Array.isArray(error.detail)) {
            const details = error.detail.map(err => {
              if (typeof err === 'object' && err.loc && err.msg) {
                return `${err.loc.join('.')}: ${err.msg}`
              }
              return String(err)
            }).join(', ')
            errorMessage += ': ' + details
          } else if (typeof error.detail === 'string') {
            errorMessage += ': ' + error.detail
          } else if (error.response?.data?.detail) {
            errorMessage += ': ' + (error.response.data.detail || error.response.data.message || '未知错误')
          }
        } else if (error.response?.data) {
          errorMessage += ': ' + (error.response.data.detail || error.response.data.message || '未知错误')
        }
      }
      ElMessage.error(errorMessage)
    }
  }
}

const deleteModel = async (modelId) => {
  try {
    await ElMessageBox.confirm('确定要删除这个模型吗？', '提示', {
      type: 'warning'
    })
    await modelsApi.deleteModel(modelId)
    ElMessage.success('模型已删除')
    loadModels()
  } catch (error) {
    if (error !== 'cancel') {
      const errorMsg = error?.detail || error?.message || '未知错误'
      ElMessage.error('删除模型失败: ' + errorMsg)
    }
  }
}



const testModelConnection = async (model) => {
  if (!model.base_url) {
    ElMessage.warning('该模型未配置 API URL，无法测试连接')
    return
  }
  
  if (!model.model_type) {
    ElMessage.warning('该模型未配置模型类型，无法测试连接')
    return
  }
  
  // 检查是否需要模型标识
  const type = modelTypes.value.find(t => t.value === model.model_type)
  const requiresModelName = type && type.requires && type.requires.includes('model_name')
  
  if (requiresModelName && !model.model_name) {
    ElMessage.warning('该模型类型需要提供模型标识才能进行测试')
    return
  }
  
  testingModels.value[model.id] = true
  
  try {
    // 传递 model_id，后端会从数据库获取真实的 api_key
    const testData = {
      model_type: model.model_type,
      base_url: model.base_url,
      api_key: model.api_key === '***' ? undefined : model.api_key,  // 如果是隐藏值，传 undefined，让后端从数据库获取
      model_name: model.model_name || undefined,
      model_id: model.id  // 传递 model_id，后端会从数据库获取真实的 api_key
    }
    
    const result = await modelsApi.testConnection(testData)
    
    if (result.success) {
      ElMessage.success(`模型 "${model.name}" 连接测试成功: ${result.message}`)
    } else {
      ElMessage.error(`模型 "${model.name}" 连接测试失败: ${result.message}`)
    }
  } catch (error) {
    const errorMsg = error?.detail || error?.message || '未知错误'
    ElMessage.error(`模型 "${model.name}" 连接测试失败: ${errorMsg}`)
  } finally {
    testingModels.value[model.id] = false
  }
}

const testConnection = async () => {
  if (!apiUrl.value || !apiUrl.value.trim()) {
    ElMessage.warning('请先输入 API URL')
    return
  }
  
  if (!modelForm.value.model_type) {
    ElMessage.warning('请先选择模型类型')
    return
  }
  
  // 检查是否需要模型标识
  const type = modelTypes.value.find(t => t.value === modelForm.value.model_type)
  const requiresModelName = type && type.requires && type.requires.includes('model_name')
  
  if (requiresModelName && !modelForm.value.model_name) {
    ElMessage.warning('该模型类型需要提供模型标识才能进行测试')
    return
  }
  
  testingConnection.value = true
  testResult.value = null
  
  try {
    // 结合模型类型和模型标识进行测试
    const testData = {
      model_type: modelForm.value.model_type,
      base_url: apiUrl.value.trim(),
      api_key: modelForm.value.api_key || undefined,
      model_name: modelForm.value.model_name || undefined
    }
    
    const result = await modelsApi.testConnection(testData)
    testResult.value = result
    
    if (result.success) {
      ElMessage.success('连接测试成功')
    } else {
      ElMessage.error('连接测试失败: ' + result.message)
    }
  } catch (error) {
    testResult.value = {
      success: false,
      message: '测试失败',
      details: error.message || '未知错误'
    }
    ElMessage.error('连接测试失败: ' + error.message)
  } finally {
    testingConnection.value = false
  }
}

onMounted(() => {
  loadModels()
  loadModelTypes()
})
</script>

<style scoped>
.models-view {
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

.form-tip {
  font-size: 12px;
  color: #999;
  margin-top: 4px;
}
</style>

