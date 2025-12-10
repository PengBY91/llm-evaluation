<template>
  <div class="models-view">
    <div class="view-header">
      <h2>模型管理</h2>
      <el-button type="primary" @click="showCreateDialog = true">
        <el-icon><Plus /></el-icon>
        添加模型
      </el-button>
    </div>

    <el-table :data="models" v-loading="loading" stripe>
      <el-table-column prop="name" label="模型名称" width="200" />
      <el-table-column prop="model_type" label="模型类型" width="180">
        <template #default="{ row }">
          <el-tag>{{ getModelTypeLabel(row.model_type) }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="model_name" label="模型标识" width="150" />
      <el-table-column prop="base_url" label="Base URL" width="200" show-overflow-tooltip />
      <el-table-column prop="port" label="端口" width="80" />
      <el-table-column prop="max_concurrent" label="最大并发" width="100" />
      <el-table-column prop="max_tokens" label="最长Token" width="100" />
      <el-table-column prop="api_key" label="API Key" width="100">
        <template #default="{ row }">
          <el-tag v-if="row.api_key" type="info">已配置</el-tag>
          <span v-else>-</span>
        </template>
      </el-table-column>
      <el-table-column prop="description" label="描述" width="200" show-overflow-tooltip />
      <el-table-column prop="created_at" label="创建时间" width="180">
        <template #default="{ row }">
          {{ formatTime(row.created_at) }}
        </template>
      </el-table-column>
      <el-table-column label="操作" width="150" fixed="right">
        <template #default="{ row }">
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
      <el-form :model="modelForm" label-width="140px" :rules="rules" ref="modelFormRef">
        <el-form-item label="模型名称" prop="name" required>
          <el-input v-model="modelForm.name" placeholder="请输入模型名称" />
        </el-form-item>
        <el-form-item label="模型类型" prop="model_type" required>
          <el-select 
            v-model="modelForm.model_type" 
            placeholder="请选择模型类型"
            style="width: 100%"
            @change="handleModelTypeChange"
          >
            <el-option 
              v-for="type in modelTypes" 
              :key="type.value" 
              :label="type.label" 
              :value="type.value"
            >
              <div>
                <div>{{ type.label }}</div>
                <div style="font-size: 12px; color: #999;">{{ type.description }}</div>
              </div>
            </el-option>
          </el-select>
        </el-form-item>
        <el-form-item label="模型标识" prop="model_name">
          <el-input 
            v-model="modelForm.model_name" 
            placeholder="例如: gpt-3.5-turbo, deepseek-chat"
          />
          <div class="form-tip">模型的具体名称或标识符{{ needsModelName ? '（必需）' : '（可选）' }}</div>
        </el-form-item>
        <el-form-item label="Base URL" prop="base_url">
          <el-input 
            v-model="modelForm.base_url" 
            placeholder="例如: https://api.example.com/v1 或 https://api.example.com/v1/chat/completions"
          />
          <div class="form-tip">API 服务的基础 URL{{ needsBaseUrl ? '（必需）' : '（可选）' }}</div>
        </el-form-item>
        <el-form-item label="API Key" prop="api_key">
          <el-input 
            v-model="modelForm.api_key" 
            type="password" 
            show-password
            placeholder="输入 API Key（可选）"
          />
          <div class="form-tip">API 密钥，如果需要认证</div>
        </el-form-item>
        <el-form-item label="端口号" prop="port">
          <el-input-number 
            v-model="modelForm.port" 
            :min="1" 
            :max="65535"
            placeholder="端口号"
            style="width: 100%"
          />
          <div class="form-tip">本地服务的端口号（可选）</div>
        </el-form-item>
        <el-form-item label="最大并发数" prop="max_concurrent">
          <el-input-number 
            v-model="modelForm.max_concurrent" 
            :min="1" 
            :max="100"
            placeholder="最大并发数"
          />
          <div class="form-tip">同时处理的最大请求数</div>
        </el-form-item>
        <el-form-item label="最长Token" prop="max_tokens">
          <el-input-number 
            v-model="modelForm.max_tokens" 
            :min="1"
            placeholder="最长Token数"
          />
          <div class="form-tip">生成的最大 token 数量</div>
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="modelForm.description" type="textarea" :rows="2" />
        </el-form-item>
        <el-form-item label="其他配置">
          <el-input 
            v-model="otherConfigStr" 
            type="textarea" 
            :rows="3"
            placeholder='JSON 格式，例如: {"temperature": 0.7, "top_p": 0.9}'
          />
          <div class="form-tip">额外的配置参数（JSON 格式）</div>
        </el-form-item>
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
import { Plus } from '@element-plus/icons-vue'
import { modelsApi } from '../api/models'

const models = ref([])
const modelTypes = ref([])
const loading = ref(false)
const showCreateDialog = ref(false)
const editingModel = ref(null)
const modelFormRef = ref(null)
const otherConfigStr = ref('{}')

const modelForm = ref({
  name: '',
  model_type: '',
  description: '',
  base_url: '',
  api_key: '',
  port: null,
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
    console.error('加载模型类型失败:', error)
    // 使用默认类型
    modelTypes.value = [
      { value: 'openai-chat-completions', label: 'OpenAI Chat Completions', description: 'OpenAI 兼容的聊天完成 API' },
      { value: 'openai-completions', label: 'OpenAI Completions', description: 'OpenAI 兼容的完成 API' },
      { value: 'hf', label: 'HuggingFace', description: 'HuggingFace Transformers 模型' },
      { value: 'vllm', label: 'vLLM', description: 'vLLM 推理服务' },
      { value: 'local-completions', label: 'Local Completions', description: '本地完成服务' }
    ]
  }
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
  otherConfigStr.value = JSON.stringify(model.other_config || {}, null, 2)
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
    port: null,
    max_concurrent: null,
    max_tokens: null,
    model_name: '',
    other_config: {}
  }
  otherConfigStr.value = '{}'
  if (modelFormRef.value) {
    modelFormRef.value.resetFields()
  }
}

const saveModel = async () => {
  if (!modelFormRef.value) return
  
  try {
    await modelFormRef.value.validate()
    
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
    
    if (editingModel.value) {
      await modelsApi.updateModel(editingModel.value.id, modelForm.value)
      ElMessage.success('模型更新成功')
    } else {
      await modelsApi.createModel(modelForm.value)
      ElMessage.success('模型创建成功')
    }
    
    showCreateDialog.value = false
    resetForm()
    editingModel.value = null
    loadModels()
  } catch (error) {
    if (error !== false) { // 验证失败时 error 为 false
      ElMessage.error('保存模型失败: ' + error.message)
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
      ElMessage.error('删除模型失败: ' + error.message)
    }
  }
}

const formatTime = (timeStr) => {
  if (!timeStr) return ''
  return new Date(timeStr).toLocaleString('zh-CN')
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

