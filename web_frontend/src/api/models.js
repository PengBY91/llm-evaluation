import api from './index'

export const modelsApi = {
  // 创建模型
  createModel(data) {
    return api.post('/models/', data)
  },
  
  // 获取模型列表
  getModels() {
    return api.get('/models/')
  },
  
  // 获取模型详情
  getModel(modelId) {
    return api.get(`/models/${modelId}`)
  },
  
  // 更新模型
  updateModel(modelId, data) {
    return api.put(`/models/${modelId}`, data)
  },
  
  // 删除模型
  deleteModel(modelId) {
    return api.delete(`/models/${modelId}`)
  },
  
  // 获取模型类型列表
  getModelTypes() {
    return api.get('/models/types/list')
  },
  
  // 获取用于评测的 model_args
  getModelArgs(modelId) {
    return api.get(`/models/${modelId}/model-args`)
  }
}

