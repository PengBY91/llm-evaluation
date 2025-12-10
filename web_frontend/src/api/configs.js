import api from './index'

export const configsApi = {
  // 创建配置
  createConfig(data) {
    return api.post('/configs/', data)
  },
  
  // 获取配置列表
  getConfigs() {
    return api.get('/configs/')
  },
  
  // 获取配置详情
  getConfig(configId) {
    return api.get(`/configs/${configId}`)
  },
  
  // 更新配置
  updateConfig(configId, data) {
    return api.put(`/configs/${configId}`, data)
  },
  
  // 删除配置
  deleteConfig(configId) {
    return api.delete(`/configs/${configId}`)
  }
}

