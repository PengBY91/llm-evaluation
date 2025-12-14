import api from './index'

export const datasetsApi = {
  // 获取数据集列表（支持分类过滤和分页）
  getDatasets(params = {}) {
    return api.get('/datasets/', { params })
  },
  
  // 获取数据集详情
  getDataset(datasetName) {
    return api.get(`/datasets/${datasetName}`)
  },
  
  // 添加数据集
  addDataset(data) {
    return api.post('/datasets/', data)
  },
  
  // 删除数据集
  deleteDataset(datasetName) {
    return api.delete(`/datasets/${datasetName}`)
  },
  
  // 获取数据集样本
  getDatasetSamples(datasetName, split = 'train', limit = 10) {
    return api.get(`/datasets/samples`, {
      params: { dataset_name: datasetName, split, limit }
    })
  },
  
  // 获取数据集 README
  getDatasetReadme(datasetId) {
    return api.get(`/datasets/${datasetId}/readme`)
  },
  
  // 刷新缓存
  refreshCache() {
    return api.post('/datasets/refresh-cache')
  }
}

