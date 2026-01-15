import api from './index'

export const datasetsApi = {
  // 获取数据集列表（支持分类过滤和分页）
  getDatasets(params = {}) {
    return api.get('/datasets/', { params })
  },

  // 获取数据集详情（ID 可能包含 /，需要 URL 编码）
  getDataset(datasetId) {
    return api.get(`/datasets/${encodeURIComponent(datasetId)}`)
  },

  // 添加数据集
  addDataset(data) {
    return api.post('/datasets/', data)
  },

  // 删除数据集（ID 可能包含 /，需要 URL 编码）
  deleteDataset(datasetId) {
    return api.delete(`/datasets/${encodeURIComponent(datasetId)}`)
  },

  // 获取数据集样本
  getDatasetSamples(datasetName, split = 'train', limit = 10) {
    return api.get(`/datasets/samples`, {
      params: { dataset_name: datasetName, split, limit }
    })
  },

  // 获取数据集 README（新路由格式，支持包含 / 的 ID）
  getDatasetReadme(datasetId) {
    return api.get(`/datasets/readme/${encodeURIComponent(datasetId)}`)
  },

  // 重建索引
  rebuildIndex() {
    return api.post('/datasets/rebuild-index/')
  }
}

