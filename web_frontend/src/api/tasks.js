import api from './index'

export const tasksApi = {
  // 创建任务
  createTask(data) {
    return api.post('/tasks/', data)
  },
  
  // 获取任务列表
  getTasks() {
    return api.get('/tasks/')
  },
  
  // 获取任务详情
  getTask(taskId) {
    return api.get(`/tasks/${taskId}`)
  },
  
  // 删除任务
  deleteTask(taskId) {
    return api.delete(`/tasks/${taskId}`)
  },
  
  // 获取任务结果
  getTaskResults(taskId) {
    return api.get(`/tasks/${taskId}/results`)
  },
  
  // 下载任务结果
  downloadTaskResults(taskId) {
    return api.get(`/tasks/${taskId}/download`, { responseType: 'blob' })
  },
  
  // 获取任务进度
  getTaskProgress(taskId) {
    return api.get(`/tasks/${taskId}/progress`)
  },
  
  // 获取可用任务列表
  getAvailableTasks() {
    return api.get('/tasks/available-tasks/list')
  }
}

