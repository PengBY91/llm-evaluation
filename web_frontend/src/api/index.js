import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000
})

// 请求拦截器
api.interceptors.request.use(
  config => {
    return config
  },
  error => {
    return Promise.reject(error)
  }
)

// 响应拦截器
api.interceptors.response.use(
  response => {
    return response.data
  },
  error => {
    // 改进错误处理，保留完整的错误信息
    if (error.response) {
      // 服务器返回了错误响应
      const data = error.response.data
      // 保留完整的错误对象，让调用方可以访问 detail 等信息
      const errorObj = new Error(data?.message || data?.detail || `请求失败: ${error.response.status} ${error.response.statusText}`)
      errorObj.response = error.response
      errorObj.detail = data?.detail || data
      errorObj.status = error.response.status
      return Promise.reject(errorObj)
    } else if (error.message) {
      return Promise.reject(error)
    } else if (error.request) {
      return Promise.reject(new Error('网络错误，请检查网络连接'))
    } else {
      return Promise.reject(new Error('请求失败'))
    }
  }
)

export default api

