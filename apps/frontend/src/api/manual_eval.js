import api from './index'

export const manualEvalApi = {
    // 生成答案
    generateAnswers(data) {
        return api.post('/manual-eval/generate', data)
    },

    // AI 评价
    evaluateAnswers(data) {
        return api.post('/manual-eval/evaluate', data)
    },

    // 保存评测结果
    saveEvaluation(data) {
        return api.post('/manual-eval/save', data)
    },

    // 获取评测记录列表
    getEvaluations() {
        return api.get('/manual-eval/list')
    },

    // 获取评测记录详情
    getEvaluation(id) {
        return api.get(`/manual-eval/${id}`)
    },

    // 删除评测记录
    deleteEvaluation(id) {
        return api.delete(`/manual-eval/${id}`)
    }
}
