<template>
  <div class="manual-eval-container">
    <div class="page-header">
      <div class="header-left">
        <h2>手动评测</h2>
        <p class="subtitle">实时对比多个模型生成的回答，进行人工打分与 AI 辅助评价</p>
      </div>
      <div class="header-actions">
        <el-button type="primary" @click="showHistory = !showHistory">
          {{ showHistory ? '返回评测' : '历史记录' }}
        </el-button>
      </div>
    </div>

    <!-- 历史记录列表 -->
    <div v-if="showHistory" class="history-section">
      <el-card shadow="never">
        <el-table :data="historyList" v-loading="loadingHistory" style="width: 100%">
          <el-table-column prop="name" label="评测名称" min-width="150" />
          <el-table-column prop="user_prompt" label="问题简述" min-width="250" show-overflow-tooltip />
          <el-table-column prop="model_count" label="模型数量" width="100" align="center" />
          <el-table-column prop="created_at" label="时间" width="180">
            <template #default="scope">
              {{ formatTime(scope.row.created_at) }}
            </template>
          </el-table-column>
          <el-table-column label="操作" width="150" align="center">
            <template #default="scope">
              <el-button link type="primary" @click="viewDetail(scope.row.id)">查看</el-button>
              <el-button link type="danger" @click="handleDelete(scope.row.id)">删除</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-card>
    </div>

    <!-- 评测主界面 -->
    <div v-else class="eval-main">
      <el-row :gutter="20">
        <!-- 左侧配置区 -->
        <el-col :span="6">
          <el-card shadow="hover" class="config-card">
            <template #header>
              <div class="card-header">
                <span><el-icon><Setting /></el-icon> 配置选项</span>
              </div>
            </template>
            
            <el-form :model="evalForm" label-position="top">
              <el-form-item label="选择模型 (多选)">
                <el-select
                  v-model="evalForm.model_ids"
                  multiple
                  placeholder="选择要对比的模型"
                  style="width: 100%"
                >
                  <el-option
                    v-for="item in availableModels"
                    :key="item.id"
                    :label="item.name"
                    :value="item.id"
                  />
                </el-select>
              </el-form-item>

              <el-form-item label="AI 评分模型">
                <el-select
                  v-model="evalForm.evaluator_id"
                  placeholder="选择辅助评分的模型"
                  style="width: 100%"
                >
                  <el-option
                    v-for="item in availableModels"
                    :key="item.id"
                    :label="item.name"
                    :value="item.id"
                  />
                </el-select>
              </el-form-item>

              <el-divider />

              <el-form-item label="Max Tokens">
                <el-input-number v-model="evalForm.max_tokens" :min="1" :max="4096" style="width: 100%" />
              </el-form-item>
              
              <el-form-item label="Temperature">
                <el-slider v-model="evalForm.temperature" :min="0" :max="1" :step="0.1" show-input />
              </el-form-item>
              
              <el-form-item label="回答数量 (每个模型)">
                <el-input-number v-model="evalForm.n" :min="1" :max="5" style="width: 100%" />
              </el-form-item>

              <div class="form-actions">
                <el-button 
                  type="primary" 
                  style="width: 100%" 
                  @click="generateAnswers"
                  :loading="generating"
                  :disabled="!canGenerate"
                >
                  开始生成
                </el-button>
              </div>
            </el-form>
          </el-card>
        </el-col>

        <!-- 右侧内容区 -->
        <el-col :span="18">
          <el-card shadow="hover" class="prompt-card">
            <el-form label-position="top">
              <el-form-item label="System Prompt">
                <el-input
                  v-model="evalForm.system_prompt"
                  type="textarea"
                  :rows="2"
                  placeholder="系统提示词..."
                />
              </el-form-item>
              <el-form-item label="User Message">
                <el-input
                  v-model="evalForm.user_prompt"
                  type="textarea"
                  :rows="4"
                  placeholder="输入测试问题或指令..."
                />
              </el-form-item>
            </el-form>
          </el-card>

          <!-- 答案详情区域 -->
          <div v-if="answers.length > 0" class="results-section">
            <div class="results-toolbar">
              <h3>生成结果对比</h3>
              <div class="toolbar-btns">
                <el-button 
                  type="success" 
                  plain 
                  @click="runAiEvaluation" 
                  :loading="evaluatingAI"
                  :disabled="!evalForm.evaluator_id"
                >
                  <el-icon><MagicStick /></el-icon> AI 辅助评价
                </el-button>
                <el-button type="primary" @click="saveEvaluation" :loading="saving">
                  保存评测记录
                </el-button>
              </div>
            </div>

            <!-- 答案列容器 -->
            <div class="answer-columns">
              <div v-for="(ans, index) in answers" :key="index" class="answer-col">
                <el-card shadow="hover" class="answer-card" :class="{ 'has-error': ans.error }">
                  <template #header>
                    <div class="answer-header">
                      <span class="model-name">{{ ans.model_name }}</span>
                      <el-tag v-if="ans.error" type="danger" size="small">生成失败</el-tag>
                    </div>
                  </template>
                  
                  <div class="answer-content">
                    <div v-if="ans.error" class="error-msg">{{ ans.error }}</div>
                    <pre v-else class="content-text">{{ ans.answer }}</pre>
                  </div>

                  <div class="answer-footer" v-if="!ans.error">
                    <div class="footer-item">
                      <span>评分：</span>
                      <el-rate v-model="ans.score" />
                    </div>
                    <div class="footer-item">
                      <span>排序：</span>
                      <el-input-number 
                        v-model="ans.rank" 
                        :min="1" 
                        :max="answers.length" 
                        size="small" 
                        style="width: 80px"
                      />
                    </div>
                  </div>
                </el-card>
              </div>
            </div>

            <!-- AI 评价展示 -->
            <el-card v-if="aiEvaluation" shadow="hover" class="ai-eval-card">
              <template #header>
                <div class="card-header">
                  <span><el-icon><Connection /></el-icon> AI 专家点评</span>
                </div>
              </template>
              <div class="ai-content">
                <pre>{{ aiEvaluation }}</pre>
              </div>
            </el-card>
          </div>

          <el-empty v-else description="暂无生成结果，请在左侧配置并点击生成" />
        </el-col>
      </el-row>
    </div>

    <!-- 保存对话框 -->
    <el-dialog v-model="saveDialogVisible" title="保存评测记录" width="400px">
      <el-form label-position="top">
        <el-form-item label="评测记录名称">
          <el-input v-model="saveName" placeholder="例如：推理能力测试 - 20241229" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="saveDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="confirmSave" :loading="saving">确认保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, computed } from 'vue'
import { Setting, MagicStick, Connection, ChatDotRound } from '@element-plus/icons-vue'
import { modelsApi } from '../api/models'
import { manualEvalApi } from '../api/manual_eval'
import { ElMessage, ElMessageBox } from 'element-plus'

// 数据定义
const showHistory = ref(false)
const availableModels = ref([])
const historyList = ref([])
const loadingHistory = ref(false)
const generating = ref(false)
const evaluatingAI = ref(false)
const saving = ref(false)
const saveDialogVisible = ref(false)
const saveName = ref('')

const evalForm = reactive({
  model_ids: [],
  evaluator_id: '',
  system_prompt: 'You are a helpful assistant.',
  user_prompt: '',
  max_tokens: 512,
  temperature: 0.7,
  n: 1
})

const answers = ref([])
const aiEvaluation = ref('')

// 计算属性
const canGenerate = computed(() => {
  return evalForm.model_ids.length > 0 && evalForm.user_prompt.trim() !== ''
})

// 生命周期挂载
onMounted(async () => {
  await fetchModels()
  if (showHistory.value) {
    await fetchHistory()
  }
})

// 方法定义
const fetchModels = async () => {
  try {
    const res = await modelsApi.getModels()
    // res 经过拦截器处理，应该直接是列表
    availableModels.value = Array.isArray(res) ? res : (res.data || [])
  } catch (error) {
    ElMessage.error('获取模型列表失败')
  }
}

const fetchHistory = async () => {
  loadingHistory.value = true
  try {
    historyList.value = await manualEvalApi.getEvaluations()
  } catch (error) {
    ElMessage.error('获取历史记录失败')
  } finally {
    loadingHistory.value = false
  }
}

const generateAnswers = async () => {
  generating.value = true
  answers.value = []
  aiEvaluation.value = ''
  
  try {
    const res = await manualEvalApi.generateAnswers({
      model_ids: evalForm.model_ids,
      system_prompt: evalForm.system_prompt,
      user_prompt: evalForm.user_prompt,
      max_tokens: evalForm.max_tokens,
      temperature: evalForm.temperature,
      n: evalForm.n
    })
    
    // 初始化分和排序
    answers.value = res.results.map(item => ({
      ...item,
      score: 0,
      rank: 1
    }))
  } catch (error) {
    console.error('生成答案失败:', error)
    ElMessage.error('生成答案失败: ' + (error.message || '未知错误'))
  } finally {
    generating.value = false
  }
}

const runAiEvaluation = async () => {
  if (!evalForm.evaluator_id) {
    ElMessage.warning('请选择评分模型')
    return
  }
  
  evaluatingAI.value = true
  try {
    const res = await manualEvalApi.evaluateAnswers({
      evaluator_model_id: evalForm.evaluator_id,
      system_prompt: evalForm.system_prompt,
      user_prompt: evalForm.user_prompt,
      answers: answers.value.filter(a => !a.error).map(a => ({
        model_name: a.model_name,
        answer: a.answer
      }))
    })
    aiEvaluation.value = res.evaluation
  } catch (error) {
    ElMessage.error('AI 辅助评价执行失败')
  } finally {
    evaluatingAI.value = false
  }
}

const saveEvaluation = () => {
  saveName.value = `评测 - ${evalForm.user_prompt.substring(0, 10)}${evalForm.user_prompt.length > 10 ? '...' : ''}`
  saveDialogVisible.value = true
}

const confirmSave = async () => {
  if (!saveName.value.trim()) {
    ElMessage.warning('请输入记录名称')
    return
  }
  
  saving.value = true
  try {
    await manualEvalApi.saveEvaluation({
      name: saveName.value,
      system_prompt: evalForm.system_prompt,
      user_prompt: evalForm.user_prompt,
      evaluations: answers.value.map(a => ({
        model_id: a.model_id,
        model_name: a.model_name,
        answer: a.answer,
        score: a.score,
        rank: a.rank
      })),
      ai_evaluation: aiEvaluation.value
    })
    ElMessage.success('保存成功')
    saveDialogVisible.value = false
  } catch (error) {
    ElMessage.error('保存失败')
  } finally {
    saving.value = false
  }
}

const viewDetail = async (id) => {
  try {
    const data = await manualEvalApi.getEvaluation(id)
    evalForm.system_prompt = data.system_prompt
    evalForm.user_prompt = data.user_prompt
    answers.value = data.evaluations
    aiEvaluation.value = data.ai_evaluation
    showHistory.value = false
    ElMessage.info('已加载历史评测数据')
  } catch (error) {
    ElMessage.error('加载详情失败')
  }
}

const handleDelete = (id) => {
  ElMessageBox.confirm('确定要删除这条记录吗？', '提示', {
    type: 'warning'
  }).then(async () => {
    try {
      await manualEvalApi.deleteEvaluation(id)
      ElMessage.success('已删除')
      await fetchHistory()
    } catch (error) {
      ElMessage.error('删除失败')
    }
  })
}

const formatTime = (timeStr) => {
  if (!timeStr) return ''
  const date = new Date(timeStr)
  return date.toLocaleString()
}

// 监听
import { watch } from 'vue'
watch(showHistory, (newVal) => {
  if (newVal) {
    fetchHistory()
  }
})
</script>

<style scoped>
.manual-eval-container {
  padding: 10px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.header-left h2 {
  margin: 0;
  color: #303133;
}

.subtitle {
  margin: 5px 0 0;
  font-size: 14px;
  color: #909399;
}

.config-card {
  height: 100%;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: bold;
}

.form-actions {
  margin-top: 20px;
}

.prompt-card {
  margin-bottom: 20px;
}

.results-section {
  margin-top: 20px;
}

.results-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.answer-columns {
  display: flex;
  gap: 15px;
  overflow-x: auto;
  padding-bottom: 15px;
  align-items: flex-start;
}

.answer-col {
  flex: 0 0 calc(33.33% - 10px);
  min-width: 300px;
}

.answer-card {
  display: flex;
  flex-direction: column;
}

.answer-card.has-error {
  border-color: #f56c6c;
}

.answer-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.model-name {
  font-weight: bold;
  color: #409eff;
}

.answer-content {
  min-height: 200px;
  max-height: 500px;
  overflow-y: auto;
  padding: 10px;
  background-color: #f8f9fa;
  border-radius: 4px;
}

.content-text {
  margin: 0;
  white-space: pre-wrap;
  font-family: inherit;
  font-size: 14px;
  line-height: 1.6;
}

.error-msg {
  color: #f56c6c;
}

.answer-footer {
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid #ebeef5;
}

.footer-item {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
}

.footer-item span {
  font-size: 14px;
  color: #606266;
  width: 50px;
}

.ai-eval-card {
  margin-top: 20px;
  border-left: 4px solid #67c23a;
}

.ai-content {
  padding: 10px;
  background-color: #f0f9eb;
  border-radius: 4px;
}

.ai-content pre {
  margin: 0;
  white-space: pre-wrap;
  line-height: 1.6;
}
</style>
