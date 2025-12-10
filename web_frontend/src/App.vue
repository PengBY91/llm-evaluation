<template>
  <el-container class="app-container">
    <el-header class="app-header">
      <div class="header-content">
        <h1>大模型评测任务管理平台</h1>
      </div>
    </el-header>
    <el-container>
      <el-aside width="200px" class="app-sidebar">
        <el-menu
          :default-active="activeMenu"
          router
          class="sidebar-menu"
        >
          <el-menu-item index="/tasks">
            <el-icon><List /></el-icon>
            <span>评测任务管理</span>
          </el-menu-item>
          <el-menu-item index="/models">
            <el-icon><Setting /></el-icon>
            <span>模型管理</span>
          </el-menu-item>
          <el-menu-item index="/datasets">
            <el-icon><FolderOpened /></el-icon>
            <span>数据集管理</span>
          </el-menu-item>
        </el-menu>
      </el-aside>
      <el-main class="app-main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { List, Setting, FolderOpened } from '@element-plus/icons-vue'

const route = useRoute()
const activeMenu = computed(() => route.path)
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

#app {
  height: 100vh;
}

.app-container {
  height: 100vh;
}

.app-header {
  background-color: #409eff;
  color: white;
  display: flex;
  align-items: center;
  padding: 0 20px;
}

.header-content h1 {
  font-size: 20px;
  font-weight: 500;
}

.app-sidebar {
  background-color: #f5f5f5;
  border-right: 1px solid #e4e7ed;
}

.sidebar-menu {
  border-right: none;
}

.app-main {
  padding: 20px;
  background-color: #fafafa;
  overflow-y: auto;
}
</style>

