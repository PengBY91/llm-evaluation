import { createRouter, createWebHistory } from 'vue-router'
import TasksView from '../views/TasksView.vue'
import ModelsView from '../views/ModelsView.vue'
import DatasetsView from '../views/DatasetsView.vue'

const routes = [
  {
    path: '/',
    redirect: '/tasks'
  },
  {
    path: '/tasks',
    name: 'Tasks',
    component: TasksView
  },
  {
    path: '/models',
    name: 'Models',
    component: ModelsView
  },
  {
    path: '/datasets',
    name: 'Datasets',
    component: DatasetsView
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router

