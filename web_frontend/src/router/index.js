import { createRouter, createWebHistory } from 'vue-router'
import TasksView from '../views/TasksView.vue'
import ModelsView from '../views/ModelsView.vue'
import DatasetsView from '../views/DatasetsView.vue'
import DatasetDetailView from '../views/DatasetDetailView.vue'
import TaskDetailView from '../views/TaskDetailView.vue'

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
    path: '/tasks/:id',
    name: 'TaskDetail',
    component: TaskDetailView
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
  },
  {
    path: '/datasets/:id',
    name: 'DatasetDetail',
    component: DatasetDetailView
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router

