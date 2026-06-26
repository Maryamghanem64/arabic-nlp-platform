import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'home', component: () => import('@/views/HomeView.vue') },
  { path: '/dashboard', redirect: '/' },
  { path: '/analyze', name: 'analyze', component: () => import('@/views/AnalyzeView.vue') },
  { path: '/smart', name: 'smart', component: () => import('@/views/SmartAnalysisView.vue') },
  { path: '/compare', name: 'compare', component: () => import('@/views/CompareView.vue') },
  { path: '/evaluate', name: 'evaluate', component: () => import('@/views/EvaluateView.vue') },
  { path: '/reports', name: 'reports', component: () => import('@/views/AboutView.vue') },
  { path: '/about', redirect: '/reports' },
]

export default createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
})
