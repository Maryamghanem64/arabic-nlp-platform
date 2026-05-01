import { createRouter, createWebHistory } from 'vue-router'
import HomeView    from '../views/HomeView.vue'
import AnalyzeView from '../views/AnalyzeView.vue'
import CompareView from '../views/CompareView.vue'
import AboutView   from '../views/AboutView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/',        component: HomeView    },
    { path: '/analyze', component: AnalyzeView },
    { path: '/compare', component: CompareView },
    { path: '/about',   component: AboutView   },
  ]
})

export default router