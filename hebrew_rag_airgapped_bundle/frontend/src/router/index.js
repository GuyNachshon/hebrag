import { createRouter, createWebHistory } from 'vue-router'
import ChatView from '../views/Chat.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      redirect: '/chat'
    },
    {
      path: '/chat',
      name: 'chat',
      component: ChatView,
    },
    {
      path: '/chat/:conversationId',
      name: 'conversation',
      component: ChatView,
      props: true
    }
  ],
})

export default router
