import { createApp } from 'vue';
import App from './App.vue';
import router from './router';
import './styles/index.css';
import api from './services';
import echarts from '@/utils/echarts';
import { createPinia } from 'pinia';
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate';
import './theme/index.js';
import 'virtual:svg-icons-register';

const app = createApp(App);
app.config.globalProperties.$echarts = echarts;
app.config.globalProperties.aui_theme = { value: 'saas' };
const pinia = createPinia();
pinia.use(piniaPluginPersistedstate);
app.use(pinia);
app.use(api);
app.use(router()).mount('#app');
