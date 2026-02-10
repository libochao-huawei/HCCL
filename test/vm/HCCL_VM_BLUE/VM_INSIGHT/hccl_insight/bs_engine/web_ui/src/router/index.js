import { createRouter, createWebHashHistory } from 'vue-router';
import commonRouter from './common';
import { useUserStore } from '@/store/app.js';
import useLoading from '@/utils/useLoading';
import { inject } from 'vue';
import { useCurrentTestCaseStore, useRunTestCaseResultStore } from '@/store/checker/index.js';

const routes = [...commonRouter];

export default function () {
  const router = createRouter({
    history: createWebHashHistory(),
    routes
  });

  const { showLoading, closeLoading } = useLoading();

  router.beforeEach(async (to, from) => {
    const { apiAssets } = inject('$api');
    showLoading();
    const userStore = useUserStore();
    const currentTestCaseStore = useCurrentTestCaseStore();
    const runTestCaseResultStore = useRunTestCaseResultStore();

    console.log('userStore.userName', userStore.userName);

    if (userStore.userName) {
      if (to.name === 'login') {
        try {
          const res = await apiAssets.fetchDataByCmd({ cmd: 'quit', user_name: userStore.userName });
          sessionStorage.clear();
          userStore.setUserName('');
          currentTestCaseStore.$reset();
          runTestCaseResultStore.$reset();
          return true;
        } catch (error) {
          return false;
        }
      } else {
        return true;
      }
    } else if (to.name !== 'login') {
      router.push({ name: 'login' });
      return false;
    }
  });

  router.afterEach((to, from) => {
    closeLoading();
  });

  return router;
}
