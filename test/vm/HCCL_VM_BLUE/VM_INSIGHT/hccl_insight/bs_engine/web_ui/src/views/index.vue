<template>
  <div class="content">
    <div class="top">
      <i>
        <svg class="logo-name">
          <use xlink:href="#icon-logoName"></use>
        </svg>
      </i>
      <div class="user-container">
        <aui-user-head type="icon" round min></aui-user-head>
        <aui-dropdown class="user" trigger="hover">
          <span>{{ userStore.userName }}</span>
          <template #dropdown>
            <aui-dropdown-menu>
              <aui-dropdown-item><aui-button type="text" @click="onQuit">注销</aui-button></aui-dropdown-item>
            </aui-dropdown-menu>
          </template>
        </aui-dropdown>
      </div>
    </div>
    <div class="router-container">
      <router-view></router-view>
    </div>
  </div>
</template>

<script setup>
import { inject, onMounted, onUnmounted } from 'vue';
import { useRouter } from 'vue-router';
import {
  UserHead as AuiUserHead,
  Dropdown as AuiDropdown,
  DropdownMenu as AuiDropdownMenu,
  DropdownItem as AuiDropdownItem,
  Button as AuiButton
} from '@aurora/vue3';
import { useUserStore } from '@/store/app.js';
import { useCurrentTestCaseStore, useRunTestCaseResultStore } from '@/store/checker/index.js';
import useLoading from '@/utils/useLoading';

const { apiAssets } = inject('$api');
const router = useRouter();
const userStore = useUserStore();
const { showLoading, closeLoading } = useLoading();
const currentTestCaseStore = useCurrentTestCaseStore();
const runTestCaseResultStore = useRunTestCaseResultStore();

const onQuit = () => {
  showLoading();
  apiAssets
    .fetchDataByCmd({ cmd: 'quit', user_name: userStore.userName })
    .then((res) => {
      closeLoading();
      sessionStorage.clear();
      currentTestCaseStore.$reset();
      runTestCaseResultStore.$reset();
      router.push({ name: 'login' });
    })
    .catch(() => closeLoading());
};

const beforeunloadHandler = (e) => {
  e.returnValue = '弹出确认框';
  return '弹出确认框';
};

const unloadHandler = (e) => {
  fetch(`/hccl/demo/checker?cmd=quit&user_name=${userStore.userName}`, {
    keepalive: true
  });
  sessionStorage.clear();
};

onMounted(() => {
  window.addEventListener('beforeunload', beforeunloadHandler);
  window.addEventListener('unload', unloadHandler);
});

onUnmounted(() => {
  window.removeEventListener('beforeunload', beforeunloadHandler);
  window.removeEventListener('unload', unloadHandler);
});
</script>

<style scoped lang="scss">
.content {
  height: 100%;
  overflow: hidden;
}
.top {
  height: 60px;
  padding: 0 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.logo-name {
  width: 180px;
  height: 60px;
}
.user-container {
  display: flex;
  align-items: center;
}
.user {
  margin-left: 8px;
}

.router-container {
  height: calc(100% - 60px);
}
</style>
