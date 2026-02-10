<template>
  <div class="login-container">
    <div class="form-container">
      <div class="logo">
        <i>
          <svg class="logo-name">
            <use xlink:href="#icon-logoName"></use>
          </svg>
        </i>
      </div>
      <aui-form ref="loginFormRef" label-position="top" validate-type="text" :model="loginForm.data" :rules="loginForm.rules">
        <aui-form-item label="用户名称" prop="userName">
          <aui-input v-model="loginForm.data.userName"></aui-input>
        </aui-form-item>
      </aui-form>
      <div class="action"><aui-button type="primary" @click="onSubmit">登录</aui-button></div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, inject, onMounted, onUnmounted } from 'vue';
import { Input as AuiInput, Form as AuiForm, FormItem as AuiFormItem, Button as AuiButton } from '@aurora/vue3';
import { useRouter } from 'vue-router';
import { useUserStore } from '@/store/app.js';
import useLoading from '@/utils/useLoading';

const { showLoading, closeLoading } = useLoading();
const { apiAssets } = inject('$api');
const router = useRouter();
const userStore = useUserStore();
const loginFormRef = ref();
const loginForm = reactive({
  data: { userName: '' },
  rules: {
    userName: [{ required: true, pattern: /^[0-9a-zA-Z_]+$/, message: '字母数字下划线组成' }]
  }
});
const onSubmit = () => {
  loginFormRef.value.validate((res) => {
    if (res) {
      showLoading();
      apiAssets
        .fetchDataByCmd({ cmd: 'register', user_name: loginForm.data.userName })
        .then((res) => {
          console.log('register');
          closeLoading();
          userStore.setUserName(loginForm.data.userName);
          router.push({ name: 'checker' });
        })
        .catch(() => closeLoading());
    }
  });
};

const onKeypress = (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    onSubmit();
  }
};

onMounted(() => {
  document.addEventListener('keypress', onKeypress);
});

onUnmounted(() => {
  document.removeEventListener('keypress', onKeypress);
});
</script>

<style scoped>
.login-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  width: 100%;
  justify-content: center;
  align-items: center;
  background: #fcf7f0;
}
.form-container {
  width: 400px;
  padding: 0 24px 24px;
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 0 4px 4px rgba(204, 204, 204, 0.2);
}
.logo {
  text-align: center;
}
.action {
  text-align: center;
  margin-top: 24px;
}
</style>
