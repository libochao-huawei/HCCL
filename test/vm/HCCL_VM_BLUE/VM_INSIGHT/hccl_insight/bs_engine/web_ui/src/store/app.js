import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useUserStore = defineStore(
  'useUser',
  () => {
    const userName = ref();

    const setUserName = (val) => {
      userName.value = val;
    };

    return { userName, setUserName };
  },
  {
    persist: {
      enabled: true,
      key: 'userInfo',
      storage: sessionStorage
    }
  }
);
