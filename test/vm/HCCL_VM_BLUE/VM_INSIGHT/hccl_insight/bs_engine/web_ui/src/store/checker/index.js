import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useCurrentTestCaseStore = defineStore(
  'currentTestCase',
  () => {
    const currentTestCase = ref({});

    const setCurrentTestCase = (val) => {
      currentTestCase.value = val;
    };

    function $reset() {
      currentTestCase.value = {};
    }

    return { currentTestCase, setCurrentTestCase, $reset };
  },
  {
    persist: false
  }
);

export const useRunTestCaseResultStore = defineStore(
  'runTestCaseResult',
  () => {
    const runTestCaseResult = ref();

    const setRunTestCaseResult = (val) => {
      runTestCaseResult.value = val;
    };

    function $reset() {
      runTestCaseResult.value = undefined;
    }

    return { runTestCaseResult, setRunTestCaseResult, $reset };
  },
  {
    persist: false
  }
);
