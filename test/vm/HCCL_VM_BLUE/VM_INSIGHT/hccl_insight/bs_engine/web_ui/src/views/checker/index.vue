<template>
  <aui-split v-model="split1" trigger-simple>
    <template #left>
      <TestCaseTree @setActiveTab="setActiveTab" />
    </template>
    <template #right>
      <aui-tabs v-model="activeTab" tab-style="card" style="height: 100%">
        <aui-tab-item title="用例配置" name="config" style="height: 100%">
          <div class="config-container">
            <TestCaseConfig v-if="currentTestCaseStore.currentTestCase.id" />
          </div>
        </aui-tab-item>
        <aui-tab-item title="执行结果" name="result" style="height: 100%" lazy>
          <div class="result-container">
            <RunResult v-if="runTestCaseResultStore.runTestCaseResult" />
          </div>
        </aui-tab-item>
      </aui-tabs>
    </template>
  </aui-split>
</template>

<script setup lang="jsx">
import { ref, defineAsyncComponent } from 'vue';
import { Tabs as AuiTabs, TabItem as AuiTabItem, Split as AuiSplit } from '@aurora/vue3';
import TestCaseTree from '@/views/checker/testCaseTree.vue';
import TestCaseConfig from '@/views/checker/testCaseConfig.vue';
import { useCurrentTestCaseStore, useRunTestCaseResultStore } from '@/store/checker/index.js';
import LoadingComponent from '@/components/LoadingComponent';

const RunResult = defineAsyncComponent({
  loader: () => import('@/views/checker/runResult/index.vue'),
  loadingComponent: LoadingComponent,
  delay: 50
});
const currentTestCaseStore = useCurrentTestCaseStore();
const runTestCaseResultStore = useRunTestCaseResultStore();
const split1 = ref('300px');

const activeTab = ref('config');

const setActiveTab = (val) => {
  activeTab.value = val;
};
</script>

<style scoped lang="scss">
.config-container {
  padding: 16px 32px;
  height: 100%;
  overflow: auto;
  background-color: #f5f5f5;
}
.result-container {
  height: 100%;
  background-color: #f5f5f5;
}
:deep() {
  .aui-tabs__content {
    height: calc(100% - 42px);
    padding: 0;
  }
  .aui-tabs.aui-tabs--card.aui-tabs--top > .aui-tabs__header .aui-tabs__item.is-active {
    background-color: #f5f5f5;
    border-bottom-color: #f5f5f5;
  }
}
</style>
