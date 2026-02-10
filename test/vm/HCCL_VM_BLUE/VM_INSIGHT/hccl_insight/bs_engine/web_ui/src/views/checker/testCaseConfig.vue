<template>
  <aui-layout class="card">
    <aui-row class="mg-b-28" v-if="testCaseConfig.errorMessage">
      <aui-col>
        <aui-alert type="error" :closable="false" :description="testCaseConfig.errorMessage"></aui-alert>
      </aui-col>
    </aui-row>
    <aui-row class="mg-b-28 title">
      <aui-col>
        <span>基本信息</span>
      </aui-col>
    </aui-row>
    <aui-row class="mg-b-28">
      <aui-col :span="6">
        <span class="config-label"> 用例分组：</span>
        <span>{{ testCaseConfig.testSuite }}</span>
      </aui-col>
      <aui-col :span="6">
        <span class="config-label"> 用例名称：</span>
        <span>{{ testCaseConfig.testCase }}</span>
      </aui-col>
    </aui-row>
    <aui-row class="mg-b-28">
      <aui-col :span="12">
        <span class="config-label"> 关闭内存冲突校验：</span>
        <span>{{ testCaseConfig.memoryConflictCheck }}</span>
      </aui-col>
    </aui-row>
    <aui-row class="mg-b-28">
      <aui-col :span="6">
        <span class="config-label"> 芯片类型：</span>
        <span>{{ testCaseConfig.devtype }}</span>
      </aui-col>
      <aui-col :span="6">
        <span class="config-label"> 算子模式：</span>
        <span>{{ testCaseConfig.opMode }}</span>
      </aui-col>
    </aui-row>
    <aui-row class="mg-b-28">
      <aui-col :span="12">
        <span class="config-label"> 算法名称：</span>
        <span>{{ testCaseConfig.algName }}</span>
      </aui-col>
    </aui-row>
    <aui-row class="mg-b-28">
      <aui-col :span="6">
        <span class="config-label"> 数据类型：</span>
        <span>{{ testCaseConfig.dataType }}</span>
      </aui-col>
      <aui-col :span="6">
        <span class="config-label"> 单个rank操作的数据量：</span>
        <span>{{ testCaseConfig.count }}</span>
      </aui-col>
    </aui-row>
    <aui-row class="mg-b-28">
      <aui-col :span="6">
        <span class="config-label"> 算子类型：</span>
        <span>{{ getLabel(testCaseConfig.opType, HcclCMDType) }}</span>
      </aui-col>
      <aui-col :span="6" v-if="showReduceType(testCaseConfig.opType)">
        <span class="config-label"> reduce操作类型：</span>
        <span>{{ testCaseConfig.reduceType }}</span>
      </aui-col>
    </aui-row>
    <aui-row class="mg-b-28" v-if="showRootRank(testCaseConfig.opType)">
      <aui-col :span="6">
        <span class="config-label"> root rank：</span>
        <span>{{ testCaseConfig.rootRank }}</span>
      </aui-col>
    </aui-row>
    <aui-row class="mg-b-28" v-if="showSrcAndDstRank(testCaseConfig.opType)">
      <aui-col :span="6">
        <span class="config-label">源rank：</span>
        <span>{{ testCaseConfig.srcRank }}</span>
      </aui-col>
      <aui-col :span="6">
        <span class="config-label"> 目的rank：</span>
        <span>{{ testCaseConfig.dstRank }}</span>
      </aui-col>
    </aui-row>
    <aui-row class="mg-b-8">
      <aui-col :span="6">
        <span class="config-label"> 网络配置方式：</span>
        <span>{{ testCaseConfig.topoConfigMode }}</span>
      </aui-col>
      <aui-col :span="6">
        <span class="config-label"> 环境变量：</span>
      </aui-col>
    </aui-row>
    <aui-row>
      <aui-col :span="3" v-if="currentTestCaseStore.currentTestCase.topoConfigMode === 'SYMMETRIC'">
        <aui-grid :data="testCaseConfig.symmetric_topo" auto-resize>
          <aui-grid-column field="superPodNum" show-overflow title="超节点个数"></aui-grid-column>
          <aui-grid-column field="serverNum" title="服务器个数"></aui-grid-column>
          <aui-grid-column field="rankNum" title="NPU个数"></aui-grid-column>
        </aui-grid>
      </aui-col>

      <aui-col :span="3" v-if="currentTestCaseStore.currentTestCase.topoConfigMode === 'ASYMMETRIC'">
        <aui-grid :data="testCaseConfig.asymmetric_topo" :row-span="[{ field: 'superPod' }]" :border="true" auto-resize>
          <aui-grid-column field="superPod" show-overflow title="超节点"></aui-grid-column>
          <aui-grid-column field="server" title="服务器"></aui-grid-column>
          <aui-grid-column field="rank" title="NPU"></aui-grid-column>
        </aui-grid>
      </aui-col>
      <aui-col :span="3"></aui-col>
      <aui-col :span="3">
        <aui-grid :data="testCaseConfig.env" :border="true" auto-resize>
          <template #empty>
            <span>暂无数据</span>
          </template>
          <aui-grid-column field="key" title="变量" width="60%" show-overflow></aui-grid-column>
          <aui-grid-column field="value" title="值" show-overflow></aui-grid-column>
        </aui-grid>
      </aui-col>
    </aui-row>
  </aui-layout>
</template>

<script setup>
import { computed } from 'vue';
import { Layout as AuiLayout, Row as AuiRow, Col as AuiCol, Grid as AuiGrid, GridColumn as AuiGridColumn, Alert as AuiAlert } from '@aurora/vue3';
import { HcclReduceOp, HcclCMDType, OpMode, DevType, HcclDataType, Envs, TopoConfigMode } from '@/utils/commonConstant.js';
import { showReduceType, showRootRank, showSrcAndDstRank } from '@/utils/commonFunction.js';
import { useCurrentTestCaseStore } from '@/store/checker/index.js';

const currentTestCaseStore = useCurrentTestCaseStore();

const getLabel = (value, options) => options.find((item) => item.value === value)?.label;

const testCaseConfig = computed(() => {
  const res = {
    errorMessage: currentTestCaseStore.currentTestCase.errorMessage,
    testSuite: currentTestCaseStore.currentTestCase.testSuite,
    testCase: currentTestCaseStore.currentTestCase.testCase,
    env: currentTestCaseStore.currentTestCase.env,
    topoConfigMode: getLabel(currentTestCaseStore.currentTestCase.topoConfigMode, TopoConfigMode),
    opType: currentTestCaseStore.currentTestCase.opType,
    opMode: getLabel(currentTestCaseStore.currentTestCase.opMode, OpMode),
    devtype: getLabel(currentTestCaseStore.currentTestCase.devtype, DevType),
    algName: currentTestCaseStore.currentTestCase.algName,
    count: currentTestCaseStore.currentTestCase.count,
    dataType: getLabel(currentTestCaseStore.currentTestCase.dataType, HcclDataType),
    reduceType: getLabel(currentTestCaseStore.currentTestCase.reduceType, HcclReduceOp),
    rootRank: currentTestCaseStore.currentTestCase.rootRank,
    srcRank: currentTestCaseStore.currentTestCase.srcRank,
    dstRank: currentTestCaseStore.currentTestCase.dstRank,
    memoryConflictCheck: currentTestCaseStore.currentTestCase.memoryConflictCheck ? '是' : '否'
  };
  if (currentTestCaseStore.currentTestCase.topoConfigMode === 'SYMMETRIC') {
    res.symmetric_topo = [
      {
        superPodNum: currentTestCaseStore.currentTestCase.symmetric_topo[0],
        serverNum: currentTestCaseStore.currentTestCase.symmetric_topo[1],
        rankNum: currentTestCaseStore.currentTestCase.symmetric_topo[2]
      }
    ];
  } else if (currentTestCaseStore.currentTestCase.topoConfigMode === 'ASYMMETRIC') {
    res.asymmetric_topo = currentTestCaseStore.currentTestCase.asymmetric_topo?.reduce((prev, cur, index) => {
      cur.forEach((item, itemIndex) => {
        prev.push({
          superPod: `超节点${index + 1}`,
          server: `server${itemIndex + 1}`,
          rank: item.sort((a, b) => a - b).join(',')
        });
      });
      return prev;
    }, []);
  }
  return res;
});
</script>

<style scoped lang="scss">
.config-label {
  color: #666666;
}
.mg-b-28 {
  margin-bottom: 28px;
}
.mg-b-8 {
  margin-bottom: 8px;
}
.card {
  background-color: #ffffff;
  padding: 26px;
  font-size: 12px;
  color: #000000;
}
.title {
  font-weight: 700;
  font-size: 14px;
  color: #333333;
}
:deep() {
  .aui-grid .aui-grid__empty-block {
    padding: 0;
  }
}
</style>
