<template>
  <div class="memory-semantic" v-if="memorySemantic">
    <template v-for="item in memorySemantic" :key="item.name">
      <SemanticChartItem v-if="item.semantics" :name="item.name" :semantics="item.semantics" :semanticsSize="item.semanticsSize" @onClickChart="onClickChart" />
    </template>
  </div>
  <aui-dialog-box :visible="dialogInfo.visible" @update:visible="dialogInfo.visible = $event" title="srcBufs">
    <aui-grid ref="dialogGridRef" :data="dialogInfo.data.srcBufs">
      <aui-grid-column field="rankId" title="rankId"></aui-grid-column>
      <aui-grid-column field="rankName" title="rankName"></aui-grid-column>
      <aui-grid-column field="bufType" title="bufType"></aui-grid-column>
      <aui-grid-column field="srcAddr" title="srcAddr"></aui-grid-column>
    </aui-grid>
    <template #footer>
      <aui-button @click="exportSrcBufs" type="primary">导出数据</aui-button>
      <aui-button @click="dialogInfo.visible = false">关闭</aui-button>
    </template>
  </aui-dialog-box>
</template>

<script setup>
import { ref, reactive, onMounted, inject, onUnmounted, onUpdated } from 'vue';
import { Button as AuiButton, Grid as AuiGrid, GridColumn as AuiGridColumn, DialogBox as AuiDialogBox } from '@aurora/vue3';
import SemanticChartItem from '@/components/SemanticChartItem.vue';
import { canGetSemanticResult, getRank } from '@/utils/commonFunction.js';
import { useCurrentTestCaseStore, useRunTestCaseResultStore } from '@/store/checker/index.js';
import { useUserStore } from '@/store/app.js';
import useLoading from '@/utils/useLoading';

const { apiAssets } = inject('$api');
const { userName: user_name } = useUserStore();
const { showLoading, closeLoading } = useLoading();
const currentTestCaseStore = useCurrentTestCaseStore();
const runTestCaseResultStore = useRunTestCaseResultStore();
const props = defineProps(['activeTab', 'rankOptions', 'rankValue', 'curReplayNode']);

let timer = null;
let curUpdateKey = props.rankValue + props.curReplayNode.nodeId;
const memorySemantic = ref();

const dialogInfo = reactive({
  visible: false,
  data: {}
});

const dialogGridRef = ref();

const onClickChart = (params) => {
  const item = params.data.item;
  if (item.srcBufs) {
    dialogInfo.data = {
      baseInfo: {
        startAddr: item.startAddr,
        size: item.size,
        isReduce: item.isReduce,
        reduceType: item.reduceType
      },
      srcBufs: item.srcBufs.map((item) => {
        return {
          rankId: item.rankId,
          rankName: getRank(props.rankOptions, item.rankId).label,
          bufType: item.bufType,
          srcAddr: item.srcAddr
        };
      })
    };
    dialogInfo.visible = true;
  }
};

const exportSrcBufs = () => {
  dialogGridRef.value.exportCsv({
    filename: 'srcBufs.csv',
    original: true,
    isHeader: true,
    data: dialogInfo.data.srcBufs
  });
};

const getSemanticResult = () => {
  showLoading();
  apiAssets
    .fetchDataByCmd({
      cmd: 'get_semantic',
      user_name,
      tc_name: currentTestCaseStore.currentTestCase.id,
      rank_id: props.rankValue,
      local_step: props.curReplayNode.localStep.localStep
    })
    .then((res) => {
      return loopFetchData(user_name).then((data) => {
        closeLoading();
        memorySemantic.value = [
          {
            name: 'input',
            semanticsSize: data.inputSize,
            semantics: data.semantic.inputBufferSemantics
          },
          {
            name: 'output',
            semanticsSize: data.outputSize,
            semantics: data.semantic.outputBufferSemantics
          },
          {
            name: 'inputCCL',
            semanticsSize: data.inputCCLSize,
            semantics: data.semantic.inputCCLBufferSemantics
          },
          {
            name: 'outputCCL',
            semanticsSize: data.outputCCLSize,
            semantics: data.semantic.outputCCLBufferSemantics
          },
          {
            name: 'scratch',
            semanticsSize: data.scratchSize,
            semantics: data.semantic.scratchBufferSemantics
          }
        ];
      });
    })
    .catch(() => closeLoading());
};

const loopFetchData = (user_name) => {
  const fetchData = (resolve, reject) =>
    apiAssets
      .fetchDataByCmd({ cmd: 'get_msg', user_name })
      .then((res) => {
        if (res.data.status) {
          timer && clearTimeout(timer);
          resolve(res.data.data);
        } else {
          timer = setTimeout(() => fetchData(resolve, reject), 1000);
        }
      })
      .catch((error) => {
        reject(error);
      });

  return new Promise((resolve, reject) => {
    fetchData(resolve, reject);
  });
};

onUpdated(() => {
  const { errorType } = runTestCaseResultStore.runTestCaseResult;
  const newUpdateKey = props.rankValue + props.curReplayNode.nodeId;
  if (props.activeTab === 'semantic' && curUpdateKey !== newUpdateKey) {
    memorySemantic.value = null;
    curUpdateKey = newUpdateKey;
    if (props.curReplayNode.nodeName !== 'Dummy_Start' && canGetSemanticResult(errorType)) {
      getSemanticResult();
    }
  }
});

onMounted(() => {
  const { errorType } = runTestCaseResultStore.runTestCaseResult;
  if (props.curReplayNode.nodeName !== 'Dummy_Start' && canGetSemanticResult(errorType)) {
    getSemanticResult();
  }
});
onUnmounted(() => {
  timer && clearTimeout(timer);
});
</script>
<style lang="scss">
.memory-semantic {
  display: flex;
  width: 100%;
  height: 100%;
  flex-wrap: nowrap;
  padding-top: 16px;
  overflow: auto;
}

.memory-semantic-item:first-child {
  padding-left: 0;
  min-width: 521px;
}
.memory-semantic-item:last-child {
  border: none;
  padding-right: 0;
  min-width: 520px;
}
.memory-semantic-item:last-child > .value-options {
  right: 0;
}
</style>
