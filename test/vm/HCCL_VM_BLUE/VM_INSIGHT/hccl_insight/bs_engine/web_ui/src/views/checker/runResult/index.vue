<template>
  <aui-split v-model="split" trigger-simple mode="vertical" collapse-right-bottom>
    <template #top>
      <div class="info">
        <div class="filter">
          <span>选择rank：</span>
          <aui-select class="rank-select" v-model="rankValue" placeholder="请选择" :options="rankOptions" @change="onChangeRank"></aui-select>
          <span class="mg-l-32">选择回放节点：</span>
          <aui-input class="rank-select" v-model="replayNode.nodeName" placeholder="请点击节点选择" disabled></aui-input>
        </div>

        <div class="tool-bar">
          <div class="tool-bar-left">
            <span class="tool-bar-left-item" title="还原" @click="onResetTopo"><AuiIconEditorUndo></AuiIconEditorUndo></span>
            <span class="tool-bar-left-item" title="聚焦" @click="onFocusNode">
              <svg style="width: 16px; height: 16px; vertical-align: top">
                <use xlink:href="#icon-focus"></use>
              </svg>
            </span>
          </div>
          <div class="tool-bar-right">
            <aui-radio-group v-model="topoViewVal" size="mini" @change="onChangeTopoViewVal">
              <aui-radio-button label="unilateral">原始图</aui-radio-button>
              <aui-radio-button label="bilateral">双边语义图</aui-radio-button>
            </aui-radio-group>
          </div>
        </div>
        <div ref="topoChartRef" class="topo-container"></div>
      </div>
    </template>
    <template #bottom>
      <div class="result">
        <aui-tabs v-model="activeTab" style="height: 100%" @click="onChangeActiveTab">
          <aui-tab-item title="日志" name="log" style="height: 100%">
            <div class="log">
              <Log
                :topologyInfo="topologyInfo"
                :rankOptions="rankOptions"
                :rankValue="rankValue"
                :errorType="runTestCaseResultStore.runTestCaseResult.errorType"
                :errorInfo="runTestCaseResultStore.runTestCaseResult.errorInfo"
                @redirectErrorSemantic="redirectErrorSemantic"
                @redirectErrorMemory="redirectErrorMemory"
              />
            </div>
          </aui-tab-item>
          <aui-tab-item title="内存冲突" name="conflict" style="height: 100%">
            <div
              ref="memoryConflictChartRef"
              class="memory-conflict"
              v-if="
                runTestCaseResultStore.runTestCaseResult.errorType === ResultStatus.MEMORY_CONFLICT &&
                runTestCaseResultStore.runTestCaseResult.errorInfo?.errorMemory?.rankId === rankValue &&
                topoViewVal === 'bilateral'
              "
            ></div>
          </aui-tab-item>
          <aui-tab-item title="语义" name="semantic" style="height: 100%" lazy>
            <SemanticChart :activeTab="activeTab" :rankOptions="rankOptions" :rankValue="rankValue" :curReplayNode="replayNode" />
          </aui-tab-item>
        </aui-tabs>
      </div>
    </template>
  </aui-split>
</template>

<script setup lang="jsx">
import { ref, computed, inject, onUnmounted, onMounted, nextTick, defineAsyncComponent, getCurrentInstance } from 'vue';
import ResizeObserver from 'resize-observer-polyfill';
import {
  Split as AuiSplit,
  Select as AuiSelect,
  Input as AuiInput,
  Tabs as AuiTabs,
  TabItem as AuiTabItem,
  RadioButton as AuiRadioButton,
  RadioGroup as AuiRadioGroup
} from '@aurora/vue3';
import { IconEditorUndo } from '@aurora/vue3-icon';
import { ResultStatus } from '@/utils/commonConstant.js';
import { needFromLogToSemantic } from '@/utils/commonFunction.js';
import { drawerTopo, onUndo, updatedTopo, focusNode, updatedNodeState } from '@/views/checker/runResult/topoChart.js';
import Log from '@/views/checker/runResult/log.jsx';
import drawerMemoryConflictChart from '@/views/checker/runResult/memConflictChart.js';
import { useUserStore } from '@/store/app.js';
import { useCurrentTestCaseStore, useRunTestCaseResultStore } from '@/store/checker/index.js';
import useLoading from '@/utils/useLoading';

const SemanticChart = defineAsyncComponent(() => import('@/views/checker/runResult/semanticChart.vue'));
const AuiIconEditorUndo = IconEditorUndo();
const instance = getCurrentInstance();
const $echarts = instance.appContext.config.globalProperties.$echarts;
const { apiAssets } = inject('$api');
const currentTestCaseStore = useCurrentTestCaseStore();
const runTestCaseResultStore = useRunTestCaseResultStore();
const { userName: user_name } = useUserStore();
const { showLoading, closeLoading } = useLoading();
let timer = null;
let infoDomObserver = null;
let topoGraph = null;
const split = ref(0.6);
let prevRankValue = runTestCaseResultStore.runTestCaseResult.topologyInfo.rankNodes.rankId;
const rankValue = ref(runTestCaseResultStore.runTestCaseResult.topologyInfo.rankNodes.rankId);
const topoViewVal = ref(runTestCaseResultStore.runTestCaseResult.errorType === ResultStatus.MEMORY_CONFLICT ? 'bilateral' : 'unilateral');
const activeTab = ref('log');
const topologyInfo = ref(runTestCaseResultStore.runTestCaseResult.topologyInfo);
const topoChartRef = ref();
const replayNode = ref(runTestCaseResultStore.runTestCaseResult.topologyInfo.rankNodes.nodes[0]);
const rankOptions = computed(() => {
  return runTestCaseResultStore.runTestCaseResult.topologyInfo.rankInfo.map((item) => {
    return {
      value: item.rankId,
      label: item.name
    };
  });
});
const memoryConflictChartRef = ref();

const onResetTopo = () => {
  onUndo({
    nodeInfo: {
      nodeId: replayNode.value.nodeId,
      localStep: replayNode.value.localStep
    },
    nodes: topologyInfo.value.rankNodes.nodes,
    links: topologyInfo.value.links,
    errorType: runTestCaseResultStore.runTestCaseResult.errorType
  });
};

const onFocusNode = () => {
  focusNode(replayNode.value.nodeId);
};

const onChangeTopoViewVal = (label) => {
  if (label === 'bilateral') {
    switchTopoViewVal('bilateral');
  } else {
    switchTopoViewVal('unilateral');
  }
};

const switchTopoViewVal = (topo_view) => {
  setActiveTab('log');
  showLoading();
  apiAssets
    .fetchDataByCmd({
      cmd: 'switch_topo_view',
      user_name,
      tc_name: currentTestCaseStore.currentTestCase.id,
      rank_id: rankValue.value,
      topo_view
    })
    .then((res) => {
      return loopFetchData(user_name).then((data) => {
        topologyInfo.value = data.topologyInfo;
        const item = data.topologyInfo.rankNodes.nodes[0];
        setReplayNode(item);
        updatedTopo({
          nodeInfo: item,
          nodes: data.topologyInfo.rankNodes.nodes,
          links: data.topologyInfo.links,
          errorType: runTestCaseResultStore.runTestCaseResult.errorType
        }).then(() => topoGraph.fitView());
        closeLoading();
      });
    })
    .catch(() => {
      closeLoading();
      topoViewVal.value = topo_view === 'bilateral' ? 'unilateral' : 'bilateral';
    });
};

const setReplayNode = (data) => {
  replayNode.value = data;
};

const setActiveTab = (val) => {
  activeTab.value = val;
};

const redirectToSemantic = (errorType) => {
  needFromLogToSemantic(errorType) && setActiveTab('semantic');
};

const redirectErrorSemantic = ({ errorSemanticItem, errorType }) => {
  if (errorSemanticItem.rankId === rankValue.value) {
    const item = topologyInfo.value.rankNodes.nodes.find((item) => item.localStep.localStep === errorSemanticItem.localStep);
    if (replayNode.value.nodeId === item.nodeId) {
      focusNode(item.nodeId).then(() => redirectToSemantic(errorType));
    } else {
      setReplayNode(item);
      updatedNodeState({
        targetNode: item,
        nodes: topologyInfo.value.rankNodes.nodes,
        errorType
      })
        .then(() => {
          return focusNode(item.nodeId);
        })
        .then(() => redirectToSemantic(errorType));
    }
  } else {
    switchRank(errorSemanticItem.rankId, rankValue.value, errorSemanticItem.localStep).then((nodeInfo) => {
      focusNode(nodeInfo.nodeId).then(() => {
        redirectToSemantic();
      });
    });
  }
};

const redirectErrorMemory = (errorMemory) => {
  if (errorMemory?.rankId === rankValue.value) {
    setActiveTab('conflict');
    nextTick(() => drawerMemoryConflict(errorMemory));
  } else {
    switchRank(errorMemory.rankId, rankValue.value, replayNode.value.localStep.localStep).then(() => {
      setActiveTab('conflict');
      nextTick(() => drawerMemoryConflict(errorMemory));
    });
  }
};

const drawerMemoryConflict = (errorMemory) => {
  drawerMemoryConflictChart({
    $echarts,
    dom: memoryConflictChartRef.value,
    errorMemory,
    nodes: topologyInfo.value.rankNodes.nodes,
    rankOptions: rankOptions.value,
    onClick: (nodeA, nodeB, targetNode) => {
      setReplayNode(targetNode);
      updatedNodeState({
        targetNode: [nodeA, nodeB],
        nodes: topologyInfo.value.rankNodes.nodes,
        errorType: runTestCaseResultStore.runTestCaseResult.errorType
      }).then(() => {
        return focusNode(targetNode.nodeId);
      });
    }
  });
};

const onChangeRank = (value) => {
  switchRank(value, prevRankValue, replayNode.value.localStep.localStep).then(() => topoGraph.fitView());
};

const switchRank = (targetRankId, oriRankId, originLocalStep) => {
  prevRankValue = targetRankId;
  rankValue.value = targetRankId;
  setActiveTab('log');
  showLoading();
  return apiAssets
    .fetchDataByCmd({
      cmd: 'switch_rank',
      user_name,
      tc_name: currentTestCaseStore.currentTestCase.id,
      origin_rank: oriRankId,
      origin_local_step: originLocalStep,
      target_rank: targetRankId
    })
    .then((res) => {
      return loopFetchData(user_name).then((data) => {
        if (runTestCaseResultStore.runTestCaseResult.errorType === ResultStatus.MEMORY_CONFLICT) {
          topoViewVal.value = 'bilateral';
        } else {
          topoViewVal.value = 'unilateral';
        }
        topologyInfo.value = data.topologyInfo;
        const item = data.topologyInfo.rankNodes.nodes.find((item) => item.localStep.localStep === originLocalStep) || data.topologyInfo.rankNodes.nodes[0];
        setReplayNode(item);
        closeLoading();
        return updatedTopo({
          nodeInfo: item,
          nodes: topologyInfo.value.rankNodes.nodes,
          links: topologyInfo.value.links,
          errorType: runTestCaseResultStore.runTestCaseResult.errorType
        });
      });
    })
    .catch(() => closeLoading());
};

const onChangeActiveTab = (tab) => {
  const { errorType, errorInfo } = runTestCaseResultStore.runTestCaseResult;
  if (
    tab.name === 'conflict' &&
    errorType === ResultStatus.MEMORY_CONFLICT &&
    topoViewVal.value === 'bilateral' &&
    errorInfo.errorMemory.rankId === rankValue.value
  ) {
    nextTick(() => drawerMemoryConflict(runTestCaseResultStore.runTestCaseResult.errorInfo.errorMemory));
  }
};

const createInfoDomObserver = () => {
  infoDomObserver = new ResizeObserver((entries) => {
    nextTick(() => {
      const topoContainer = document.querySelector('.topo-container');
      topoGraph.resize(topoContainer.offsetWidth, topoContainer.offsetHeight);
    });
  });

  infoDomObserver.observe(document.querySelector('.info'));
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

onMounted(() => {
  drawerTopo({
    container: topoChartRef.value,
    nodeInfo: {
      nodeId: replayNode.value.nodeId,
      localStep: replayNode.value.localStep
    },
    nodes: topologyInfo.value.rankNodes.nodes,
    links: topologyInfo.value.links,
    errorType: runTestCaseResultStore.runTestCaseResult.errorType,
    setTopoGraph: (graph) => {
      topoGraph = graph;
    },
    onCliclNode: (e) => {
      const node = topoGraph.getElementData(e.target.id);
      if (replayNode.value.nodeId != node.id) {
        setReplayNode && setReplayNode(node.data);
        updatedNodeState({
          targetNode: node.data,
          nodes: topologyInfo.value.rankNodes.nodes,
          errorType: runTestCaseResultStore.runTestCaseResult.errorType
        });
      }
    },
    onCliclEdge: (e) => {
      const updateEdgeState = {};
      const nodes = topoGraph.getElementDataByState('edge', 'selectedEdge');
      if (nodes.length) {
        updateEdgeState[nodes[0].id] = '';
      }
      updateEdgeState[e.target.id] = 'selectedEdge';
      topoGraph.setElementState(updateEdgeState);
    }
  });

  createInfoDomObserver();
});

onUnmounted(() => {
  timer && clearTimeout(timer);
  infoDomObserver.disconnect();
});
</script>
<style scoped lang="scss">
.info {
  padding: 16px 32px 0;
  height: 100%;
}
.rank-select {
  width: 270px;
}
.filter {
  background: #ffffff;
  padding: 16px 24px;
  color: #666666;
  font-size: 12px;
}
.mg-l-32 {
  margin-left: 32px;
}

.topo-container {
  height: calc(100% - 122px);
}
.topo-node-item {
  width: 50px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.tool-bar {
  margin: 16px 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.tool-bar-left-item {
  background: #fff;
  padding: 6px;
  cursor: pointer;
}

.aui-split-wrapper {
  box-shadow: none;
}
.result {
  padding: 0;
  height: 100%;
  overflow: hidden;
  background: #ffffff;
  padding: 0 16px;
  font-size: 14px;
}
.log {
  padding-top: 16px;
  height: 100%;
  overflow: auto;
}
.memory-conflict {
  height: 100%;
  overflow: auto;
}
</style>
<style>
.log-ul li {
  margin-bottom: 8px;
  line-height: 20px;
}
</style>
