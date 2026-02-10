<template>
  <div class="memory-semantic-item">
    <div class="tool">
      <span>Y轴类型：</span>
      <aui-radio-group v-model="toolData.yAxisType" :options="yAxisTypeOptions"></aui-radio-group>
      <span class="mg-l-24">地址范围：</span>
      <aui-numeric class="numeric-width" v-model="toolData.startValue" :controls="false" size="mini" allow-empty show-left></aui-numeric>
      <span class="scope-line">-</span>
      <aui-numeric class="numeric-width" v-model="toolData.endValue" :controls="false" size="mini" allow-empty show-left></aui-numeric>
      <aui-button class="mg-l-16" size="mini" type="primary" @click="onSubmit">提交</aui-button>
    </div>
    <div class="chart-container" ref="chartRef"></div>
    <div class="value-options">
      <aui-checkbox-group v-model="checked" @change="change" vertical>
        <aui-checkbox v-for="item in options" :key="item.name" :label="item.name" :disabled="checked.length >= 2 && !checked.includes(item.name)">
          <div class="option-name" :title="item.name">{{ item.name }}</div>
        </aui-checkbox>
      </aui-checkbox-group>
    </div>
  </div>
</template>
<script setup>
import { ref, getCurrentInstance, nextTick, reactive, onMounted, inject, onUnmounted, onUpdated } from 'vue';
import {
  Button as AuiButton,
  RadioGroup as AuiRadioGroup,
  Numeric as AuiNumeric,
  Checkbox as AuiCheckbox,
  CheckboxGroup as AuiCheckboxGroup
} from '@aurora/vue3';
import { Colors, BufferType, ResultStatus } from '@/utils/commonConstant.js';
import { useRunTestCaseResultStore } from '@/store/checker/index.js';

const runTestCaseResultStore = useRunTestCaseResultStore();
const props = defineProps(['semantics', 'semanticsSize', 'name']);
const emits = defineEmits(['onClickChart']);
const instance = getCurrentInstance();
const $echarts = instance.appContext.config.globalProperties.$echarts;

const chartRef = ref();

const yAxisTypeOptions = ref([
  { label: 'value', text: '数值' },
  { label: 'log', text: '对数' }
]);
const toolData = reactive({
  yAxisType: 'value',
  startValue: null,
  endValue: null
});

const checked = ref([]);
const options = ref([]);

const drawerChart = ({ node, semantics, semanticsSize, name }) => {
  const { errorType, errorInfo } = runTestCaseResultStore.runTestCaseResult;
  const errorSemantic = errorInfo?.errorSemantic.find((item) => item.rankId === props.rankValue && item.localStep === props.curReplayNode.localStep.localStep);
  const myChart = $echarts.init(node);
  const series = semantics
    .map((item) => {
      const { startAddr, size } = item;
      return {
        startAddr,
        end: startAddr + size,
        ...item,
        srcBufs: item.srcBufs
      };
    })
    .sort((a, b) => a.startAddr - b.startAddr)
    .reduce((prev, cur, index, arr) => {
      const prevEndItem = prev[prev.length - 1];
      if (prevEndItem) {
        if (cur.startAddr > prevEndItem.end) {
          prev.push({ startAddr: prevEndItem.end, end: cur.startAddr }, cur);
        } else {
          prev.push(cur);
        }
      } else {
        if (cur.startAddr > 0) {
          prev.push({ startAddr: 0, end: cur.startAddr }, cur);
        } else {
          prev.push(cur);
        }
      }
      if (index === arr.length - 1 && cur.end < semanticsSize) {
        prev.push({ startAddr: cur.end, end: semanticsSize });
      }
      return prev;
    }, [])
    .reduce((prev, item, index, arr) => {
      let isMissingSemantic = false;
      if (
        errorType === ResultStatus.CHECK_FAILED_MISSING_SEMANTIC &&
        errorSemantic &&
        name.toUpperCase() === BufferType[errorSemantic?.bufferType] &&
        item.startAddr == errorSemantic?.startAddr
      ) {
        isMissingSemantic = true;
      }
      if (!isMissingSemantic && (index === 0 || index === arr.length - 1) && !item.srcBufs) {
        return prev;
      }
      let itemStyle = {};
      if (isMissingSemantic) {
        itemStyle = {
          color: '#ccc',
          decal: {
            color: 'rgba(255,255,255,.7)'
          },
          borderColor: 'red',
          borderWidth: 2
        };
      } else if (!item.srcBufs) {
        itemStyle = {
          color: '#ccc',
          decal: {
            color: 'rgba(255,255,255,.7)'
          }
        };
      } else if (item.isReduce) {
        itemStyle = {
          ...(item.invalid ? { borderColor: 'red', borderWidth: 2 } : {}),
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 1,
            y2: 0,
            global: false,
            colorStops: item.srcBufs.reduce((prev, cur, index, arr) => {
              const color = Colors[index % Colors.length];
              console.log(index, index % Colors.length, color);
              const k = 1 / arr.length;
              prev.push({ offset: k * index, color }, { offset: k * index + k, color });
              return prev;
            }, [])
          }
        };
      } else {
        itemStyle = item.invalid ? { borderColor: 'red', borderWidth: 2 } : {};
      }

      prev.push({
        name: `${item.startAddr}-${item.end}`,
        type: 'bar',
        stack: 'total',
        data: [{ value: item.end - item.startAddr, item }],
        tooltip: {
          confine: true,
          formatter: (params) => {
            const { value, data } = params;
            if (data.item.srcBufs) {
              return `<ul><li>size：${value}</li><li>startAddr：${data.item.startAddr}</li><li>endAddr：${data.item.end}</li><li>isReduce：${
                data.item.isReduce
              }</li><li>reduceType：${data.item.reduceType}</li><li>srcBufs：点击查看详情</li></ul>`;
            } else {
              return `<ul><li>size：${value}</li><li>startAddr：${data.item.startAddr}</li><li>endAddr：${data.item.end}</li></ul>`;
            }
          }
        },
        itemStyle
      });
      return prev;
    }, []);

  const option = {
    color: Colors,
    tooltip: { show: true },
    dataZoom: [
      {
        type: 'inside',
        filterMode: 'none',
        yAxisIndex: 0
      }
    ],
    grid: {
      left: 80,
      right: 140,
      top: 10,
      bottom: 22
    },
    xAxis: {
      type: 'category',
      data: [name],
      axisTick: { show: false },
      axisLabel: { formatter: name, fontWeight: 600, fontSize: 14 }
    },
    yAxis: {
      type: 'value',
      axisLabel: { formatter: (value) => (value >= 10000 ? value.toExponential(2) : value) }
    },
    series: series
  };
  memorySemanticChartEvents(myChart);
  myChart.setOption(option);
  options.value = series;
};

const memorySemanticChartEvents = (chart) => {
  chart.on('click', (params) => {
    emits('onClickChart', params);
  });
};

const onSubmit = () => {
  let chart = $echarts.getInstanceByDom(chartRef.value);
  let yAxisType = toolData.yAxisType;
  let startValue = toolData.startValue;
  let endValue = toolData.endValue;

  const option = chart.getOption();
  option.yAxis[0].type = yAxisType;
  if (typeof startValue === 'number') {
    option.dataZoom[0].startValue = startValue;
    delete option.dataZoom[0].start;
  } else {
    delete option.dataZoom[0].startValue;
    option.dataZoom[0].start = 0;
  }
  if (typeof endValue === 'number') {
    option.dataZoom[0].endValue = endValue;
    delete option.dataZoom[0].end;
  } else {
    delete option.dataZoom[0].endValue;
    option.dataZoom[0].end = 100;
  }

  chart.setOption(option);
};

const change = (value) => {
  let start;
  let end;
  if (value.length === 0) {
    start = null;
    end = null;
  } else if (value.length === 1) {
    [start, end] = value[0].split('-').map((item) => Number(item));
  } else if (value.length === 2) {
    start = Number(value[0].split('-')[0]);
    end = Number(value[1].split('-')[1]);
  }

  toolData.startValue = start;
  toolData.endValue = end;

  onSubmit();
};

onMounted(() => {
  drawerChart({
    node: chartRef.value,
    semantics: props.semantics,
    semanticsSize: props.semanticsSize,
    name: props.name
  });
});
</script>
<style scoped lang="scss">
.memory-semantic-item {
  min-height: 264px;
  min-width: 545px;
  padding: 0 24px;
  border-right: 1px solid #d9d9d9;
  position: relative;
}

.mg-l-24 {
  margin-left: 24px;
}
.mg-l-16 {
  margin-left: 16px;
}
.numeric-width {
  width: 60px;
}
.tool {
  font-size: 12px;
  margin-bottom: 16px;
  color: #666666;
}
.chart-container {
  height: calc(100% - 44px);
}
.scope-line {
  padding: 0 4px;
}
.value-options {
  position: absolute;
  top: 42px;
  right: 24px;
  max-height: calc(100% - 44px);
  overflow-y: scroll;
  padding: 4px;
  border: 1px solid #d9d9d9;
}
.option-name {
  max-width: 80px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  display: inline-block;
}
</style>
