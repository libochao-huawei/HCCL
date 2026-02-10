import { Category, Colors } from '@/utils/commonConstant.js';
import { getRank, getNode } from '@/utils/commonFunction.js';

export default function drawerMemoryConflictChart({ $echarts, dom, errorMemory, nodes, rankOptions, onClick }) {
  const instance = $echarts.getInstanceByDom(dom);
  if (instance) {
    return;
  }
  const width = 80;
  const height = 160;
  const { dataSliceA, dataSliceB, nodeIdA, nodeIdB, rankId } = errorMemory;
  const conflictScope = getConflictScope(dataSliceA, dataSliceB);
  const conflictValue = conflictScope[1] - conflictScope[0];
  const conflictScopeHeight = height * (conflictValue / dataSliceA.size);
  const conflictScopeStart = height - conflictScopeHeight;
  const nodeA = getNode(nodes, nodeIdA);
  const nodeB = getNode(nodes, nodeIdB);
  const rankInfo = getRank(rankOptions, rankId);
  const nodeATooltip = `
  <ul><li><strong>节点A</strong></li><li>Rank：${rankInfo.label}</li><li>节点名称：${nodeA?.nodeName}</li><li>memOp：${dataSliceA.memOp}</li><li>startAddr：${dataSliceA.startAddr}</li><li>endAddr：${dataSliceA.endAddr}</li><li>size：${dataSliceA.size}</li><li>冲突区域：${JSON.stringify(conflictScope)}</li></ul>
  `;
  const nodeBTooltip = `
  <ul><li><strong>节点B</strong></li><li>Rank：${rankInfo.label}</li><li>节点名称：${nodeB?.nodeName}</li><li>memOp：${dataSliceB.memOp}</li><li>startAddr：${dataSliceB.startAddr}</li><li>endAddr：${dataSliceB.endAddr}</li><li>size：${dataSliceB.size}</li><li>冲突区域：${JSON.stringify(conflictScope)}</li></ul>
  `;
  const myChart = $echarts.init(dom, null, { height: 210 + conflictScopeStart, width: 320 });
  const option = {
    tooltip: {},
    series: [
      {
        type: 'custom',
        coordinateSystem: 'none',
        renderItem: (params, api) => {
          if (api.value(2) === 'rect') {
            return {
              type: 'rect',
              transition: ['shape'],
              shape: { x: api.value(0), y: api.value(1), width: api.value(3).width, height: api.value(3).height },
              style: api.style()
            };
          } else if (api.value(2) === 'text') {
            return { type: 'text', style: { x: api.value(0), y: api.value(1), text: api.value(3), width: 60, overflow: 'break' } };
          }
        },
        data: [
          { value: [50, 110, 'text', '节点A'] },
          { value: [115, 15, 'text', 'startAddr'] },
          { value: [115, 195, 'text', 'endAddr'] },
          {
            value: [100, 30, 'rect', { width, height }],
            name: 'nodeA',
            itemStyle: { color: Category[nodeA?.category]?.color || Colors[0] },
            tooltip: {
              position: [182, 80],
              formatter: () => nodeATooltip
            }
          },
          {
            value: [100, 30 + conflictScopeStart, 'rect', { width, height: conflictScopeHeight }],
            name: 'conflictScope_nodeA',
            itemStyle: { color: 'red' },
            tooltip: {
              position: [182, 80],
              formatter: () => nodeATooltip
            }
          },
          { value: [277, 105 + conflictScopeStart, 'text', '节点B'] },
          { value: [197, 15 + conflictScopeStart, 'text', 'startAddr'] },
          {
            value: [197, 195 + conflictScopeStart, 'text', 'endAddr']
          },
          {
            value: [182, 30 + conflictScopeStart, 'rect', { width, height }],
            name: 'nodeB',
            itemStyle: {
              color: Category[nodeB?.category]?.color || Colors[1]
            },
            tooltip: {
              position: [264, 80],
              formatter: () => nodeBTooltip
            }
          },
          {
            value: [182, 30 + conflictScopeStart, 'rect', { width, height: conflictScopeHeight }],
            name: 'conflictScope_nodeB',
            itemStyle: { color: 'red' },
            tooltip: {
              position: [264, 80],
              formatter: () => nodeBTooltip
            }
          }
        ]
      }
    ]
  };
  myChart.setOption(option);
  memoryConflictChartEvents(myChart, nodeA, nodeB, onClick);
}

const memoryConflictChartEvents = (chart, nodeA, nodeB, onClick) => {
  chart.on('click', function (params) {
    const { name } = params;
    let node;
    if (name === 'nodeA') {
      node = nodeA;
    } else if (name === 'nodeB') {
      node = nodeB;
    } else if (name === 'conflictScope_nodeA') {
      node = nodeA;
    } else if (name === 'conflictScope_nodeB') {
      node = nodeB;
    }
    onClick && onClick(nodeA, nodeB, node);
  });
};

const getConflictScope = (a, b) => {
  return [Math.max(a.startAddr, b.startAddr), Math.min(a.endAddr, b.endAddr)];
};
