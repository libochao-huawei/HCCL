import { Category } from '@/utils/commonConstant.js';
import { canGetSemanticResult } from '@/utils/commonFunction.js';
import { Graph } from '@antv/g6';

let graph = null;

export const drawerTopo = ({ container, nodeInfo, nodes, links, errorType, setTopoGraph, onCliclNode, onCliclEdge }) => {
  graph = new Graph({
    container: container,
    autoFit: 'view',
    autoResize: true,
    data: getTopoData({ nodeInfo, nodes, links, errorType }),
    node: {
      type: 'rect',
      state: {
        selectedNode: {
          lineWidth: 3,
          halo: true,
          haloLineWidth: 30
        },
        ltSelectedNodeLocalStep: {
          lineWidth: 3
        }
      },
      style: {
        size: [100, 30],
        radius: 8,
        labelText: (d) => d.data.nodeName,
        labelPlacement: 'center',
        labelFill: '#fff',
        labelWordWrap: true,
        labelWordWrapWidth: 80
      }
    },
    edge: {
      state: {
        selectedEdge: {
          lineWidth: 3,
          halo: true,
          haloStrokeOpacity: 0.3
        }
      },
      style: {
        endArrow: true
      }
    },
    plugins: [
      {
        key: 'mini-map',
        type: 'minimap',
        size: [240, 120],
        position: 'right-bottom',
        containerStyle: {
          border: '1px solid #ddd',
          background: '#fff',
          opacity: 0.8
        }
      },
      {
        type: 'tooltip',
        position: 'bottom-right',
        enable: (e) => e.targetType === 'node',
        getContent: (e, items) => {
          const { nodeId, nodeName, localStep, info } = items[0].data;
          return `<ul>
              <li>
                <span>nodeId：</span>
                <span>${nodeId}</span>
              </li>
              <li>
                <span>nodeName：</span>
                <span>${nodeName}</span>
              </li>
              <li>
                <span>localStep：</span>
                <span>${localStep.localStep}</span>
              </li>
              <li>
                <span>nodeDescribe：</span>
                <span>${info}</span>
              </li>
            </ul>`;
        }
      }
    ],
    behaviors: ['drag-element', 'drag-canvas', 'zoom-canvas']
  });
  setTopoGraph && setTopoGraph(graph);
  topoChartEvents(onCliclNode, onCliclEdge);
  return graph.render();
};
const getTopoNodes = ({ nodeInfo, nodes, errorType }) => {
  return nodes.map((item) => {
    const states = [];

    if (canGetSemanticResult(errorType) && item.localStep.localStep < nodeInfo.localStep.localStep) {
      states.push('ltSelectedNodeLocalStep');
    } else if (item.nodeId === nodeInfo.nodeId) {
      states.push('selectedNode');
    }

    return {
      id: `${item.nodeId}`,
      style: {
        x: item.x,
        y: item.y,
        fill: Category[item.category].color,
        ...(item.genSemanticError || item.isLoop
          ? {
              stroke: 'red',
              lineWidth: 3
            }
          : {})
      },
      data: item,
      states
    };
  });
};
const getTopoEdges = (links) => {
  return links.map((item) => {
    return {
      source: `${item.source}`,
      target: `${item.target}`,
      style: {
        lineWidth: 3,
        ...(item.isLoop
          ? {
              stroke: 'red'
            }
          : {})
      }
    };
  });
};

const getTopoData = ({ nodeInfo, nodes, links, errorType }) => {
  return {
    nodes: getTopoNodes({ nodeInfo, nodes, errorType }),
    edges: getTopoEdges(links)
  };
};

const topoChartEvents = (onCliclNode, onCliclEdge) => {
  graph.on('node:click', onCliclNode);
  graph.on('edge:click', onCliclEdge);
};

export const onUndo = (data) => {
  if (graph) {
    graph.setData(getTopoData(data));
    graph.render();
  }
};

export const updatedTopo = (data) => {
  graph.setData(getTopoData(data));
  return graph.draw();
};

export const focusNode = (id) => {
  return graph.focusElement(id).then(() => graph.zoomTo(0.8));
};

export const updatedNodeState = ({ nodes, targetNode, errorType }) => {
  let updatedNodes = {};
  if (canGetSemanticResult(errorType)) {
    nodes.forEach((node) => {
      if (node.localStep.localStep < targetNode.localStep.localStep) {
        updatedNodes[node.nodeId] = 'ltSelectedNodeLocalStep';
      } else if (node.nodeId === targetNode.nodeId) {
        updatedNodes[node.nodeId] = 'selectedNode';
      } else {
        updatedNodes[node.nodeId] = '';
      }
    });
  } else {
    const nodes = graph.getElementDataByState('node', 'selectedNode');
    if (nodes.length) {
      nodes.forEach((node) => {
        updatedNodes[node.id] = '';
      });
    }

    if (Array.isArray(targetNode)) {
      targetNode.forEach((node) => {
        updatedNodes[node.nodeId] = 'selectedNode';
      });
    } else {
      updatedNodes[targetNode.nodeId] = 'selectedNode';
    }
  }
  return graph.setElementState(updatedNodes);
};
