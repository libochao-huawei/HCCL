<template>
  <div class="left-container">
    <div class="tool">
      <aui-file-upload ref="upload" :http-request="uploadTestCase" :before-upload="beforeUpload" :show-file-list="false" accept=".json">
        <template #trigger>
          <aui-button type="primary">上传用例</aui-button>
        </template>
      </aui-file-upload>
      <aui-button class="mg-l-16" @click="onOpenExportDialog">导出用例</aui-button>
    </div>
    <div class="tree">
      <aui-tree
        ref="tcTreeRef"
        node-key="id"
        :data="treeData"
        :indent="12"
        :expand-on-click-node="false"
        @current-change="onChangeCurrentTreeNode"
        default-expand-all
        highlight-current
      >
        <template #default="sourceData">
          <div class="tree-node-label" :title="sourceData.data.label">
            {{ sourceData.data.label }}
          </div>
        </template>
        <template #operation="{ node }">
          <template v-if="node.data.isLeaf">
            <i title="执行用例" v-if="!node.data.errorMessage" @click="onRunTestCase($event, node)">
              <svg class="icon-normal" style="vertical-align: middle">
                <use xlink:href="#icon-run"></use>
              </svg>
            </i>
            <i title="执行用例" v-if="node.data.errorMessage">
              <svg class="icon-normal icon-disabled" style="vertical-align: middle">
                <use xlink:href="#icon-runDisabled"></use>
              </svg>
            </i>
            <template v-if="!node.data.isDefault">
              <i title="编辑用例" @click="onEditTestCase($event, node)">
                <svg class="icon-normal" style="vertical-align: middle">
                  <use xlink:href="#icon-edit"></use>
                </svg>
              </i>
              <i title="删除用例" @click="onDelTestCase($event, node)">
                <svg class="icon-normal" style="vertical-align: middle">
                  <use xlink:href="#icon-del"></use>
                </svg>
              </i>
            </template>
          </template>
          <i v-else title="新建用例" @click="onCreateCase($event, node)">
            <svg class="icon-normal" style="vertical-align: middle">
              <use xlink:href="#icon-add"></use>
            </svg>
          </i>
        </template>
      </aui-tree>
    </div>
  </div>

  <aui-drawer
    :title="drawerInfo.isEdit ? '修改用例' : '新建用例'"
    width="360px"
    :show-footer="true"
    :visible="drawerInfo.visible"
    :mask-closable="false"
    @close="onCloseDrawer"
  >
    <div style="padding: 0 16px">
      <aui-form ref="drawerFormRef" label-position="top" label-width="100px" validate-type="text" :model="drawerInfo.formData" :rules="drawerFormRules">
        <aui-form-item label="用例分组" prop="testSuite">
          <aui-select
            v-model="drawerInfo.formData.testSuite"
            :options="drawerInfo.testSuiteOptions"
            :disabled="drawerInfo.isEdit"
            allow-create
            filterable
            default-first-option
          ></aui-select>
        </aui-form-item>

        <aui-form-item label="用例名称" prop="testCase">
          <aui-input v-model="drawerInfo.formData.testCase" :disabled="drawerInfo.isEdit"> </aui-input>
        </aui-form-item>

        <aui-form-item label="关闭内存冲突校验" prop="memoryConflictCheck">
          <aui-switch v-model="drawerInfo.formData.memoryConflictCheck">
            <template #open>
              <span>是</span>
            </template>
            <template #close>
              <span style="position: relative; left: 4px">否</span>
            </template>
          </aui-switch>
        </aui-form-item>

        <aui-form-item label="环境变量">
          <aui-grid
            ref="envRef"
            :data="drawerInfo.formData.env"
            :edit-config="{ trigger: 'click', mode: 'row', showStatus: false }"
            :edit-rules="envValidRules"
          >
            <template #toolbar>
              <aui-grid-toolbar style="padding-top: 0">
                <template #buttons>
                  <aui-button @click="onInsertEnv">新增</aui-button>
                </template>
              </aui-grid-toolbar>
            </template>
            <template #empty>
              <span>暂无数据</span>
            </template>
            <aui-grid-column field="key" title="变量" :editor="{ ignoreFocus: true, type: 'visible' }">
              <template #edit="data">
                <aui-select
                  v-model="data.row.key"
                  :options="
                    Envs.filter((item) => {
                      const tableData = data.$table.getTableData().tableData;
                      return !tableData?.find((ele) => ele.key === item.value);
                    })
                  "
                  allowCreate
                  filterable
                  defaultFirstOption
                ></aui-select>
              </template>
            </aui-grid-column>
            <aui-grid-column field="value" title="值" :editor="{ component: AuiInput, type: 'visible' }"></aui-grid-column>
            <aui-grid-column title="操作" width="40">
              <template v-slot="{ row }">
                <div class="demo-custom-column">
                  <i title="删除" @click="onDelEnv(row)">
                    <svg class="icon-normal" style="vertical-align: middle">
                      <use xlink:href="#icon-del"></use>
                    </svg>
                  </i>
                </div>
              </template>
            </aui-grid-column>
          </aui-grid>
        </aui-form-item>

        <aui-form-item label="芯片类型" prop="devtype">
          <aui-select v-model="drawerInfo.formData.devtype" :options="drawerInfo.devTypeOptions" @change="onChangeDevType"></aui-select>
        </aui-form-item>

        <aui-form-item label="网络配置方式" prop="topoConfigMode">
          <aui-select v-model="drawerInfo.formData.topoConfigMode" :options="drawerInfo.topoConfigModeOptions" @change="onChangeTopoConfigMode"></aui-select>
        </aui-form-item>

        <aui-form-item label="网络配置参数" prop="symmetric_topo" v-if="drawerInfo.formData.topoConfigMode === 'SYMMETRIC'">
          <aui-grid
            ref="networkConfigRef"
            :data="drawerInfo.formData.symmetric_topo"
            :edit-config="{ trigger: 'click', mode: 'cell', showStatus: false }"
            :edit-rules="drawerNetworkConfigValidRules"
            @edit-closed="onEditClosedSymmetricTopo"
          >
            <aui-grid-column
              field="superPodNum"
              show-overflow
              title="超节点个数"
              :editor="{
                component: 'input',
                type: 'visible',
                isValidAlways: true,
                attrs: { type: 'number' }
              }"
            ></aui-grid-column>
            <aui-grid-column
              field="serverNum"
              title="服务器个数"
              :editor="{
                component: 'input',
                type: 'visible',
                isValidAlways: true,
                attrs: { type: 'number' }
              }"
            ></aui-grid-column>
            <aui-grid-column
              field="rankNum"
              title="NPU个数"
              :editor="{
                component: 'input',
                type: 'visible',
                isValidAlways: true,
                validNoFocus: true,
                attrs: { type: 'number' }
              }"
            ></aui-grid-column>
          </aui-grid>
        </aui-form-item>

        <aui-form-item label="网络配置参数" prop="asymmetric_topo" v-if="drawerInfo.formData.topoConfigMode === 'ASYMMETRIC'">
          <div style="margin-bottom: 10px">
            <aui-button @click="onAddSuperPodDrawerAsymmetricTopo">添加超节点</aui-button>
          </div>
          <aui-tree
            :data="drawerInfo.formData.asymmetric_topo"
            default-expand-all
            :expand-on-click-node="false"
            :render-content="renderDrawerAsymmetricTopoContent"
          >
          </aui-tree>
        </aui-form-item>

        <aui-form-item label="算子模式" prop="opMode">
          <aui-select v-model="drawerInfo.formData.opMode" :options="drawerInfo.opModeOptions"></aui-select>
        </aui-form-item>

        <aui-form-item label="算法名称" prop="algName">
          <aui-input v-model="drawerInfo.formData.algName"></aui-input>
        </aui-form-item>

        <aui-form-item label="数据类型" prop="dataType">
          <aui-select v-model="drawerInfo.formData.dataType" :options="drawerInfo.dataTypeOptions"></aui-select>
        </aui-form-item>

        <aui-form-item prop="count">
          <template #label>
            <span>count</span>
            <aui-popover trigger="hover" content="一个rank操作的数据量" placement="top">
              <template #reference> <aui-icon-help-circle style="margin-left: 4px; vertical-align: top"> </aui-icon-help-circle></template>
            </aui-popover>
          </template>
          <aui-numeric v-model="drawerInfo.formData.count" :controls="false" :show-left="true" string-mode allow-empty></aui-numeric>
        </aui-form-item>

        <aui-form-item label="算子类型" prop="opType">
          <aui-select v-model="drawerInfo.formData.opType" :options="drawerInfo.opTypeOptions"></aui-select>
        </aui-form-item>

        <aui-form-item label="root rank" prop="rootRank" v-if="showRootRank(drawerInfo.formData.opType)">
          <aui-select v-model="drawerInfo.formData.rootRank" :options="drawerInfo.rankOptions"></aui-select>
        </aui-form-item>

        <aui-form-item label="源rank" prop="srcRank" v-if="showSrcAndDstRank(drawerInfo.formData.opType)">
          <aui-select v-model="drawerInfo.formData.srcRank" :options="drawerInfo.rankOptions"></aui-select>
        </aui-form-item>

        <aui-form-item label="目的rank" prop="dstRank" v-if="showSrcAndDstRank(drawerInfo.formData.opType)">
          <aui-select v-model="drawerInfo.formData.dstRank" :options="drawerInfo.rankOptions"></aui-select>
        </aui-form-item>

        <aui-form-item label="reduce操作类型" prop="reduceType" v-if="showReduceType(drawerInfo.formData.opType)">
          <aui-select v-model="drawerInfo.formData.reduceType" :options="drawerInfo.reduceTypeOptions"></aui-select>
        </aui-form-item>
      </aui-form>
    </div>
    <template #footer>
      <aui-button @click="onCloseDrawer">取消</aui-button>
      <aui-button @click="onConfirmDrawer" type="primary" :loading="drawerInfo.confirmLoading">保存</aui-button>
    </template>
  </aui-drawer>

  <aui-dialog-box :visible="dialogInfo.visible" title="导出用例" @close="onCloseDialog">
    <aui-tree ref="exportTreeRef" node-key="id" :data="treeData" :expand-on-click-node="false" check-on-click-node default-expand-all show-checkbox> </aui-tree>

    <template #footer>
      <aui-button @click="onCloseDialog">取消</aui-button>
      <aui-button type="primary" @click="onExportTestCase" :loading="dialogInfo.loading"> 导出 </aui-button>
    </template>
  </aui-dialog-box>
</template>

<script setup lang="jsx">
import { onMounted, ref, reactive, inject, nextTick, onUnmounted } from 'vue';
import {
  Button as AuiButton,
  FileUpload as AuiFileUpload,
  Select as AuiSelect,
  Input as AuiInput,
  Form as AuiForm,
  FormItem as AuiFormItem,
  Numeric as AuiNumeric,
  Grid as AuiGrid,
  GridColumn as AuiGridColumn,
  GridToolbar as AuiGridToolbar,
  Tree as AuiTree,
  Drawer as AuiDrawer,
  RadioGroup as AuiRadioGroup,
  DialogBox as AuiDialogBox,
  Modal,
  Radio as AuiRadio,
  Popover as AuiPopover,
  Switch as AuiSwitch
} from '@aurora/vue3';
import { HcclReduceOp, HcclCMDType, OpMode, DevType, HcclDataType, Envs, TopoConfigMode } from '@/utils/commonConstant.js';
import { showReduceType, showRootRank, showSrcAndDstRank } from '@/utils/commonFunction.js';
import downloadFile from '@/utils/downloadFile';
import { cloneDeep, omit, unionBy, differenceBy } from 'lodash-es';
import { useCurrentTestCaseStore, useRunTestCaseResultStore } from '@/store/checker/index.js';
import { useUserStore } from '@/store/app.js';
import JSZip from 'jszip';
import useLoading from '@/utils/useLoading';
import { IconHelpCircle } from '@aurora/vue3-icon';

const AuiIconHelpCircle = IconHelpCircle();

const { apiAssets } = inject('$api');
const commonRequiredParams = ['testSuite', 'testCase', 'devtype', 'topoConfigMode', 'opMode', 'dataType', 'count', 'opType'];
const extraRequiredParamsMap = {
  HCCL_CMD_ALLREDUCE: ['reduceType'],
  HCCL_CMD_REDUCE_SCATTER: ['reduceType'],
  HCCL_CMD_REDUCE: ['reduceType', 'rootRank'],
  HCCL_CMD_BROADCAST: ['rootRank'],
  HCCL_CMD_SCATTER: ['rootRank'],
  HCCL_CMD_SEND: ['srcRank', 'dstRank']
};

let timer = null;
const currentTestCaseStore = useCurrentTestCaseStore();
const runTestCaseResultStore = useRunTestCaseResultStore();
const { userName: user_name } = useUserStore();
const { showLoading, closeLoading } = useLoading();
const emits = defineEmits(['setActiveTab']);
const uploadPolicyOptions = [
  { label: 'overwrite', text: '覆盖' },
  { label: 'unoverwrite', text: '不覆盖' }
];
const uploadPolicy = ref('overwrite');
const tcTreeRef = ref();
const treeData = ref([
  {
    id: 'all',
    label: '全部',
    children: []
  }
]);
const drawerFormRef = ref();
const envRef = ref();
const networkConfigRef = ref();
const commonRules = [{ required: true, message: '必填' }];
const drawerFormRules = {
  testSuite: commonRules,
  testCase: [
    ...commonRules,
    {
      trigger: 'blur',
      validator: (rule, value, callback) => {
        if (!drawerInfo.isEdit && drawerInfo.formData.testSuite) {
          const item = treeData.value[0].children.find((item) => item.id === drawerInfo.formData.testSuite);
          if (item && item.children.some((item) => item.testCase === value)) {
            callback(new Error('用例名不能重复'));
            return;
          }
        }
        callback();
      }
    }
  ],
  topoConfigMode: commonRules,
  opType: commonRules,
  opMode: commonRules,
  reduceType: commonRules,
  devtype: commonRules,
  algName: [{ pattern: /^[0-9a-zA-Z_]+$/, message: '字母数字下划线组成' }],
  count: commonRules,
  dataType: commonRules,
  rootRank: commonRules,
  srcRank: commonRules,
  dstRank: commonRules,
  symmetric_topo: [
    ...commonRules,
    {
      validator: (rule, value, callback) => {
        const dataItem = value[0];

        if (!dataItem.superPodNum) {
          callback(new Error('超节点个数必填'));
          return;
        }
        if (!dataItem.serverNum) {
          callback(new Error('服务器个数必填'));
          return;
        }
        if (!dataItem.rankNum) {
          callback(new Error('NPU个数必填'));
          return;
        }

        const devtype = drawerInfo.formData.devtype;
        if (!devtype) {
          callback(new Error('请先选择芯片类型'));
          return;
        }
        const npu = DevType.find((item) => item.value === devtype).npu;
        if (dataItem.rankNum > npu) {
          callback(new Error(`所选芯片类型的最大卡数为${npu}`));
          return;
        }

        let num;
        if (dataItem.superPodNum && dataItem.serverNum && dataItem.rankNum) {
          num = dataItem.superPodNum * dataItem.serverNum * dataItem.rankNum;
        }

        if (num > 128) {
          callback(new Error('npu总个数不能大于128'));
          return;
        }

        if (drawerInfo.formData.opType === 'HCCL_CMD_SEND' && num !== 2) {
          callback(new Error('算子类型为 SendRecv 时，npu个数只能为2'));
          return;
        }
        callback();
      }
    }
  ],
  asymmetric_topo: [
    ...commonRules,
    {
      validator: (rule, value, callback) => {
        let arr = [];
        value.forEach((superPodItem) => {
          superPodItem.children.forEach((serverItem) => {
            if (!serverItem.children[0].values || !serverItem.children[0].values.length) {
              arr.push(`${superPodItem.label}的${serverItem.label}`);
            }
          });
        });
        if (arr.length) {
          callback(new Error(`${arr.join('、')}未选择npu`));
          return;
        }

        const num = value.reduce((prev, cur) => {
          cur.children.forEach((server) => {
            prev += server.children[0].values.length;
          });
          return prev;
        }, 0);

        if (num > 128) {
          callback(new Error('npu总个数不能大于128'));
          return;
        }

        if (drawerInfo.formData.opType === 'HCCL_CMD_SEND' && num !== 2) {
          callback(new Error('算子类型为 SendRecv 时，npu个数只能为2'));
          return;
        }

        callback();
      }
    }
  ]
};
const envValidRules = {
  key: commonRules,
  value: commonRules
};
const drawerNetworkConfigValidRules = {
  superPodNum: commonRules,
  serverNum: commonRules,
  rankNum: [
    { required: true, message: '必填' },
    {
      validator: ({ row }, value) => {
        return new Promise((resolve, reject) => {
          const devtype = drawerInfo.formData.devtype;
          if (!devtype) {
            reject('请先选择芯片类型');
          }
          const npu = DevType.find((item) => item.value === devtype).npu;
          if (value > npu) {
            reject(`所选芯片类型的最大卡数为${npu}`);
          }
          resolve();
        });
      }
    }
  ]
};
const drawerSymmetricTopoDefalut = [{ superPodNum: '', serverNum: '', rankNum: '' }];
const drawerAsymmetricTopoDefalut = [
  {
    label: '超节点1',
    type: 'superPod',
    children: [
      {
        label: 'server1',
        type: 'server',
        children: [
          {
            label: 'npu',
            type: 'npu',
            values: []
          }
        ]
      }
    ]
  }
];
const drawerInfo = reactive({
  visible: false,
  isEdit: false,
  confirmLoading: false,
  formData: {},
  testSuiteOptions: [],
  topoConfigModeOptions: TopoConfigMode,
  opTypeOptions: HcclCMDType,
  opModeOptions: OpMode,
  reduceTypeOptions: HcclReduceOp,
  devTypeOptions: DevType,
  dataTypeOptions: HcclDataType,
  npuOptions: {},
  rankOptions: []
});
const exportTreeRef = ref();
const dialogInfo = reactive({
  visible: false,
  loading: false
});

const onChangeCurrentTreeNode = (data, node) => {
  if (node.isLeaf) {
    if (data.id !== currentTestCaseStore.currentTestCase.id) {
      emits('setActiveTab', 'config');
      currentTestCaseStore.setCurrentTestCase(data);
      runTestCaseResultStore.$reset();
    }
  } else {
    tcTreeRef.value.setCurrentKey(currentTestCaseStore.currentTestCase.id);
  }
};

const getTreeData = () => {
  showLoading();
  apiAssets
    .fetchDataByCmd({ cmd: 'start', user_name })
    .then((res) => {
      return loopFetchData(user_name).then((data) => {
        treeData.value[0].children = convertToTree([
          ...data.defaultTcList.map((item) => {
            const errorMessage = validateTestCase(item);
            return {
              ...item,
              isDefault: true,
              errorMessage
            };
          }),
          ...data.userTcList.map((item) => {
            const errorMessage = validateTestCase(item);
            return {
              ...item,
              isDefault: false,
              errorMessage
            };
          })
        ]);
        currentTestCaseStore.setCurrentTestCase(treeData.value[0].children[0].children[0]);
        nextTick(() => {
          tcTreeRef.value.setCurrentKey(currentTestCaseStore.currentTestCase.id);
        });
        closeLoading();
      });
    })
    .catch(() => closeLoading());
};

const setTestSuiteOptions = () => {
  drawerInfo.testSuiteOptions = treeData.value[0].children.map((item) => ({
    value: item.label
  }));
};

const convertToTree = (dataList) => {
  let dataMap = new Map();
  dataList.forEach((item) => {
    const child = {
      id: `${item.testSuite}.${item.testCase}`,
      label: item.testCase,
      isLeaf: true,
      isDefault: false,
      ...item
    };
    const value = dataMap.get(item.testSuite);
    if (value) {
      value.children.push(child);
    } else {
      const parent = {
        id: item.testSuite,
        label: item.testSuite,
        children: [child]
      };
      dataMap.set(item.testSuite, parent);
    }
  });
  const res = [];
  dataMap.forEach((item) => res.push(item));
  return res;
};

const onOpenExportDialog = () => {
  exportTreeRef.value.setCheckedKeys([]);
  dialogInfo.visible = true;
};
const onCloseDialog = () => {
  dialogInfo.visible = false;
};
const onExportTestCase = () => {
  const checkedNodes = exportTreeRef.value.getCheckedNodes();
  const checkedTestCase = checkedNodes.filter((item) => item.isLeaf);
  if (checkedTestCase.length > 0) {
    let exportData = checkedTestCase.map((item) => {
      const testCase = omit(item, ['id', 'label', 'isLeaf', 'isDefault', 'errorMessage']);
      if (!testCase.algName) {
        testCase.algName = '';
      }
      return testCase;
    });
    exportData = JSON.stringify({ caselist: exportData }, null, 2);
    let blob = new Blob([exportData], { type: 'application/json;charset=utf-8' });
    if (checkedTestCase.length > 1) {
      const zip = new JSZip();
      zip.file('testCase.json', blob);
      zip
        .generateAsync({ type: 'blob' })
        .then((blob) => {
          const url = window.URL.createObjectURL(blob);
          return downloadFile(url, 'testCase.zip');
        })
        .then(() => onCloseDialog());
    } else {
      const url = window.URL.createObjectURL(blob);
      downloadFile(url, 'testCase.json').then(() => onCloseDialog());
    }
  } else {
    Modal.alert({ message: '请选择用例', title: '错误', status: 'error' });
  }
};

const beforeUpload = (file) => {
  return new Promise((resolve, reject) => {
    const fileReader = new FileReader();
    fileReader.readAsText(file);
    fileReader.onload = (e) => {
      const fileStr = e.target.result;
      try {
        const fileData = JSON.parse(fileStr);
        const errorList = [];
        fileData.caselist.forEach((item) => {
          const errorMessage = validateTestCase(item);
          if (errorMessage) {
            errorList.push(errorMessage);
          }
        });
        if (errorList.length) {
          Modal.alert({
            status: 'error',
            message: () => {
              return (
                <ul>
                  {errorList.map((item) => (
                    <li style="margin-top:8px;font-size: 14px;font-weight: normal;">{item}</li>
                  ))}
                </ul>
              );
            }
          }).then((res) => {
            reject();
          });
        } else {
          Modal.confirm({
            title: '上传策略',
            message: () => {
              return (
                <div style={{ lineHeight: '22px' }}>
                  <span style={{ fontSize: '12px', color: '#5e6d82' }}>重复用例策略：</span>
                  <AuiRadioGroup
                    modelValue={uploadPolicy.value}
                    onChange={(val) => {
                      if (val === 'overwrite') {
                        uploadPolicy.value = 'unoverwrite';
                      } else {
                        uploadPolicy.value = 'overwrite';
                      }
                    }}
                  >
                    <AuiRadio label="overwrite">
                      <span style={{ marginRight: '4px' }}>覆盖</span>
                      <AuiPopover trigger="hover" content="默认用例不会被覆盖" placement="top">
                        {{ reference: () => <AuiIconHelpCircle style={{ fontSize: '14px' }} /> }}
                      </AuiPopover>
                    </AuiRadio>
                    <AuiRadio label="unoverwrite">
                      <span>不覆盖</span>
                    </AuiRadio>
                  </AuiRadioGroup>
                </div>
              );
            }
          }).then((res) => {
            res === 'confirm' ? resolve() : reject();
          });
        }
      } catch (error) {
        console.log(error);
        Modal.alert({ message: () => <div style={{ wordBreak: 'break-word' }}>上传文件有误，请检查：{error.message}</div>, title: '错误', status: 'error' });
      }
    };
  });
};

const validateTestCase = (testCase) => {
  let errorMessage;
  const itemKeys = Object.keys(testCase);
  const missingKeys = commonRequiredParams.filter((val) => !itemKeys.includes(val));

  if (testCase.topoConfigMode === 'SYMMETRIC' && !('symmetric_topo' in testCase)) {
    missingKeys.push('symmetric_topo');
  } else if (testCase.topoConfigMode === 'ASYMMETRIC' && !('asymmetric_topo' in testCase)) {
    missingKeys.push('asymmetric_topo');
  }

  (extraRequiredParamsMap[testCase.opType] || []).forEach((val) => {
    if (!itemKeys.includes(val)) {
      missingKeys.push(val);
    }
  });

  if (missingKeys.length) {
    errorMessage = `${testCase.testSuite} 的 ${testCase.testCase} 用例缺少 ${missingKeys.join(',')} 参数`;
  }
  return errorMessage;
};

const uploadTestCase = (file) => {
  showLoading();
  const formData = new FormData();
  formData.append('file_info', file.file);
  formData.append('user_name', user_name);
  apiAssets
    .uploadCase(formData)
    .then((res) => {
      const fileReader = new FileReader();
      fileReader.readAsText(file.file);
      fileReader.onload = (e) => {
        const fileStr = e.target.result;
        const fileData = JSON.parse(fileStr);
        const resData = convertToTree(fileData.caselist);
        resData.forEach((item1) => {
          const resItem = treeData.value[0].children.find((item2) => item2.id === item1.id);
          if (resItem) {
            const defaultCases = resItem.children.filter((item) => item.isDefault);
            if (uploadPolicy.value === 'overwrite') {
              let arr = differenceBy(resItem.children, item1.children, 'id');
              resItem.children = unionBy(defaultCases, [...arr, ...item1.children], 'id');
              const target = resItem.children.find((item) => item.id === currentTestCaseStore.currentTestCase.id);
              if (target) {
                currentTestCaseStore.setCurrentTestCase(target);
              }
            } else {
              resItem.children = unionBy(resItem.children, item1.children, 'id');
            }
          } else {
            treeData.value[0].children = [...treeData.value[0].children, item1];
          }
        });
        Modal.message({
          message: '上传成功',
          status: 'success',
          duration: '500'
        });
        closeLoading();
      };
    })
    .catch(() => closeLoading());
};

const onRunTestCase = (e, node) => {
  e.stopPropagation();
  currentTestCaseStore.setCurrentTestCase(node.data);
  runTestCaseResultStore.$reset();
  tcTreeRef.value.setCurrentKey(node.data.id);
  emits('setActiveTab', 'result');
  runTestcase();
};

const runTestcase = () => {
  showLoading();
  apiAssets
    .fetchDataByCmd({
      cmd: 'run_tc',
      user_name,
      tc_name: currentTestCaseStore.currentTestCase.id
    })
    .then((res) => {
      return loopFetchData(user_name).then((data) => {
        runTestCaseResultStore.setRunTestCaseResult(data);
        closeLoading();
      });
    })
    .catch(() => closeLoading());
};

const onEditTestCase = (e, node) => {
  e.stopPropagation();
  setTestSuiteOptions();
  drawerInfo.isEdit = true;
  if (currentTestCaseStore.currentTestCase.id !== node.data.id) {
    currentTestCaseStore.setCurrentTestCase(node.data);
    tcTreeRef.value.setCurrentKey(node.data.id);
    runTestCaseResultStore.$reset();
    emits('setActiveTab', 'config');
  }
  drawerInfo.formData = convertToFormDataFromCurrentTestCase(node.data);
  setNpuOptions(drawerInfo.formData.devtype);
  setRankOptions();
  drawerInfo.visible = true;
};

const convertToFormDataFromCurrentTestCase = (value) => {
  const res = {
    testSuite: value.testSuite,
    testCase: value.testCase,
    env: cloneDeep(value.env),
    topoConfigMode: value.topoConfigMode,
    opType: value.opType,
    opMode: value.opMode,
    reduceType: value.reduceType || 'HCCL_REDUCE_SUM',
    devtype: value.devtype,
    algName: value.algName,
    count: value.count,
    dataType: value.dataType,
    rootRank: value.rootRank ?? 0,
    srcRank: value.srcRank ?? 0,
    dstRank: value.dstRank ?? 1,
    memoryConflictCheck: value.memoryConflictCheck
  };

  if (value.topoConfigMode === 'SYMMETRIC') {
    res.symmetric_topo = [
      {
        superPodNum: value.symmetric_topo && value.symmetric_topo[0],
        serverNum: value.symmetric_topo && value.symmetric_topo[1],
        rankNum: value.symmetric_topo && value.symmetric_topo[2]
      }
    ];
    res.asymmetric_topo = cloneDeep(drawerAsymmetricTopoDefalut);
  } else if (value.topoConfigMode === 'ASYMMETRIC') {
    res.symmetric_topo = cloneDeep(drawerSymmetricTopoDefalut);
    res.asymmetric_topo = value.asymmetric_topo?.map((superPodItem, superPodIndex) => {
      return {
        label: `超节点${superPodIndex + 1}`,
        type: 'superPod',
        children: superPodItem.map((item, index) => {
          return {
            label: `server${index + 1}`,
            type: 'server',
            children: [
              {
                label: 'npu',
                type: 'npu',
                values: item
              }
            ]
          };
        })
      };
    });
  }

  return res;
};

const onDelTestCase = (e, node) => {
  e.stopPropagation();
  Modal.confirm('确定要删除吗？').then((res) => {
    if (res === 'confirm') {
      showLoading();
      apiAssets
        .deleteCase({ user_name, tc_name: node.data.id })
        .then((res) => {
          const item = treeData.value[0].children.find((item) => item.id === node.data.testSuite);
          item.children = item.children.filter((item) => item.id !== node.data.id);
          currentTestCaseStore.$reset();
          runTestCaseResultStore.$reset();
          const currentTestCase = treeData.value[0]?.children[0]?.children[0];
          if (currentTestCase) {
            currentTestCaseStore.setCurrentTestCase(currentTestCase);
            tcTreeRef.value.setCurrentKey(currentTestCase.id);
          }
        })
        .finally(() => closeLoading());
    }
  });
};

const onCreateCase = (e, node) => {
  e.stopPropagation();
  setTestSuiteOptions();
  drawerFormRef.value.resetFields();
  drawerInfo.isEdit = false;
  drawerInfo.formData = {
    testSuite: node.data.id !== 'all' ? node.data.id : drawerInfo.testSuiteOptions[0]?.value,
    env: [],
    symmetric_topo: cloneDeep(drawerSymmetricTopoDefalut),
    asymmetric_topo: cloneDeep(drawerAsymmetricTopoDefalut),
    dataType: 'HCCL_DATA_TYPE_FP32',
    rootRank: 0,
    srcRank: 0,
    dstRank: 1,
    reduceType: 'HCCL_REDUCE_SUM',
    memoryConflictCheck: false
  };
  drawerInfo.visible = true;
};

const onInsertEnv = () => {
  envRef.value.insert({}).then((res) => {
    envRef.value.setActiveRow(res.row);
  });
};

const onDelEnv = (row) => {
  envRef.value.remove(row);
};

const onChangeDevType = (value) => {
  setNpuOptions(value);
  if (drawerInfo.formData.topoConfigMode === 'ASYMMETRIC') {
    drawerInfo.formData.asymmetric_topo = drawerInfo.formData.asymmetric_topo.map((item) => {
      return {
        ...item,
        children: item.children.map((server) => {
          return {
            ...server,
            children: server.children.map((npu) => {
              return {
                ...npu,
                values: []
              };
            })
          };
        })
      };
    });
  }
};

const onChangeTopoConfigMode = (val) => {
  resetRank();
};

const onEditClosedSymmetricTopo = () => {
  drawerFormRef.value.validateField('symmetric_topo');
  resetRank();
};

const onCloseDrawer = () => {
  drawerInfo.rankOptions = [];
  drawerInfo.visible = false;
};

const onConfirmDrawer = () => {
  drawerFormRef.value.validate((valid) => {
    if (valid) {
      drawerInfo.confirmLoading = true;
      const config = omit(drawerInfo.formData, ['symmetric_topo', 'asymmetric_topo', 'reduceType', 'rootRank', 'srcRank', 'dstRank']);
      if (drawerInfo.formData.topoConfigMode === 'SYMMETRIC') {
        config.symmetric_topo = [
          Number(drawerInfo.formData.symmetric_topo[0].superPodNum),
          Number(drawerInfo.formData.symmetric_topo[0].serverNum),
          Number(drawerInfo.formData.symmetric_topo[0].rankNum)
        ];
      } else if (drawerInfo.formData.topoConfigMode === 'ASYMMETRIC') {
        config.asymmetric_topo = drawerInfo.formData.asymmetric_topo.map((superPodItem) => {
          return superPodItem.children.map((serverItem) => serverItem.children[0].values);
        });
      }

      if (showReduceType(config.opType)) {
        config.reduceType = drawerInfo.formData.reduceType;
      }

      if (showRootRank(config.opType)) {
        config.rootRank = drawerInfo.formData.rootRank;
      }

      if (showSrcAndDstRank(config.opType)) {
        config.srcRank = drawerInfo.formData.srcRank;
        config.dstRank = drawerInfo.formData.dstRank;
      }

      config.env = envRef.value.getTableData().tableData.map((item) => ({ key: item.key, value: item.value }));

      if (drawerInfo.formData.devtype === 'DEV_TYPE_310P3_V') {
        config.devtype = 'DEV_TYPE_310P3';
        config.is310P3V = true;
      } else if (drawerInfo.formData.devtype === 'DEV_TYPE_310P3_Duo') {
        config.devtype = 'DEV_TYPE_310P3';
        config.is310P3V = false;
      } else {
        'is310P3V' in config && delete config.is310P3V;
      }

      config.count = Number(config.count);

      const promise = drawerInfo.isEdit ? apiAssets.modifyCase({ user_name, config }) : apiAssets.createCase({ user_name, config });
      promise
        .then(() => {
          const newTestCase = {
            id: `${config.testSuite}.${config.testCase}`,
            label: config.testCase,
            isLeaf: true,
            ...config,
            devtype: config.devtype === 'DEV_TYPE_310P3' ? (config.is310P3V ? 'DEV_TYPE_310P3_V' : 'DEV_TYPE_310P3_Duo') : config.devtype
          };
          const item = treeData.value[0].children.find((item) => item.id === config.testSuite);
          if (drawerInfo.isEdit) {
            item.children = item.children.map((cur) => {
              if (cur.id === newTestCase.id) {
                return newTestCase;
              } else {
                return cur;
              }
            });
          } else {
            if (item) {
              item.children = [...item.children, newTestCase];
            } else {
              treeData.value[0].children = [
                ...treeData.value[0].children,
                {
                  id: config.testSuite,
                  label: config.testSuite,
                  children: [newTestCase]
                }
              ];
            }
          }
          currentTestCaseStore.setCurrentTestCase(newTestCase);
          runTestCaseResultStore.$reset();
          nextTick(() => tcTreeRef.value.setCurrentKey(newTestCase.id));
          drawerInfo.confirmLoading = false;
          drawerInfo.visible = false;
        })
        .finally(() => (drawerInfo.confirmLoading = false));
    }
  });
};

const onAddSuperPodDrawerAsymmetricTopo = () => {
  const len = drawerInfo.formData.asymmetric_topo.length;
  const suffix = len + 1;
  drawerInfo.formData.asymmetric_topo.push({
    label: `超节点${suffix}`,
    type: 'superPod',
    children: cloneDeep(drawerAsymmetricTopoDefalut[0].children)
  });
};
const onAddServerDrawerAsymmetricTopo = (data) => {
  const label = `server${data.children.length + 1}`;
  data.children = [
    ...data.children,
    {
      label,
      type: 'server',
      children: [
        {
          label: 'npu',
          type: 'npu',
          values: []
        }
      ]
    }
  ];
};
const onRemoveNodeDrawerAsymmetricTopo = (node) => {
  if (node.data.type === 'server') {
    const children = node.parent.data.children;
    node.parent.data.children = children.filter((d) => d.label !== node.data.label);
  } else {
    const parentData = node.parent.data;
    drawerInfo.formData.asymmetric_topo = parentData.filter((d) => d.label !== node.data.label);
  }
  resetRank();
};

const renderDrawerAsymmetricTopoContent = (h, { node, data }) => {
  return (
    <div>
      {data.type === 'npu' ? (
        <AuiSelect
          size="mini"
          options={drawerInfo.npuOptions[drawerInfo.formData.devtype] || []}
          modelValue={data.values}
          onChange={(values) => {
            data.values = values;
            drawerFormRef.value.validateField('asymmetric_topo');
            resetRank();
          }}
          multiple={true}
          placeholder="请选择"
          no-data-text="请先选择芯片类型"
        />
      ) : (
        <span style={{ verticalAlign: 'text-top' }}>{node.label}</span>
      )}
      {data.type === 'superPod' ? (
        <i title="添加server" onClick={() => onAddServerDrawerAsymmetricTopo(data)}>
          <svg class="icon-normal" style="margin-left:4px">
            <use xlink:href="#icon-add"></use>
          </svg>
        </i>
      ) : null}
      {(data.type === 'server' && node.parent.data.children.length > 1) || (data.type === 'superPod' && node.parent.data.length > 1) ? (
        <i title={`${data.type === 'superPod' ? '删除超节点' : '删除server'}`} onClick={() => onRemoveNodeDrawerAsymmetricTopo(node)}>
          <svg class="icon-normal" style="margin-left:4px">
            <use xlink:href="#icon-del"></use>
          </svg>
        </i>
      ) : null}
    </div>
  );
};

const generateOptionsByNum = (num) => {
  return new Array(num).fill('').map((item, index) => ({ value: index, label: `${index}` }));
};

const setNpuOptions = (devtype) => {
  if (!drawerInfo.npuOptions[devtype]) {
    drawerInfo.npuOptions[devtype] = generateOptionsByNum(DevType.find((item) => item.value === devtype).npu);
  }
};

const setRankOptions = () => {
  let num = 0;
  if (drawerInfo.formData.topoConfigMode === 'SYMMETRIC') {
    const dataItem = drawerInfo.formData.symmetric_topo[0];
    if (dataItem.superPodNum && dataItem.serverNum && dataItem.rankNum) {
      num = dataItem.superPodNum * dataItem.serverNum * dataItem.rankNum;
    }
  } else if (drawerInfo.formData.topoConfigMode === 'ASYMMETRIC') {
    num = drawerInfo.formData.asymmetric_topo.reduce((prev, cur) => {
      cur.children.forEach((server) => {
        prev += server.children[0].values.length;
      });
      return prev;
    }, 0);
  }
  if (num <= 128) {
    drawerInfo.rankOptions = generateOptionsByNum(num);
  }
};

const resetRank = () => {
  drawerInfo.formData.rootRank = 0;
  drawerInfo.formData.srcRank = 0;
  drawerInfo.formData.dstRank = 1;
  setRankOptions();
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
  getTreeData();
});
onUnmounted(() => {
  timer && clearTimeout(timer);
});
</script>

<style scoped lang="scss">
.left-container {
  height: 100%;
  overflow: auto;
  position: relative;
  background: #fcf7f0;
}
.tool {
  position: sticky;
  top: 0;
  z-index: 1;
  padding: 16px;
  display: flex;
  background: #fcf7f0;
}
.mg-l-16 {
  margin-left: 16px;
}
.tree {
  padding: 0 0 16px 8px;
}
.tree-node-label {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  padding-right: 16px;
  height: 32px;
  line-height: 32px;
}

:deep() {
  .aui-grid .aui-grid__empty-block {
    padding: 0;
  }
  input::-webkit-outer-spin-button,
  input::-webkit-inner-spin-button {
    -webkit-appearance: none !important;
  }
}
</style>
