import { ResultStatus } from '@/utils/commonConstant.js';
import { getRank } from '@/utils/commonFunction.js';
import { Button, Grid } from '@aurora/vue3';

export default function Log({ topologyInfo, rankOptions, rankValue, errorType, errorInfo }, { emit }) {
  const reg = /[\w+\.[a-z]+\:[0-9]+]/;
  const errorMessage = errorInfo?.errorMessage;

  const renderContent = () => {
    let content = '';
    if (errorType === ResultStatus.CHECK_SUCCESS) {
      const hasLoop = topologyInfo.rankNodes.nodes.some((item) => item.isLoop);
      content = (
        <>
          <li>用例执行成功</li>
          {hasLoop ? <li>{getRank(rankOptions, rankValue)?.label}的task拓扑图中存在环</li> : null}
        </>
      );
    } else if (errorType === ResultStatus.MEMORY_CONFLICT) {
      content = (
        <>
          <li>内存冲突错误：</li>
          <li>{renderErrorMemory()}</li>
        </>
      );
    } else if (errorType === ResultStatus.GEN_FAILED_INCOMPLETE_SLICE) {
      content = (
        <>
          <li>slice中的语义块不完整：</li>
          <li>{renderErrorSemantic()}</li>
        </>
      );
    } else if (errorType === ResultStatus.GEN_FAILED_MODIFY_SEMANTIC_FAILED) {
      content = (
        <>
          <li>修改语义块失败：</li>
          <li>{renderErrorSemantic()}</li>
        </>
      );
    } else if (errorType === ResultStatus.CHECK_FAILED_MISSING_SEMANTIC) {
      content = (
        <>
          <li>校验过程中，语义块有缺失：</li>
          <li>{renderErrorSemantic()}</li>
        </>
      );
    } else if (errorType === ResultStatus.CHECK_FAILED_UNEXPECTED_SEMANTIC) {
      content = (
        <>
          <li>校验过程中，出现了不期望的语义块：</li>
          <li>{renderErrorSemantic()}</li>
        </>
      );
    }
    return content;
  };

  const renderErrorSemantic = () => {
    const errorSemantic = errorInfo.errorSemantic;
    return (
      <Grid
        style="width:240px"
        column-width={80}
        data={(errorSemantic || []).map((item) => ({ ...item, rankName: getRank(rankOptions, item.rankId)?.label }))}
        columns={[
          { field: 'rankName', title: 'rankName' },
          { field: 'localStep', title: 'localStep' },
          {
            title: '操作',
            slots: {
              default: ({ row }) => {
                return (
                  <Button
                    type="text"
                    style="padding:0;min-width:30px"
                    onClick={() => {
                      emit('redirectErrorSemantic', { errorSemanticItem: row, errorType });
                    }}
                  >
                    跳转
                  </Button>
                );
              }
            }
          }
        ]}
      />
    );
  };

  const renderErrorMemory = () => {
    const errorMemory = errorInfo?.errorMemory;
    const rankName = getRank(rankOptions, errorMemory?.rankId)?.label;
    return (
      <div>
        <span>
          {rankName}中节点{errorMemory?.nodeIdA}和节点{errorMemory?.nodeIdB}冲突
        </span>
        &nbsp;&nbsp;
        <Button
          type="text"
          style="padding:0;min-width:30px"
          onClick={() => {
            emit('redirectErrorMemory', errorMemory);
          }}
        >
          跳转
        </Button>
      </div>
    );
  };

  const renderErrorMessage = () => {
    if (errorMessage && errorMessage.length) {
      return errorMessage.map((item) => {
        const matchRes = item.match(reg);
        const str = item.replace(reg, '');
        return (
          <li>
            {matchRes ? <strong>{matchRes[0]}</strong> : null}
            {str}
          </li>
        );
      });
    }
    return null;
  };

  return (
    <ul className="log-ul">
      {renderContent()}
      {renderErrorMessage()}
    </ul>
  );
}
