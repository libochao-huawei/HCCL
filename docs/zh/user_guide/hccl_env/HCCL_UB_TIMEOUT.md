# HCCL_UB_TIMEOUT

## 功能描述

用于配置UB协议下的Jetty超时时间的系数timeout。

UB协议下的Jetty超时时间分为4档，档位的计算公式为：timeout / 8，其中timeout为该环境变量配置值，0档：512ms；1档：1s；2档：8s；3档：32s。软件内部会有拦截校验机制，在创建Jetty前，先查出TP的超时配置，如果环境变量配置值比TP小，将Jetty超时时间强制拉成TP超时时间；若环境变量配置值比TP大，基于环境变量配置Jetty超时时间。若为UBC_CTP协议，直接基于环境变量值配置。时间配置建议按照0/8/16/24选择配置。

针对Ascend 950PR/Ascend 950DT，该环境变量配置为整数，取值范围为\[0,31\]，默认值为8。

## 配置示例

```bash
# UB超时时间的系数配置为8，则网卡启用UB功能时，超时时间档位为：8 / 8 = 1，对应1s
export HCCL_UB_TIMEOUT=8
```

## 使用约束

无

## 支持的型号

Ascend 950PR/Ascend 950DT
