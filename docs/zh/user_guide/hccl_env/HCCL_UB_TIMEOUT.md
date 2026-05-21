# HCCL_UB_TIMEOUT

## 功能描述

用于配置UB协议下的Jetty重传超时时间的系数timeout。

UB协议下的Jetty重传超时时间分为4挡，挡位的计算公式为：timeout / 8，其中timeout为该环境变量配置值，0挡：512ms；1挡：1s；2挡：8s；3挡：32s。软件内部会有拦截校验机制，在创建Jetty前，先查出TP的超时配置，如果环境变量配置值比TP小，将Jetty超时时间强制拉成TP超时时间；若环境变量配置值比TP大，基于环境变量配置Jetty超时时间。若为UBC_CTP协议，直接基于环境变量值配置。

- 针对Ascend 950PR/Ascend 950DT，该环境变量配置为整数，取值范围为\[0,31\]，默认值为16。

## 配置示例

```bash
# UB重传超时时间的系数配置为16，则网卡启用UB功能时，重传超时时间最小值为：16 / 8 = 2，对应8s
export HCCL_UB_TIMEOUT=16
```

## 使用约束

无

## 支持的型号

Ascend 950PR/Ascend 950DT

<cann-filter npu-type="910">Atlas 训练系列产品</cann-filter>

<cann-filter npu-type="310p">Atlas 推理系列产品（针对Atlas 推理系列产品，仅支持Atlas 300I Duo 推理卡。）</cann-filter>
