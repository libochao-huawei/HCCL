/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: qos stub
 */

#include "qos_stub.h"

QosErrorCode QosGetStreamEngineQos(
    QosStreamType label, QosEngineType engine, const std::string &op, int devId, QosConfig *info)
{
    return QosErrorCode::QOS_SUCCESS;
}
