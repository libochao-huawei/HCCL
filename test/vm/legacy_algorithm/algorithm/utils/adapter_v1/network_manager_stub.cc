/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
 * Description: 集合通信算子插件
 * Author: lilianlin
 * Create: 2019-11-28
 */

#include "network_manager_pub.h"


namespace hccl {

NetworkManager* NetworkManager::nmInstance[MAX_DEV_NUM] = {nullptr};
std::atomic<unsigned> NetworkManager::InitTool::initCount(0);

NetworkManager::InitTool::InitTool()
{
    if (initCount.load() == 0) {
        for (u32 i = 0; i < MAX_DEV_NUM; i++) {
            NetworkManager::nmInstance[i] = new NetworkManager;
        }
    }
    ++initCount;
}

NetworkManager::InitTool::~InitTool()
{
    --initCount;
    if (initCount.load() == 0) {
        for (u32 i = 0; i < MAX_DEV_NUM; i++) {
            if (NetworkManager::nmInstance[i] != nullptr) {
                delete NetworkManager::nmInstance[i];
                NetworkManager::nmInstance[i] = nullptr;
            }
        }
    }
}

NetworkManager::NetworkManager()
    : deviceLogicId_(INVALID_INT),
      devicePhyId_(INVALID_UINT),
      vnicListened_(false),
      isHostUseDevNic_(false),
      notifyType_(NO_USE)
{
}

NetworkManager::~NetworkManager()
{
}


}


