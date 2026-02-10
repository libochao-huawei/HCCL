#!/bin/bash
# .
# Copyright (c) Huawei Technologies Co., Ltd. 2010-2018. All rights reserved.
set -ex
# set buildVersion
echo "Release is ${ENV_IS_RELEASE}"
if [ "${ENV_IS_RELEASE}" == "false" ];then
    echo "buildVersion=${ENV_PIPELINE_STARTTIME}">${WORKSPACE}/buildInfo.properties
else
    if [ "${ENV_IS_RELEASE}" == "true" ];then
        echo "buildVersion=${ENV_RELEASE_VERSION}">${WORKSPACE}/buildInfo.properties
    fi
fi
#如果jenkins的$buildNumber为空，获取时间戳，获取build随机数
if [ -z "${BUILD_SERIAL_NUMBER}" ];then
    if [ -e /proc/sys/kernel/random/uuid ] && [ -r /proc/sys/kernel/random/uuid ];then
        build=$(cat /proc/sys/kernel/random/uuid| cksum | cut -f1 -d" ")
    else
        build=${RANDOM}
    fi
    datetime=$(date +%Y%m%d%H%M%S)
    build_number="${datetime}.${build}"
else
    build_number="${BUILD_SERIAL_NUMBER}"
fi

ls -l
#包名称
PACKAGE_NAME="cic"
PROJECT_NAME="cic"
npm config set registry https://mirrors.tools.huawei.com/npm/
npm config set @panshi:registry https://cmc.centralrepo.rnd.huawei.com/artifactory/api/npm/product_npm/

cd ${WORKSPACE}/${PROJECT_NAME}
echo "### install dependency"
npm install --unsafe-perm=true --allow-root --debug

cd ${WORKSPACE}
mkdir -p package
tar -zcf ${PACKAGE_NAME}_${build_number}.tar.gz ${PROJECT_NAME}/
#删除历史包
rm -rf package/${PACKAGE_NAME}*.tar.gz package/${PACKAGE_NAME}*.tar.gz.cms
mv ${PACKAGE_NAME}*.tar.gz package/${PACKAGE_NAME}_${build_number}.tar.gz