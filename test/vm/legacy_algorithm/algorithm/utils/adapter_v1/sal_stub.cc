/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2017-2022. All rights reserved.
 * Description: 系统抽象层实现
 * Author: xiaoshizhong
 * Create: 2017-07-26
 */

#include "sal.h"

#include <cmath>
#include <cstdlib>
#include <fcntl.h>
#include <mutex>
#include <syscall.h>
#include <sys/time.h> /* 获取时间 */
#include <slog.h>
#include <slog_api.h>
#include <securec.h>
#include <unistd.h>

#include "driver/ascend_hal.h"
#include "adapter_rts.h"
#include "adapter_hccp_common.h"
#include "adapter_hal.h"
#include "externalinput.h"
#include "dlhal_function.h"
#include "device_capacity.h"

using namespace std;
constexpr uint32_t HOST = 1;

#if T_DESC("C字符串处理函数适配", true)

u32 SalStrLen(const char *s, u32 maxLen)
{
    return strnlen(s, maxLen);
}

#endif

#if T_DESC("时间处理接口适配", true)

void SaluSleep(u32 usec)
{
    /* usleep()可能会因为进程收到信号(比如alarm)而提前返回EINTR, 后续优化  */
    s32 iRet = usleep(usec);
    if (iRet != 0) {
        HCCL_WARNING("Sleep: usleep failed[%d]: %s [%d]", iRet, strerror(errno), errno);
    }
}

void SalSleep(u32 sec)
{
    /* sleep()可能会因为进程收到信号(比如alarm)而提前返回EINTR, 后续优化  */
    s32 iRet = sleep(sec);
    if (iRet != 0) {
        HCCL_WARNING("Sleep: sleep failed[%d]: %s [%d]", iRet, strerror(errno), errno);
    }
}

HcclResult SalGetCurrentTimestamp(u64& timestamp)
{
    struct timeval tv;
    int ret = gettimeofday(&tv, nullptr);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("[Get][tCurrentTimestamp]errNo[0x%016llx] get timestamp fail, return[%d].",
        HCCL_ERROR_CODE(HCCL_E_SYSCALL), ret), HCCL_E_SYSCALL);
    timestamp = tv.tv_sec * 1000000 + tv.tv_usec; // 1000000: 单位转换 秒 -> 微秒
    return HCCL_SUCCESS;
}

u64 GetCurAicpuTimestamp()
{
    struct timespec timestamp;
    (void)clock_gettime(1, &timestamp);
    return static_cast<u64>((timestamp.tv_sec * 1000000000U) + (timestamp.tv_nsec));
}

#endif

#if T_DESC("跨进程处理函数", true)

// 去除字符串中的首位空格
std::string SalTrim(const std::string &s)
{
    std::string tempStr = s;
    if (!tempStr.empty()) {
        auto fiFirst = tempStr.find_first_not_of(" ");
        if (fiFirst != std::string::npos) {
            (void)tempStr.erase(0, fiFirst);
        }

        auto fiLast = tempStr.find_last_not_of(" ");
        if (fiLast != std::string::npos) {
            (void)tempStr.erase(fiLast + 1);
        }
    }

    return tempStr;
}

// 返回当前进程ID
s32 SalGetPid()
{
    return getpid();
}

HcclResult SalGetBareTgid(u32 *pid)
{
    CHK_PTR_NULL(pid);
    CHK_RET(hrtDeviceGetBareTgid(pid));
    return HCCL_SUCCESS;
}

// 返回当前线程ID
s32 SalGetTid()
{
    return syscall(SYS_gettid);
}

// 获取当前用户ID
u32 SalGetUid()
{
    return getuid();
}

#endif

#if T_DESC("环境变量处理适配", true)

std::string SalGetEnv(const char *name)
{
    if (name == nullptr || getenv(name) == nullptr) {
        return "EmptyString";
    }

    return getenv(name);
}

s32 SalSetEnv(const char *name, const char *value, int overwrite)
{
    return setenv(name, value, overwrite);
}
#endif

#if T_DESC("系统时间处理适配", true)

// 获取系统当前时间
s64 SalGetSysTime()
{
    // 获取当前系统时间,将时分秒清零
    time_t curTime = time(&curTime);  // time_t是一种时间类型，一般用来存放自1970年1月1日0点0时0分开始的秒数

    return static_cast<s64>(curTime);
}

#endif

#if T_DESC("库函数封装", true)
// 字符串转换成整型
HcclResult SalStrToInt(const std::string str, int base, s32 &val)
{
    try {
        val = std::stoi(str, nullptr, base);
    }
    catch (std::invalid_argument&) {
        HCCL_ERROR("[Transform][StrToInt]strtoi invalid argument, str[%s] base[%d] val[%d]", str.c_str(), base, val);
        return HCCL_E_PARA;
    }
    catch (std::out_of_range&) {
        HCCL_ERROR("[Transform][StrToInt]strtoi out of range, str[%s] base[%d] val[%d]", str.c_str(), base, val);
        return HCCL_E_PARA;
    }
    catch (...) {
        HCCL_ERROR("[Transform][StrToInt]strtoi catch errror, str[%s] base[%d] val[%d]", str.c_str(), base, val);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

// 字符串转换成浮点数
HcclResult SalStrToDouble(const std::string str, double &val)
{
    try {
        val = std::stod(str);
    }
    catch (std::invalid_argument&) {
        HCCL_ERROR("[Transform][StrToDouble]stod invalid argument, str[%s] val[%f]", str.c_str(), val);
        return HCCL_E_PARA;
    }
    catch (std::out_of_range&) {
        HCCL_ERROR("[Transform][StrToDouble]stod out of range, str[%s] val[%f]", str.c_str(), val);
        return HCCL_E_PARA;
    }
    catch (...) {
        HCCL_ERROR("[Transform][StrToDouble]stod catch error, str[%s] val[%f]", str.c_str(), val);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

// 字串符转换成无符号整型
HcclResult SalStrToULong(const std::string str, int base, u32 &val)
{
    try {
        u64 tmp = std::stoull(str, nullptr, base);
        if (tmp > INVALID_UINT) {
            HCCL_ERROR("[Transform][StrToULong]sal_strtoul out of range[%x], str[%s]", INVALID_UINT, str.c_str());
            return HCCL_E_PARA;
        } else {
            val = static_cast<u32>(tmp);
        }
    }
    catch (std::invalid_argument&) {
        HCCL_ERROR("[Transform][StrToULong]stoul invalid argument, str[%s] base[%d] val[%u]", str.c_str(), base, val);
        return HCCL_E_PARA;
    }
    catch (std::out_of_range&) {
        HCCL_ERROR("[Transform][StrToULong]stoul out of range, str[%s] base[%d] val[%u]", str.c_str(), base, val);
        return HCCL_E_PARA;
    }
    catch (...) {
        HCCL_ERROR("[Transform][StrToULong]stoul catch errror, str[%s] base[%d] val[%u]", str.c_str(), base, val);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

// 字串符转换成无符号长整型
HcclResult SalStrToULonglong(const std::string str, int base, u64 &val)
{
    try {
        val = std::stoull(str, nullptr, base);
    }
    catch (std::invalid_argument&) {
        HCCL_ERROR("[Transform][StrToULonglong]stoull invalid argument, str[%s] base[%d] val[%llu]",
            str.c_str(), base, val);
        return HCCL_E_PARA;
    }
    catch (std::out_of_range&) {
        HCCL_ERROR("[Transform][StrToULonglong]stoull out of range, str[%s] base[%d] val[%llu]",
            str.c_str(), base, val);
        return HCCL_E_PARA;
    }
    catch (...) {
        HCCL_ERROR("[Transform][StrToULonglong]stoull catch errror, str[%s] base[%d] val[%llu]",
            str.c_str(), base, val);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

// 字串符转换成长整型
HcclResult SalStrToLonglong(const std::string str, int base, s64 &val)
{
    try {
        val = std::stoll(str, nullptr, base);
    }
    catch (std::invalid_argument&) {
        HCCL_ERROR("[Transform][SalStrToLonglong]stoll invalid argument, str[%s] base[%d] val[%lld]",
            str.c_str(), base, val);
        return HCCL_E_PARA;
    }
    catch (std::out_of_range&) {
        HCCL_ERROR("[Transform][SalStrToLonglong]stoll out of range, str[%s] base[%d] val[%lld]",
            str.c_str(), base, val);
        return HCCL_E_PARA;
    }
    catch (...) {
        HCCL_ERROR("[Transform][SalStrToLonglong]stoll catch errror, str[%s] base[%d] val[%lld]",
            str.c_str(), base, val);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}
#endif

#if T_DESC("路径信息函数", true)
HcclResult SalIsDirExist(const std::string &dir, s32 &status)
{
    // 文件存在：0，不存在：-1，异常：1
    if (dir.length() == 0) {
        HCCL_ERROR("[Check][DirExist]invalid path length:%d", dir.length());
        status = 1;
        return HCCL_E_PARA;
    }
    char realPath[PATH_MAX] = {0};
    if (realpath(dir.c_str(), realPath) == nullptr) {
        // 如果错误码是文件不存在，记录状态，否则报错
        if (errno == ENOENT) {
            status = -1;
            return HCCL_SUCCESS;
        } else {
            status = 1;
            HCCL_ERROR("[Check][DirExist]path %s is invalid errno(%d):%s", dir.c_str(), errno, strerror(errno));
            return HCCL_E_PARA;
        }
    } else {
        status = 0;
    }
    return HCCL_SUCCESS;
}
#endif

#if T_DESC("数学计算处理函数", true)
s32 SalLog2(s32 data)
{
    return static_cast<s32>(log2(data));
}
#endif

#if T_DESC("计算类型占用内存大小函数", true)
HcclResult SalGetDataTypeSize(HcclDataType dataType, u32 &dataTypeSize)
{
    if ((dataType >= HCCL_DATA_TYPE_INT8) &&
        (dataType < HCCL_DATA_TYPE_RESERVED)) {
        dataTypeSize = SIZE_TABLE[dataType];
    } else {
        HCCL_ERROR("[Get][DataTypeSize]errNo[0x%016llx] get date size failed. dataType[%s] is invalid.", \
            HCOM_ERROR_CODE(HCCL_E_PARA), GetDataTypeEnumStr(dataType).c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}
#endif

#if T_DESC("设置指定位值函数", true)
void SalSetBitOne(u64 &value, u64 index)
{
    u64 bit = static_cast<u64>(1) << index;
    value |= bit;
    return;
}
#endif


#if T_DESC("json处理函数", true)
HcclResult SalParseInformation(nlohmann::json &parseInformation, const std::string &information)
{
    try {
        parseInformation = nlohmann::json::parse(information);
    } catch (...) {
        HCCL_ERROR("[Parse][Information] errNo[0x%016llx] load allocated resource to json fail. "\
            "please check json input!", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult SalGetJsonProperty(const nlohmann::json &obj, const std::string &propName, std::string &propValue)
{
    /* 查找json对象中是否有该属性, 不存在的属性不能直接访问 */
    CHK_PRT_RET(obj.find(propName) == obj.end(),
        HCCL_ERROR("[Get][JsonProperty]json object has no property called %s", propName.c_str()), HCCL_E_INTERNAL);

    /* 所有属性值都必须是字符串 */
    if (obj[propName].is_string()) {
        propValue = obj[propName];
        return HCCL_SUCCESS;
    } else {
        printf("property value of Name[%s] is not string!", propName.c_str());
        return HCCL_E_INTERNAL;
    }
}
#endif

HcclResult GetLocalHostIP(hccl::HcclIpAddress &ip, u32 devPhyId)
{
    if (!ip.IsInvalid()) {
        return HCCL_SUCCESS;
    }
    std::vector<std::pair<std::string, hccl::HcclIpAddress>> ifInfos;
    CHK_RET(hrtGetHostIf(ifInfos, devPhyId));
    CHK_PRT_RET(ifInfos.empty(), HCCL_ERROR("[Get][LocalHostIP]there is no valid host if."), HCCL_E_NOT_FOUND);

    CHK_RET(FindLocalHostIP(ifInfos, ip));

    return HCCL_SUCCESS;
}

bool FindHostIPByNicClass(const std::map<std::string, std::map<std::string, hccl::HcclIpAddress>> &nicClassifyInfo,
    const std::string &nicClass, hccl::HcclIpAddress &ip)
{
    auto iterClass = nicClassifyInfo.find(nicClass);
    if (iterClass != nicClassifyInfo.end()) {
        if (iterClass->second.empty()) {
            HCCL_WARNING("nic class[%s]: no valid ip.", nicClass.c_str());
            return false;
        }
        ip = iterClass->second.begin()->second;
        HCCL_INFO("get host ip success. host ifname[%s] ip[%s]", iterClass->second.begin()->first.c_str(),
            ip.GetReadableAddress());
        return true;
    }
    return false;
}

HcclResult FindLocalHostIPByIfname(std::vector<std::pair<std::string, hccl::HcclIpAddress>> &ifInfos, s32 family,
    hccl::HcclIpAddress &ip)
{
    for (auto &ifInfo : ifInfos) {
        if (ifInfo.second.GetFamily() != family) {
            continue;
        }
        u32 matchLen = ifInfo.first.size();
        bool configIfNamesFlag = false;
        for (u32 i = 0; i < GetExternalInputHcclSocketIfName().configIfNames.size(); i++) {
            matchLen = GetExternalInputHcclSocketIfName().searchExact ?
                ifInfo.first.size() :
                GetExternalInputHcclSocketIfName().configIfNames[i].size();
            if (ifInfo.first.compare(0, matchLen, GetExternalInputHcclSocketIfName().configIfNames[i], 0,
                matchLen) == 0) {
                configIfNamesFlag = true;
            }
        }
        if ((configIfNamesFlag) ^ (GetExternalInputHcclSocketIfName().searchNot)) {
            configIfNamesFlag = false;
            ip = ifInfo.second;
            HCCL_RUN_INFO("get host ip success. name[%s] ip[%s]", ifInfo.first.c_str(),
                ifInfo.second.GetReadableAddress());
            return HCCL_SUCCESS;
        }
    }
    return HCCL_E_NOT_FOUND;
}

HcclResult FindLocalHostIPByIfname(std::vector<std::pair<std::string, hccl::HcclIpAddress>> &ifInfos,
    hccl::HcclIpAddress &ip)
{
    s32 firstFamily = (GetExternalInputHcclSocketFamily() == -1) ? AF_INET :
        GetExternalInputHcclSocketFamily();
    HcclResult ret = FindLocalHostIPByIfname(ifInfos, firstFamily, ip);
    if (ret == HCCL_E_NOT_FOUND) {
        s32 family = (firstFamily == AF_INET) ? AF_INET6 : AF_INET;
        ret = FindLocalHostIPByIfname(ifInfos, family, ip);
    }
    return ret;
}

HcclResult FindLocalHostIPDefault(std::vector<std::pair<std::string, hccl::HcclIpAddress>> &ifInfos, s32 family,
    hccl::HcclIpAddress &ip)
{
    std::map<std::string, std::map<std::string, hccl::HcclIpAddress>> nicClassify;
    for (auto &ifInfo : ifInfos) {
        if (ifInfo.second.GetFamily() != family) {
            continue;
        }
        if (ifInfo.first.find("lo") == 0) {
            nicClassify["lo"].insert({ ifInfo.first, ifInfo.second });
        } else if (ifInfo.first.find("docker") == 0) {
            nicClassify["docker"].insert({ ifInfo.first, ifInfo.second });
        } else {
            nicClassify["normal"].insert({ ifInfo.first, ifInfo.second });
        }
        HCCL_DEBUG("ifname[%s] addr[%s]", ifInfo.first.c_str(), ifInfo.second.GetReadableAddress());
    }

    if (FindHostIPByNicClass(nicClassify, "normal", ip)) {
        HCCL_RUN_INFO("nic class[normal]: find nic[%s] success.", ip.GetReadableAddress());
        return HCCL_SUCCESS;
    } else if (FindHostIPByNicClass(nicClassify, "docker", ip)) {
        HCCL_RUN_INFO("nic class[docker]: find nic[%s] success.", ip.GetReadableAddress());
        return HCCL_SUCCESS;
    } else if (FindHostIPByNicClass(nicClassify, "lo", ip)) {
        HCCL_RUN_INFO("nic class[lo]: find nic[%s] success.", ip.GetReadableAddress());
        return HCCL_SUCCESS;
    }
    return HCCL_E_NOT_FOUND;
}

HcclResult FindLocalHostIPDefault(std::vector<std::pair<std::string, hccl::HcclIpAddress>> &ifInfos,
    hccl::HcclIpAddress &ip)
{
    s32 firstFamily = (GetExternalInputHcclSocketFamily() == -1) ? AF_INET :
        GetExternalInputHcclSocketFamily();
    HcclResult ret = FindLocalHostIPDefault(ifInfos, firstFamily, ip);
    if (ret == HCCL_E_NOT_FOUND) {
        s32 family = (firstFamily == AF_INET) ? AF_INET6 : AF_INET;
        ret = FindLocalHostIPDefault(ifInfos, family, ip);
    }
    return ret;
}

HcclResult FindLocalHostIP(std::vector<std::pair<std::string, hccl::HcclIpAddress>> &ifInfos, hccl::HcclIpAddress &ip)
{
    CHK_PRT_RET(ifInfos.empty(),
        HCCL_ERROR("[Find][LocalHostIP]there is no valid host if. (host if is not exist or not in whitelist)"),
        HCCL_E_NOT_FOUND);

    hccl::HcclIpAddress tmpIp;
    std::string ipModleInfo;
    if (!GetExternalInputMasterInfo().agentIp.IsInvalid()) {
        tmpIp = GetExternalInputMasterInfo().agentIp;
        ipModleInfo = "WORKER IP";
    } else if (!GetExternalInputHcclControlIfIp().IsInvalid()) {
        tmpIp = GetExternalInputHcclControlIfIp();
        ipModleInfo = "IF IP";
    }
    if (!tmpIp.IsInvalid()) {
        // 匹配指定IP的网卡信息
        for (auto &ifInfo : ifInfos) {
            if (ifInfo.second == tmpIp) {
                ip = ifInfo.second;
                HCCL_RUN_INFO("get host ip success by if IP of [%s]. name[%s] ip[%s]", ipModleInfo.c_str(),
                    ifInfo.first.c_str(), ifInfo.second.GetReadableAddress());
                return HCCL_SUCCESS;
            }
        }
        HCCL_ERROR("[Find][LocalHostIP]ip [%s] of [%s] is not found in the nic list.", tmpIp.GetReadableAddress(),
            ipModleInfo.c_str());
        return HCCL_E_NOT_FOUND;
    } else if (!GetExternalInputHcclSocketIfName().configIfNames.empty()) {
        // 使用Host网卡名和环境变量HCCL_SOCKET_IFNAME配置的网卡名进行比较
        HcclResult ret = FindLocalHostIPByIfname(ifInfos, ip);
        if (ret != HCCL_SUCCESS) {
            for (auto &ifInfo : ifInfos) {
                HCCL_ERROR("[Get][LocalServerId]get host ip fail by socket Ifname. name[%s] ip[%s]",
                    ifInfo.first.c_str(), ifInfo.second.GetReadableAddress());
            }
            return HCCL_E_NOT_FOUND;
        }
    } else {
        CHK_PRT_RET(FindLocalHostIPDefault(ifInfos, ip), HCCL_ERROR("[Find][LocalHostIP]there is no host if."),
            HCCL_E_NOT_FOUND);
    }
    return HCCL_SUCCESS;
}

std::string GetLocalServerId(std::string &serverId)
{
    hccl::HcclIpAddress hostIP;
    HcclResult ret = GetLocalHostIP(hostIP);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[Get][ServerId]GetLocalHostIP Failed, Use INVALID value");
        serverId = "0.0.0.0";
    } else {
        serverId = hostIP.GetReadableAddress();
    }
    return serverId;
}

HcclResult IsAllDigit(const char *strNum)
{
    // 参数有效性检查
    CHK_PTR_NULL(strNum);

    u32 nLength = SalStrLen(strNum);
    for (u32 index = 0; index < nLength; index++) {
        if (!isdigit(strNum[index])) {
            HCCL_ERROR("[Check][Isdigit]errNo[0x%016llx] In judge all digit, check isdigit failed."
                "ensure that the number is an integer. strNum[%u] is [%d](Dec)",
                HCCL_ERROR_CODE(HCCL_E_PARA), index, strNum[index]);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

void SetThreadName(const std::string &threadStr){
    s32 sRet = pthread_setname_np(pthread_self(), threadStr.c_str());
    CHK_PRT_CONT(sRet != 0, HCCL_WARNING("err[%d] link[%s] nameSet failed.", sRet, threadStr.c_str()));
}

HcclResult CheckHexUInt(const std::string& str)
{
    if (str.length() != 10) {  // 有效的16进制无符号整型数如0xFFFFFFFF共10个字符
        HCCL_ERROR("[Check][HexUInt]string[%s] is not a valid hexadecimal uint value.", str.c_str());
        return HCCL_E_PARA;
    }
    if (str.substr(0, 2) != "0x" && str.substr(0, 2) != "0X") {  // 字符串前两2个字符，有效的16进制数以0x或者0X开头
        HCCL_ERROR("[Check][HexUInt]string[%s] is not a valid hexadecimal uint value.", str.c_str());
        return HCCL_E_PARA;
    }
    for (int i = 2; i < 10; i++) {  // 从第2个字符到第10个字符判断是否是有效字符
        if ((str[i] >= '0' && str[i] <= '9') ||
            (str[i] >= 'a' && str[i] <= 'f') ||
            (str[i] >= 'A' && str[i] <= 'F')) {
            continue;
        } else {
            HCCL_ERROR("[Check][HexUInt]string[%s] is not a valid hexadecimal uint value.", str.c_str());
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

bool IsGeneralServer()
{
    return false;
}

// 判断host侧是否需要使用device网卡
HcclResult IsHostUseDevNic(bool &isHdcMode)
{
    CHK_RET(hccl::DlHalFunction::GetInstance().DlHalFunctionInit());
    // 如果不位于host侧直接返回
    uint32_t info = 0;
    CHK_RET(hrtDrvGetPlatformInfo(&info));
    if (info != HOST) {
        HCCL_INFO("[IsHostUseDevNic] : now on device, info: [%u]", info);
        isHdcMode = false;
        return HCCL_SUCCESS;
    }

    // 通用服务器直接返回
    if (IsGeneralServer()) {
        isHdcMode = false;
        HCCL_INFO("[IsHostUseDevNic] : universal server, isHdcMode[%u]", isHdcMode);
        return HCCL_SUCCESS;
    }

    // 在aiserver上判断该环境变量是否设置
    isHdcMode = true;
    HCCL_INFO("IsHostUseDevNic[%u]", isHdcMode);

    return HCCL_SUCCESS;
}

u32 GetNicPort(u32 devicePhyId, const std::vector<u32> &ranksPort, u32 userRank, bool isUseRanksPort)
{
    if (!ranksPort.size() || (userRank < ranksPort.size() && ranksPort[userRank] == HCCL_INVALID_PORT) ||
        (!isUseRanksPort && !hccl::Is310PDevice())) {
        // 使用device nic时且无外部配置的port(ranksPort长度为0或者有port但为无效值)时，默认16666
        return HETEROG_CCL_PORT;
    }

    if (isUseRanksPort && ranksPort.size() && userRank < ranksPort.size()) {
        return ranksPort[userRank];
    }

    if (GetExternalInputHcclIfBasePort() == HCCL_INVALID_PORT) {
        HCCL_INFO("[Init][Nic] port is set to HOST_PARA_BASE_PORT");
        return (HOST_PARA_BASE_PORT + devicePhyId);
    }else {
        return (GetExternalInputHcclIfBasePort() + HCCL_AISERVER_DEVICE_NUM + devicePhyId);
    }
    // peer及hdc模式下listen_start/batch_connect/listen_stop调用支持指定端口
    // server及client按照此相同规则指定端口
}