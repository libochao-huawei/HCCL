/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu context signature header file
 */

#ifndef CCU_KERNEL_SIGNATURE_H
#define CCU_KERNEL_SIGNATURE_H

#include <sstream>
#include <string>

namespace hcomm {

class CcuKernelSignature {
public:
    CcuKernelSignature()  = default;
    ~CcuKernelSignature() = default;
    CcuKernelSignature(const CcuKernelSignature &other)
    {
        // 实现复制构造函数
        data << other.data.str();
    }
 
    void operator=(const CcuKernelSignature &other)
    {
        // 实现赋值操作
        data << other.data.str();
    }
 
    bool operator==(const CcuKernelSignature &rhs) const
    {
        return this == &rhs || data.str() == rhs.data.str();
    }
 
    // 用法Append<T>(t)
    template <typename T> void Append(T t)
    {
        data << t;
    }
 
    void Append(const CcuKernelSignature &other)
    {
        data << other.data.str();
    }

    std::string Describe() const
    {
        return "CcuKernelSignature[data=" + data.str() + "]";
    }
 
    // 下掉CcuContext GetSignatrue
    std::string GetData() const
    {
        return data.str();
    }
 
    friend class std::hash<hcomm::CcuKernelSignature>;

private:
    std::ostringstream data;
};

} // namespace hcomm

namespace std {

template <> class hash<hcomm::CcuKernelSignature> {
public:
    size_t operator()(const hcomm::CcuKernelSignature &signature) const
    {
        auto dataHash = hash<string>{}(signature.GetData());
        constexpr size_t res     = 17;
        constexpr size_t padding = 31;
        constexpr size_t preHash = res * padding;
        return preHash + dataHash;
    }
};

} // namespace std

#endif // _CCU_KERNEL_SIGNATURE_H