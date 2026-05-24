/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_ALGO_RULE_H
#define HCCLV2_ALGO_RULE_H

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include "auto_selector_base.h"

namespace ops_hccl {

/**
 * @brief 算法选择规则的结果
 */
struct RuleResult {
    bool matched;
    std::string algoName;
    
    RuleResult() : matched(false), algoName("") {}
    RuleResult(bool m, const std::string& name) : matched(m), algoName(name) {}
    
    static RuleResult Match(const std::string& algoName) { return RuleResult(true, algoName); }
    static RuleResult NoMatch() { return RuleResult(false, ""); }
};

/**
 * @brief 规则匹配的上下文信息
 */
struct RuleContext {
    const TopoInfoWithNetLayerDetails* topoInfo;
    const OpParam* opParam;
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>* configAlgMap;
    const AutoSelectorBase* selector;  // 用于调用基类辅助方法
    u64 dataSize;
    
    RuleContext(const TopoInfoWithNetLayerDetails* topo, const OpParam* param,
                const std::map<HcclCMDType, std::vector<HcclAlgoType>>* configMap,
                const AutoSelectorBase* sel)
        : topoInfo(topo), opParam(param), configAlgMap(configMap), selector(sel), dataSize(0) {
        if (opParam) {
            u64 perDataSize = DATATYPE_SIZE_TABLE[opParam->DataDes.dataType];
            dataSize = opParam->DataDes.count * perDataSize;
        }
    }
};

/**
 * @brief 算法选择规则基类
 * 
 * 每个规则代表一个算法选择条件，支持链式匹配
 */
class AlgoRule {
public:
    virtual ~AlgoRule() = default;
    
    /**
     * @brief 尝试匹配当前规则
     * @param context 匹配上下文
     * @return RuleResult 匹配结果，包含是否匹配及算法名称
     */
    virtual RuleResult TryMatch(const RuleContext& context) const = 0;
    
    /**
     * @brief 获取规则名称（用于日志）
     */
    virtual std::string GetRuleName() const = 0;
};

/**
 * @brief 函数式规则 - 使用 lambda 函数定义匹配逻辑
 */
class FunctionalRule : public AlgoRule {
public:
    using MatchFunc = std::function<RuleResult(const RuleContext&)>;
    
    FunctionalRule(const std::string& name, MatchFunc func)
        : name_(name), matchFunc_(std::move(func)) {}
    
    RuleResult TryMatch(const RuleContext& context) const override {
        return matchFunc_(context);
    }
    
    std::string GetRuleName() const override { return name_; }
    
private:
    std::string name_;
    MatchFunc matchFunc_;
};

/**
 * @brief 条件构建器 - 提供声明式的条件构建方法
 */
class ConditionBuilder {
public:
    using Predicate = std::function<bool(const RuleContext&)>;
    
    ConditionBuilder() : predicate_([](const RuleContext&) { return true; }) {}
    
    // 拓扑层级条件
    ConditionBuilder& WithTopoLevelNums(u32 levelNums) {
        AddCondition([levelNums](const RuleContext& ctx) {
            return ctx.topoInfo->topoLevelNums == levelNums;
        });
        return *this;
    }
    
    ConditionBuilder& WithTopoLevelNumsGreaterThan(u32 levelNums) {
        AddCondition([levelNums](const RuleContext& ctx) {
            return ctx.topoInfo->topoLevelNums > levelNums;
        });
        return *this;
    }
    
    ConditionBuilder& WithSingleLevel() {
        AddCondition([](const RuleContext& ctx) {
            return ctx.topoInfo->topoLevelNums == 1;
        });
        return *this;
    }
    
    ConditionBuilder& WithMultiLevel() {
        AddCondition([](const RuleContext& ctx) {
            return ctx.topoInfo->topoLevelNums > 1;
        });
        return *this;
    }
    
    // Level0 拓扑条件
    ConditionBuilder& WithLevel0Topo(Level0Shape topo) {
        AddCondition([topo](const RuleContext& ctx) {
            return ctx.topoInfo->level0Topo == topo;
        });
        return *this;
    }
    
    ConditionBuilder& WithLevel0MeshType(Level0MeshType meshType) {
        AddCondition([meshType](const RuleContext& ctx) {
            return ctx.topoInfo->level0MeshType == meshType;
        });
        return *this;
    }
    
    // 数据大小条件
    ConditionBuilder& WithDataSizeGreaterThan(u64 size) {
        AddCondition([size](const RuleContext& ctx) {
            return ctx.dataSize > size;
        });
        return *this;
    }
    
    ConditionBuilder& WithDataSizeLessThan(u64 size) {
        AddCondition([size](const RuleContext& ctx) {
            return ctx.dataSize <= size;
        });
        return *this;
    }
    
    ConditionBuilder& WithDataSizeInRange(u64 minSize, u64 maxSize) {
        AddCondition([minSize, maxSize](const RuleContext& ctx) {
            return ctx.dataSize > minSize && ctx.dataSize <= maxSize;
        });
        return *this;
    }
    
    // Rank 大小条件
    ConditionBuilder& WithRankSizeLessOrEqual(u32 rankSize) {
        AddCondition([rankSize](const RuleContext& ctx) {
            return ctx.topoInfo->userRankSize <= rankSize;
        });
        return *this;
    }
    
    ConditionBuilder& WithRankSizeGreaterThan(u32 rankSize) {
        AddCondition([rankSize](const RuleContext& ctx) {
            return ctx.topoInfo->userRankSize > rankSize;
        });
        return *this;
    }
    
    // PCIe 混合拓扑条件
    ConditionBuilder& WithPcieMix() {
        AddCondition([](const RuleContext& ctx) {
            return ctx.topoInfo->level0PcieMix;
        });
        return *this;
    }
    
    ConditionBuilder& WithoutPcieMix() {
        AddCondition([](const RuleContext& ctx) {
            return !ctx.topoInfo->level0PcieMix;
        });
        return *this;
    }
    
    // NHR 条件
    ConditionBuilder& WithLevel1Nhr() {
        AddCondition([](const RuleContext& ctx) {
            return ctx.topoInfo->Level1Nhr;
        });
        return *this;
    }
    
    ConditionBuilder& WithLevel0Nhr() {
        AddCondition([](const RuleContext& ctx) {
            return ctx.topoInfo->Level0Nhr;
        });
        return *this;
    }
    
    // 2 Die 条件
    ConditionBuilder& With2DieFullMesh() {
        AddCondition([](const RuleContext& ctx) {
            return ctx.topoInfo->is2DieFullMesh;
        });
        return *this;
    }
    
    // 网络实例条件
    ConditionBuilder& WithLocalNetInsSizeOfLayer0(u32 size) {
        AddCondition([size](const RuleContext& ctx) {
            return ctx.topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == size;
        });
        return *this;
    }
    
    // 自定义条件
    ConditionBuilder& WithCustomCondition(Predicate pred) {
        AddCondition(std::move(pred));
        return *this;
    }
    
    // 构建最终的规则
    std::unique_ptr<AlgoRule> Build(const std::string& ruleName, const std::string& algoName) {
        auto pred = predicate_;
        return std::make_unique<FunctionalRule>(ruleName, 
            [pred, algoName](const RuleContext& ctx) -> RuleResult {
                if (pred(ctx)) {
                    return RuleResult::Match(algoName);
                }
                return RuleResult::NoMatch();
            });
    }
    
private:
    void AddCondition(Predicate pred) {
        auto existing = predicate_;
        predicate_ = [existing, pred](const RuleContext& ctx) {
            return existing(ctx) && pred(ctx);
        };
    }
    
    Predicate predicate_;
};

/**
 * @brief 规则链 - 按优先级顺序尝试匹配规则
 */
class RuleChain {
public:
    /**
     * @brief 添加规则（按添加顺序作为优先级）
     */
    RuleChain& AddRule(std::unique_ptr<AlgoRule> rule) {
        rules_.push_back(std::move(rule));
        return *this;
    }
    
    /**
     * @brief 使用条件构建器添加规则
     */
    RuleChain& AddRule(ConditionBuilder&& builder, const std::string& ruleName, const std::string& algoName) {
        rules_.push_back(builder.Build(ruleName, algoName));
        return *this;
    }
    
    /**
     * @brief 执行规则链匹配
     * @return 第一个匹配的规则结果，如果没有匹配则返回 NoMatch
     */
    RuleResult Execute(const RuleContext& context) const {
        for (const auto& rule : rules_) {
            auto result = rule->TryMatch(context);
            if (result.matched) {
                HCCL_DEBUG("[RuleChain] Matched rule [%s], algo [%s]", 
                           rule->GetRuleName().c_str(), result.algoName.c_str());
                return result;
            }
        }
        return RuleResult::NoMatch();
    }
    
    /**
     * @brief 获取规则数量
     */
    size_t Size() const { return rules_.size(); }
    
    /**
     * @brief 清空所有规则
     */
    void Clear() { rules_.clear(); }
    
private:
    std::vector<std::unique_ptr<AlgoRule>> rules_;
};

/**
 * @brief 规则链构建器 - 提供流式 API 构建规则链
 */
class RuleChainBuilder {
public:
    RuleChainBuilder() = default;
    
    /**
     * @brief 开始构建一个新规则
     */
    ConditionBuilder When() {
        return ConditionBuilder();
    }
    
    /**
     * @brief 添加规则到链中
     */
    RuleChainBuilder& Then(std::unique_ptr<AlgoRule> rule) {
        chain_.AddRule(std::move(rule));
        return *this;
    }
    
    /**
     * @brief 构建并返回规则链
     */
    RuleChain Build() {
        return std::move(chain_);
    }
    
private:
    RuleChain chain_;
};

}  // namespace ops_hccl

#endif  // HCCLV2_ALGO_RULE_H
