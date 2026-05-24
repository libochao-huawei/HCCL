/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RULE_BASED_SELECTOR_H
#define HCCLV2_RULE_BASED_SELECTOR_H

#include "algo_rule.h"
#include "auto_selector_base.h"

namespace ops_hccl {

/**
 * @brief 基于规则链的算法选择器基类
 * 
 * 该类提供了一种声明式的方式来定义算法选择逻辑。
 * 子类只需要实现 BuildRules() 方法来定义规则链，无需编写复杂的 if-else 逻辑。
 * 
 * 使用示例:
 * @code
 * class MySelector : public RuleBasedSelector {
 * protected:
 *     void BuildCcuScheduleRules(RuleChain& chain) override {
 *         chain.AddRule(ConditionBuilder()
 *             .WithMultiLevel()
 *             .WithLevel0Topo(Level0Shape::MESH_1D)
 *             .WithLevel1Nhr(),
 *             "Rule_MultiLevel_NHR", "CcuAllGatherNHR1DMem2Mem");
 *             
 *         chain.AddRule(ConditionBuilder()
 *             .WithSingleLevel()
 *             .WithLevel0Topo(Level0Shape::MESH_1D),
 *             "Rule_SingleLevel_Mesh1D", "CcuAllGatherMesh1DMem2Mem");
 *     }
 * };
 * @endcode
 */
class RuleBasedSelector : public AutoSelectorBase {
public:
    RuleBasedSelector() : rulesBuilt_(false) {}
    
protected:
    /**
     * @brief 构建 CCU Schedule 模式的规则链
     * 子类应重写此方法以添加自定义规则
     */
    virtual void BuildCcuScheduleRules(RuleChain& chain) {}
    
    /**
     * @brief 构建 CCU MS 模式的规则链
     * 子类应重写此方法以添加自定义规则
     */
    virtual void BuildCcuMsRules(RuleChain& chain) {}
    
    /**
     * @brief 构建 AICPU 模式的规则链
     * 子类应重写此方法以添加自定义规则
     */
    virtual void BuildAicpuRules(RuleChain& chain) {}
    
    /**
     * @brief 构建 AIV 模式的规则链
     * 子类应重写此方法以添加自定义规则
     */
    virtual void BuildAivRules(RuleChain& chain) {}
    
    /**
     * @brief 构建 DPU 模式的规则链
     * 子类应重写此方法以添加自定义规则
     */
    virtual void BuildDPURules(RuleChain& chain) {}
    
    /**
     * @brief 确保规则已构建（懒加载）
     */
    void EnsureRulesBuilt() const {
        if (!rulesBuilt_) {
            auto* self = const_cast<RuleBasedSelector*>(this);
            self->BuildCcuScheduleRules(self->ccuScheduleRules_);
            self->BuildCcuMsRules(self->ccuMsRules_);
            self->BuildAicpuRules(self->aicpuRules_);
            self->BuildAivRules(self->aivRules_);
            self->BuildDPURules(self->dpuRules_);
            self->rulesBuilt_ = true;
        }
    }
    
    /**
     * @brief 执行规则链选择算法
     */
    SelectorStatus ExecuteRuleChain(const RuleChain& chain, const RuleContext& context,
                                    std::string& selectAlgName) const {
        auto result = chain.Execute(context);
        if (result.matched) {
            selectAlgName = result.algoName;
            return SelectorStatus::MATCH;
        }
        return SelectorStatus::NOT_MATCH;
    }
    
    // 重写基类方法，使用规则链
    SelectorStatus SelectCcuScheduleAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                         const OpParam& opParam,
                                         const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap,
                                         std::string& selectAlgName) const override {
        EnsureRulesBuilt();
        if (ccuScheduleRules_.Size() == 0) {
            // 如果没有定义规则，回退到基类实现
            return AutoSelectorBase::SelectCcuScheduleAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
        }
        RuleContext context(topoInfo, &opParam, &configAlgMap, this);
        return ExecuteRuleChain(ccuScheduleRules_, context, selectAlgName);
    }
    
    SelectorStatus SelectCcuMsAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                   const OpParam& opParam,
                                   const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap,
                                   std::string& selectAlgName) const override {
        EnsureRulesBuilt();
        if (ccuMsRules_.Size() == 0) {
            return AutoSelectorBase::SelectCcuMsAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
        }
        RuleContext context(topoInfo, &opParam, &configAlgMap, this);
        return ExecuteRuleChain(ccuMsRules_, context, selectAlgName);
    }
    
    SelectorStatus SelectAicpuAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                   const OpParam& opParam,
                                   const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap,
                                   std::string& selectAlgName) const override {
        EnsureRulesBuilt();
        if (aicpuRules_.Size() == 0) {
            return AutoSelectorBase::SelectAicpuAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
        }
        RuleContext context(topoInfo, &opParam, &configAlgMap, this);
        return ExecuteRuleChain(aicpuRules_, context, selectAlgName);
    }
    
    SelectorStatus SelectAivAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                 const OpParam& opParam,
                                 const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap,
                                 std::string& selectAlgName) const override {
        EnsureRulesBuilt();
        if (aivRules_.Size() == 0) {
            return AutoSelectorBase::SelectAivAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
        }
        RuleContext context(topoInfo, &opParam, &configAlgMap, this);
        return ExecuteRuleChain(aivRules_, context, selectAlgName);
    }
    
    SelectorStatus SelectDPUAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                 const OpParam& opParam,
                                 const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap,
                                 std::string& selectAlgName) const override {
        EnsureRulesBuilt();
        if (dpuRules_.Size() == 0) {
            return AutoSelectorBase::SelectDPUAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
        }
        RuleContext context(topoInfo, &opParam, &configAlgMap, this);
        return ExecuteRuleChain(dpuRules_, context, selectAlgName);
    }

    // 规则链存储
    mutable RuleChain ccuScheduleRules_;
    mutable RuleChain ccuMsRules_;
    mutable RuleChain aicpuRules_;
    mutable RuleChain aivRules_;
    mutable RuleChain dpuRules_;
    mutable bool rulesBuilt_;
};

/**
 * @brief 辅助函数：创建简单的条件规则
 * 
 * 用法:
 * @code
 * auto rule = MakeRule("MyRule", "MyAlgo",
 *     [](const RuleContext& ctx) { return ctx.topoInfo->topoLevelNums == 1; });
 * @endcode
 */
inline std::unique_ptr<AlgoRule> MakeRule(const std::string& ruleName, const std::string& algoName,
                                          std::function<bool(const RuleContext&)> predicate) {
    return std::make_unique<FunctionalRule>(ruleName,
        [predicate, algoName](const RuleContext& ctx) -> RuleResult {
            if (predicate(ctx)) {
                return RuleResult::Match(algoName);
            }
            return RuleResult::NoMatch();
        });
}

}  // namespace ops_hccl

#endif  // HCCLV2_RULE_BASED_SELECTOR_H
