# RFC：HCCL Plugin —— HCCL自定义算法扩展框架

- 起始日期：2026-05-29
- RFC PR编号：(待填写)
- 相关Issue：126

---

## 1. 概要

　　HCCL Plugin旨在为HCCL提供一个自定义算法扩展框架。其目标是在不修改HCCL核心代码的前提下，针对已有算子（如AllReduce、AllGather等）通过动态库形式添加自定义算法，使自定义算法能够无缝接入现有的算法选择和执行流程。

---

## 2. 背景与动机

### 2.1 背景

　　HCCL算子仓包含多种算子（AllReduce、AllGather、Broadcast、Reduce等），每个算子可包含多种算法实现。当前算法通过以下方式选择：

- **拓扑感知选择**：根据集群拓扑结构（1D Mesh、2D Mesh、CLOS等）自动选择最适合的算法。
- **数据大小阈值**：根据传输数据大小选择不同算法（如小数据用OneShot，大数据用TwoShot/NHR）。
- **硬件形态适配**：不同硬件形态（如950）需要不同算法实现。

　　当前添加新算法存在以下限制：

- **代码侵入**：需要修改HCCL源码，添加注册代码。
- **构建耦合**：新算法需要与HCCL源码一起编译。
- **发布依赖**：算法更新需要重新编译和发布整个HCCL。
- **选择逻辑封闭**：新算法难以接入现有的算法选择流程。

　　针对以上限制，HCCL Plugin的设计方案逐一予以解决：

- **解决代码侵入问题**：用户开发新算法只需实现标准接口并打包为动态库，**无需修改任何HCCL源码**。
- **解决构建耦合问题**：自定义算法以独立动态库（`.so`）形式交付，通过 `dlopen` 在运行时加载，**与HCCL主库完全解耦编译**，用户可使用独立的构建脚本单独编译算法包。
- **解决发布依赖问题**：HCCL Plugin动态库和自定义算法动态库均可独立安装到指定目录下，**更新算法只需替换对应`.so`文件**，无需重新编译或发布整个HCCL。
- **解决选择逻辑封闭问题**：HCCL Plugin在HCCL原有算法选择流程的入口处插入优先匹配逻辑，**自定义算法可无缝参与现有的选择流程**，未命中时自动回退到原有逻辑，两套机制互不干扰。

### 2.2 支持的场景

　　HCCL Plugin可支持以下扩展自定义算法的场景。
- **新算法实现**：用户希望添加自己设计的全新算法（如优化的Ring、Tree算法变体） 。
- **新硬件支持**：支持新硬件形态或新拓扑结构，需要添加对应的算法实现。
- **定制化优化**：针对特定业务场景（如特定网络环境、特定数据模式）进行算法定制。
- **实验性算法**：在生产环境外验证新算法性能。

---

## 3. HCCL通信库代码结构及算子执行流程解读

### 3.1 HCCL通信库代码结构

　　HCCL通信库的关键目录如下所示：
```
│── src                          # HCCL算子源码目录
|    ├── common                  # 通用逻辑，包括类型定义、日志模块等
|    └── ops                     # HCCL算子实现
|        ├── all_gather          # AllGather算子实现
|        ├── all_reduce          # AllReduce算子实现
|        ├── broadcast           # Broadcast算子实现
|        |   ├── executor        # Broadcast算子执行器
|        |   ├── selector        # Broadcast算法选择器
|        |   ├── template        # Broadcast算法模板
|        |   └── broadcast_op.cc # Broadcast算子对外提供的API实现
|        ├── ......              # 其他算子实现
|        └──  op_common          # 算子通用组件
|            ├── executor        # 执行器
|            ├── selector        # 算法选择器
|            ├── template        # 算法模板
|            ├── topo            # 通信域拓扑信息获取和转换 
|            └── op_common.cc    # 算子通用函数文件
├── include                      # HCCL对外头文件
├── test                         # 测试代码目录
├── examples                     # 样例代码目录
├── build.sh                     # 编译构建脚本
└── .......                      # 其他目录

```
　　`/ops`目录定义了HCCL算子实现，包含`all_gather`、`all_reduce`等常见集合通信算子，每个算子实现其执行器（`executor`）、算法选择器（`selector`）、算法模板（`template`）和对外提供的API文件（`XX_op.cc`）。  

　　`/ops`目录下的`/op_common`目录定义了算子通用组件，包括算子执行器基类、算法选择器公共逻辑、算法模板基类、通信域拓扑处理等各算子共用的基础设施。

### 3.2 HCCL通信库算子执行流程

　　以`Broadcast`算子的执行流程为例，当应用调用`HcclBroadcast()`后，首先判断是否为910_95或950设备，若非910_95或950设备则回退至旧逻辑，即调用`HcclBroadcastInner()`，按照旧逻辑实现`Broadcast`算子，不再执行以下流程。  
　　否则，主要通过 **算法选择** 和 **算法执行** 两步骤完成`Broadcast`操作，具体流程如下：  
　　（1）调用`Selector()`函数进行算法选择：  
　　　　1) `Selector()`函数位于`src/ops/op_common/op_common.cc`文件中，其主要逻辑为：  
　　　　　　创建算法选择执行器实例`collAlgSelector`（ExecuteSelector 类），调用`collAlgSelector->Run()`。  
　　　　2) `Run()`函数位于`src/ops/op_common/selector/execute_selector.cc`文件中，其主要逻辑为：  
　　　　　　① 从全局选择器注册表获取所有已注册的选择器；  
　　　　　　② 若算子为Mc2模式（Multi-Channel v2），则将其选择器集合置为仅含优先级为18的选择器；若算子为非Mc2模式，按操作类型获取选择器集合；  
　　　　　　③ 按照优先级从高到低遍历选择器集合，调用选择器的`Select()`方法检查是否匹配；  
　　　　　　④ 若`Select()`返回`SelectorStatus::MATCH`，则说明已选定执行算法，退出遍历；否则继续遍历下一选择器。  
　　　　3) `Select()`函数位于`src/ops/op_common/selector/auto_selector_base.cc`文件中，其主要逻辑为：  
　　　　　　根据运行模式（DPU、CCU_MS、AIV、AICPU等）调用对应的选择函数，例如当为AICPU模式时，调用`SelectAicpuAlgo()`函数进行算法选择。  
　　　　4) `Broadcast`算子的`SelectAicpuAlgo()`函数位于`src/ops/broadcast/selector/broadcast_auto_selector.cc`文件下，其主要逻辑为：  
　　　　　　根据拓扑信息（如Level0Topo的具体形状、层级数目等）选定算法名称，例如多层级下若Level0Topo形状为Mesh，选择`ParallelMesh1DNHR`算法。  
　　（2）调用`HcclExecOp()`函数执行`Selector()`函数选中的算法。  
　　　　1) `HcclExecOp()`函数位于`src/ops/op_common/op_common.cc`文件中，其主要逻辑为：  
　　　　　　① 根据操作类型和选定的算法名称(以下都假定选中`ParallelMesh1DNHR`算法)，获取对应的`executor`实例。  
　　　　　　② 创建线程、计算通信所需资源等。  
　　　　　　③ 算法执行：调用`executor`的算法编排，即调用`executor->Orchestrate()`函数。  
　　　　2) `executor`的`Orchestrate()`函数位于`src/ops/broadcast/executor/ins_v2_broadcast_sole_executor.cc`中，主要逻辑为：  
　　　　　　① 进一步计算资源、进行数据分片等步骤。  
　　　　　　② 调用`ParallelMesh1DNHR`算法模板的`KernelRun`函数，完成该次通信操作，即`algTemplate->KernelRun()`。  
　　　　3) `ParallelMesh1DNHR`算法的`KernelRun`函数位于`src/ops/broadcast/template/aicpu/ins_temp_broadcast_nhr.cc`中，其主要逻辑为：  
　　　　　　根据NHR算法逻辑执行远端读、远端写、本地线程同步、本地数据拷贝等操作来完成该次通信。

---

## 4. 详细设计

### 4.1 总体架构

　　整个Plugin系统由三大部分构成：

　　**(1) Plugin的管理和自定义算法调用**

　　该部分内嵌在HCCL代码仓中，通过设计实现`HcclPluginManager`，获取并保存Plugin动态库的句柄，在HCCL代码仓中进行Plungin生命周期的管理以及自定义算法的选择和执行调用。

　　**(2) Plugin动态库**

　　该部分为独立模块，作为HCCL与自定义算法之间的桥梁，它定义并实现一套标准接口，主要包括：

- **算法注册与管理**：在初始化时自动扫描指定目录下的算法库和策略文件，解析自定义算法的策略条件，将用户算法实现注册到plugin的算法注册表中。  
- **算法选择**：接收来自HCCL的通信请求，根据此次通信操作参数和拓扑等信息，遍历算法注册表进行算法匹配，选择分数最高的算法返回其算法名。
- **算法执行**：根据选定的算法名，动态加载对应的自定义算法库，并调用其标准的执行函数，完成实际的通信操作。

　　**(3) 自定义算法策略与算法实现**

- **策略动态库**：将算法选择条件（消息大小、网络拓扑类型、设备型号、Rank个数等）以硬编码字符串常量的形式编译进 `.so`，通过导出函数 `GetHcclPluginStrategy()` 供Plugin调用。一个算子对应一个策略库。
- **算法实现动态库**：以AICPU算子开发为例，包含host侧算法编排逻辑和device侧Kernel执行逻辑，预编译为动态库，该库须导出一个标准的执行入口函数`HcclPluginAlgExecute`，供Plugin调用。

　　整体数据流如下：

```
【初始化阶段】（通信操作初始化时执行）

InitEnvConfig()
  └─► HcclPluginManager::Init()
        └─► dlopen(HCCL_PLUGIN_PATH)        → 加载Plugin动态库
        └─► dlsym → GetHcclPlugin()         → 获取HcclPlugin_t函数表
        └─► plugin_->Init(HCCL_PLUGIN_ALG_DIR)
              └─► 扫描算子子目录（AllReduce/、AllGather/、...）
              └─► 对每个算子目录：
                    dlopen(libhccl_plugin_{op}_strategy.so)
                    dlsym → GetHcclPluginStrategy() → 策略JSON字符串
                    PluginRegisterAlg()             → 注册该算子的算法
                          ├─►ParseOpStrategy()               → 解析rules、priority
                          │   └─►填充algLibPath（AlgName/libAlgName.so）
                          └─► 写入context->registry[op]   → 写入PluginContext注册表
                    dlclose(libhccl_plugin_{op}_strategy.so)


【算法选择阶段】（每次通信调用时执行）

用户应用
  └─► HcclAllReduce() / HcclBroadcast() / ...
        └─► op_common::Selector()
              │
              ├─► [Plugin 路径] HcclPluginManager::IsLoaded() == true
              │     └─► plugin_->SelectAlg(context, param, topoInfo)
              │           └─► BuildEvalContext()    → 从param + topoInfo提取匹配字段
              │           └─► 遍历注册表（按priority降序）
              │                 ├─► TryLoadScoreFn()  → 懒加载打分函数
              │                 ├─► [有打分函数] scoreFn(&evalCtx) → 获取动态分数
              │                 │     └─► 返回-1 → 跳过该算法
              │                 └─► [无打分函数] EvaluateAlg() → 静态rules匹配
              │                       └─► 命中 → 以priority作为分数
              │           └─► 选分数最高的算法；同分时按注册顺序
              │           └─► 命中 → param.pluginSelected=true，返回算法名
              │
              └─► [回退路径] Plugin 未命中/未加载
                    └─► 原有HCCL选择逻辑（ExecuteSelector::Run() → Select() → ...）


【算法执行阶段】（每次通信调用时执行）

        └─► op_common::HcclExecOp()
              │
              ├─► [Plugin 路径] param.pluginSelected == true
              │     └─► plugin_->ExecuteAlg(context, algName, param, topoInfo)
              │           └─► FindAlgEntry()              → 从注册表定位算法条目
              │           └─► dlopen(算法.so)             → 懒加载算法动态库
              │           └─► dlsym → HcclPluginAlgExecute → 懒加载函数指针
              │           └─► HcclPluginAlgExecute(param, commResources, topoInfo)
              │                 → 执行自定义算法，完成通信
              │
              └─► [回退路径] param.pluginSelected == false
                    └─► 原有 HCCL 执行逻辑（executor->Orchestrate() → KernelRun() → ...）


【销毁阶段】（通信域销毁时执行）

HcclCommDestroy()
  └─► HcclPluginManager::Destroy()
        └─► plugin_->Destroy(ctx)  → dlclose所有算法.so句柄，释放注册表
        └─► dlclose(Plugin动态库)
```

### 4.2 接口设计

#### 4.2.1 HcclPluginManager

　　`HcclPluginManager`以单例模式实现，负责Plugin动态库的完整生命周期管理并保存Plugin动态库句柄。`Init()`通过`call_once`保证全局只执行一次，内部读取环境变量`HCCL_PLUGIN_PATH`，若未配置则跳过，不影响原有流程；若已配置则通过`dlopen`加载Plugin动态库，`dlsym`获取`GetHcclPlugin()`函数表，并调用Plugin的初始化函数。`Destroy()`在通信域销毁时调用，负责释放Plugin上下文和动态库句柄，并重置`once_flag`以支持下次重新初始化。

```cpp
/**
 * 使用方式：
 *   1. HCCL通信操作初始化里调用HcclPluginManager::Instance().Init()
 *   2. op_common.cc Selector()里调用pluginMgr.GetPlugin()->SelectAlg(...)
 *   3. op_common.cc HcclExecOp()里调用pluginMgr.GetPlugin()->ExecuteAlg(...)
 *   4. HCCL销毁通信域里调用HcclPluginManager::Instance().Destroy()
 */
class HcclPluginManager {
public:
    static HcclPluginManager& Instance()
    {
        static HcclPluginManager instance;
        return instance;
    }

    /** Init: 在 HCCL 算子初始化阶段调用。内部通过 call_once 保证全局只执行一次，多次调用安全。*/
    HcclResult Init()
    {
        std::call_once(initFlag_, [this]() { DoInit(); });
        return initResult_;
    }

    /** 是否已成功加载 Plugin */
    bool IsLoaded() const { return plugin_ != nullptr; }

    /** 获取 Plugin 函数表指针（仅 IsLoaded() == true 时调用）*/
    HcclPlugin_t* GetPlugin() { return plugin_; }

    /** 获取 Plugin 上下文（Init 时由 Plugin 分配）*/
    void* GetContext() { return pluginCtx_; }

    /** 销毁 Plugin */
    void Destroy()
    {
        std::lock_guard<std::mutex> lock(destroyMutex_);
        if (plugin_ && plugin_->Destroy) {
            plugin_->Destroy(pluginCtx_);
            pluginCtx_ = nullptr;
        }
        if (libHandle_) {
            dlclose(libHandle_);
            libHandle_ = nullptr;
        }
        plugin_ = nullptr;

        // 重置 once_flag，允许下次重新初始化
        std::destroy_at(&initFlag_);
        new (&initFlag_) std::once_flag;
        initResult_ = HCCL_SUCCESS;
    }

    ~HcclPluginManager() { Destroy(); }

private:
    HcclPluginManager() = default;
    HcclPluginManager(const HcclPluginManager&) = delete;
    HcclPluginManager& operator=(const HcclPluginManager&) = delete;

    void DoInit()
    {
        const char* pluginPath = getenv("HCCL_PLUGIN_PATH");
        if (!pluginPath || pluginPath[0] == '\0') {
            // 未配置，静默跳过，不影响原有流程
            HCCL_INFO("[HcclPluginManager] HCCL_PLUGIN_PATH not set, skip plugin loading");
            return;
        }

        libHandle_ = dlopen(pluginPath, RTLD_LAZY | RTLD_LOCAL);
        if (!libHandle_) {
            HCCL_WARNING("[HcclPluginManager] dlopen plugin [%s] failed: %s",
                pluginPath, dlerror());
            return;
        }
        

        using GetHcclPluginFn = HcclPlugin_t*(*)();
        auto getter = reinterpret_cast<GetHcclPluginFn>(
            dlsym(libHandle_, "GetHcclPlugin"));
        if (!getter) {
            HCCL_WARNING("[HcclPluginManager] symbol GetHcclPlugin not found in [%s]",
                pluginPath);
            dlclose(libHandle_); 
            libHandle_ = nullptr; 
            return;
        }

        plugin_ = getter();
        if (!plugin_ || !plugin_->Init) {
            HCCL_WARNING("[HcclPluginManager] Plugin or Init function is null");
            plugin_ = nullptr;
            dlclose(libHandle_); 
            libHandle_ = nullptr; 
            return;
        }

        const char* algDir = getenv("HCCL_PLUGIN_ALG_DIR");
        int ret = plugin_->Init(algDir ? algDir : "", &pluginCtx_);
        if (ret != 0) {
            HCCL_WARNING("[HcclPluginManager] Plugin->Init failed ret[%d], "
                "fallback to original selector", ret);
            plugin_ = nullptr;
            pluginCtx_ = nullptr;
            dlclose(libHandle_);
            libHandle_ = nullptr;
            return;
        }

        HCCL_INFO("[HcclPluginManager] Plugin [%s] loaded and initialized. "
            "algDir[%s]",
            plugin_->name ? plugin_->name : "?",
            algDir ? algDir : "(empty)");
    }

    std::once_flag    initFlag_;
    HcclResult        initResult_ = HCCL_SUCCESS;
    void*             libHandle_  = nullptr;
    HcclPlugin_t*     plugin_     = nullptr;
    void*             pluginCtx_  = nullptr;
    std::mutex        destroyMutex_;
};
```

#### 4.2.2 Plugin动态库接口（`HcclPlugin_t`）

　　Plugin 对外暴露以下接口（即 `HcclPlugin_t` 函数表中的成员）：
- `Init()`：根据算子根目录路径，扫描各算子的策略动态库，构建算法注册表。
- `Destroy()`：释放注册表，关闭所有已加载的算法动态库句柄。
- `RegisterAlg()`：解析策略JSON字符串，将算法条目写入Plugin注册表。
- `SelectAlg()`：根据当前通信参数和拓扑信息，遍历注册表进行算法匹配并返回算法名。
- `ExecuteAlg()`：根据算法名定位注册条目，懒加载算法动态库并调用`HcclPluginAlgExecute`执行自定义算法。
- `QueryAlgs()`：查询已注册的算法列表，返回包含算法名、优先级、rules详情等信息的JSON字符串。

　　此外`ParseOpStrategy()`负责将策略JSON字符串解析为内部注册表数据结构，为Plugin内部静态函数，不对外暴露，由`RegisterAlg()`在注册时调用。

```cpp
// 算法选择参数（复用HCCL的OpParam，扩展Plugin相关字段）
struct OpParam {
    void*           hcclComm;
    aclrtStream     stream;
    void*           inputPtr        = nullptr;
    OpExecuteConfig opExecuteConfig;
    u32             root            = INVALID_VALUE_RANKID;
    HcclCMDType     opType          = HcclCMDType::HCCL_CMD_INVALID;
    HcclReduceOp    reduceType      = HcclReduceOp::HCCL_REDUCE_RESERVED;
    // ...原有字段...

    bool pluginSelected     = false;   // 标记本次算法选择是否由Plugin命中
    char pluginAlgName[256] = {0};     // Plugin命中时选出的算法名
};

struct HcclPlugin_t {
    // Plugin初始化：从策略动态库加载策略内容，构建算法注册表；algDir为算子根目录
    int (*Init)(const char* algDir, void** context);

    // 销毁Plugin
    int (*Destroy)(void* ctx);

    // 注册算法：将算法实现.so与策略内容绑定，注册到Plugin算法注册表
    int (*RegisterAlg)(void* context,
                       HcclCMDType opType,
                       const char* algDir,
                       const char* strategyJson); // 策略JSON字符串

    // 算法选择：返回true表示命中，algName填入选中算法名；返回false表示未命中，HCCL走原有逻辑
    bool (*SelectAlg)(void* context,
                      const OpParam* param,
                      const TopoInfoWithNetLayerDetails* topoInfo,
                      char* algName,
                      size_t algNameLen);

    // 执行选中的算法：调用预编译.so中的函数
    int (*ExecuteAlg)(void* ctx,
                      const char* algName,
                      const OpParam* param,
                      const TopoInfoWithNetLayerDetails* topoInfo,
                      void* resources); // HcclComm 指针
    
    // 算法查询：查询已注册的算法列表，返回 JSON 字符串。
    const char* (*QueryAlgs)(void* ctx, int opType);
};

// Plugin .so必须导出此符号
extern "C" HcclPlugin_t* GetHcclPlugin();

```

#### 4.2.3 自定义算法动态库导出接口
　　
　　自定义算法`.so`须向Plugin暴露以下导出接口，Plugin在算法选择和算法执行阶段分别通过`dlsym`进行加载：

```cpp
// 算法执行入口，Plugin的ExecuteAlg()通过dlsym找到并调用
typedef int (*HcclPluginAlgExecuteFn)(const OpParam* param,
                                      void* commResources,
                                      const TopoInfoWithNetLayerDetails* topoInfo);

// 打分函数接口，Plugin的SelectAlg()通过dlsym加载；不导出时退回静态rules匹配
// 返回值 >= 0 表示优先级分数，返回 -1 表示不适用当前场景，函数须是确定性的
// 参数EvalContext为算法选择的精简上下文，定义于4.3.2节
typedef int (*HcclPluginScoreFn)(const EvalContext* ctx);
```

---

### 4.3 数据结构

#### 4.3.1 Plugin内部算法注册表

```cpp
// 单条策略规则（字段之间 AND 关系）
struct AlgRule {
    double dataSizeMin     = -1, dataSizeMax     = -1; // -1 表示不限
    double userRankSizeMin = -1, userRankSizeMax = -1;
    double moduleNumMin    = -1, moduleNumMax    = -1;
    double serverNumMin    = -1, serverNumMax    = -1;
    std::vector<std::string> level0TopoValues;         // 空集合表示不限
    std::string comment;
};

// 一个算法的完整注册项
struct AlgEntry {
    std::string            algName;                // 算法名称
    std::string            algLibPath;             // 预编译算法.so路径
    void*                  algLibHandle = nullptr; // dlopen句柄，懒加载
    HcclPluginAlgExecuteFn executeFn    = nullptr; // dlsym得到的函数指针
    int                    priority;               // 静态优先级；无打分函数时作为匹配分数，越大越优先
    std::vector<AlgRule>   rules;                  // OR关系：任一命中即可
    int                    registrationOrder = 0;  // 注册时写入，用于保序
    HcclPluginScoreFn      scoreFn = nullptr;      // 动态打分函数
};

// 算子级别的注册表（一个算子一个）
struct OpRegistry {
    std::string           opType;
    std::vector<AlgEntry> algorithms; // 按priority降序排列
};

// Plugin上下文
struct PluginContext {
    std::map<int, OpRegistry>  registry; // key: opType
    std::map<int, std::string> opNames;
};
```

#### 4.3.2 算法选择上下文

　　`EvalContext`是用于算法选择的精简上下文，该结构体记录了从通信操作参数和拓扑信息中提取的消息大小、Rank数、拓扑类型等字段，作为算法选择阶段打分函数和规则匹配的统一输入：

```cpp
struct EvalContext {
    double      dataSize;
    double      userRankSize;
    double      moduleNum;
    double      serverNum;
    std::string level0Topo;
    std::string opType;
};
```

#### 4.3.3 算法目录结构

　　`HCCL_PLUGIN_ALG_DIR` 是用户通过环境变量指定的算法根目录，Plugin初始化时以此为入口扫描所有算子子目录。每个算子子目录下包含两类文件：一是该算子的策略动态库（`libhccl_plugin_{op}_strategy.so`），负责描述各自定义算法的选择条件；二是各自定义算法的实现动态库，每个算法独占一个子目录。目录结构简单描述如下：

```
HCCL_PLUGIN_ALG_DIR/
├── AllReduce/
│   ├── libhccl_plugin_allreduce_strategy.so      ← 策略动态库（策略内容编译进.so）
│   ├── MyRingAlg/
│   │   └── libhccl_plugin_my_ring_alg.so
│   └── MyTreeAlg/
│       └── libhccl_plugin_my_tree_alg.so
├── AllGather/
│   ├── libhccl_plugin_allgather_strategy.so
│   └── MyMeshAlg/
│       └── libhccl_plugin_my_mesh_alg.so
```
　　每个算子目录下提供一个独立的策略动态库`libhccl_plugin_{op}_strategy.so`，将策略内容以硬编码字符串常量的形式编译进`.so`，并导出`GetHcclPluginStrategy()`函数供 Plugin调用，如下所示：

```cpp
// 编译为 libhccl_plugin_allreduce_strategy.so
extern "C" const char* GetHcclPluginStrategy() {
    return R"({
        "opType": "AllReduce",
        "version": "1.0",
        "algorithms": [
            {
                "algName": "MyRingAlg",
                "priority": 100,
                "rules": [
                    {
                        "comment": "小消息 MESH 拓扑",
                        "dataSize":     { "min": 0,     "max": 65536 },
                        "userRankSize": { "min": 2,     "max": 8     },
                        "level0Topo":   { "values": ["MESH", "DOUBLE_RING"] },
                        "deviceType":   { "values": ["910B", "910_93"] }
                    }
                ]
            },
            {
                "algName": "MyTreeAlg",
                "priority": 50,
                "rules": [
                    {
                        "comment": "通用大消息",
                        "dataSize":     { "min": 65537, "max": 999999999 },
                        "userRankSize": { "min": 4,     "max": 16 }
                    }
                ]
            }
        ]
    })";
}
```

　　策略规则语义：`rules`之间为OR关系，`rule`内部字段为AND关系；`priority`决定多算法同时命中时的优先顺序（值越大越优先）；某字段缺失表示"不限"。

### 4.4 关键逻辑

#### 4.4.1 Plugin初始化调用点与Plugin初始化实现逻辑

　　**(1) Plugin初始化调用点**

　　在集合通信操作进行算法选择和执行前需要执行`InitEnvConfig()`函数，该函数负责解析环境变量（如算子展开模式、调试配置等），为了支持Plugin的运行，拟在`InitEnvConfig()`函数中调用Plugin初始化代码：

```cpp
// src/common/alg_env_config.cc
HcclResult InitEnvConfig() {
    // 原有代码...
    HcclResult pluginRet = HcclPluginManager::Instance().Init(); // 全局只执行一次
}
```
　　**(2) Plugin初始化实现逻辑**

　　`PluginInit()`遍历`HCCL_PLUGIN_ALG_DIR`下的每个算子子目录，通过`dlopen`+`dlsym`读取`libhccl_plugin_{op}_strategy.so`中的策略内容，自动调用 `PluginRegisterAlg()`完成各算子的算法注册表构建：

```cpp
static int PluginInit(const char* algDir, void** ctx)
{
    if (!ctx) return -1;
    auto* context = new PluginContext();
    *ctx = context;
    if (!algDir || algDir[0] == '\0' || !fs::exists(algDir)) return 0;

    // 遍历算子目录（AllReduce/、AllGather/ 等）
    for (const auto& opDirEntry : fs::directory_iterator(algDir)) {
        if (!opDirEntry.is_directory()) continue;

        std::string opDirName = opDirEntry.path().filename().string();
        int opType = static_cast<int>(OpNameToType(opDirName));
        if (opType < 0) continue;

        // 从策略动态库中读取策略内容
        std::string opDirNameLower = ToLower(opDirName); // 如 "AllReduce" → "allreduce"
        std::string strategyLibPath = opDirEntry.path().string() + "/libhccl_plugin_" + opDirNameLower + "_strategy.so";

        if (!fs::exists(strategyLibPath)) continue;

        void* strategyHandle = dlopen(strategyLibPath.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (!strategyHandle) continue;

        using GetStrategyFn = const char*(*)();
        auto getStrategy = reinterpret_cast<GetStrategyFn>(
            dlsym(strategyHandle, "GetHcclPluginStrategy"));
        if (!getStrategy) { dlclose(strategyHandle); continue; }

        const char* strategyJson = getStrategy();

        int ret = PluginRegisterAlg(context, opType,
                                     opDirEntry.path().string().c_str(),
                                     strategyJson);
        if (ret != 0) {
            PLUGIN_WARNING("PluginInit: opType[%d] register failed, skip", opType);
        }

        dlclose(strategyHandle);
    }

    PLUGIN_INFO("PluginInit done. algDir[%s]", algDir);
    return 0;
}
```

#### 4.4.2 Plugin销毁调用点与Plugin销毁实现逻辑

　　**(1) Plugin销毁调用点**

　　在通信域销毁的函数`HcclCommDestroy()`中进行Plugin的销毁。

```cpp
// 通信域销毁
HcclResult HcclCommDestroy(HcclComm comm) {
    // 原有代码...
    HcclPluginManager::Instance().Destroy();
}
```

　　**(2) Plugin销毁实现逻辑**

　　`PluginDestroy()` 遍历注册表中所有算子的算法条目，对已通过懒加载打开的算法动态库逐一调用`dlclose()`释放句柄，最后释放`PluginContext`对象，完成Plugin的完整清理。

```cpp
static int PluginDestroy(void* ctx)
{
    if (!ctx) return 0;
    auto* context = static_cast<PluginContext*>(ctx);
    for (auto& [opType, opReg] : context->registry) {
        for (auto& entry : opReg.algorithms) {
            if (entry.algLibHandle) {
                dlclose(entry.algLibHandle);
                entry.algLibHandle = nullptr;
            }
        }
    }
    delete context;
    PLUGIN_INFO("PluginDestroy done.");
    return 0;
}
```

#### 4.4.3 Plugin算法注册调用点与Plugin算法注册实现逻辑

　　**(1) Plugin算法注册调用点**
　　

　　`PluginRegisterAlg()`由`PluginInit()`在初始化阶段自动调用——每扫描到一个算子子目录，调用一次`PluginRegisterAlg()`完成该算子的算法注册，无需用户手动触发。

　　**(2) Plugin算法注册实现逻辑**

　　`PluginRegisterAlg()`接收算子类型、算子目录路径和策略JSON字符串，调用`ParseOpStrategy()`将策略内容解析为内部`OpRegistry`结构（包含各算法的rules列表和优先级），并将结果写入`PluginContext`注册表，完成该算子下所有自定义算法的统一注册。

```cpp
static int PluginRegisterAlg(void* ctx, int opType,
                               const char* algDir, const char* strategyJson)
{
    if (!ctx || !algDir || !strategyJson) return -1;
    auto* context = static_cast<PluginContext*>(ctx);

    OpRegistry opReg;
    if (!ParseOpStrategy(strategyJson, algDir, opType, opReg)) {
        PLUGIN_WARNING("PluginRegisterAlg: failed to parse strategy");
        return -1;
    }

    context->registry[opType] = std::move(opReg);
    context->opNames[opType]  = OpTypeToName(static_cast<HcclCMDType>(opType));

    PLUGIN_INFO("PluginRegisterAlg: opType[%d/%s] registered %zu algorithm(s)",
        opType,
        OpTypeToName(static_cast<HcclCMDType>(opType)).c_str(),
        context->registry[opType].algorithms.size());
    return 0;
}
```

　**(3) ParseOpStrategy()实现逻辑**

　　`ParseOpStrategy()`对策略JSON字符串进行解析，逐条提取算法名、优先级和rules列表，并按约定填充算法动态库路径。解析完成后先按解析顺序为每条算法记录原始注册序号`registrationOrder`，再以`stable_sort`按`priority`降序排序写入`OpRegistry`——`priority`相同时保持原始注册顺序。

```cpp
static bool ParseOpStrategy(const std::string& strategyJson,
                              const std::string& algDir,
                              int opType, OpRegistry& opReg)
{
    nlohmann::json j = nlohmann::json::parse(strategyJson, nullptr, false);
    if (j.is_discarded()) return false;

    opReg.opType = j.value("opType", "");

    for (const auto& algJson : j.value("algorithms", nlohmann::json::array())) {
        AlgEntry entry;
        entry.algName  = algJson.value("algName", "");
        entry.priority = algJson.value("priority", 0);
        if (entry.algName.empty()) continue;

        // 约定：算法 .so 位于 algDir/<AlgName>/lib<AlgName>.so
        std::string algSubDir = algDir + "/" + entry.algName;
        entry.algLibPath = algSubDir + "/lib" + entry.algName + ".so";
        if (algJson.contains("soPath")) {
            entry.algLibPath = algJson["soPath"].get<std::string>();
        }

        for (const auto& ruleJson : algJson.value("rules", nlohmann::json::array())) {
            entry.rules.push_back(ParseRule(ruleJson));
        }
        opReg.algorithms.push_back(std::move(entry));
    }

    // 先记录原始注册顺序
    for (int i = 0; i < static_cast<int>(opReg.algorithms.size()); ++i) {
        opReg.algorithms[i].registrationOrder = i;
    }
    // 按priority降序排列，priority相同时保持原始注册顺序
    std::stable_sort(opReg.algorithms.begin(), opReg.algorithms.end(),
        [](const AlgEntry& a, const AlgEntry& b) {
            return a.priority > b.priority;
        });
    return true;
}
```

　　`ParseOpStrategy()`由三个辅助函数协作完成：
- **`ParseRangeField()`函数**：负责提取JSON中的数值范围字段（`min`/`max`）,字段缺失时保持默认值`-1`（表示不限）；
- **`ParseEnumField()`函数**：负责提取枚举字段的候选值列表，列表为空时表示不限；
- **`ParseRule()`函数**：统一调用`ParseRangeField()`和`ParseEnumField()`，将一条JSON规则对象完整解析为`AlgRule`结构体。

```cpp
static void ParseRangeField(const nlohmann::json& ruleJson,
                             const std::string& field,
                             double& minVal, double& maxVal)
{
    if (ruleJson.contains(field)) {
        const auto& f = ruleJson[field];
        if (f.contains("min")) minVal = f["min"].get<double>();
        if (f.contains("max")) maxVal = f["max"].get<double>();
    }
}

static void ParseEnumField(const nlohmann::json& ruleJson,
                            const std::string& field,
                            std::vector<std::string>& values)
{
    if (ruleJson.contains(field) && ruleJson[field].contains("values")) {
        for (const auto& v : ruleJson[field]["values"]) {
            values.push_back(v.get<std::string>());
        }
    }
}

static AlgRule ParseRule(const nlohmann::json& ruleJson)
{
    AlgRule rule;
    ParseRangeField(ruleJson, "dataSize",     rule.dataSizeMin,     rule.dataSizeMax);
    ParseRangeField(ruleJson, "userRankSize", rule.userRankSizeMin, rule.userRankSizeMax);
    ParseRangeField(ruleJson, "moduleNum",    rule.moduleNumMin,    rule.moduleNumMax);
    ParseRangeField(ruleJson, "serverNum",    rule.serverNumMin,    rule.serverNumMax);
    ParseEnumField (ruleJson, "level0Topo",   rule.level0TopoValues);
    rule.comment = ruleJson.value("comment", "");
    return rule;
}
```

#### 4.4.4 Plugin算法选择调用点与Plugin算法选择实现逻辑

　　**(1) Plugin算法选择调用点**

　　为减少对HCCL代码仓的修改，设计在`op_common.cc`的`Selector() `函数中添加Plugin算法选择的调用，即`SelectAlg()`。首先进行Plugin的算法选择，若无法找到匹配，则回退到原HCCL代码仓的算法选择逻辑：

```cpp
HcclResult Selector(HcclComm comm, OpParam &param,
                    std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
                    std::string &algName)
{
    // ......原有逻辑不变

    // ===================== 新增：Plugin 算法选择 =====================
    auto& pluginMgr = HcclPluginManager::Instance();
    if (pluginMgr.IsLoaded()) {
        char selectedAlg[PLUGIN_ALG_NAME_MAX_LEN] = {0};
        bool hit = pluginMgr.GetPlugin()->SelectAlg(
            pluginMgr.GetContext(),
            &param,
            topoInfo.get(), 
            selectedAlg,
            sizeof(selectedAlg));

        if (hit) {
            HCCL_INFO("[Selector] Plugin hit algorithm: [%s]", selectedAlg);
            param.pluginSelected = true;
            strncpy_s(param.pluginAlgName, sizeof(param.pluginAlgName),
                      selectedAlg, sizeof(selectedAlg) - 1);
            algName = std::string(selectedAlg);
            HCCL_INFO("[Selector] Plugin selected, skip original selector.");
            return HCCL_SUCCESS;
        }
        HCCL_INFO("[Selector] Plugin not hit, fallback to original selector.");
    }

    // ......原有算法选择逻辑
}
```

　　**(2) Plugin算法选择实现逻辑**

　　`PluginSelectAlg()`从注册表中找到当前算子对应的算法集合，通过调用`BuildEvalContext()`从操作参数和拓扑信息中提取消息大小、Rank数、拓扑类型等匹配字段，构建评估上下文`EvalContext`。遍历所有算法条目，优先调用算法的打分函数`HcclPluginAlgScore`获取动态分数；若算法未导出打分函数，则退回静态rules匹配，命中时以该算法的`priority`字段作为分数，未命中则跳过。遍历结束后选择分数最高的算法写入`algName`；若多个算法分数相同，则按注册顺序选择最先注册的算法；若所有算法均未命中则返回`false`，由HCCL回退至原有选择逻辑。

```cpp

// 懒加载算法.so的打分函数
static void TryLoadScoreFn(AlgEntry& entry)
{
    if (entry.scoreFn != nullptr) return;             // 已加载
    if (entry.algLibHandle == nullptr) {
        entry.algLibHandle = dlopen(entry.algLibPath.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (!entry.algLibHandle) return;
    }
    entry.scoreFn = reinterpret_cast<HcclPluginScoreFn>(
        dlsym(entry.algLibHandle, "HcclPluginAlgScore"));
}

static bool PluginSelectAlg(void* context, const OpParam* param,
                              const TopoInfoWithNetLayerDetails* topoInfo,
                              char* algName, size_t algNameLen)
{
    if (!context || !param || !algName || algNameLen == 0 || !topoInfo) return false;

    auto* ctx = static_cast<PluginContext*>(context);
    auto it = ctx->registry.find(static_cast<int>(param->opType));
    if (it == ctx->registry.end()) return false;

    EvalContext evalCtx = BuildEvalContext(param, topoInfo);

    int    bestScore = -1;
    int    bestOrder = INT_MAX;
    const AlgEntry* bestEntry = nullptr;

    for (AlgEntry& entry : it->second.algorithms) {
        int score = -1;

        if (entry.scoreFn != nullptr || !entry.algLibPath.empty()) {
            // 尝试懒加载打分函数
            TryLoadScoreFn(entry);
        }

        if (entry.scoreFn != nullptr) {
            // ① 有打分函数：调用它获取分数
            score = entry.scoreFn(&evalCtx);
            // ④ 返回 -1 表示不适用，直接跳过
            if (score < 0) continue;
        } else {
            // 无打分函数：退回静态rules匹配
            if (!EvaluateAlg(entry, evalCtx)) continue;
            // 静态命中时，以priority字段作为分数
            score = entry.priority;
        }

        // ② ③ 选分数最高的；同分时按注册顺序（registrationOrder 越小越优先）
        if (score > bestScore || (score == bestScore && entry.registrationOrder < bestOrder)) {
            bestScore = score;
            bestOrder = entry.registrationOrder;
            bestEntry = &entry;
        }
    }

    if (!bestEntry) {
        PLUGIN_DEBUG("SelectAlg: opType[%d/%s] no match, fallback",
            static_cast<int>(param->opType), OpTypeToName(param->opType).c_str());
        return false;
    }

    snprintf(algName, algNameLen, "%s", bestEntry->algName.c_str());
    PLUGIN_INFO("SelectAlg: opType[%d/%s] hit alg[%s] score[%d] order[%d] "
        "dataSize[%.0f] userRankSize[%.0f] topo[%s]",
        static_cast<int>(param->opType), OpTypeToName(param->opType).c_str(),
        bestEntry->algName.c_str(), bestScore, bestOrder,
        evalCtx.dataSize, evalCtx.userRankSize, evalCtx.level0Topo.c_str());
    return true;
}

```
　　**(3) Plugin算法选择辅助函数实现逻辑**

　　算法选择辅助函数包括：
- **BuildEvalContext()函数**：将原始通信参数和拓扑信息转换为`EvalContext`结构体。
- **EvaluateRule()函数**：对单条规则进行AND匹配，数值字段通过范围检查（`-1`表示不限），枚举字段通过集合成员检查（空集合表示不限）；
- **EvaluateAlg()函数**：对一个算法条目的所有规则执行OR匹配，任一规则命中即返回 `true`。

```cpp
static EvalContext BuildEvalContext(const OpParam* param,
                                     const TopoInfoWithNetLayerDetails* topoInfo)
{
    EvalContext ctx;
    ctx.dataSize     = static_cast<double>(CalculateDataSize(*param));
    ctx.userRankSize = static_cast<double>(topoInfo->userRankSize);
    ctx.moduleNum    = static_cast<double>(topoInfo->moduleNum);
    ctx.serverNum    = static_cast<double>(topoInfo->serverNum);
    ctx.level0Topo   = TopoTypeToStr(topoInfo->level0Topo);
    ctx.opType       = OpTypeToName(param->opType);
    return ctx;
}

static bool EvaluateRule(const AlgRule& rule, const EvalContext& ctx)
{
    auto checkRange = [](double val, double lo, double hi) -> bool {
        if (lo >= 0 && val < lo) return false;
        if (hi >= 0 && val > hi) return false;
        return true;
    };
    if (!checkRange(ctx.dataSize,     rule.dataSizeMin,     rule.dataSizeMax))     return false;
    if (!checkRange(ctx.userRankSize, rule.userRankSizeMin, rule.userRankSizeMax)) return false;
    if (!checkRange(ctx.moduleNum,    rule.moduleNumMin,    rule.moduleNumMax))     return false;
    if (!checkRange(ctx.serverNum,    rule.serverNumMin,    rule.serverNumMax))     return false;

    auto checkEnum = [](const std::string& val,
                         const std::vector<std::string>& allowed) -> bool {
        if (allowed.empty()) return true;
        return std::find(allowed.begin(), allowed.end(), val) != allowed.end();
    };
    if (!checkEnum(ctx.level0Topo, rule.level0TopoValues)) return false;
    return true;
}

static bool EvaluateAlg(const AlgEntry& entry, const EvalContext& ctx)
{
    for (const auto& rule : entry.rules) {
        if (EvaluateRule(rule, ctx)) return true;
    }
    return false;
}
```

#### 4.4.5 Plugin算法执行调用点与Plugin算法执行实现逻辑

　　**(1) Plugin算法执行调用点**

　　在`op_common.cc`的`HcclExecOp()`函数中添加Plugin算法执行调用，当`param.pluginSelected == true`，即Plugin算法选中时走Plugin算法执行路径：

```cpp
HcclResult HcclExecOp(HcclComm comm, OpParam &param,
                       std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
                       std::string &algName, const ResPackGraphMode &resPack)
{
    // ......原有逻辑不变

    // ===================== 新增：Plugin 算法执行 =====================
    if (param.pluginSelected) {
        HCCL_INFO("[HcclExecOp] Plugin algorithm [%s] executing.",
            param.pluginAlgName);
        auto& pluginMgr = HcclPluginManager::Instance();
        if (!pluginMgr.IsLoaded()){
            HCCL_ERROR("[HcclExecOp] Plugin was selected but is no longer loaded.");
            return HCCL_E_INTERNAL;
        } 

        int pluginRet = pluginMgr.GetPlugin()->ExecuteAlg(
            pluginMgr.GetContext(),
            param.pluginAlgName,
            &param,
            topoInfo.get(),
            static_cast<void*>(comm));

        if (pluginRet != 0) {
            HCCL_ERROR("[HcclExecOp] Plugin ExecuteAlg failed, algName[%s], ret[%d]",
                param.pluginAlgName, pluginRet);
            return HCCL_E_INTERNAL;
        }
        
        HCCL_INFO("[HcclExecOp] Plugin ExecuteAlg success.");
        return HCCL_SUCCESS;
    }

    // ......原有执行逻辑
}
```
　　**(2) Plugin算法执行实现逻辑**

　　`PluginExecuteAlg()`接收Plugin选择阶段确定的算法名，从注册表中定位对应的`AlgEntry`条目。算法动态库采用懒加载策略——仅在首次执行该算法时通过`dlopen`加载对应的`.so`并通过`dlsym`解析 `HcclPluginAlgExecute`函数指针，后续调用直接复用已缓存的句柄和函数指针，避免重复加载的开销。最终调用`HcclPluginAlgExecute()`完成实际的通信操作。

```cpp
static int PluginExecuteAlg(void* ctx, const char* algName,
                              const OpParam* param,
                              const TopoInfoWithNetLayerDetails* topoInfo,
                              void* resources)
{
    if (!ctx || !algName || !param || !topoInfo) {
        PLUGIN_ERROR("ExecuteAlg: invalid arguments");
        return -1;
    }

    auto* context = static_cast<PluginContext*>(ctx);
    AlgEntry* entry = FindAlgEntry(context, static_cast<int>(param->opType), algName);
    if (!entry) {
        PLUGIN_ERROR("ExecuteAlg: algName[%s] opType[%d] not found in registry", algName, static_cast<int>(param->opType));
        return -1;
    }

    // 懒加载算法 .so
    if (!entry->algLibHandle) {
        entry->algLibHandle = dlopen(entry->algLibPath.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (!entry->algLibHandle) {
            PLUGIN_ERROR("ExecuteAlg: dlopen [%s] failed: %s", entry->algLibPath.c_str(), dlerror());
            return -1;
        }
        PLUGIN_INFO("ExecuteAlg: dlopen [%s] success", entry->algLibPath.c_str());
    }

    // 懒加载函数指针
    if (!entry->executeFn) {
        entry->executeFn = reinterpret_cast<HcclPluginAlgExecuteFn>(
            dlsym(entry->algLibHandle, "HcclPluginAlgExecute"));
        if (!entry->executeFn) {
            PLUGIN_ERROR("ExecuteAlg: dlsym HcclPluginAlgExecute failed in [%s]: %s", entry->algLibPath.c_str(), dlerror());
            dlclose(entry->algLibHandle);
            entry->algLibHandle = nullptr;
            return -1;
        }
    }
    PLUGIN_INFO("ExecuteAlg: calling HcclPluginAlgExecute, alg[%s]", algName);
    return entry->executeFn(param, resources, topoInfo);
}
```

#### 4.4.6 Plugin算法查询调用点与Plugin算法查询实现逻辑

　　**(1) Plugin算法查询调用点**

　　`PluginQueryAlgs()`可供调试模块等主动调用，用于查看当前Plugin注册表的内容：

```cpp
auto& pluginMgr = HcclPluginManager::Instance();
if (pluginMgr.IsLoaded()) {
    const char* json = pluginMgr.GetPlugin()->QueryAlgs(
        pluginMgr.GetContext(),
        static_cast(HcclCMDType::HCCL_CMD_INVALID)); // 传HCCL_CMD_INVALID查询所有算子
    if (json) HCCL_INFO("[QueryAlgs] registered algorithms:\n%s", json);
}
```

　　**(2) Plugin算法查询实现逻辑**

　　`PluginQueryAlgs()`接收算子类型参数，从注册表中找到对应算子的所有已注册算法条目，逐条将算法名、完整的rules列表等序列化为JSON字符串输出。当opType传入HCCL_CMD_INVALID时，遍历全部算子一并返回其算法条目。查询结果由Plugin内部的静态缓冲区`g_queryAlgsBuffer`持有，生命周期到下次调用算法查询前有效。

```cpp
// Plugin 内部持有上次查询结果
static std::string g_queryAlgsBuffer;

static nlohmann::json SerializeRule(const AlgRule& rule)
{
    nlohmann::json j;
    j["dataSize"]     = { {"min", rule.dataSizeMin},     {"max", rule.dataSizeMax} };
    j["userRankSize"] = { {"min", rule.userRankSizeMin}, {"max", rule.userRankSizeMax} };
    j["moduleNum"]    = { {"min", rule.moduleNumMin},    {"max", rule.moduleNumMax} };
    j["serverNum"]    = { {"min", rule.serverNumMin},    {"max", rule.serverNumMax} };
    j["level0Topo"]   = rule.level0TopoValues;   // 空数组表示不限
    if (!rule.comment.empty()) j["comment"] = rule.comment;
    return j;
}

static const char* PluginQueryAlgs(void* ctx, int opType)
{
    if (!ctx) return nullptr;
    auto* context = static_cast<PluginContext*>(ctx);

    nlohmann::json result = nlohmann::json::object();

    auto writeOp = [&](int type, const OpRegistry& opReg) {
        std::string opName = OpTypeToName(static_cast<HcclCMDType>(type));
        nlohmann::json algList = nlohmann::json::array();
        for (const auto& entry : opReg.algorithms) {
            nlohmann::json algJson;
            algJson["algName"]  = entry.algName;
            algJson["priority"] = entry.priority;
            algJson["hasDynamicScore"] = (entry.scoreFn != nullptr);
            nlohmann::json ruleArray = nlohmann::json::array();
            for (const auto& rule : entry.rules) {
                ruleArray.push_back(SerializeRule(rule));
            }
            algJson["rules"] = std::move(ruleArray);
            algList.push_back(std::move(algJson));
        }
        result[opName] = std::move(algList);
    };

    if (opType == static_cast<int>(HcclCMDType::HCCL_CMD_INVALID)) {
        for (const auto& [type, opReg] : context->registry) {
            writeOp(type, opReg);
        }
    } else {
        auto it = context->registry.find(opType);
        if (it != context->registry.end()) {
            writeOp(opType, it->second);
        }
    }

    g_queryAlgsBuffer = result.dump(2);
    return g_queryAlgsBuffer.c_str();
}
```

#### 4.4.7 自定义算法实现规范

　　自定义算法以独立目录的形式组织，编译后产出算法实现动态库（`lib{AlgName}.so`），并安装到对应的算子目录下。以AICPU算子为例，源码目录结构如下：

```
ALgName/
├── CMakeLists.txt
├── op_host/
│   └── my_alg.cc                  ← Host侧算法编排，须导出HcclPluginAlgExecute
├── op_kernel_aicpu/
│   └── my_kernel.cc               ← Device侧Kernel
│   └── libmy_alg.json
└── inc/
    └── my_alg.h
```

　　`op_host/my_alg.cc` 实现host侧算法编排逻辑，负责任务下发和资源调度；`op_kernel_aicpu/my_kernel.cc`实现device侧Kernel，完成实际的数据通信操作。`op_host/my_alg.cc`须导出标准执行入口：

```cpp
extern "C" int HcclPluginAlgExecute(const OpParam* param,
                                     void* commResources,
                                     const TopoInfoWithNetLayerDetails* topoInfo) {
}
```
　　此外，`op_host/my_alg.cc`可选导出打分函数`HcclPluginAlgScore`，Plugin在算法选择阶段通过`dlsym`加载并调用。若不导出，Plugin退回到静态rules匹配：

```cpp
extern "C" int HcclPluginAlgScore(const EvalContext* ctx) {
}
```

---
#### 4.4.8 自定义算法完整使用流程

　　假设用户希望新增算法`MyRingAlg`用于`AllReduce`，需经历以下完整流程：

　　**（1）编写自定义算法逻辑和策略库**
　　

　　用户需要分别编写算法实现和策略库两部分代码。此处以新增AICPU算法为例，算法实现目录 `MyRingAlg` 包含host侧编排逻辑和device侧Kernel执行逻辑；策略库`allreduce_strategy`独立于算法实现，描述`AllReduce`算子下所有自定义算法的选择条件。

```
AllReduce/
├── build.sh     
├── allreduce_strategy/
│   └── strategy.cc                             # 实现GetHcclPluginStrategy()，编译为libhccl_plugin_allreduce_strategy.so
└──  MyRingAlg/
    ├── CMakeLists.txt
    ├── op_host/
    │   └── my_ring_alg.cc                      # Host侧算法编排，实现HcclPluginAlgExecute()和HcclPluginAlgScore()
    ├── op_kernel_aicpu/ 
    │   ├── libmy_ring_alg_aicpu_kernel.json    # AICPU Kernel算子描述文件
    │   ├── my_ring_alg_kernel.cc               # AICPU Kernel实现逻辑
    │   └── exec_op.cc                          # AICPU算子编排逻辑
    └── inc/
        └── my_ring_alg.h
```
　　如需添加多个算法（如`MyTreeAlg`），在`AllReduce/`下并行创建算法子目录即可，策略库 `strategy.cc`中统一描述所有算法的选择条件。

　　**（2）编译**

　　执行工程AllReduce/目录下的下的 `build.sh` 进行编译：
```bash
bash build.sh --cann-path=${ASCEND_HOME_PATH}
```
　　编译产物位于 `AllReduce/build/` 目录下，主要包含：

| 产物 | 说明 |
|------|------|
| `libhccl_plugin_allreduce_strategy.so` | AllReduce算子策略动态库 |
| `libhccl_plugin_my_ring_alg.so` | MyRingAlg host侧算法实现动态库 |
| `aicpu_hccl_custom_my_ring_alg.tar.gz` | AICPU Kernel算子包（打包后） |

　　**（3）安装**

　　执行工程自带的安装脚本，将编译产物部署到CANN安装目录：

```bash
bash install.sh --install-path=${ASCEND_HOME_PATH}/opp/vendors/cust
```

　　安装后的目录结构如下：

```
${ASCEND_HOME_PATH}/opp/vendors/cust/
└── AllReduce/
    ├── libhccl_plugin_allreduce_strategy.so      ← AllReduce算子策略动态库
    └── MyRingAlg/
        └── libhccl_plugin_my_ring_alg.so          ← MyRingAlg算法实现动态库
        └── aicpu/
            ├── config/
            │   └── libmy_ring_alg_aicpu_kernel.json       ← AICPU Kernel算子描述文件
            └── kernel/
                └── aicpu_hccl_custom_my_ring_alg.tar.gz  ← AICPU Kernel算子包
```

　　**（4）配置环境变量**

　　`libhccl_plugin.so` 随HCCL安装，通过环境变量`HCCL_PLUGIN_PATH`决定是否启用Plugin功能——不设置则HCCL行为与原有完全一致：

```bash
# 启用Plugin
export HCCL_PLUGIN_PATH=${ASCEND_HOME}/opp/vendors/cust/lib64/libhccl_plugin.so
# 自定义算法根目录
export HCCL_PLUGIN_ALG_DIR=${ASCEND_HOME_PATH}/opp/vendors/cust/
```

　　配置完成后，HCCL将自动加载Plugin动态库，并从`libhccl_plugin_allreduce_strategy.so`中读取`AllReduce`算子的策略内容完成注册。此后上层应用的`HcclAllReduce()`调用都将经过Plugin的算法选择和执行流程，若策略命中则执行`MyRingAlg`，否则回退至原有HCCL逻辑。

---

## 5. 兼容性考虑

**向后兼容性**：本方案不修改任何现有HCCL代码逻辑，只在`op_common.cc`的`Selector()`和`HcclExecOp()`函数中新增可选分支。当`HCCL_PLUGIN_PATH`未配置时，`HcclPluginManager::IsLoaded()` 返回false，所有新增分支直接跳过，HCCL行为与原有完全一致。

**特性开关**：通过环境变量`HCCL_PLUGIN_PATH`控制——不设置即为禁用，无需代码改动。

**接口向前兼容**：`HcclPlugin_t`以函数指针结构体暴露接口，后续扩展只需在结构体末尾追加新函数指针，老版本Plugin自动忽略。

## 6. 测试场景

　　**(1) 单元测试**： 
- `HcclPluginManager` 初始化、销毁的幂等性测试（多次Init/Destroy）
- Plugin加载失败（文件不存在、符号解析失败、`GetHcclPluginStrategy`不存在）时的降级行为测试
- `ParseOpStrategy()` 对合法/非法JSON字符串的解析正确性测试;
- `PluginSelectAlg()` 的打分函数路径的正确性测试（返回-1跳过、同分按注册顺序）和静态规则匹配正确性测试（覆盖AND/OR边界条件、priority排序）
- `PluginExecuteAlg()` 懒加载路径的正确性测试
- `PluginQueryAlgs()` 的输出正确性测试（覆盖单算子查询、全量查询、rules字段完整性）

　　**(2)集成测试**：
- 正常场景：配置Plugin后`Hccl{op}()`能选中并执行自定义算法
- 回退场景：Plugin未命中时回退到原有HCCL算法，结果一致
- 禁用场景：不设置`HCCL_PLUGIN_PATH`时，HCCL行为与原有完全相同

　　**(3)端到端验证**：
- 编译示例自定义算法（如 MyRingAlg），按完整流程安装并执行，验证通信结果正确性
- 多个自定义算法并存时的优先级选择验证

---