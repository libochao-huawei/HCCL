# HCCL 模拟运行器数据模型设计

## 软硬件资源交互关系建模
数据流交互示意图
```mermaid
sequenceDiagram
    participant HostThread as Host CPU (Runner)
    participant StreamQueue as Stream (Memory Queue)
    participant DeviceScheduler as Device Hardware (TS)
    participant EventMem as Event Status (Memory)

    Note over HostThread: 1. aclrtRecordEvent(evt1)
    HostThread->>StreamQueue: Push CMD: [Write Event1=Done]
    
    Note over HostThread: 2. aclrtStreamWaitEvent(evt1)
    HostThread->>StreamQueue: Push CMD: [Wait Event1==Done]
    
    Note over HostThread: 3. aclrtLaunchKernel(MatMul)
    HostThread->>StreamQueue: Push CMD: [Execute MatMul]
    
    
    Note over DeviceScheduler: 异步执行阶段 (Device侧)
    
    StreamQueue->>DeviceScheduler: Pop CMD: [Write Event1]
    DeviceScheduler->>EventMem: Update Status to DONE
    
    StreamQueue->>DeviceScheduler: Pop CMD: [Wait Event1]
    DeviceScheduler->>EventMem: Check Status?
    Note right of DeviceScheduler: 发现是 DONE，通过！<br/>(如果是 NotReady，硬件会在这里空转等待)
    
    StreamQueue->>DeviceScheduler: Pop CMD: [MatMul]
    DeviceScheduler->>DeviceScheduler: Start AI Core Computing...
```
只要是塞进 Stream 里的东西，都是由 Device 硬件执行的。

## Device、Context、Stream 等硬件资源关系建模

Device、Context、Stream 与用户主机线程之间的关系
```mermaid
graph TD
    subgraph Server1[Server 1]
        Host1
        Device
        Device2
    end

    subgraph Server2[Server 2]
        Host3
        Host2
        Device3
        Device4
        
    end

    subgraph Host1[Host1]
        Runner1[runner<br>用户线程1]
        Runner2[runner<br>用户线程2]
    end

    subgraph Host2[Host2]
        Runner3[Runner...]
    end

    subgraph Host3[Device CPU<br>边缘计算/嵌入式: atlas 500]
        embedRunner[Embed Runner...]
    end

    subgraph Device[Device 1]
        Context1
    end

    subgraph Context1[run-Context1]
        Stream1
    end

    subgraph Stream1[ctx-Stream1]
        TaskKernel
    end

    subgraph TaskKernel[Task/Kernel]

    end


    subgraph Device2[Device 2]
        Context2
    end

    subgraph Context2[run-Context2]
        Stream2
    end

    subgraph Stream2[ctx-Stream2]
        Kernel
    end

    subgraph Kernel[Kernel]
    end

    subgraph Device3[Device 3]
        ctxM[...]
    end

    subgraph Device4[Device 4]
        ctxN[...]
    end

    Runner1-->Device
    Runner2-->Device
    Runner2-->Device2
    Runner3-->Device3
    Runner3-.->|???sdid|Device
    embedRunner-->Device4
 

```

### 基础设备关系建模


```mermaid
erDiagram

    Server {
        typ server-id PK
        typ pod-id
        typ version
    }

    Host {
        typ host-id PK
        typ server-id FK
        typ ip
        typ arch
    }

    Runner {
        typ run-id PK
        typ host-id FK
        typ pid
        typ timeout-config-ms
        typ current-ctx-id FK
    }

    Device {
        typ device-id PK
        typ server-id FK
        typ logic-id "当前可用设备的序号"
        typ physical-id
        typ ccu-die-num "910D目前双die"
        typ overflow-mode
        typ status
        typ soc-version "A3"
        typ max-stream-cnt "1984"
    }

    Port {
        typ port-id PK
        typ device-id FK
        typ ccu-id    FK
        typ eid "IP地址"
        typ func-id "框内固定填2，框外固定填3"
    }

    Ccu {
        typ ccu-id PK
        typ device-id FK
        typ status
    }

    EndPointPair {
        typ endpoint-pair-id PK
        typ src-port-id FK
        typ dst-port-id FK
    }

    DeviceConnection {
        typ connection-id PK
        typ src-dev-id FK
        typ dst-dev-id FK
        typ link-type
        typ access-by-remote
    }

    CcuChannel {
        typ channel-id PK
        typ endpoint-pair-id FK
    }
   
    Context {
        typ ctx-id PK
        typ run-id FK
        typ thread-id 
        typ device-id FK
        typ is-default
        typ float-overflow-addr
        typ capture-mode
    }

    Stream {
        typ stream-id PK
        typ ctx-id FK
        typ sq-base-addr
        typ is-primary-default
        typ is-other-default
        typ priority
        typ schedule-strategy
        typ failure-mode
        typ user-tag
        typ overflow-switch
        typ activated
        typ capture-status
        typ task-complete-status
    }

    Server ||--|{ Host : contains
    Server ||--o{ Device : contains

    Host ||--o{ Runner : runs
    Runner ||--o{ Context : "creates / owns"
    Runner |o--o{ Context : "current activates"
    Context }o --|| Device : binds

    %% Runner ||--o{ RunnerCallback : registers
    %% RunnerCallback }|--|| Stream : linkedTo
    Context ||--|{ Stream : owns
    Ccu ||--|{ Port : "1:N"
    Port ||--o{ EndPointPair : "1:N"
    EndPointPair ||--o{ CcuChannel : hasLogicLink
    Device ||--o{ Port : "belongs to"
    Device ||--o{ Ccu : "1:2"
    Device ||--o{ DeviceConnection : hasLink
    Device ||..|{ Stream : "Hardware Constraint"
    

    %% Stream ||--o{ Notify : signals
    %% Notify ||--o{ IPCNotify : extends
    %% Stream ||--o{ Event : triggers
```
#### 关键关系说明：
|关系	|含义	|说明/注意事项|
|------|--------------|--------------|
|Server → Host|一个Server可能包含多个 Host 实例|比如多路 CPU 或虚拟化环境|
|Server → Device|一个Server包含多个 AI 设备|对应 /dev/davinci0, /dev/davinci1 等|
|Host → Runner|	每个主机上有多个应用线程（Runner）|每个 Runner 可以创建多个 Context|
|Runner → Context|线程创建或切换到不同的 Context|使用 aclrtCreateContext() 和 aclrtSetCurrentContext()|
|Context → Device|Context 绑定 Device|一旦创建后不可跨设备|
|Context → Stream|每个 Context 可创建多个 Stream|对应 aclrtCreateStream()|
|Stream → TaskKernel|Stream 承载推理/算子任务|aclrtLaunchKernel() / aclrtMemcpyAsync()|
|Device ↔ DeviceConnection|设备间通信通道|aclrtDeviceCanAccessPeer(), aclrtDeviceEnablePeerAccess()|
|Runner ↔ Context (current)|当前上下文激活状态|aclrtGetCurrentContext()、aclrtSetCurrentContext()|
|Device .. Stream|资源上限约束关系|Stream 数量受硬件限制（max-stream-cnt）|
|Host ↔ Device|Host 通过驱动与 Device 通信|	逻辑上是 PCIe 或者 HCCS 连接|

#### 配套接口映射表：
##### 基础与设备层 (Device / Server)
| 实体/属性 | 关键API接口 |
|----------- |-------------|
|Server/Host |`aclInit`, `aclFinalize` |
|Runner.pid |`rtDeviceGetBareTgid` |
|Device|`rtGetDeviceCount`  |
|Device.logic-id|`rtSetDevice`,`rtResetDevice`,`rtGetDevice`,`rtsGetLogicDevIdByPhyDevId`|
|Device.physical-id|`rtGetPhyDevIdByLogicDevId`|
|Device.soc-name |`rtGetSocName` |
|Device.overflow-mode|`rtSetDeviceSatMode`,`rtGetDeviceSatMode`|
|DeviceConnection|`rtGetDevicesTopo`,`rtDeviceDisablePeerAccess`,`rtDeviceEnablePeerAccess`,`rtDevicePeerAccessStatus`|
|Context.context-id|`rtCreateContext`,`rtDestroyContext`|
|Context.is-default |`rtSetCurrentContext`,`rtGetCurrentContext`|
|Context.float-overflow-addr |`rtCtxGetFloatOverflowAddr`|
|Stream表|`aclrtGetStreamAvailableNum`|
|Stream.stream-id |`rtCreateStream`,`rtCreateStreamWithConfig`,`rtDestroyStream`,`rtDestroyStreamForce`|
|Stream.task-complete-status |`rtSynchronizeStream`, `rtSynchronizeStreamWithTimeout`|
|Stream.activated |`rtStreamStop`|
|Stream.failure-mode |`rtSetStreamAttribute`,`rtGetStreamAttribute`|

### 基础内存管理关系建模
```mermaid
erDiagram
    PhyMemBlock ||--o{ VirtualPointerTable : "物理到虚拟映射"
    PhyMemBlock ||--o{ FdMemRecord : "文件描述符映射"
    PhyMemBlock {
        typ phy-mem-id PK "自增ID"
        typ device-id FK "0,1...或 -1(host)"
        typ size
        typ type 
        typ ref-count
    }

    VirtualPointerTable {
        typ start-ptr PK
        typ size
        typ context-id FK
        typ phy-mem-id FK
        typ owner-pid  "创建进程"
        typ source-type ""
        typ policy
    }

    VirtualPointerTable ||--o{ IpcMemRecord : "共享内存注册"
    IpcMemRecord {
        typ ipc-id PK
        typ vir_mem_id FK
        typ phy_mem_id FK
        typ name-or-key
        typ create-pid
    }

    IpcMemRecord ||--o{ IpcMemWhiteList : "进程白名单"
    IpcMemWhiteList {
        typ ipc-id FK
        typ pid
        typ create-pid
    }

    FdMemRecord {
        typ fd PK
        typ phy-mem-id FK
        typ name
        typ type
    }

    FdMemRecord ||--o{ FdMemWhiteList : "进程白名单"
    FdMemWhiteList {
        typ fd-id FK
        typ pid
        typ create-pid
    }

    %%VirtualPointerTable ||--o{ MemMapRecord : "映射关系"
    %%MemMapRecord {
    %%    typ ptr FK
    %%    typ phy-mem-id FK
    %%}
```
#### 关键关系说明：
1. **物理内存核心地位**  
   `PhyMemBlock`作为基础实体，通过`phy_mem_id`与所有其他实体关联，体现华为昇腾"物理内存池化"的设计理念[3]。

2. **三层映射体系**：
   - 物理→虚拟（`VirtualPointerTable`）
   - 物理→IPC共享（`IpcMemRecord`）
   - 物理→文件描述符（`fdMemRecord`）

3. **安全控制**：
   `IpcMemWhiteList`通过进程PID白名单机制实现华为HCCS（Huawei Collective Communication Service）的安全共享[3]。

4. **特殊映射类型**：
   `MemMapRecord`记录双虚拟地址映射场景（如`aclrtMapMem`产生的映射），支持华为NPU的零拷贝数据传输[3]。

#### 配套接口映射表：
| 实体 | 关键管理接口 |
|--------------|--------------|
|PhyMemBlock|`rtMallocPhysical`, `rtFreePhysical`|
|VirtualPointerTable|`rtMallocWithCfg`,`rtMallocForTaskScheduler`,`rtMallocHostWithCfg`,`rtFree`,`rtReserveMemAddress`,`ReleaseMemAddress`, `rtMapMem`, `rtUnmapMem`|
|VirtualPointerTable.context-id| `rtPointerGetAttributes`|
|FdMemRecord.fd | `rtMemExportToShareableHandle`, `rtMemImportFromShareableHandle`|
|FdMemWhiteList.pid| `rtMemSetPidToShareableHandle`|
|IpcMemRecord.name-or-key| `rtIpcMemGetExportKey` |
|IpcMemRecord.ipc-id|`rtIpcMemImportByKey`,`IpcMemClose` |
|IpcMemWhiteList.pid | `rtIpcMemSetImportPid` |
|MemMapRecord |`rtHostRegister`, `rtHostUnRegister` |

## 在基础模型上扩展数据任务模型

### 数据 / 任务流建模
```mermaid
erDiagram
    Context {
        typ ctx-id PK
        typ run-id FK
    }

    Stream {
        typ stream-id PK
        typ ctx-id FK
        typ state "Running/Idle"
    }

    %% Stream 包含有序的任务列表
    Context ||--o{ Stream : "manages/submits"
    Stream ||--o{ Task : "queues [1..*]"

    Task {
        typ task-id PK
        typ stream-id FK
        typ seq-number "stream内自增"
        typ type "Kernel/Memcpy/Callback"
    }

    %% 各种具体的 Task 类型 (逻辑上的继承关系)
    MemcpyTask {
        typ task-id FK
        typ src-addr
        typ dst-addr
        typ size
    }

    %% 继承关系的逻辑表达 (Task 分为多种)
    Task ||--|{ MemcpyTask : "is a"

    
    %% MemcpyTask 的地址应该在 VirtualPointerTable 可寻址
    MemcpyTask }o..|{ VirtualPointerTable : "Range Constraint"
    VirtualPointerTable {
        typ start-ptr PK
        typ ctx-id FK
    }

    VirtualPointerTable }o--|| Context : "belongs to"

```
### ccu资源建模
一个NPU device包含2个CCU，分别为die0和die1。

```mermaid
graph RL
    subgraph DavidDevice0[David 0]
        direction RL
        Memory0[Memory]
        David0Die0[Die0_ccu]
        David0Die1[Die1_ccu]
    end


    subgraph David0Die0[Die0_ccu]

        CcuBuf00[CcuBuf]
        Variable00[Variable]
        Notify00[Notify]
        CompletedEvent00[CompletedEvent]
        Local/Rmt-Addr00[Local/Rmt-Addr]
    end

    subgraph David0Die1[Die1_ccu]

        CcuBuf01[CcuBuf]
        Variable01[Variable]
        Notify01[Notify]
        CompletedEvent01[CompletedEvent]
        Local/Rmt-Addr01[Local/Rmt-Addr]
    end

    David0Die0---Memory0
    David0Die1---Memory0

```

#### 配套接口映射表：
| 实体 | 关键管理接口 |
|------|--------------|
||`xx`, ``|
  
### 异步 / 同步执行建模

#### [Notify资源管理](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/appdevg/acldevg/aclcppdevg_000524.html)
```mermaid
erDiagram
    Device ||..o{ Notify : "Hardware Limit"
    Device ||--o{ Context : "referred by"
    Device {
        typ device-id PK
        typ device-type "A3"
        typ max-notify-cnt "8192"
    }
    Context {
        typ ctx-id PK
        typ device-id FK
    }

    Notify ||--o| IpcNotify : "is a"
    Notify o|--|| Context : "record"
    Notify {
        typ notify-id PK
        typ create-ctx-id FK
        typ device-notify-seq "0~8191"
        typ value "notify读写寄存器"
    }

    IpcNotify {
        typ ipc-id PK
        typ notify-id FK
        typ name-or-key
        typ create-pid
    }

    IpcNotify ||--o{ IpcNotifyVistorList : "has"
    IpcNotifyVistorList {
        typ ipc-id FK
        typ vistior-pid
    }
```
##### 关键关系说明：

##### 配套接口映射表：
| 实体 | 关键管理接口 |
|------|--------------|
|Notify.notify-id |`rtCreateNotify`,`rtDestroyNotify`，`rtGetNotifyId`|
|Notify.state|`lrtWaitAndResetNotify`, `rtWaitAndResetNotify`|
|IpcNotify.name-or-key|`rtNotifyGetExportKey`,`rtNotifyImportByKey`|
|IpcNotifyVistorList.ipc-id|`rtNotifySetImportPid`|

#### Notify同步控制

```mermaid
erDiagram
    Context {
        typ ctx-id PK
        typ run-id FK
    }

    Stream {
        typ stream-id PK
        typ ctx-id FK
        typ state "Running/Idle"
    }

    %% Stream 包含有序的任务列表
    Context ||--o{ Stream : "manages/submits"
    Stream ||--o{ Task : "queues [1..*]"

    Task {
        typ task-id PK
        typ stream-id FK
        typ seq-number "stream内自增"
        typ type "Notify"
    }

    %% 各种具体的 Task 类型 (逻辑上的继承关系)
    NotifyRecordTask {
        typ notify-id  FK

    }

    NotifyWaitTask {
        typ notify-id FK
    }

    NotifyRecordTask }o..|| Notify : "use"
    NotifyWaitTask }o..|| Notify : "use"
    Notify {
        typ notify-id  PK
        typ value
    }
    %% 继承关系的逻辑表达 (Task 分为多种)
    Task ||--|{ NotifyRecordTask : "is a"
    Task ||--|{ NotifyWaitTask : "is a"



```
##### 关键关系说明：
##### 配套接口映射表：
| 实体 | 关键管理接口 |
|------|--------------|
|NotifyRecordTask|`rtRecordNotify` |
|NotifyWaitTask|`lrtWaitAndResetNotify`|
#### Event资源管理

```mermaid
erDiagram
    Device ||..|{ Event : "Hardware Limit"
    Device ||--o{ Context : "refered by"
    Device {
        typ device-id PK
        typ device-type "A3"
        typ max-event-cnt "65536"
    }

    Event }o..|| Context : "created by"
    Event {
        typ event-id PK
        typ created-ctx-id FK
        typ event-flag
        typ device-res-seq "0~65535"
        typ created-time
        typ status
    }
    Context {
        typ ctx-id PK
        typ device-id FK
    }
```

##### 关键关系说明：

##### 配套接口映射表：
| 实体 | 关键管理接口 |
|------|--------------|
| Event.event-id | `rtCreateEvent`, `rtCreateEventWithFlag`,`rtDestroyEvent`,`rtGetEventId`|
| Event.status | `rtRecordEvent`,`rtQueryEventStatus`|
#### Event 流程控制
```mermaid
erDiagram
    Context {
        typ ctx-id PK
        typ run-id FK
    }

    Stream {
        typ stream-id PK
        typ ctx-id FK
        typ state "Running/Idle"
    }

    %% Stream 包含有序的任务列表
    Context ||--o{ Stream : "manages/submits"
    Stream ||--o{ Task : "queues [1..*]"

    Task {
        typ task-id PK "自增"
        typ stream-id FK
        typ seq-number "stream内自增"
        typ type "EVENT"
    }

    %% 各种具体的 Task 类型 (逻辑上的继承关系)
    EventTask {
        typ task-id FK
        typ event-id FK
        typ excute-time
        typ finish-time 
        typ first-capture-taskid  FK  
    }

    EventRICaptureTask {
        typ updated-time     
    }    

    EventSyncTask {
        typ event-id FK
        typ excute-time
        typ finish-time 
        typ op-timeout-s 
    }
    EventRecordTask {
        typ event-id FK
        typ excute-time
        typ finish-time 
    }
    EventWaitTask {
        typ event-id FK
        typ excute-time
        typ finish-time 
    }      
    EventTimeTask {
        typ event-id FK
        typ excute-time       
    }
    EventTraceTask {
        typ event-id FK
        typ start-task-id  FK  
    }        

    %% 继承关系的逻辑表达 (Task 分为多种)
    Task ||--|{ EventTask : "is a "
    EventTask ||--|{ EventRICaptureTask : "is a EXTERNAL"
    EventTask ||--|{ EventSyncTask : "is a EX"
    EventTask ||--|{ EventTimeTask : "is a EX"
    EventTask ||--|{ EventTraceTask : "is a EX"
    EventSyncTask ||--|{ EventRecordTask : "is a EX"
    EventSyncTask ||--|{ EventWaitTask : "is a EX"
    EventRecordTask ||--|{ EventWaitTask : "mapped to"    
    EventTask ||--|{ EventTraceTask : "recorded by"

```
##### 关键关系说明：
##### 配套接口映射表：
| 实体 | 关键管理接口 |
|------|--------------|
| EventTask | `rtRecordEvent`, `rtResetEvent`,`rtSynchronizeEvent`|
| EventRecordTask | `rtRecordEvent`, `rtResetEvent`|
| EventWaitTask | `rtStreamWaitEvent`, `rtQueryEventWaitStatus`|
| EventTimeTask | `rtResetEvent`, `rtRecordEvent` |
| EventTraceTask | `rtResetEvent`, `rtRecordEvent` |

## 通信域建模

跨机通信涉及到多通信域混合任务编排。
通信域的本质是HCCL在框架层通过Rdma_Agent提供的建链能力，维护的一张多卡的网络拓扑，保存在host进程中。



```mermaid

erDiagram

    Device {
        int rank-id PK
        int world-size
        string ip-addr
        int port
        typ host-id FK
        typ device-id FK
    }

    %% ==========================================
    %% 控制面 (对应 RaSocket 系列接口)
    %% ==========================================
    RaDevice ||--o{ RaSocket : 创建
    RaSocket {
        typ socket_handle  PK "模拟的文件句柄"
        typ state "LISTENING/CONNECTED"
        typ role "SERVER/CLIENT"
        typ rank-id
    }

    RaSocket ||--o{ RaSocketPair : owns
    RaSocketPair ||--o{ VirtualPointerTable : contains
    RaSocketPair {
        typ id          PK
        typ client-id   FK
        typ server-id   FK
    }


    
    %% ==========================================
    %% 数据面 (对应 RaRdev, RaQp, RaMr 接口)
    %% ==========================================
    Device ||--|{ RaDevice : "has virtual NIC"
    
    RaDevice {
        typ rdev-handle PK
        typ device-id FK "关联的NPU Device"
        typ mac-addr
        typ state
    }

    RaDevice ||--o{ RaMR : 注册
    RaDevice ||--o{ RaQP : 拥有
    RaDevice ||--o{ RaCQ : 拥有

    %% RDMA 核心概念：QP (Queue Pair)
    RaQP {
        typ qp-handle PK
        typ ra-dev-id FK
        typ qp-num "QPN"
        typ type "RC/UC/UD"
        typ state "RESET/INIT/RTR/RTS"
        typ peer-qpn "对端QPN"
        typ peer-lid "对我段LID/IP"
        typ send_cq_handle FK
        typ recv_ce_handle FK
    }

    %% RDMA 内存注册 (MR)
    RaMR ||--|{ VirtualPointerTable : 映射到虚拟内存
    RaMR {
        typ mr-handle PK
        typ lkey "Local Key"
        typ rkey "Remote Key"
        typ vptr_id "虚拟地址 关联到virtualPointTable"
        typ length
        typ addr 
    }

    RaQP ||--o{ RaCQ : Send_CQ关联
    RaQP ||--o{ RaCQ : recv_CQ关联
    RaQP |o--o| RaQP : 逻辑链接（ConnectAsync）
    RaCQ {
        typ handle-id PK
        typ ra-dev-id FK
        typ cqn 
        typ size
    }

    RaCQ ||--o{ RaCQe : 包含
    RaCQe {
        typ id PK
        typ cq-handle FK 
        typ wr-id "WorkRequest ID"
        typ status "SUCESS/FLUSH_ERR/.."
    }
```

##### 关键关系说明：
我们需要将 RA 模块拆解为两部分：IB Verbs 资源池（用于数据平面）和 Socket 资源池（用于控制平面）。

关键点说明：

RaContext/RaDevice: 模拟一张网卡（RNIC）。
RaQP (Queue Pair): 模拟发送/接收队列，这是 RDMA 通信的核心。
RaCQ (Completion Queue): 模拟完成队列，存放通信结果。
RaMR (Memory Region): 这是连接你现有内存模型的桥梁。它必须关联到你定义的 VirtualPointerTable
##### 配套接口映射表：
| 实体 | 关键管理接口 |
|------|--------------|
|RaDevice | `RaTlvInit`, `RaGetSockets`，`RaRdevInit`|
|ControlSocket|`RaSocketBatchConnect`,`RaSocketListenStart`,`RaSocketSend/Recv`|
|RaMR|`RaMrReg`,`RaRegisterMr`,`RaGetNotifyMrInfo`|
|RqQP | `RaQpCreate`,`RaQpDestroy`,`RaQpModify`,`RaQpConnectAsync`|
|RqCQ  | `RaSendWr`, `RaRecvWrlist`,`RaPollCq`|

## 回调与报告关系建模

```mermaid
erDiagram
    Runner ||--o{ Context: "creates"
    Runner {
        typ run-id PK
    }

    Context {
        typ ctx-id PK
        typ run-id FK
    }

    Stream {
        int stream-id PK
        int ctx-id FK
        string state "Running/Idle"
    }

    %% Stream 包含有序的任务列表
    Context ||--o{ Stream : "manages/submits"
    Stream ||--o{ Task : "queues [1..*]"

    Task {
        int task-id PK
        int stream-id FK
        typ seq-number "stream内自增"
        string type "Kernel/Memcpy/Callback"
    }

    %% 各种具体的 Task 类型 (逻辑上的继承关系)
    CallbackTask {
        typ report-id FK
        typ callback-fn
        typ user-data
    }    

    %% 继承关系的逻辑表达 (Task 分为多种)
    Task ||--|{ CallbackTask : "is a"

    ReportChannel {
        typ report-id
        typ stream-id
        typ run-id
    }    


    CallbackTask }o ..|| ReportChannel : "push"  
    ReportChannel ||--o{ Runner : "trigger and called by"

```


##### 关键关系说明：
==**设备**==执行到 CallbackTask 时，触发 Host Runner 线程执行回调
##### 配套接口映射表：
| 实体 | 关键管理接口 |
|------|--------------|
| CallbackTask | `rtSetExceptionInfoCallback`, `rtLaunchCallback`,`rtSynchronizeEvent`|
| ReportChannel | `rtSubscribeReport`, `rtUnSubscribeReport`|
| Runner | `rtProcessReport`|


## 基础`设备`模型细粒度底层扩展
```mermaid
erDiagram

    Device ||--|| DeviceStatus : "has a"
    Device ||--|{ TaskSchedulerDevice : "has "
    Device {
        typ device-id PK
    }
    DeviceStatus {
        typ device-id FK
        typ overflow-status
        typ synchronize-strategy
        typ synchronize-timeout
        typ capability-mask
        typ run-by-host
        typ ts-core
        typ online-status
    }

    TaskSchedulerDevice  ||--|| Scalar :"is a"
    TaskSchedulerDevice  ||--|| CCU :"is a"
    TaskSchedulerDevice  ||--|| CPU :"is a"
    TaskSchedulerDevice {
        typ ts-id PK
        typ device-id FK
        typ type "Scalar"
    }

   CPU {
   }  

   Scalar ||--|| ComputeDie :"schedule"
   Scalar {
   }

   CCU {
        typ ccu-id PK
        typ ts-id FK
        typ version "v1/v2"
        typ xnNum
        typ ckeNum
        typ msNum
        typ channelNum
    }

    ComputeDie  ||--|| Cube :"is a"
    ComputeDie  ||--|| Vector :"is a"
    ComputeDie  ||--|| HybridComputeDie :"is a  (vector+cube)"
    ComputeDie {
        typ compute-id PK
        typ ts-id FK
        typ type
    }  
    Cube {
    }
    Vector {
    }
    HybridComputeDie {
    }

```
##### 关键关系说明：

##### 配套接口映射表：
| 实体 | 关键管理接口 |
|------|--------------|
| TaskSchedulerDevice | `rtGetDeviceInfo`, `rtSetTsDevice`|
| DeviceStatus      |`rtGetRunMode`     |

## Kernel 运行时关系建模

```mermaid
erDiagram
    KernelBinary {
        typ id PK
        typ file
        typ create-pid
    }

    KernelBinary  ||--|| KernelBinaryHandle :"loaded"
    KernelBinaryHandle  ||--o{ KernelFuncHandle :"contains"
    KernelBinaryHandle {
        typ handle-id PK
        typ kernel-id FK
    }

    KernelFuncHandle  ||--o{ KernelFuncArgsHandle :"has a"
    KernelFuncHandle  }o--|| Task :"Called "
    KernelFuncHandle  }o--|| KernelLaunchCfg :"launch config"
    KernelFuncHandle {
        typ handle-id PK
        typ binary-id FK
        typ func-name
        typ kernel-name
        typ aic-addr
        typ aiv-addr
    }

    KernelFuncArgsHandle  ||--o{ KernelFuncArgsParamHandle :"append"
    KernelFuncArgsHandle {
        typ args-handle-id PK
        typ func-id FK
        typ args-size
        typ type "device/host"
    }

    KernelFuncArgsParamHandle {
        typ args-param-id PK
        typ args-id FK
        typ param-size
        typ is-place-holder
    }


    KernelLaunchCfg {

    }



```
##### 关键关系说明：

##### 配套接口映射表：
| 实体 | 关键管理接口 |
|------|--------------|
| KernelBinary | `rtCreateBinary`, `rtDestroyBinary`|
| KernelBinaryHandle|`rtBinaryLoad`, `rtBinaryUnLoad`,`rtBinaryLoadFromFile`,`rtBinaryLoadFromData`|
| KernelFuncHandle|`rtBinaryGetFunction`, `rtBinaryGetFunctionByEntry`,`rtGetFunctionAddr`,`rtGetFunctionName`,`rtRegisterCpuFunc`|

## 模型加载关系建模

