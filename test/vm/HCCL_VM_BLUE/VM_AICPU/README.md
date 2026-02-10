# HOST CLI README
[toc]

## 核心功能介绍

### 1.启动模拟环境
#### 第二阶段（当前）：
```cpp
./hccl-vm start configFile --level $(level)
// configFile : 用户建模配置配置yaml文件名
// level : 模拟层级，checker或runner模式

// 举例：
root$> ./hccl-vm start config --level 2 //runner模式
```
第二阶段目标实现解析配置文件，获取建模信息，进行ipc层初始化。

**注意：**
**1.启动模拟环境仅即可完成初始化shm和ipc；**
**2.hccl-vm启动时，默认加载checker和dumper插件，如果想运行runner模式，需要手动加载runner插件，加载完毕后将自动启用后台进程运行virtual runtime；**
**3.加载runner插件将自动卸载checker插件，两者功能目前处于互斥状态**

**yaml文件示例**
```yaml
# 1. 全局统计信息 podNum, serNum, rankNum 都小于 1024
meta:
  podNum: 1       
  serNum: 1     
  rankNum: 4     

# 2. 详细拓扑结构 
topology:
  - podId: 0
    servers:
      - serId: 0
        ranks: [0, 1, 2, 3]
```


### 2.运行用户算例
**模拟器当前有两种运行算例的方法**
#### (1) 子bash命令透传：start
**hccl-vm完成环境初始化后，将启用被proxy.so劫持的子bash，用户只需像正常使用shell一样使用子bash即可，其中模拟器专属的子命令将以hccl-vm作为trigger标识**
命令选项|选项后接参数|描述
------------ | ------------- | ------------ |
--level|必须|level 1：安装checker和可视化插件；level 2：卸载checker插件，安装runner插件; 默认仿真等级 2 

```cpp
// 举例：(hvm$)> 为模拟器子bash 标识
root$> ./hccl-vm start config --level 2     // start命令将启用子bash
(hvm)$> mpirun -n 8 mpi_hccl_test_bin ... // hccl_test算例符号已被劫持，将在模拟器走完校验流程
(hvm)$> pwd                               // linux系统调用
(hvm)$> hccl-vm plugin list                 // 模拟器管理插件子命令
(hvm)$> plugin list                         // 不被视作模拟器子命令，将交由linux系统执行 
```
#### (2) one_shot模式：run
**one_shot模式下，模拟器启动指令后接hccl算例执行指令，可以直接完成环境建立->运行算例->完成校验全流程，跳过bash交互；one_shot模式当前仅支持runner模式，来完成hccl_test校验**
```cpp
// 举例：(hvm$)> 为模拟器子bash 标识
root$> ./hccl-vm run config --level 2 mpirun -n 8 mpi_hccl_test_bin ... // runner模式，将在模拟器走完校验流程
```
### 3.子命令

(1) 统一命令标识
**hccl-vm所有子命令由统一的标识管理(hccl-vm)，若用户输入的指令无标识，将被交由linux系统调用**
```cpp
// 举例：
(hvm)$> hccl-vm plugin list // 被识别为hccl-vm子命令
(hvm)$> plugin list         // 不被视作hccl-vm子命令，将交由linux系统执行 
```

(2) 插件管理：plugin
子命令|选项后接参数|描述
------------ | ------------ | ------------ |
install|必须|为模拟器虚拟环境注册插件,插件注册成功后，将在hccl-vm环境下以插件文件名命名该插件
uninstall|必须|为模拟器虚拟环境卸载插件，参数为'@'符号+插件名，支持多插件卸载，多插件时插件名用','隔开
list|无须|展示模拟器虚拟环境已经注册的插件，打印插件名
run|必须|运行插件，参数为'@'符号+插件名，支持多插件运行，多插件时插件名用','隔开

```cpp
// 举例：
(hvm)$> hccl-vm plugin install ${pluginName} // pluginName可以去plugin路径下检索，一般plugin路径下的插件文件夹名就是pluginName
(hvm)$> hccl-vm plugin uninstall @checker
(hvm)$> hccl-vm plugin list // 打印结果为checker
(hvm)$> hccl-vm plugin run @checker,@otherPlugin // 运行checker.so和otherPlugin
(hvm)$> hccl-vm plugin -h  // 子命令帮助
```

(3) model查询：model 
子命令|选项后接参数|描述
------------ | ------------- | ------------ |
list|无须|查看当前模拟器已存在的建模配置yaml文件

```cpp
// 举例：
hvm$> hccl-vm model list
```

(4) 模拟器日志查询：log **（暂时未支持，将随模拟器开发过程中完善）**
命令选项|选项后接参数|选项名|描述
------------ | ------------- | ------------ | ------------ |
-l|无须|log|查看模拟器运行日志
-h|无须|help|展示HOST CLI子命令向导

```cpp
// 举例：
hvm$> hccl-vm log -l
hvm$> hccl-vm log -h
```

### 4.退出模拟器子bash
```cpp
(hvm)$> exit
ctrl + D
```
**退出模拟环境时，将销毁共享内存和卸载所有已安装插件。**

### 5.最简功能指令集
(1) 功能：one_shot模式直接模拟器runner模式运行算例，完成hccl_test校验后退出
```cpp
// 举例：以podNum = 1, serNum = 1, rankNum = 2 为例
root$> ./hccl-vm run configFile_112 --level 2 mpirun -n 2 /srv/hccl_host/hccl_test/scatter_test_openmpi -p 2 -w 0 -n 1
```
(2) 功能：在子bash中运行算例，使用checker校验算法（checker模式）
```cpp
// 举例：
root$> ./hccl-vm start configFile_112                                                // 启动hccl-vm 默认level 2
(hvm)$> mpirun -n 2 /srv/hccl_host/hccl_test/scatter_test_openmpi -p 2 -w 0 -n 1     // 在hccl-vm环境中运行用户hccl_test算例，但其中所有数据搬运操作使用假地址，hccl_test本身校验不通过
(hvm)$> hccl-vm plugin run @checker                                                  // hccl_test算例运行结束后，运行checker插件
```