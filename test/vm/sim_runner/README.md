### hccl_sim_runner模拟运行器
**简介：**
- hccl_sim_runner提供hccl业务代码单算子API测试所需的仿真框架和底层打桩，对外提供virt_runtime_fwk.so和stub_runtime.so，由llt/ace/comop/hccl/sim_runner/st目录下的hccl_sim_runner二进制程序调用。

**编译&&运行：**
- 在项目根目录下(`work_code`)创建文件夹st: `mkdir st`
- 进入st文件夹(`cd st`)并执行如下命令:
```shell
rm -rf ../st/* && cmake ../cmake/superbuild/ -DHOST_PACKAGE=st -DBUILD_MOD=hccl_sim_runner && make -j8
```
- 生成的`hccl_sim_runner`二进制程序所在文件路径如下:
```shell
st/llt_gccnative-prefix/src/llt_gccnative-build/llt/ace/comop/hccl/sim_runner/st/hccl_sim_runner
```