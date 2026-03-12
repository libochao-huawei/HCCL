# 编译 hccl_proxy
1. 安装CANN包
   ~~~
   # install-path参数写自己期望的安装路径
   ./Ascend-cann-toolkit_8.5.0_linux-x86_64.run --full --install-path=/home/l30069909/Ascend
   ~~~
2. 在海思极速空间代码根目录`work_code`下拉HCCL_Checker_L2代码
4. 将 `hccl_vm/hccl_proxy/CMakeLists.txt` 中的 `ASCEND_CANN_PACKAGE_PATH` 参数修改为自己本机CANN包的安装路径
5. 进入hccl_vm目录 `cd .../work_code/HCCL_Checker_L2/hccl_vm`, 执行以下命令
    ~~~
    cmake -S . -B build
    cmake --build build --target hccl_proxy
    ~~~