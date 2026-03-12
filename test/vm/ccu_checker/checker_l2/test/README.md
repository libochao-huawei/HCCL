1. 目录结构： 如下所示，各个模块在test目录建立各自的用例目录。CMakeLists.txt文件和main.cc文件，可直接参考已有的模块拷贝，修改
    ~~~
    test
        ├── CMakeLists.txt
        ├── googletest
        ├── hccl_virtual_runtime_test
        │   ├── CMakeLists.txt
        │   ├── main.cc
        │   └── testcast_xxx.cc
        └── README.txt
    ~~~

2. 写用例： 模块用例testcast_xxx.cc，参考标准gtest用例写法即可

3. 运行： 在test目录下，执行如下命令
    ~~~
    # 构建
    mkdir build
    cd build
    cmake ..
    make
    
    # 运行用例
    ctest -V                    # 运行所有模块的用例
    ctest -V -R hccl_ipc_test   # 运行指定模块的用例 (例如 hccl_ipc_test)
    ~~~

可优化点：后续可以将用例执行命令归纳为脚本文件，一键式编译&执行