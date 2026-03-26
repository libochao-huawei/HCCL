cmake_minimum_required(VERSION 3.16.0)

# 这个样例既包含 Host 侧动态库，也包含 AICPU kernel，
# 所以这里同时声明 CXX / AICPU / ASC 三种语言能力。
find_package(ASC REQUIRED)
find_package(AICPU REQUIRED)

project(hccl_examples_custom_ops_allgather_aicpu_2in2out LANGUAGES CXX AICPU ASC)
set(CMAKE_SYSTEM_PROCESSOR "x86_64")

# HCCL 主仓在 ENABLE_CUSTOM 模式下会从顶层注入这些安装目录。
# 如果是单独调试这个样例，这里给一组默认值，避免变量未定义。
if(NOT DEFINED BUILD_DEVICE_DIR)
    set(BUILD_DEVICE_DIR ${CMAKE_BINARY_DIR})
endif()

if(NOT DEFINED CUSTOM_OPS_OPP_LIB_PATH)
    set(CUSTOM_OPS_OPP_LIB_PATH ${CMAKE_INSTALL_PREFIX}/lib)
endif()

if(NOT DEFINED CUSTOM_OPS_OPP_INC_PATH)
    set(CUSTOM_OPS_OPP_INC_PATH ${CMAKE_INSTALL_PREFIX}/include)
endif()

if(NOT DEFINED CUSTOM_OPS_OPP_AICPU_CONFIG_PATH)
    set(CUSTOM_OPS_OPP_AICPU_CONFIG_PATH ${CMAKE_INSTALL_PREFIX}/aicpu/config)
endif()

if(NOT DEFINED CUSTOM_OPS_OPP_AICPU_KERNEL_PATH)
    set(CUSTOM_OPS_OPP_AICPU_KERNEL_PATH ${CMAKE_INSTALL_PREFIX}/aicpu/kernel)
endif()

# 统一维护样例内部可复用的 include 路径。
# 后面的 Host 库和 AICPU kernel 都直接复用这组目录，
# 这样你以后把 allgather 换成 allreduce 时，外层骨架基本可以原样保留。
set(HCCL_CUSTOM_OPS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(HCCL_CUSTOM_OPS_INCLUDE_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include/mmpa
    ${ASCEND_CANN_PACKAGE_PATH}/asc/include/aicpu_api
)

install(FILES
    inc/hccl_custom_allgather_2in2out.h
    DESTINATION ${CUSTOM_OPS_OPP_INC_PATH}
)

# Host 侧动态库：对外导出的自定义算子接口在这里。
add_subdirectory(op_host)

# AICPU kernel 侧最终会被打成 tar.gz，并配合 json 一起安装到 opp 路径。
set(AICPU_KERNEL_TAR_FILE ${CMAKE_BINARY_DIR}/aicpu_hccl_custom_allgather_2in2out.tar.gz)
set(AICPU_KERNEL_JSON_FILE ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu/liballgather_2in2out_aicpu_kernel.json)

if(KERNEL_MODE)
    # 这里定义 device 侧 AICPU kernel 的共享库目标。
    # op_kernel_aicpu/CMakeLists.txt 只负责往这个目标里追加源文件。
    add_library(allgather_2in2out_aicpu_kernel SHARED)
    target_include_directories(allgather_2in2out_aicpu_kernel PRIVATE
        ${HCCL_CUSTOM_OPS_INCLUDE_LIST}
    )
    target_compile_definitions(allgather_2in2out_aicpu_kernel PRIVATE
        -DAICPU_COMPILE
    )
    target_compile_options(allgather_2in2out_aicpu_kernel PRIVATE
        -g
        -O0
    )
    target_link_directories(allgather_2in2out_aicpu_kernel PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/lib64/device/lib64
        ${ASCEND_CANN_PACKAGE_PATH}/devlib/device
    )
    target_link_libraries(allgather_2in2out_aicpu_kernel PRIVATE
        ccl_kernel
    )

    # 下面这三个 helper 通常由 HCCL 主仓的 cmake/func.cmake 提供。
    # 如果你是从主仓 ENABLE_CUSTOM 流程进来，这里会完整执行“注册 -> 打包 -> 签名”。
    # 如果你只是单独调这个样例目录，则允许退化成“只编 AICPU .so，不做 tar/sign”。
    if(TARGET custom_aicpu AND COMMAND register_custom_aicpu_target)
        register_custom_aicpu_target(allgather_2in2out_aicpu_kernel)
    else()
        message(STATUS "custom_aicpu target not found, skip register_custom_aicpu_target for standalone build")
    endif()

    if(COMMAND package_aicpu_kernel AND COMMAND sign_aicpu_kernel)
        package_aicpu_kernel(
            allgather_2in2out_aicpu_kernel
            ${AICPU_KERNEL_TAR_FILE}
            ${AICPU_KERNEL_JSON_FILE}
        )
        sign_aicpu_kernel(
            allgather_2in2out_aicpu_kernel
            ${AICPU_KERNEL_TAR_FILE}
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/hccl_custom_allgather_2in2out_check_cfg.xml
            SIGNED_AICPU_KERNEL_TAR_FILE
        )
    else()
        message(STATUS "package_aicpu_kernel/sign_aicpu_kernel not found, skip tar/sign in standalone build")
    endif()
else()
    # 非 KERNEL_MODE 下不编 device .so，但仍保留安装规则，
    # 这样以后接入主仓的自定义算子打包流程时，目录结构是完整的。
    install(FILES
        ${BUILD_DEVICE_DIR}/signatures/aicpu_hccl_custom_allgather_2in2out.tar.gz
        DESTINATION ${CUSTOM_OPS_OPP_AICPU_KERNEL_PATH}
        OPTIONAL
    )
    install(FILES
        ${AICPU_KERNEL_JSON_FILE}
        DESTINATION ${CUSTOM_OPS_OPP_AICPU_CONFIG_PATH}
    )
endif()

add_subdirectory(op_kernel_aicpu)

# testcase 只在 Host 侧调试时有意义；进入 KERNEL_MODE 交叉编译时不参与。
option(BUILD_EXAMPLE_TESTCASE "Build testcase for the custom allgather example" ON)
if(BUILD_EXAMPLE_TESTCASE AND NOT KERNEL_MODE)
    add_subdirectory(testcase)
endif()
