# 静态库重打包脚本
cmake_minimum_required(VERSION 3.16.0)

# 校验入参
foreach(var _STATIC_LIB _AICPU_TAR _AIV_SEARCH_DIR _EXTRACT_DIR _FINAL_STATIC_LIB)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "Required variable ${var} is not defined")
    endif()
endforeach()

# 1.解出目标文件
message(STATUS "[static_postprocess] Extracting object files from ${_STATIC_LIB}")
execute_process(
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_EXTRACT_DIR}
    RESULT_VARIABLE ret
)
if(NOT ret EQUAL 0)
    message(FATAL_ERROR "Failed to create extract directory ${_EXTRACT_DIR}")
endif()

execute_process(
    COMMAND ar -x ${_STATIC_LIB}
    WORKING_DIRECTORY ${_EXTRACT_DIR}
    RESULT_VARIABLE ret
)
if(NOT ret EQUAL 0)
    message(FATAL_ERROR "Failed to extract object files from ${_STATIC_LIB}")
endif()

file(GLOB extracted_objs "${_EXTRACT_DIR}/*.o")
list(LENGTH extracted_objs obj_count)
message(STATUS "[static_postprocess] Extracted ${obj_count} object files")

# 2.AICPU包转二进制
message(STATUS "[static_postprocess] Embedding AICPU tar as binary object")
execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy ${_AICPU_TAR} ${_EXTRACT_DIR}/aicpu_hccl.tar.gz
    RESULT_VARIABLE ret
)
if(NOT ret EQUAL 0)
    message(FATAL_ERROR "Failed to copy AICPU tar to extract directory")
endif()

execute_process(
    COMMAND ld -r -b binary -o aicpu_hccl_tar.o aicpu_hccl.tar.gz
    WORKING_DIRECTORY ${_EXTRACT_DIR}
    RESULT_VARIABLE ret
)
if(NOT ret EQUAL 0)
    message(FATAL_ERROR "Failed to convert AICPU tar to binary object")
endif()

# 3.AIV内核转二进制
message(STATUS "[static_postprocess] Embedding AIV device kernel objects")
file(GLOB_RECURSE aiv_objects "${_AIV_SEARCH_DIR}/hccl_aiv_*_op_910_95.o")
list(LENGTH aiv_objects aiv_count)
message(STATUS "[static_postprocess] Found ${aiv_count} AIV kernel objects")

foreach(aiv_o ${aiv_objects})
    get_filename_component(aiv_basename "${aiv_o}" NAME)
    string(REGEX REPLACE "\.o$" "" aiv_stem "${aiv_basename}")
    set(aiv_bin "${_EXTRACT_DIR}/${aiv_stem}.bin")
    set(aiv_embed "${_EXTRACT_DIR}/${aiv_stem}_embed.o")

    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy ${aiv_o} ${aiv_bin}
        RESULT_VARIABLE ret
    )
    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to copy AIV kernel: ${aiv_basename}")
    endif()

    execute_process(
        COMMAND ld -r -b binary -o ${aiv_stem}_embed.o ${aiv_stem}.bin
        WORKING_DIRECTORY ${_EXTRACT_DIR}
        RESULT_VARIABLE ret
    )
    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to embed AIV kernel: ${aiv_basename}")
    endif()

    file(REMOVE ${aiv_bin})
    message(STATUS "[static_postprocess] Embedded AIV kernel: ${aiv_basename}")
endforeach()

# 4.重打包最终库
message(STATUS "[static_postprocess] Creating final static library")
get_filename_component(_final_lib_dir "${_FINAL_STATIC_LIB}" DIRECTORY)
execute_process(
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_final_lib_dir}
    RESULT_VARIABLE ret
)

# 目录内打包，避免路径嵌入
execute_process(
    COMMAND sh -c "cd ${_EXTRACT_DIR} && ar rcs ${_FINAL_STATIC_LIB} *.o"
    RESULT_VARIABLE ret
)
if(NOT ret EQUAL 0)
    message(FATAL_ERROR "Failed to create final static library")
endif()

# 验证
execute_process(
    COMMAND ar t ${_FINAL_STATIC_LIB}
    OUTPUT_VARIABLE ar_output
    RESULT_VARIABLE ret
)
if(NOT ret EQUAL 0)
    message(FATAL_ERROR "Failed to verify final static library")
endif()
message(STATUS "[static_postprocess] Final static library created: ${_FINAL_STATIC_LIB}")
