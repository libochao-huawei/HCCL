set(allgatherbatch_host_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/../profiling_dl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/allgather_batch.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/load_kernel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/launch_kernel.cc
)

target_sources(hccl_allgatherbatch PRIVATE
    ${allgatherbatch_host_src_list}
)
