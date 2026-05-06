add_library(scatter_aicpu_kernel SHARED
 	         ${CMAKE_CURRENT_SOURCE_DIR}/common/utils.cc
 	         # ${CMAKE_CURRENT_SOURCE_DIR}/common/adapter_acl.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/common/config_log.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/common/sal.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/common/log.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/common/adapter_error_manager_pub.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/common/alg_env_config.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/common/device_compat.cc
 	 
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor/channel/channel.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor/channel/channel_request.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor/registry/coll_alg_exec_registry.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor/registry/coll_alg_v2_exec_registry.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor/executor_base.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor/executor_v2_base.cc
 	 
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/alg_template_base.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/alg_v2_template_base.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/template_utils.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/aicpu/kernel_launch.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/aicpu/dfx/task_exception_fun.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/registry/alg_template_register.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/wrapper/alg_data_trans_wrapper.cc
 	 
 	 
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo/topo.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo/topo_match_1d.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo/topo_match_base.cc
 	 
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/scatter_comm_executor.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/scatter_executor_base.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/scatter_mesh_executor.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/scatter_ring_executor.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/scatter_single_executor.cc
 	 
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template/nhr_base.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template/scatter_mesh.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template/scatter_nb.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template/scatter_nhr.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template/scatter_ring_direct.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template/scatter_ring.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/executor/ins_v2_scatter_sole_executor.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/executor/ins_v2_scatter_parallel_executor.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template/aicpu/ins_temp_scatter_mesh_1D.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template/aicpu/ins_temp_scatter_nhr.cc
 	 
 	 
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sole_executor.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/executor/ins_reduce_scatter_parallel_executor.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_mesh_1D.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_nhr.cc
 	         ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_mesh_1D_meshchunk.cc
@@ -232,106 +233,110 @@	 
 	             ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/executor/ins_v2_broadcast_sequence_executor.cc
 	             ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/aicpu/ins_temp_allgather_nhr_dpu_inter.cc
 	             ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/aicpu/ins_temp_scatter_nhr_dpu_inter.cc
 	             ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/executor/ins_v2_all_gather_sequence_executor.cc
 	             ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/aicpu/ins_temp_all_gather_nhr_dpu.cc
 	             ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/executor/ins_v2_reduce_sequence_executor.cc
 	             ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/aicpu/ins_temp_gather_dpu_inter.cc
 	             ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template/aicpu/ins_temp_dpu_alltoall_mesh.cc
 	             ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor/ins_v2_all_reduce_sequence_executor.cc
 	             ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu/ins_temp_reduce_scatter_mesh_1D_dpu_inter.cc
 	             ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu/ins_temp_all_gather_nhr_dpu_inter.cc
 	             ${CMAKE_CURRENT_SOURCE_DIR}/ops/send/template/ins_temp_send_dpu.cc
 	             ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv/template/ins_temp_recv_dpu.cc
 	         )
 	     endif()
 	 
 	     target_include_directories(scatter_aicpu_kernel PRIVATE
 	         ${INCLUDE_LIST}
 	     )
 	 
 	     target_compile_options(scatter_aicpu_kernel PRIVATE
 	         $<$<CONFIG:Debug>:-g>
 	         $<$<CONFIG:Release>:-O3>
 	         -fstack-protector-all
 	         -Werror
 	     )
 	 
 	     target_link_options(scatter_aicpu_kernel PRIVATE
 	         -Wl,-z,relro
 	         -Wl,-z,now
 	         -Wl,-z,noexecstack
 	         $<$<CONFIG:Release>:-s>
 	     )
 	 
 	     target_compile_definitions(scatter_aicpu_kernel PRIVATE
 	         -DAICPU_COMPILE
 	     )
 	 
 	     if(HCCL_CANN_COMPAT_850)
 	         target_compile_definitions(scatter_aicpu_kernel PRIVATE HCCL_CANN_COMPAT_850)
 	     endif()
 	 
 	     target_link_directories(scatter_aicpu_kernel PRIVATE
 	       ${ASCEND_CANN_PACKAGE_PATH}/devlib/device
 	   )
 	 
 	     target_link_libraries(scatter_aicpu_kernel PRIVATE
 	         -Wl,--no-as-needed
 	         ccl_kernel
 	         hccl_kernel_compat
 	         -Wl,--no-as-needed
 	     )
 	     add_dependencies(scatter_aicpu_kernel hccl_kernel_compat)