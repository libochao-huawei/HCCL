target_sources(stub_runtime PRIVATE
    llt_hccl_stub_mmpa.cc
    llt_hccl_stub.cc
    hccl_sim_pub_stub.cc
    hccl_sim_aicpu_subprocess.cc
    hccl_sim_comm_stub.cc
    llt_hccl_stub_ascendcl.cc
    llt_hccl_stub_ascend_hal.cc

    llt_hccl_stub_gert.cc
    llt_hccl_stub_fe.cc
    llt_hccl_stub_ge.cc
    llt_hccl_stub_profiling_plugin.cc
    llt_hccl_stub_error_manager.cc
    ./device/hccl_sim_aicpu_sqe.cc
)