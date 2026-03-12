target_sources(hccl_proxy_level1 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/sim_context_manager.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sim_thread.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sim_thread_manager.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sim_channel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sim_channel_manager.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sim_communicator.cc
)