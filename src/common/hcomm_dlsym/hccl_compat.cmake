if(STATIC_MODE)
    target_sources(${HCCT_TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_rank_graph_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_res_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hcomm_primitives_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hcomm_dlsym.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_inner_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hcomm_host_profiling_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_host_comm_dl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_res_expt_dl.cc
    )
else()
    add_library(hccl_compat SHARED)

    if(BUILD_OPEN_PROJECT)
        target_compile_definitions(hccl_compat PRIVATE
            OPEN_BUILD_PROJECT
            $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
        )
    else()
        target_compile_definitions(hccl_compat PRIVATE
            $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
        )
    endif()

    if(HCCL_CANN_COMPAT_850)
        target_compile_definitions(hccl_compat PRIVATE HCCL_CANN_COMPAT_850)
    endif()

    target_include_directories(hccl_compat PRIVATE
        ${INCLUDE_LIST}
    )

    target_compile_options(hccl_compat PRIVATE
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-g>
        -std=c++14
        -fstack-protector-all
    )

    target_link_options(hccl_compat PRIVATE
        -Wl,-z,relro
        -Wl,-z,now
        -Wl,-z,noexecstack
        $<$<CONFIG:Release>:-s>
    )

if(BUILD_OPEN_PROJECT)
    target_link_libraries(hccl_compat PRIVATE
        -Wl,--no-as-needed
        unified_dlog
        acl_rt
        -Wl,--no-as-needed
    )
else()
    target_link_libraries(hccl_compat PRIVATE
        $<BUILD_INTERFACE:slog_headers>
        -Wl,--no-as-needed
        unified_dlog
        acl_rt
        -Wl,--no-as-needed
    )
endif()

    target_link_directories(hccl_compat PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/lib64
    )

    install(TARGETS hccl_compat
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} 
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )

    target_sources(hccl_compat PRIVATE
        hccl_rank_graph_dl.cc
        hccl_res_dl.cc
        hcomm_primitives_dl.cc
        hcomm_dlsym.cc
        hccl_inner_dl.cc
        hcomm_host_profiling_dl.cc
        hccl_host_comm_dl.cc
        hccl_res_expt_dl.cc
    )
endif()