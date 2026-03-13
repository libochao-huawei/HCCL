get_filename_component(ALG_ANALYZER_UI_PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ui/proto ABSOLUTE)

add_custom_target(alg_analyzer_proto_legacy
    DEPENDS protobuf    # 依赖 protoc
)
add_custom_command(TARGET alg_analyzer_proto_legacy
    COMMAND mkdir -p ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto
    COMMAND cd ${ALG_ANALYZER_UI_PROTO_DIR} && ${PROTOC_PROGRAM} --cpp_out=${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto analysis_result.proto
)

add_dependencies(hccl_alg_analyzer_legacy alg_analyzer_proto_legacy)

target_sources(hccl_alg_analyzer_legacy
    PRIVATE
    check_utils.cc
    checker.cc
    task_stub.cc
    task_graph_generator.cc
    topo_meta.cc
    ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto/analysis_result.pb.cc
    aiv_task_stub.cc
)

add_subdirectory(task_collector)
add_subdirectory(rank_info_recorder)
add_subdirectory(mem_layout)
add_subdirectory(link_type_recorder)
add_subdirectory(semantics_check)
add_subdirectory(ui)
add_subdirectory(param_trans)
add_subdirectory(singletask_check)
add_subdirectory(mem_conflict_check)