target_include_directories(stub_runtime PRIVATE
    ${TOP_DIR}/abl/qos
    ${TOP_DIR}/abl/libc_sec/include
)

target_sources(stub_runtime PRIVATE
    qos_stub.cc
    tsd_stub.cc
)