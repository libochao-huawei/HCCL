function(generate_stub_with_output_name STUB STUB_OUTPUT_NAME)
    if(EXISTS ${DOWNLOAD_LIB_DIR}/lib${STUB_OUTPUT_NAME}.so)
        add_library(${STUB} SHARED IMPORTED GLOBAL)
        set_target_properties(${STUB} PROPERTIES
            IMPORTED_LOCATION "${DOWNLOAD_LIB_DIR}/lib${STUB_OUTPUT_NAME}.so"
            INTERFACE_LINK_OPTIONS "-Wl,-rpath-link=${DOWNLOAD_LIB_DIR}"
        )
        message(STATUS "Imported library lib${STUB_OUTPUT_NAME}.so")
    else()
        string(FIND ${STUB_OUTPUT_NAME} "::" temp)
        if (temp EQUAL "-1")
            set(target_plain_name ${STUB_OUTPUT_NAME})
        else()
            string(REPLACE "::" ";" temp_list ${STUB_OUTPUT_NAME})
            list(GET temp_list 1 target_plain_name)
        endif()

        if (NOT TARGET ${target_plain_name}_stub_tmp)
            add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stub/${target_plain_name}.c
                COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/stub
                COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/stub/${target_plain_name}.c)
            add_library(${target_plain_name}_stub_tmp SHARED ${CMAKE_CURRENT_BINARY_DIR}/stub/${target_plain_name}.c)
            set_target_properties(${target_plain_name}_stub_tmp PROPERTIES
                WINDOWS_EXPORT_ALL_SYMBOLS TRUE
                LIBRARY_OUTPUT_NAME ${target_plain_name}
                RUNTIME_OUTPUT_NAME ${target_plain_name}
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/stub
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/stub)
        endif()

        add_library(${STUB} SHARED IMPORTED GLOBAL)
        if (UNIX)
            set_target_properties(${STUB} PROPERTIES
                IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/stub/lib${target_plain_name}.so")
        endif()
        if (WIN32)
            set_target_properties(${STUB} PROPERTIES
                IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/stub/${target_plain_name}.dll"
                IMPORTED_IMPLIB "${CMAKE_CURRENT_BINARY_DIR}/stub/${target_plain_name}.lib")
        endif()
        add_dependencies(${STUB} ${target_plain_name}_stub_tmp)

        message(STATUS "Stub library lib${STUB_OUTPUT_NAME}.so")
    endif()
endfunction()

function(generate_stub STUB)
    if(DEFINED STUB_OUTPUT_NAME_${STUB})
        set(STUB_OUTPUT_NAME ${STUB_OUTPUT_NAME_${STUB}})
    else()
        set(STUB_OUTPUT_NAME ${STUB})
    endif()

    generate_stub_with_output_name(${STUB} ${STUB_OUTPUT_NAME})

    if(DEFINED STUB_LINK_LIBRARIES_${STUB})
        foreach(LIB ${STUB_LINK_LIBRARIES_${STUB}})
            if(TARGET ${LIB})
                target_link_libraries(${STUB} INTERFACE ${LIB})
            endif()
        endforeach()
    endif()
endfunction(generate_stub)
set(STUBS
    ascend_hal
    c_sec
    slog
    unified_dlog
    aicpu_sharder
    mmpa
    hcomm
    ccl_kernel
)
foreach(STUB ${STUBS})
    if(NOT TARGET ${STUB})
        generate_stub(${STUB})
    endif()
endforeach()