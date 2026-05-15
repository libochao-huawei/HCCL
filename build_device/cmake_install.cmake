# Install script for directory: /home/helloworld/code/hccl-ccu

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/helloworld/code/hccl-ccu/build_out")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/info/hccl/script" TYPE DIRECTORY PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE DIR_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE FILES "/home/helloworld/code/hccl-ccu/scripts/package/hccl/scripts/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/info/hccl/script" TYPE FILE FILES
    "/home/helloworld/code/hccl-ccu/scripts/package/common/sh/check_version_required.awk"
    "/home/helloworld/code/hccl-ccu/scripts/package/common/sh/common_func.inc"
    "/home/helloworld/code/hccl-ccu/scripts/package/common/sh/common_interface.bash"
    "/home/helloworld/code/hccl-ccu/scripts/package/common/sh/common_interface.csh"
    "/home/helloworld/code/hccl-ccu/scripts/package/common/sh/common_interface.fish"
    "/home/helloworld/code/hccl-ccu/scripts/package/common/sh/version_compatiable.inc"
    "/home/helloworld/code/hccl-ccu/scripts/package/common/py/merge_binary_info_config.py"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/info/hccl" TYPE FILE RENAME "version.info" FILES "/home/helloworld/code/hccl-ccu/build_device/version.hccl.info")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/hccl/conf" TYPE FILE FILES "/home/helloworld/code/hccl-ccu/scripts/package/common/cfg/path.cfg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/info/hccl/script" TYPE FILE FILES
    "/home/helloworld/code/hccl-ccu/scripts/package/common/sh/install_common_parser.sh"
    "/home/helloworld/code/hccl-ccu/scripts/package/common/sh/common_func_v2.inc"
    "/home/helloworld/code/hccl-ccu/scripts/package/common/sh/common_installer.inc"
    "/home/helloworld/code/hccl-ccu/scripts/package/common/sh/script_operator.inc"
    "/home/helloworld/code/hccl-ccu/scripts/package/common/sh/version_cfg.inc"
    "/home/helloworld/code/hccl-ccu/scripts/package/common/sh/multi_version.inc"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/latest_manager" TYPE FILE FILES
    "/home/helloworld/code/hccl-ccu/scripts/package/common/sh/install_common_parser.sh"
    "/home/helloworld/code/hccl-ccu/scripts/package/common/sh/common_func_v2.inc"
    "/home/helloworld/code/hccl-ccu/scripts/package/common/sh/common_installer.inc"
    "/home/helloworld/code/hccl-ccu/scripts/package/common/sh/script_operator.inc"
    "/home/helloworld/code/hccl-ccu/scripts/package/common/sh/version_cfg.inc"
    "/home/helloworld/code/hccl-ccu/scripts/package/common/sh/common_func.inc"
    "/home/helloworld/code/hccl-ccu/scripts/package/common/sh/version_compatiable.inc"
    "/home/helloworld/code/hccl-ccu/scripts/package/common/sh/check_version_required.awk"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/latest_manager" TYPE DIRECTORY FILES "/home/helloworld/code/hccl-ccu/scripts/package/latest_manager/scripts/")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/helloworld/code/hccl-ccu/build_device/src/cmake_install.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/hccl/include/hccl" TYPE FILE FILES
    "/home/helloworld/code/hccl-ccu/include/hccl.h"
    "/home/helloworld/code/hccl-ccu/include/hccl_mc2.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/helloworld/code/hccl-ccu/build_device/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
