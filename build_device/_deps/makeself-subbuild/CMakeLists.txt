# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.22.1)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(makeself-populate NONE)



include(ExternalProject)
ExternalProject_Add(makeself-populate
                     "UPDATE_DISCONNECTED" "False" "URL" "https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz" "URL_HASH" "SHA256=bfa730a5763cdb267904a130e02b2e48e464986909c0733ff1c96495f620369a" "DOWNLOAD_NO_PROGRESS" "TRUE" "DOWNLOAD_DIR" "/home/helloworld/code/hccl-ccu/third_party"
                    SOURCE_DIR          "/home/helloworld/code/hccl-ccu/third_party/makeself"
                    BINARY_DIR          "/home/helloworld/code/hccl-ccu/build_device/_deps/makeself-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)


