#!/bin/bash
echo "#############################工具说明##################################"
echo "# 【注意】本脚本默认用户处于hccl_tools仓的work_code目录"
echo "# 【功能描述】编译&构建业务代码，并且构建和启动HCCL_Insight工具服务"
echo "#  HCCL Inisight工具默认部署位置：/srv/workspace/Hccl_Code/work_code/asl/devtools/hccl_tools/hccl_insight/bs_engine/server"
echo "#  用户可以通过参数指定HCCL Inisight工具部署路径"
echo "######################################################################"
 
current_dir=$(pwd)
default_server_dir="$current_dir/asl/devtools/hccl_tools/hccl_insight/bs_engine/server"
server_dir=$default_server_dir

#解析参数
envConfig=false
skip_compile_binary=false
bin_package_config=false
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -skip_compile_binary)
      echo "跳过编译insight_adapter二进制文件(一般本地源码部署)..."
      skip_compile_binary=true;;
    -env_config)
      echo "获取-env_config参数，开启配置前端编译环境功能..."
      envConfig=true;;
    -bin_package)
      echo "获取-bin_package参数，开启工具二进制打包..."
      bin_package_config=true;;
    -server_dir)
      echo "获取-server_dir参数，确认本地工具服务路径为: $2"
      if [ $2 == "" ]; then
        echo "本地工具服务路径不能为空..."
        exit 1
      fi
      server_dir="$2"
      shift
      ;;
    *)
      echo "未知参数: $key"
      exit 1
      ;;
  esac
  shift
done
 
echo -e "\n<<<< 开始启动hccl insight工具本地构建任务...>>>>\n"
echo -e "工具服务的启动路径为： $server_dir\n"
 
# 非默认路径，则需要将给定的servr文件夹清除，并创建一个新的server目录
if [ "$server_dir" != "$default_server_dir" ]; then
  echo -e "\n启动本地服务，且工具路径非默认路径，开始创建server目录...\n"
  if [ -d $server_dir ]; then
    rm -rf $server_dir
  fi
  mkdir $server_dir
  cp -rf $default_server_dir $server_dir/..
 
  if [ -d $server_dir/node_modules ]; then
    rm -rf $server_dir/node_modules
  fi
fi
 
comm_work_space_dir="$server_dir/work_space/common"
echo -e "用户在hccl insight工具中work space路径为: ${comm_work_space_dir}\n"
 
if [ $bin_package_config == false ] && [ $skip_compile_binary == false ]; then
  echo -e "\n开始编译业务代码...\n"
  # 编译和链接业务代码
  rm -rf tmp
  mkdir -p tmp
  cd tmp
  cmake ../cmake/superbuild/ -DCUSTOM_PYTHON=python3 -DHOST_PACKAGE=ut -DBUILD_MOD=hccl_checker -DFULL_COVERAGE=false -DCOVERAGE_RC_CONFIG=false && TARGETS=open_hccl_test  make -j20
  
  if [ ! -d $comm_work_space_dir ]; then
    mkdir $comm_work_space_dir
  fi
  cp ./llt_gccnative-prefix/src/llt_gccnative-build/asl/devtools/hccl_tools/hccl_insight/cpp_engine/insight_adapter $comm_work_space_dir
fi
 
# 开始构建hccl insight工具，并启动本地服务...
cd "$current_dir/asl/devtools/hccl_tools/hccl_insight/bs_engine"
xx=$(pwd)
echo "pwd: $xx"
if [ $envConfig == true ]; then
  if [ $bin_package_config == true ]; then
    ./hccl_insight_tool_run.sh -env_config -action package -server_dir $server_dir
  else
    ./hccl_insight_tool_run.sh -env_config -action run -server_dir $server_dir
  fi
else
  if [ $bin_package_config == true ]; then
    ./hccl_insight_tool_run.sh -action package -server_dir $server_dir
  else
    ./hccl_insight_tool_run.sh -action run -server_dir $server_dir
  fi
fi