#!/bin/bash
echo "#############################工具说明##################################"
echo "# 【注意】本脚本默认用户处于hccl_tools仓的work_code目录"
echo "# 【功能描述】编译&构建业务代码，并且将业务可执行Binary拷贝至远端服务器工具的用户工作空间路径"
echo "#  用户必须指定远端服务器IP, server路径及用户名"
echo "######################################################################"

#解析参数
rem_server_dir=""
rem_server_ip=""
rem_user_name=$USER
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -server_ip)
      echo "获取-server_ip参数，确认远端服务器的ip地址为: $2"
      # IP 地址的正则表达式
      ip_regex="^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
      if [[ $2 =~ $ip_regex ]]; then
        rem_server_ip="$2"
      else
        echo "IP地址$2无效，请输入正确的IP地址..."
        exit 1
      fi
      shift
      ;;
    -server_dir)
      echo "获取-server_dir参数，确认远端服务器的工具路径为: $2"
      if [ $2 == "" ]; then
        echo "远端服务器工具路径不能为空..."
        exit 1
      fi
      rem_server_dir="$2"
      shift
      ;;
    -server_user)
      echo "获取-server_user参数，确认远端服务器的用户名为: $2"
      if [ $2 == "" ]; then
        echo "远端服务器工具用户名不能为空..."
        exit 1
      fi
      rem_user_name="$2"
      shift
      ;;
    *)
      echo "未知参数: $key"
      exit 1
      ;;
  esac
  shift
done

echo -e "\n<<<< 开始启动算法业务的构建任务...>>>>\n"
echo -e "HCCL Insight工具配置信息如下:"
echo -e "IP地址:$rem_server_ip, 用户名:$rem_user_name"
echo -e "工具路径:$rem_server_dir\n"

user_work_space_dir="$rem_server_dir/work_space/$rem_user_name"

# 编译和链接
rm -rf tmp
mkdir -p tmp
cd tmp
cmake ../cmake/superbuild/ -DCUSTOM_PYTHON=python3 -DHOST_PACKAGE=ut -DBUILD_MOD=hccl_checker -DFULL_COVERAGE=false -DCOVERAGE_RC_CONFIG=false && TARGETS=open_hccl_test  make -j20
cd ./llt_gccnative-prefix/src/llt_gccnative-build/ace/comop/hccl/open_source/test
cp ../../../../../asl/devtools/hccl_tools/hccl_insight/cpp_engine/libhccl_insight.so .
cp ../../../../../asl/devtools/hccl_tools/hccl_alg_analyzer/libhccl_alg_analyzer.so .
# 判断远程target目录是否存在，如果不存在，则创建target目录
echo -e "\n算法业务模块构建完毕，开始向远端服务器server拷贝可执行binary文件...\n"
echo -e "IP: $rem_server_ip"
echo -e "USER Addr: $user_work_space_dir\n"
ssh root@$rem_server_ip "[ -d $user_work_space_dir ] || mkdir -p $user_work_space_dir"
scp open_hccl_test libhccl_insight.so libhccl_alg_analyzer.so root@$rem_server_ip:$user_work_space_dir