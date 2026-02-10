#!/bin/bash
echo "#############################工具说明##################################"
echo "# 【注意】本脚本默认用户处于hccl_tools仓的work_code目录"
echo "# 【功能描述】构建和启动本地HCCL_Insight工具服务"
echo "#  HCCL Inisight工具默认部署位置：/srv/workspace/Hccl_Code/work_code/asl/devtools/hccl_tools/hccl_insight/bs_engine/server"
echo "#  用户可以通过参数指定HCCL Inisight工具部署路径"
echo "######################################################################"
 
#解析参数
bs_engine_dir=$(pwd)
serverDir="$bs_engine_dir/server"
envConfig=false
action="run"
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -env_config)
      echo "获取-env_config参数，开启配置前端编译环境功能..."
      envConfig=true;;
    -action)
      if [ $2 != "run" ] && [ $2 != "package" ]; then
        echo "参数-action的合法值是{run, bin_run, package}"
        exit 1
      fi
      if [ $2 == "run" ]; then
        echo "获取-action参数值为run，hccl insight工具将直接开启服务..."
      fi
 
 
 
      if [ $2 == "package" ]; then
        echo "获取-action参数值为package，将打包hccl insight工具，请拷贝至目标服务器部署服务..."
      fi
      action="$2"
      shift
      ;;
    -server_dir)
      if [ $2 == "" ]; then
        echo "本地服务路径不能为空..."
        exit 1
      fi
      serverDir=$2
      shift
      ;;
    *)
      echo "未知参数: $key"
      exit 1
      ;;
  esac
  shift
done
 
if [ action == "run" ]; then
  echo "hccl insight工具启动服务路径为:$serverDir"
elif [ action == "package" ] && [ "$serverDir" != "$bs_engine_dir/server" ]; then
  echo "本次执行action任务为'package'，自动忽略用户配置的hccl insight工具服务本地启动路径:$serverDir"
fi
 
# 安装前端编译依赖库
frontDir="$bs_engine_dir/web_ui"
cd $frontDir
if [ $envConfig == true ]; then
    echo "开始配置hccl insight工具前端编译依赖库..."
    node -v
    npm config set strict-ssl false
    echo "开始执行前端npm i..."
    npm i --no-audit
fi
# 编译&打包前端代码
echo "开始编译hccl insight工具前端代码..."
npm run build:production
 
# 打包后端代码
if [ $action == "run" ]; then
  echo "开始配置hccl insight工具后端依赖库..."
  # 进入server目录，安装依赖&启动服务
  cd $serverDir
  if [ ! -d $serverDir/node_modules ]; then
     npm install
  fi
  # 拷贝前端包
  if [ ! -d $serverDir/dist ]; then
     cp -r $frontDir/dist .
  fi
  export NODE_ENV=dev
  echo "hccl insight工具环境已配置完毕，现在开始启动工具服务..."
  npm start
else
  echo "开始打包hccl insight工具..."
  # 打包insight adapter工具
  echo "开始配置hccl insight工具后端依赖库..."
  # 进入server目录，安装依赖&启动服务
  cd $serverDir
  if [ ! -d $serverDir/node_modules ]; then
     npm install
  fi
  # 拷贝前端包
  if [ ! -d $serverDir/dist ]; then
     cp -r $frontDir/dist .
  fi
  export NODE_ENV=dev
  echo "hccl insight工具环境已配置完毕，现在开始启动工具服务..."
  # 打包为二进制文件
  npm run pkg
  
  echo -e "\n#############################工具说明##################################"
  echo -e "Hccl Insight工具打包完成，路径为: $serverDir/hccl_insight"
  echo -e "#############################工具说明##################################\n"
  hcclLibDir=$bs_engine_dir/../../../../../output/llt/hccl_lib
  if [ -d $hcclLibDir ]; then
    echo "dir: $hcclLibDir"
    cp $serverDir/hccl_insight $hcclLibDir
  fi
fi