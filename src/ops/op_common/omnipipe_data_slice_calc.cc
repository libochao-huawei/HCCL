#include"omnipipe_data_slice_calc.h"
#undef HCCL_INFO
#define HCCL_INFO(...)

namespace ops_hccl{
//bufferinfo赋值//
void BuffInfoAssign(BuffInfo &bi,u64 inBuffBaseOff,u64 outBuffBaseOff,u64 hcclBuffBaseOff)
{
    bi.inBuffBaseOff=inBuffBaseOff;
    bi.outBuffBaseOff=outBuffBaseOff;
    bi.hcclBuffBaseOff=hcclBuffBaseOff;
}

//stepsliceinfo赋值
void StepSliceInfoAssign(StepSliceInfo &stepSliceInfotmp,BuffInfo buffInfo,u64 count, u64 sliceSize,u64 inputSliceStride,u64 outputSliceStride)
{
    stepSliceInfotmp.buffInfo=buffInfo;
    stepSliceInfotmp.count=count;
    stepSliceInfotmp.sliceSize=sliceSize;
    stepSliceInfotmp.inputSliceStride=inputSliceStride;
    stepSliceInfotmp.outputSliceStride=outputSliceStride;
}

//数组求和，算偏移用
double SumDataSize(u64* datastart,u64 num)
{
    double result=0;
    for(u64 i=0;i<num;i++)
    {
        result=result+datastart[i];
    }
    return result;
}

//2DAG偏移计算,y轴快
//参数,xAGOffect x轴偏移，yAGOffect y轴偏移，stepNum步数，xRankSize x轴大小，yRankSize y轴大小，xAGDataSize x轴每步每一片数据大小，yAGDataSize y轴每步每一片数据大小
void CalAllgather2DOffset(u64* xAGOffect,u64*yAGOffect,u64 stepNum,u64 xRankSize,u64 yRankSize, u64*xAGDataSize,u64*yAGDataSize)
{
    HCCL_INFO("[CalcDataSliceInfo.cc][CalAllgather2DOffset] start");
    xAGOffect[0]=0;//第一步发同轴，偏移为0
    yAGOffect[0]=0;//第一步发同轴，偏移为0
    if(stepNum>1)
    {
        yAGOffect[1]=0;//第二步y轴发斜对角，偏移0
        for(u64 sn=2;sn<stepNum;sn++)
        {
            yAGOffect[sn]=yAGOffect[sn-1]+yAGDataSize[sn-1];//后面y轴每一步都是接着上一步发
        }
        for(u64 sn=1;sn<stepNum-1;sn++)
        {
            xAGOffect[sn]=xAGOffect[sn-1]+xAGDataSize[sn-1];//后面x轴每一步都是接着上一步发
        }
        //最后一步单独处理下，慢轴发前半块，快轴发后半块
        xAGOffect[stepNum-1]=yAGOffect[stepNum-1];
        yAGOffect[stepNum-1]= yAGOffect[stepNum-1]+xAGDataSize[stepNum-1];
    }
    for(int i=0;i<stepNum;i++)
    {
        HCCL_INFO("[CalcDataSliceInfo.cc][CalAllgather2DOffset] xAGOffect[%d]=[%u],yAGOffect[%d]=[%u]",i,xAGOffect[i],i,yAGOffect[i]);
    }
    HCCL_INFO("[CalcDataSliceInfo.cc][CalAllgather2DOffset] end");
}

//2D rs发送数据片偏移计算,y轴快
//参数,xAGOffect x轴偏移，yAGOffect y轴偏移，stepNum步数，xRankSize x轴大小，yRankSize y轴大小，xAGDataSize x轴每步每一片数据大小，yAGDataSize y轴每步每一片数据大小
//同轴最后一步拆成两步，需要单独处理
void CalReducescatter2DOffset(u64* xRSOffect,u64*yRSOffect,u64 stepNum,u64 xRankSize,u64 yRankSize, u64*xRSDataSize,u64*yRSDataSize)
{
    HCCL_INFO("[CalcDataSliceInfo.cc][CalReducescatter2DOffset] start");
    xRSOffect[0]=0;//第一步发斜对角，偏移为0
    yRSOffect[0]=0;
    if(stepNum>2)//最后一步拆两步，所以只有3步及以上
    {
        xRSOffect[1]=0;//第二步x轴发同轴，偏移0
        //y轴前n-1步发斜对角
        yRSOffect[0]=xRSOffect[0]+xRSDataSize[0];//第一步单独处理下，慢轴发前半块，快轴发后半块
        for(u64 sn=1;sn<stepNum-2;sn++)
        {
            yRSOffect[sn]=yRSOffect[sn-1]+yRSDataSize[sn-1];//后面y轴每一步都是接着上一步发
        }
        for(u64 sn=2;sn<stepNum-2;sn++)
        {
            xRSOffect[sn]=xRSOffect[sn-1]+xRSDataSize[sn-1];//后面x轴每一步都是接着上一步发
        }
        //最后拆成两步了单独处理下
        yRSOffect[stepNum-2]= 0;
        yRSOffect[stepNum-1]= yRSOffect[stepNum-2]+yRSDataSize[stepNum-2];
        xRSOffect[stepNum-1]=0;
        if(stepNum>3)
        {
            xRSOffect[stepNum-1]=xRSOffect[stepNum-3]+xRSDataSize[stepNum-3];
        }
        xRSOffect[stepNum-2]=xRSOffect[stepNum-1]+xRSDataSize[stepNum-1];
        for(int i=0;i<stepNum;i++)
        {
            HCCL_INFO("[CalcDataSliceInfo.cc][CalReducescatter2DOffset] xRSOffect[%d]=[%u],yRSOffect[%d]=[%u]",i,xRSOffect[i],i,yRSOffect[i]);
        }
        HCCL_INFO("[CalcDataSliceInfo.cc][CalReducescatter2DOffset] end");
    }
}

//计算2Dag每步数据量比例存进数组，返回通信步数，用于scratchbuffer计算，返回的是比例，double类型
//参数：xStepP2pDataSize慢轴数据量，yStepP2pDataSize快轴数据量，xB慢轴带宽，yB快轴带宽，j慢轴卡数，i快轴卡数，dataSize单卡数据量（ag做之前每张卡数据量），maxStep设定的最大步数
u64 CalAllgatherDataSizeRatio2D(double* xStepP2pDataSize,double* yStepP2pDataSize,double xB,double yB,u64 xRankSize,u64 yRankSize,double dataSize,u64 maxStep)
{
    HCCL_INFO("[CalcDataSliceInfo.cc][CalAllgatherDataSizeRatio2D] start");
    u64 step=1;
    if(yRankSize==1)
    {
        xStepP2pDataSize[0]=dataSize;
    }
    else if(xRankSize==1)
    {
        yStepP2pDataSize[0]=dataSize;;
    }
    else
    {
        dataSize=dataSize*xRankSize*yRankSize;
        double bandwidthRatio = yB / xB;  // 带宽比例
        u64 wholeRankSize = yRankSize * xRankSize;
        // 计算斜对角等比
        double omniPipeRatio = (xRankSize - 1) / bandwidthRatio;
        // 计算放大系数
        double scale=0;
        for(u64 t=0;t<maxStep-1;t++)
        {
            scale=scale+std::pow(omniPipeRatio,t);
        }
        scale=bandwidthRatio/scale;
        // 计算通信步数,计算固定max步
        step=maxStep;
        if(xRankSize - bandwidthRatio>0)
        {
            if(omniPipeRatio==1)
            {
                //等比为1时需要单独算步数
                step = bandwidthRatio+1;
            }
            else
            {
                step = ceil(std::log(xRankSize - bandwidthRatio) / std::log(omniPipeRatio))+1;
            }
            //如果步数小于最大步数，就不需要放大
            if(step<=maxStep)
            {
                scale=1;
            }
        }
        HCCL_INFO("[CalcDataSliceInfo.cc][CalAllgatherDataSizeRatio2D] bandwidthRatio=[%f],omniPipeRatio=[%f],scale=[%f],step=[%u]",bandwidthRatio,omniPipeRatio,scale,step);
        // 1. 计算第一步的通信数据
        xStepP2pDataSize[0] = scale * dataSize / (wholeRankSize * bandwidthRatio);
        yStepP2pDataSize[0] = dataSize / wholeRankSize;
        double sumXDataSzie = xStepP2pDataSize[0];
        double sumYDataSzie = yStepP2pDataSize[0];
        // 2. 计算后续的通信数据
        for (u64 index = 1; index < step - 1; index++) {
            if (index == step - 2) {
                // 循环最后一轮特殊处理
                xStepP2pDataSize[index] = dataSize / wholeRankSize - sumXDataSzie;
                yStepP2pDataSize[index] = bandwidthRatio * xStepP2pDataSize[index];
                sumXDataSzie += xStepP2pDataSize[index];
                sumYDataSzie += yStepP2pDataSize[index];
                continue;
            }
            yStepP2pDataSize[index] = xStepP2pDataSize[index - 1] * (xRankSize - 1);
            xStepP2pDataSize[index] = yStepP2pDataSize[index] / bandwidthRatio;
            sumXDataSzie += xStepP2pDataSize[index];
            sumYDataSzie += yStepP2pDataSize[index];
        }
        // 3. 剩余数据切分转发
        xStepP2pDataSize[step - 1] = (dataSize * (yRankSize - 1) * (xRankSize - 1) / wholeRankSize - (sumYDataSzie - yStepP2pDataSize[0])*(yRankSize-1))  / ((xRankSize - 1) + (yRankSize - 1) * bandwidthRatio);
        yStepP2pDataSize[step - 1] = ((dataSize * (yRankSize - 1) * (xRankSize - 1) / wholeRankSize - (sumYDataSzie - yStepP2pDataSize[0])*(yRankSize-1)) - xStepP2pDataSize[step - 1]*(xRankSize - 1))/(yRankSize - 1);
    }
    HCCL_INFO("[CalcDataSliceInfo.cc][CalAllgatherDataSizeRatio2D] step=[%u]",step);
    for(int i=0;i<step;i++)
    {
        HCCL_INFO("[CalcDataSliceInfo.cc][CalAllgatherDataSizeRatio2D] xStepP2pDataSize[%d]=[%f],yStepP2pDataSize[%d]=[%f],",i,xStepP2pDataSize[i],i,yStepP2pDataSize[i]);
    }
    HCCL_INFO("[CalcDataSliceInfo.cc][CalAllgatherDataSizeRatio2D] end");
    return step;
}

//计算2Drs每步数据量比例存进数组，返回通信步数，用于scratchbuffer计算。
//参数：xStepP2pDataSize慢轴数据量，yStepP2pDataSize快轴数据量，xB慢轴带宽，yB快轴带宽，xRankSize慢轴卡数，yRankSize快轴卡数，dataSize单卡数据量(rs做完每张卡的数据量)，maxStep设定的最大步数
u64 CalReducescatterDataSizeRatio2D(double* xStepP2pDataSize,double* yStepP2pDataSize,double xB,double yB,u64 xRankSize,u64 yRankSize,double dataSize,u64 maxStep)
{
    HCCL_INFO("[CalcDataSliceInfo.cc][CalReducescatterDataSizeRatio2D] start");
    u64 step=1;
    maxStep=maxStep-1;//为了确定性计算这里最后一步实际上被拆成两步，所以算比例的时候先减一步。
    if(yRankSize==1)
    {
        xStepP2pDataSize[0]=dataSize;
    }
    else if(xRankSize==1)
    {
        yStepP2pDataSize[0]=dataSize;;
    }
    else
    {
        dataSize=dataSize*yRankSize*xRankSize;
        double bandwidthRatio = yB / xB;  // 带宽比例
        u64 wholeRankSize = yRankSize * xRankSize;
        // 计算斜对角等比
        double omniPipeRatio = (xRankSize - 1) / bandwidthRatio;
        // 计算放大系数
        double scale=0;
        for(u64 t=0;t<maxStep-1;t++)
        {
            scale=scale+std::pow(omniPipeRatio,t);
        }
        scale=bandwidthRatio/scale;
        // 计算通信步数,计算固定5步
        step=maxStep;
        if(xRankSize - bandwidthRatio>0)
        {
            if(omniPipeRatio==1)
            {
                //等比为1时需要单独算步数
                step = bandwidthRatio+1;
            }
            else
            {
                step = ceil(std::log(xRankSize - bandwidthRatio) / std::log(omniPipeRatio))+1;
            }
            //如果步数小于最大步数，就不需要放大
            if(step<=maxStep)
            {
                scale=1;
            }
        }
        HCCL_INFO("[CalcDataSliceInfo.cc][CalReducescatterDataSizeRatio2D] bandwidthRatio=[%f],omniPipeRatio=[%f],scale=[%f],step=[%u]",bandwidthRatio,omniPipeRatio,scale,step);
        // 1. 计算第一步的通信数据
        xStepP2pDataSize[0]=(yRankSize-1)*(xRankSize-bandwidthRatio)*dataSize/(yRankSize*xRankSize*((yRankSize-1)*bandwidthRatio+xRankSize-1));
        yStepP2pDataSize[0]=bandwidthRatio*(yRankSize-1)*(xRankSize-bandwidthRatio)*dataSize/(yRankSize*xRankSize*((yRankSize-1)*bandwidthRatio+xRankSize-1));
        double sumXDataSzie = 0;
        double sumYDataSzie = yStepP2pDataSize[0]/(xRankSize-1)+xStepP2pDataSize[0]/(yRankSize-1);
        // 2. 计算后续的通信数据
        for (u64 index = 1; index < step - 1; index++) {
            if (index == step - 2) {
                // 循环最后一轮特殊处理
                // xStepP2pDataSize[index] = dataSize / wholeRankSize - sumXDataSzie;
                yStepP2pDataSize[index] = (dataSize/(yRankSize*xRankSize)-sumYDataSzie)*(xRankSize-1);
                xStepP2pDataSize[index]=yStepP2pDataSize[index]/bandwidthRatio;
                sumXDataSzie += xStepP2pDataSize[index];
                sumYDataSzie += yStepP2pDataSize[index];
                continue;
            }
            if(index==1)
            {
                xStepP2pDataSize[index]=sumYDataSzie;
                yStepP2pDataSize[index]=xStepP2pDataSize[index]*bandwidthRatio;
                sumXDataSzie += xStepP2pDataSize[index];
                sumYDataSzie += yStepP2pDataSize[index]/(xRankSize-1);
            }
            xStepP2pDataSize[index] = yStepP2pDataSize[index-1] / (xRankSize-1);
            yStepP2pDataSize[index] = xStepP2pDataSize[index]*bandwidthRatio;
            sumXDataSzie += xStepP2pDataSize[index];
            sumYDataSzie += yStepP2pDataSize[index]/(xRankSize-1);
        }
        // 3. 剩余同轴数据转发
        xStepP2pDataSize[step - 1] = dataSize/(yRankSize*xRankSize)-sumXDataSzie;
        yStepP2pDataSize[step - 1] = dataSize/(yRankSize*xRankSize);
    }
    HCCL_INFO("[CalcDataSliceInfo.cc][CalReducescatterDataSizeRatio2D] step=[%u]",step);
    for(int i=0;i<step;i++)
    {
        HCCL_INFO("[CalcDataSliceInfo.cc][CalReducescatterDataSizeRatio2D] xStepP2pDataSize[%d]=[%f],yStepP2pDataSize[%d]=[%f],",i,xStepP2pDataSize[i],i,yStepP2pDataSize[i]);
    }
    HCCL_INFO("[CalcDataSliceInfo.cc][CalReducescatterDataSizeRatio2D] end");
    return step;
}

//计算2Dag每步数据片大小存进数组，返回通信步数，y轴快，数据需要整除对齐，注意是每一小片的大小。按照现有executor的HCCL_MIN_SLICE_ALIGN对齐
//参数：xStepP2pDataSize慢轴数据量，yStepP2pDataSize快轴数据量，xB慢轴带宽，yB快轴带宽，xRankSize慢轴卡数，yRankSize快轴卡数，dataSizeEachRank单卡数据量，maxStep设定的最大步数
u64 CalAllgatherDataSize2D(u64* xStepP2pDataSize,u64* yStepP2pDataSize,double xB,double yB,u64 xRankSize,u64 yRankSize,u64 dataSizeEachRank,u64 maxStep)
{
    HCCL_INFO("[CalcDataSliceInfo.cc][CalAllgatherDataSize2D] start");
    u64 step=1;
    u64 justifyLen=HCCL_MIN_SLICE_ALIGN;
    if(yRankSize==1)
    {
        xStepP2pDataSize[0]=dataSizeEachRank;
    }
    else if(xRankSize==1)
    {
        yStepP2pDataSize[0]=dataSizeEachRank;;
    }
    else
    {
        double bandwidthRatio = yB / xB;  // 带宽比例
        u64 wholeRankSize = xRankSize * yRankSize;
        // 计算斜对角等比
        double omniPipeRatio = (xRankSize - 1) / bandwidthRatio;
        // 计算放大系数,
        double scale=0;
        for(u64 t=0;t<maxStep-1;t++)
        {
            scale=scale+std::pow(omniPipeRatio,t);
        }
        scale=bandwidthRatio/scale;
        // 计算通信步数,计算固定max步
        step=maxStep;
        if(xRankSize - bandwidthRatio>0)
        {
            if(omniPipeRatio==1)
            {
                //等比为1时需要单独算步数
                step = bandwidthRatio+1;
            }
            else
            {
                step = ceil(std::log(xRankSize - bandwidthRatio) / std::log(omniPipeRatio))+1;
            }
            //如果步数小于最大步数，就不需要放大
            if(step<=maxStep)
            {
                scale=1;
            }
        }
        HCCL_INFO("[CalcDataSliceInfo.cc][CalAllgatherDataSize2D] bandwidthRatio=[%f],omniPipeRatio=[%f],scale=[%f],step=[%u]",bandwidthRatio,omniPipeRatio,scale,step);
        // 1. 计算第一步的通信数据
        //step为2单独处理一下
        if(step==2)
        {
            xStepP2pDataSize[0] = dataSizeEachRank;
        }
        else
        {
            xStepP2pDataSize[0] = scale * dataSizeEachRank /  bandwidthRatio;//有个double，下一行对齐
            xStepP2pDataSize[0]=xStepP2pDataSize[0]/justifyLen*justifyLen;//对齐
        }
        yStepP2pDataSize[0] = dataSizeEachRank;
        u64 sumXDataSzie = xStepP2pDataSize[0];
        u64 sumYDataSzie = 0;
        // 2. 计算后续的通信数据
        for (u64 index = 1; index < step - 1; index++) {
            if (index == step - 2) {
                // 循环最后一轮特殊处理
                xStepP2pDataSize[index] = dataSizeEachRank - sumXDataSzie;
                yStepP2pDataSize[index] = bandwidthRatio * xStepP2pDataSize[index]/(xRankSize-1);
                if(yStepP2pDataSize[index]>xStepP2pDataSize[index - 1])
                {
                    yStepP2pDataSize[index]=xStepP2pDataSize[index - 1];
                }
                yStepP2pDataSize[index]=yStepP2pDataSize[index]/justifyLen*justifyLen;
                sumXDataSzie += xStepP2pDataSize[index];
                sumYDataSzie += yStepP2pDataSize[index];
                continue;
            }
            yStepP2pDataSize[index] = xStepP2pDataSize[index - 1];
            xStepP2pDataSize[index] = yStepP2pDataSize[index]*(xRankSize-1)/ bandwidthRatio;
            xStepP2pDataSize[index]=xStepP2pDataSize[index]/justifyLen*justifyLen;
            sumXDataSzie += xStepP2pDataSize[index];
            sumYDataSzie += yStepP2pDataSize[index];
        }
        // 3. 剩余数据切分转发
        xStepP2pDataSize[step - 1] = (dataSizeEachRank - sumYDataSzie )*(xRankSize - 1) / ((xRankSize - 1) + (yRankSize - 1) * bandwidthRatio);
        xStepP2pDataSize[step - 1] = xStepP2pDataSize[step - 1]/justifyLen*justifyLen;
        yStepP2pDataSize[step - 1] = (dataSizeEachRank - sumYDataSzie )-xStepP2pDataSize[step - 1];

    }
    HCCL_INFO("[CalcDataSliceInfo.cc][CalAllgatherDataSize2D] step=[%u]",step);
    for(int i=0;i<step;i++)
    {
        HCCL_INFO("[CalcDataSliceInfo.cc][CalAllgatherDataSize2D] xStepP2pDataSize[%d]=[%u],yStepP2pDataSize[%d]=[%u],",i,xStepP2pDataSize[i],i,yStepP2pDataSize[i]);
    }
    HCCL_INFO("[CalcDataSliceInfo.cc][CalAllgatherDataSize2D] end");
    return step;
}

//计算2Drs每步数据量存进数组，返回通信步数，y轴快，数据需要整除对齐。补充计算两轴数据量最大的一步的数据量。按照现有executor128对齐，为满足确定性计算，这里最后一步拆为两步，两片
//参数：xStepP2pDataSize慢轴数据量，yStepP2pDataSize快轴数据量，xB慢轴带宽，yB快轴带宽，xRankSize慢轴卡数，yRankSize快轴卡数，dataSize总数据量，maxStep设定的最大步数,两轴数据量最大的一步的数据量
u64 CalReducescatterDataSize2D(u64* xStepP2pDataSize,u64* yStepP2pDataSize,double xB,double yB,u64 xRankSize,u64 yRankSize,u64 dataSizeEachRank,u64 maxStep)
{
    HCCL_INFO("[CalcDataSliceInfo.cc][CalReducescatterDataSize2D] start");
    u64 step=1;
    u64 justifyLen=HCCL_MIN_SLICE_ALIGN;
    if(yRankSize==1)
    {
        xStepP2pDataSize[0]=dataSizeEachRank;
    }
    else if(xRankSize==1)
    {
        yStepP2pDataSize[0]=dataSizeEachRank;
    }
    else
    {
        double bandwidthRatio = yB /  xB;  // 带宽比例
        // u64 wholeRankSize = yRankSize * xRankSize;
        // 计算斜对角等比
        double omniPipeRatio = (xRankSize - 1) / bandwidthRatio;
        // 计算放大系数
        double scale=0;
        for(u64 t=0;t<maxStep-2;t++)
        {
            scale=scale+std::pow(omniPipeRatio,t);
        }
        scale=bandwidthRatio/scale;
        // 计算通信步数,计算固定5步
        step=maxStep;
        if(xRankSize - bandwidthRatio>0)
        {
            if(omniPipeRatio==1)
            {
                //等比为1时需要单独算步数，最后一步拆成两步，所以加2
                step = bandwidthRatio+2;
            }
            else
            {
                //最后一步拆成两步，所以加2
                step = ceil(std::log(xRankSize - bandwidthRatio) / std::log(omniPipeRatio))+2;
            }
            //如果步数小于最大步数，就不需要放大
            if(step<=maxStep)
            {
                scale=1;
            }
        }
        HCCL_INFO("[CalcDataSliceInfo.cc][CalReducescatterDataSize2D] bandwidthRatio=[%f],omniPipeRatio=[%f],scale=[%f],step=[%u]",bandwidthRatio,omniPipeRatio,scale,step);
        // 1. 计算第一步的通信数据
        if(scale>1)
        {
            xStepP2pDataSize[0]=dataSizeEachRank*scale*std::pow(xRankSize-1,step-2)/(((yRankSize-1)*bandwidthRatio+xRankSize-1)*std::pow(bandwidthRatio,step-2));
            xStepP2pDataSize[0]=xStepP2pDataSize[0]/justifyLen*justifyLen;
        }
        else
        {
            xStepP2pDataSize[0]=(xRankSize-bandwidthRatio)*dataSizeEachRank/((yRankSize-1)*bandwidthRatio+xRankSize-1);
            xStepP2pDataSize[0]=xStepP2pDataSize[0]/justifyLen*justifyLen;
        }
        if(step==3)
        {
            yStepP2pDataSize[0]=dataSizeEachRank-xStepP2pDataSize[0];
        }
        else
        {
            yStepP2pDataSize[0]=xStepP2pDataSize[0]*bandwidthRatio*(yRankSize-1)/(xRankSize-1);
            yStepP2pDataSize[0]=yStepP2pDataSize[0]/justifyLen*justifyLen;
        }
        u64 sumXDataSzie = 0;
        u64 sumYDataSzie = yStepP2pDataSize[0]+xStepP2pDataSize[0];
        // 2. 计算后续的通信数据
        for (u64 index = 1; index < step - 2; index++) {
            if (index == step - 3) {
                // 循环最后一轮特殊处理
                // xStepP2pDataSize[index] = dataSize / wholeRankSize - sumXDataSzie;
                yStepP2pDataSize[index] = dataSizeEachRank-sumYDataSzie;
                xStepP2pDataSize[index]=yStepP2pDataSize[index]*(xRankSize-1)/bandwidthRatio;
                if(index==1 && xStepP2pDataSize[index]>sumYDataSzie)
                {
                    xStepP2pDataSize[index]=sumYDataSzie;
                }
                else if(xStepP2pDataSize[index]>yStepP2pDataSize[index-1])
                {
                    xStepP2pDataSize[index]=yStepP2pDataSize[index-1];
                }
                xStepP2pDataSize[index]=xStepP2pDataSize[index]/justifyLen*justifyLen;
                sumXDataSzie += xStepP2pDataSize[index];
                sumYDataSzie += yStepP2pDataSize[index];
                continue;
            }
            if(index==1)
            {
                xStepP2pDataSize[index]=sumYDataSzie;
                yStepP2pDataSize[index]=xStepP2pDataSize[index]*bandwidthRatio/(xRankSize-1);
                yStepP2pDataSize[index]=yStepP2pDataSize[index]/justifyLen*justifyLen;
                sumXDataSzie += xStepP2pDataSize[index];
                sumYDataSzie += yStepP2pDataSize[index];
                continue;
            }
            xStepP2pDataSize[index] = yStepP2pDataSize[index-1];
            yStepP2pDataSize[index] = xStepP2pDataSize[index]*bandwidthRatio/(xRankSize-1);
            yStepP2pDataSize[index]=yStepP2pDataSize[index]/justifyLen*justifyLen;
            sumXDataSzie += xStepP2pDataSize[index];
            sumYDataSzie += yStepP2pDataSize[index];
        }
        // 3. 剩余同轴数据转发
        //拆两步
        if(bandwidthRatio>10)
        {
            xStepP2pDataSize[step - 2]=dataSizeEachRank-sumXDataSzie;
        }
        else
        {
            xStepP2pDataSize[step - 2]=dataSizeEachRank/(1+bandwidthRatio);
            xStepP2pDataSize[step - 2]=xStepP2pDataSize[step - 2]/justifyLen*justifyLen;
        }
        xStepP2pDataSize[step - 1] = dataSizeEachRank-sumXDataSzie-xStepP2pDataSize[step - 2];
        yStepP2pDataSize[step - 2] = dataSizeEachRank-xStepP2pDataSize[step - 2];
        yStepP2pDataSize[step - 1] = dataSizeEachRank-yStepP2pDataSize[step - 2];
    }
    HCCL_INFO("[CalcDataSliceInfo.cc][CalReducescatterDataSize2D] step=[%u]",step);
    for(int i=0;i<step;i++)
    {
        HCCL_INFO("[CalcDataSliceInfo.cc][CalReducescatterDataSize2D] xStepP2pDataSize[%d]=[%u],yStepP2pDataSize[%d]=[%u],",i,xStepP2pDataSize[i],i,yStepP2pDataSize[i]);
    }
    HCCL_INFO("[CalcDataSliceInfo.cc][CalReducescatterDataSize2D] end");
    return step;
}

//2d打平带宽计算,y轴快,不考虑两个轴大小都为1，都为1应该走1D
//参数：xB x轴带宽，yB y轴带宽，xRankSize慢轴卡数，yRankSize快轴卡数，maxStepNum设定的最大步数
double CalcBandwidth2D(double xB,double yB,u64 xRankSize,u64 yRankSize,int maxStepNum)
{
    HCCL_INFO("[CalcDataSliceInfo.cc][CalcBandwidth2D] start");
    if(yRankSize==1)
    {
        return xB;
        HCCL_INFO("[CalcDataSliceInfo.cc][CalcBandwidth2D] xB=[%f]",xB);
    }
    else if(xRankSize==1)
    {
        return yB;
        HCCL_INFO("[CalcDataSliceInfo.cc][CalcBandwidth2D] yB=[%f]",yB);
    }
    else
    {
        double xAGDataSize[maxStepNum];
        double yAGDataSize[maxStepNum];
        double ds=1;
        //根据数据量为1计算每步数据比例
        int stepNum2d=CalAllgatherDataSizeRatio2D(xAGDataSize,yAGDataSize,xB,yB,xRankSize,yRankSize,ds,maxStepNum);
        double xds=0;
        //根据慢轴总时间计算等效带宽
        for(int i=0;i<stepNum2d;i++)
        {
            xds=xds+xAGDataSize[i];
        }
        HCCL_INFO("[CalcDataSliceInfo.cc][CalcBandwidth2D] Bandwidth2D=[%f]",xB*ds/xds);
        HCCL_INFO("[CalcDataSliceInfo.cc][CalcBandwidth2D] end");
        return xB*ds/xds;
    }
}

//cclbuffer计算接口,RS用，返回maxDataCountPerLoop和loopTimes
//levelRankSize三轴大小，dataSize单卡数据量，dataTypeSize数据类型的大小，endpointAttrBw三轴带宽，maxTmpMemSize cclbuffer大小
std::vector<u64> CalcOmniPipeScratchInfo(OmniPipeScratchParam &omniPipeScratchParam)
{
    HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] start");
    u64 transportBoundDataSize=UB_MAX_DATA_SIZE;
    u64 justifyLen=HCCL_MIN_SLICE_ALIGN;//128对齐
    u64 maxDataSizePerLoop=0;
    u64 loopTimes=1;
    std::vector<u64> scratchInfo={0,0};
    int maxStepNum=MAX_STEP_NUM+1;//这个函数只有rs使用，rs多拆了一步出来所以是+1
    HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] justifyLen=[%u],transportBoundDataSize=[%u],maxStepNum=[%u],",justifyLen,transportBoundDataSize,maxStepNum);
    std::vector<u64> levelRankSize=omniPipeScratchParam.levelRankSize;
    u64 xRankSize=levelRankSize[0];//x轴卡数
    u64 yRankSize=levelRankSize[1];//y轴卡数
    u64 zRankSize=levelRankSize[2];//z轴卡数
    HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] xRankSize=[%u],yRankSize=[%u],zRankSize=[%u],",xRankSize,yRankSize,zRankSize);
    double xB=omniPipeScratchParam.endpointAttrBw[0];
    double yB=omniPipeScratchParam.endpointAttrBw[1];
    double zB=omniPipeScratchParam.endpointAttrBw[2];
    u64 dataSize=omniPipeScratchParam.dataSize;
    u64 dataTypeSize=omniPipeScratchParam.dataTypeSize;
    u64 maxTmpMemSize=omniPipeScratchParam.maxTmpMemSize;
    std::vector<u64> levelAlgType=omniPipeScratchParam.levelAlgType;
    OpMode opMode=omniPipeScratchParam.opMode;
    CommEngine engine=omniPipeScratchParam.engine;
    HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] dataSize=[%u],dataTypeSize=[%u],maxTmpMemSize=[%u],opMode=[%u],engine=[%u],levelAlgType.size()=[%u]",dataSize,dataTypeSize,maxTmpMemSize,opMode,engine,levelAlgType.size());

    double xyB=CalcBandwidth2D(xB,yB,xRankSize,yRankSize,maxStepNum-1);
    HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] xB=[%f],yB=[%f],zB=[%f],xyB=[%f]",xB,yB,zB,xyB);

    u64 zRSDataSize[maxStepNum];
    u64 xyRSDataSize[maxStepNum];
    u64 xRSDataSize[maxStepNum][maxStepNum];
    u64 yRSDataSize[maxStepNum][maxStepNum];

    int outerStepNum=0;
    //先计算数据量
    if(zB>xyB)
    {
        outerStepNum=CalReducescatterDataSize2D(xyRSDataSize,zRSDataSize,xyB,zB,xRankSize*yRankSize,zRankSize,dataSize,maxStepNum);
        HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] zB>xyB,outerStepNum=[%u]",outerStepNum);
    }
    else
    {
        outerStepNum=CalReducescatterDataSize2D(zRSDataSize,xyRSDataSize,zB,xyB,zRankSize,xRankSize*yRankSize,dataSize,maxStepNum);
        HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] zB<=xyB,outerStepNum=[%u]",outerStepNum);
    }
    int innerStepNum=0;
    for(u64 i=0;i<outerStepNum;i++)
    {
        innerStepNum=CalReducescatterDataSize2D(xRSDataSize[i],yRSDataSize[i],xB,yB,xRankSize,yRankSize,xyRSDataSize[i],maxStepNum);
        HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] innerStepNum=[%u]",innerStepNum);
    }
    //根据数据量和算法类型计算scratch大小
    std::vector<u64> scratchSize;
    u64 zConnerStep=0;
    if(zB>xyB)
    {
        if(outerStepNum>2)
        {
            zConnerStep=outerStepNum-2;
        }
        scratchSize=CalScratchSize((u64*)xRSDataSize,(u64*)yRSDataSize,zRSDataSize,levelRankSize,zConnerStep,outerStepNum,innerStepNum,maxStepNum,levelAlgType,engine);
        HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] zB>xyB,scratchSize=[%u]",scratchSize);
    }
    else
    {
        if(outerStepNum>2)
        {
            zConnerStep=1;
        }
        scratchSize=CalScratchSize((u64*)xRSDataSize,(u64*)yRSDataSize,zRSDataSize,levelRankSize,zConnerStep,outerStepNum,innerStepNum,maxStepNum,levelAlgType,engine);
        HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] zB<=xyB,scratchSize=[%u]",scratchSize);
    }

    //算总的scratch再按比例除得到loop
    u64 allCclBufferSize=0;
    if(opMode==OpMode::OPBASE && engine==CommEngine::COMM_ENGINE_AICPU_TS)
    {
        allCclBufferSize=dataSize*xRankSize*yRankSize*zRankSize;
    }
    allCclBufferSize=allCclBufferSize+scratchSize[0]+scratchSize[1]+scratchSize[2];
    double bufferRatio=1;
    if(allCclBufferSize!=0)
    {
        //单独报错需要buffer但是cclbuffer为零的情况
        if(maxTmpMemSize!=0)
        {
            bufferRatio=allCclBufferSize*1.0/maxTmpMemSize;
        }
        else
        {
            HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] maxTmpMemSize=0,allCclBufferSize!=0,allCclBufferSize=[%u]",allCclBufferSize);
            return scratchInfo;
        }
    }
    HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] allCclBufferSize=[%u],bufferRatio=[%f],",allCclBufferSize,bufferRatio);

    //按比例计算loop
    if(bufferRatio<1)
    {
        maxDataSizePerLoop=dataSize;
    }
    else
    {
        maxDataSizePerLoop=dataSize/bufferRatio;
        maxDataSizePerLoop=maxDataSizePerLoop/justifyLen*justifyLen;
    }

    //校验
    HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] start check");
    while((allCclBufferSize!=0) && (allCclBufferSize>maxTmpMemSize))
    {
        if(zB>xyB)
        {
            outerStepNum=CalReducescatterDataSize2D(xyRSDataSize,zRSDataSize,xyB,zB,xRankSize*yRankSize,zRankSize,maxDataSizePerLoop,maxStepNum);
            HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] zB>xyB,outerStepNum=[%u]",outerStepNum);
        }
        else
        {
            outerStepNum=CalReducescatterDataSize2D(zRSDataSize,xyRSDataSize,zB,xyB,zRankSize,xRankSize*yRankSize,maxDataSizePerLoop,maxStepNum);
            HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] zB<=xyB,outerStepNum=[%u]",outerStepNum);
        }
        for(u64 i=0;i<outerStepNum;i++)
        {
            innerStepNum=CalReducescatterDataSize2D(xRSDataSize[i],yRSDataSize[i],xB,yB,xRankSize,yRankSize,xyRSDataSize[i],maxStepNum);
            HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] innerStepNum=[%u]",innerStepNum);
        }
        if(zB>xyB)
        {
            scratchSize=CalScratchSize((u64*)xRSDataSize,(u64*)yRSDataSize,zRSDataSize,levelRankSize,zConnerStep,outerStepNum,innerStepNum,maxStepNum,levelAlgType,engine);
            HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] zB>xyB,scratchSize=[%u]",scratchSize);
        }
        else
        {
            scratchSize=CalScratchSize((u64*)xRSDataSize,(u64*)yRSDataSize,zRSDataSize,levelRankSize,zConnerStep,outerStepNum,innerStepNum,maxStepNum,levelAlgType,engine);
            HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] zB<=xyB,scratchSize=[%u]",scratchSize);
        }
        if(opMode==OpMode::OPBASE && engine==CommEngine::COMM_ENGINE_AICPU_TS)
        {
            allCclBufferSize=maxDataSizePerLoop*xRankSize*yRankSize*zRankSize;
        }
        allCclBufferSize=allCclBufferSize+scratchSize[0]+scratchSize[1]+scratchSize[2];
        HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] allCclBufferSize=[%u],",allCclBufferSize);
        if(allCclBufferSize>maxTmpMemSize)
        {
            maxDataSizePerLoop=maxDataSizePerLoop*0.9;//大了就小一点
            maxDataSizePerLoop=maxDataSizePerLoop/justifyLen*justifyLen;
        }
    }
    HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] end check");

    if(maxDataSizePerLoop>transportBoundDataSize)
    {
        maxDataSizePerLoop=transportBoundDataSize;
        maxDataSizePerLoop=maxDataSizePerLoop/justifyLen*justifyLen;
    }
    if(maxDataSizePerLoop!=0)
    {
        loopTimes = dataSize / maxDataSizePerLoop + static_cast<u64>(dataSize % maxDataSizePerLoop != 0);
    }
    u64 maxDataCountPerLoop=maxDataSizePerLoop/dataTypeSize;

    scratchInfo[0]=maxDataCountPerLoop;
    scratchInfo[1]=loopTimes;
    HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] maxDataCountPerLoop=[%u],loopTimes=[%u],",maxDataCountPerLoop,loopTimes);
    HCCL_INFO("[CalcDataSliceInfo.cc][CalcOmniPipeScratchInfo] end");
    return scratchInfo;
}

//算总共需要的scratch大小,返回scratch大小,rs用
//xRSDataSize，yRSDataSize，zRSDataSize三轴每一步小片数据量，levelRankSize三轴卡数，cornerStep斜对角是哪一步，outerStepNum整体步数，innerStepNum机内步数，maxStepNum最大步数，levelAlgType三轴算法类型，engine引擎
std::vector<u64> CalScratchSize(u64*xRSDataSize,u64*yRSDataSize,u64*zRSDataSize,std::vector<u64> levelRankSize,u64 cornerStep,u64 outerStepNum,u64 innerStepNum,u64 maxStepNum,std::vector<u64> levelAlgType,CommEngine engine)
{
    HCCL_INFO("[CalcDataSliceInfo.cc][CalScratchSize] start");
    //返回3个值，xBuffer,yBuffer,zBuffer,大小
    std::vector<u64> scratchSize={0,0,0};
    u64 xRankSize=levelRankSize[0];//x轴卡数
    u64 yRankSize=levelRankSize[1];//y轴卡数
    u64 zRankSize=levelRankSize[2];//z轴卡数
    HCCL_INFO("[CalcDataSliceInfo.cc][CalScratchSize] xRankSize=[%u],yRankSize=[%u],zRankSize=[%u],",xRankSize,yRankSize,zRankSize);

    u64 xTopo=levelAlgType[0];
    u64 yTopo=levelAlgType[1];
    u64 zTopo=levelAlgType[2];
    HCCL_INFO("[CalcDataSliceInfo.cc][CalScratchSize] xTopo=[%u],yTopo=[%u],zTopo=[%u],",xTopo,yTopo,zTopo);

    std::vector <std::vector<u64>> rsStepDataSize=CalRSDataSizeStep(xRSDataSize,yRSDataSize,zRSDataSize,levelRankSize,cornerStep, outerStepNum,innerStepNum,maxStepNum);

    for(int axis=0;axis<levelAlgType.size();axis++)
    {
        //判断是不是aicpu+mesh，是的话需要预留scratch
        if(levelAlgType[axis]>0 && engine==CommEngine::COMM_ENGINE_AICPU_TS)
        {
            for(int i=0;i<rsStepDataSize[axis].size();i++)
            {
                if(scratchSize[axis]<rsStepDataSize[axis][i]*levelRankSize[axis] && levelRankSize[axis]>1)
                {
                    scratchSize[axis]=rsStepDataSize[axis][i]*levelRankSize[axis];
                }
            }
        }
    }
    HCCL_INFO("[CalcDataSliceInfo.cc][CalScratchSize] scratchSize[0]=[%u],scratchSize[1]=[%u],scratchSize[2]=[%u]",scratchSize[0],scratchSize[1],scratchSize[2]);
    HCCL_INFO("[CalcDataSliceInfo.cc][CalScratchSize] end");
    return scratchSize;
}


//根据数据片大小得到RS每步数据量
//同轴拆为两步
std::vector<std::vector<u64>> CalRSDataSizeStep(u64*xRSDataSize,u64*yRSDataSize,u64*zRSDataSize,std::vector<u64> levelRankSize,u64 cornerStep,u64 outerStepNum,u64 innerStepNum,u64 maxStepNum)
{
    HCCL_INFO("[CalcDataSliceInfo.cc][CalRSDataSizeStep] start");
    std::vector <std::vector<u64>> rsStepDataSize={};
    std::vector<u64> xSize={};
    rsStepDataSize.push_back(xSize);
    std::vector<u64> ySize={};
    rsStepDataSize.push_back(ySize);
    std::vector<u64> zSize={};
    rsStepDataSize.push_back(zSize);
    u64 xRankSize=levelRankSize[0];//x轴卡数
    u64 yRankSize=levelRankSize[1];//y轴卡数
    u64 zRankSize=levelRankSize[2];//z轴卡数
    HCCL_INFO("[CalcDataSliceInfo.cc][CalRSDataSizeStep] xRankSize=[%u],yRankSize=[%u],zRankSize=[%u],",xRankSize,yRankSize,zRankSize);
    int zConnerStep=cornerStep;
    int xyConnerStep=0;
    int xInCornerStep=0;
    int yInCornerStep=0;
    //只存在一步和3步以上的情况
    if(outerStepNum>2)
    {
        xyConnerStep=outerStepNum-zConnerStep-1;
    }
    if(innerStepNum>2)//这里判断下，步数为1的时候只进下面的循环，否则这里走一步
    {
        xInCornerStep=1;//步数为1的时候只走一步，否则走innerStepNum-1步
        yInCornerStep=innerStepNum-2;//步数为1的时候只走一步，否则走innerStepNum-2步
    }
    HCCL_INFO("[CalcDataSliceInfo.cc][CalRSDataSizeStep] xInCornerStep=[%u],yInCornerStep=[%u],cornerStep=[%u],",xInCornerStep,yInCornerStep,cornerStep);

    //斜对角需要计算多片
    for(int osn=0;osn<zConnerStep;osn++)
    {
        rsStepDataSize[2].push_back(zRSDataSize[osn]*(xRankSize*yRankSize-1));
    }
    //同轴只需计算一片
    for(int osn=zConnerStep;osn<outerStepNum;osn++)
    {
        rsStepDataSize[2].push_back(zRSDataSize[osn]);
    }
    //x
    for(int osn=0;osn<xyConnerStep;osn++)
    {
        for(int isn=0;isn<xInCornerStep;isn++)
        {
            if(yRankSize>1)
            {
                rsStepDataSize[0].push_back(xRSDataSize[osn*maxStepNum+isn]*(zRankSize-1)*(yRankSize-1));
            }
            else
            {
                rsStepDataSize[0].push_back(xRSDataSize[osn*maxStepNum+isn]*(zRankSize-1));
            }
        }
        for(int isn=xInCornerStep;isn<innerStepNum;isn++)
        {
            if(yRankSize>1)
            {
               rsStepDataSize[0].push_back(xRSDataSize[osn*maxStepNum+isn]*(zRankSize-1));
            }
            else
            {
                rsStepDataSize[0].push_back(xRSDataSize[osn*maxStepNum+isn]*(zRankSize-1));
            }
        }
    }
    for(int osn=xyConnerStep;osn<outerStepNum;osn++)
    {
        for(int isn=0;isn<xInCornerStep;isn++)
        {
            if(yRankSize>1)
            {
                rsStepDataSize[0].push_back(xRSDataSize[osn*maxStepNum+isn]*(yRankSize-1));
            }
            else
            {
                rsStepDataSize[0].push_back(xRSDataSize[osn*maxStepNum+isn]);
            }
        }
        for(int isn=xInCornerStep;isn<innerStepNum;isn++)
        {
            if(yRankSize>1)
            {
                rsStepDataSize[0].push_back(xRSDataSize[osn*maxStepNum+isn]);
            }
            else
            {
                rsStepDataSize[0].push_back(xRSDataSize[osn*maxStepNum+isn]);
            }
        }
    }

    //y
    for(int osn=0;osn<xyConnerStep;osn++)
    {
        for(int isn=0;isn<yInCornerStep;isn++)
        {
            if(xRankSize>1)
            {
                rsStepDataSize[1].push_back(yRSDataSize[osn*maxStepNum+isn]*(zRankSize-1)*(xRankSize-1));
            }
            else
            {
                rsStepDataSize[1].push_back(yRSDataSize[osn*maxStepNum+isn]*(zRankSize-1));
            }
        }
        for(int isn=yInCornerStep;isn<innerStepNum;isn++)
        {
            if(xRankSize>1)
            {
                rsStepDataSize[1].push_back(yRSDataSize[osn*maxStepNum+isn]*(zRankSize-1));
            }
            else
            {
                rsStepDataSize[1].push_back(yRSDataSize[osn*maxStepNum+isn]*(zRankSize-1));
            }
        }
    }
    for(int osn=xyConnerStep;osn<outerStepNum;osn++)
    {
        for(int isn=0;isn<yInCornerStep;isn++)
        {
            if(xRankSize>1)
            {
               rsStepDataSize[1].push_back(yRSDataSize[osn*maxStepNum+isn]*(xRankSize-1));
            }
            else
            {
                rsStepDataSize[1].push_back(yRSDataSize[osn*maxStepNum+isn]);
            }
        }
        for(int isn=yInCornerStep;isn<innerStepNum;isn++)
        {
            if(xRankSize>1)
            {
                rsStepDataSize[1].push_back(yRSDataSize[osn*maxStepNum+isn]);
            }
            else
            {
                rsStepDataSize[1].push_back(yRSDataSize[osn*maxStepNum+isn]);
            }
        }
    }

    for(int i=0;i<outerStepNum;i++)
    {
        HCCL_INFO("[CalcDataSliceInfo.cc][CalRSDataSizeStep] rsStepDataSize[2][%d]=[%u],",i,rsStepDataSize[2][i]);
    }
    for(int i=0;i<outerStepNum*innerStepNum;i++)
    {
        HCCL_INFO("[CalcDataSliceInfo.cc][CalRSDataSizeStep] rsStepDataSize[0][%d]=[%u],",i,rsStepDataSize[0][i]);
    }
    for(int i=0;i<outerStepNum*innerStepNum;i++)
    {
        HCCL_INFO("[CalcDataSliceInfo.cc][CalRSDataSizeStep] rsStepDataSize[1][%d]=[%u],",i,rsStepDataSize[1][i]);
    }
    HCCL_INFO("[CalcDataSliceInfo.cc][CalRSDataSizeStep] end");
    return rsStepDataSize;
}

//ag数据偏移计算，可能还需要赋值inputSliceStride，还需例子验证。优化点1，uI向uO拷贝可以和其他卡读写并发，省去本地拷贝的时间
//levelRankSize三轴大小（x,y,z），dataSize单卡数据量，dataSize数据类型大小用于计算count，endpointAttrBw三轴带宽（x,y,z），levelRankId三轴坐标（x,y,z）
OmniPipeSliceInfo CalcAGOmniPipeSliceInfo(OmniPipeSliceParam &omniPipeSliceParam)
{
    //公共拓扑参数
    HCCL_INFO("[CalcDataSliceInfo.cc][CalcAGOmniPipeSliceInfo] Run start");
    int maxStepNum=MAX_STEP_NUM;
    u64 processedDataEachRank=0;//预留偏移参数，现在填0
    std::vector<u64> levelRankSize=omniPipeSliceParam.levelRankSize;
    u64 dataSize=omniPipeSliceParam.dataWholeSize;
    u64 dataSizePerLoop=omniPipeSliceParam.dataSizePerLoop;
    u64 dataTypeSize=omniPipeSliceParam.dataTypeSize;
    std::vector<EndpointAttrBwCoeff> endpointAttrBw=omniPipeSliceParam.endpointAttrBw;
    std::vector<u64> levelRankId=omniPipeSliceParam.levelRankId;
    u64 xRankSize=levelRankSize[0];//x轴卡数，机内mesh
    u64 yRankSize=levelRankSize[1];//y轴卡数，机内clos
    u64 zRankSize=levelRankSize[2];//z轴卡数，机间
    double xB=endpointAttrBw[0]*1.0;
    double yB=endpointAttrBw[1]*1.0;
    double zB=endpointAttrBw[2]*1.0;
    double xyB=CalcBandwidth2D(xB,yB,xRankSize,yRankSize,maxStepNum);//2d等效带宽计算
    HCCL_INFO("[CalcDataSliceInfo.cc][CalcAGOmniPipeSliceInfo] xRankSize=[%u],yRankSize=[%u],zRankSize=[%u],",xRankSize,yRankSize,zRankSize);
    HCCL_INFO("[CalcDataSliceInfo.cc][CalcAGOmniPipeSliceInfo] xB=[%f],yB=[%f],zB=[%f],xyB=[%f]",xB,yB,zB,xyB);
    u64 xAxis=levelRankId[0];//当前卡x坐标
    u64 yAxis=levelRankId[1];//当前卡y坐标
    u64 zAxis=levelRankId[2];//当前卡z坐标
    u64 rankid;//当前卡rankid
    rankid=xAxis+yAxis*xRankSize+zAxis*xRankSize*yRankSize;//当前卡rankid计算
    HCCL_INFO("[CalcDataSliceInfo.cc][CalcAGOmniPipeSliceInfo] xAxis=[%u],yAxis=[%u],zAxis=[%u],rankid=[%u]",xAxis,yAxis,zAxis,rankid);
    u64 zAGDataSize[maxStepNum];//存数据片大小
    u64 xyAGDataSize[maxStepNum];
    u64 xAGDataSize[maxStepNum][maxStepNum];
    u64 yAGDataSize[maxStepNum][maxStepNum];
    u64 outerStepNum;//机内机间步数
    u64 innerStepNum;//机内两轴步数
    u64 zAGOffect[maxStepNum];//z轴偏移
    u64 xAGOffect[maxStepNum][maxStepNum];//x轴偏移
    u64 yAGOffect[maxStepNum][maxStepNum];//y轴偏移
    u64 xyAGOffect[maxStepNum];//xy整体偏移

    int zConnerStep=1;
    int xyConnerStep=1;
    int xInCornerStep=1;
    int yInCornerStep=1;
    //机内快和机间快分开写，但实际上这俩逻辑一样
    if(xyB>zB)
    {
        HCCL_INFO("[CalcDataSliceInfo.cc][CalcAGOmniPipeSliceInfo] xyB>zB");
        //先计算通信步数和每步每一小片数据量
        outerStepNum=CalAllgatherDataSize2D(zAGDataSize,xyAGDataSize,zB,xyB,zRankSize,xRankSize*yRankSize,dataSizePerLoop,maxStepNum);
        //这里认为y一定大
        for(u64 i=0;i<outerStepNum;i++)
        {
            innerStepNum=CalAllgatherDataSize2D(xAGDataSize[i],yAGDataSize[i],xB,yB,xRankSize,yRankSize,xyAGDataSize[i],maxStepNum);
        }
        //计算2d数据片的偏移，下面变成3d时用
        CalAllgather2DOffset(zAGOffect,xyAGOffect,outerStepNum,zRankSize,xRankSize*yRankSize,zAGDataSize,xyAGDataSize);

        if(outerStepNum>1)
        {
            zConnerStep=outerStepNum-1;
        }
        if(innerStepNum>1)//这里判断下，步数为1的时候只进上面的循环，否则这里走一步
        {
            xInCornerStep=innerStepNum-1;//步数为1的时候只走一步，否则走innerStepNum-1步
        }
        HCCL_INFO("[CalcDataSliceInfo.cc][CalcAGOmniPipeSliceInfo] xInCornerStep=[%u],yRanyInCornerStepkSize=[%u],zConnerStep=[%u],",xInCornerStep,yInCornerStep,zConnerStep);
    }
    else
    {
        HCCL_INFO("[CalcDataSliceInfo.cc][CalcAGOmniPipeSliceInfo] xyB<=zB");
        outerStepNum=CalAllgatherDataSize2D(xyAGDataSize,zAGDataSize,xyB,zB,xRankSize*yRankSize,zRankSize,dataSizePerLoop,maxStepNum);
        for(u64 i=0;i<outerStepNum;i++)
        {
            innerStepNum=CalAllgatherDataSize2D(xAGDataSize[i],yAGDataSize[i],xB,yB,xRankSize,yRankSize,xyAGDataSize[i],maxStepNum);
        }

        CalAllgather2DOffset(xyAGOffect,zAGOffect,outerStepNum,xRankSize*yRankSize,zRankSize,xyAGDataSize,zAGDataSize);

        if(outerStepNum>1)
        {
            xyConnerStep=outerStepNum-1;
        }
        if(innerStepNum>1)//这里判断下，步数为1的时候只进上面的循环，否则这里走一步
        {
            xInCornerStep=innerStepNum-1;//步数为1的时候只走一步，否则走innerStepNum-1步
        }
        HCCL_INFO("[CalcDataSliceInfo.cc][CalcAGOmniPipeSliceInfo] xInCornerStep=[%u],yRanyInCornerStepkSize=[%u],zConnerStep=[%u],",xInCornerStep,yInCornerStep,zConnerStep);
    }
        //z是慢轴，n-1步同轴+1步斜对角
    std::vector<StepSliceInfo> dataSliceLevelz;
    //z的同轴
    for(u64 osn=0;osn<zConnerStep;osn++)
    {
        struct StepSliceInfo stepSliceInfotmp;
        struct BuffInfo bitmp;
        //同轴数据搬运需要算算自己的rankid再加上偏移得到起始地址，x+y*xR)*dataSize
        BuffInfoAssign(bitmp,(xAxis+yAxis*xRankSize)*dataSize+processedDataEachRank+zAGOffect[osn],(xAxis+yAxis*xRankSize)*dataSize+processedDataEachRank+zAGOffect[osn],0);
        StepSliceInfoAssign(stepSliceInfotmp,bitmp,zAGDataSize[osn]/dataTypeSize,zAGDataSize[osn],xRankSize*yRankSize*dataSize,xRankSize*yRankSize*dataSize);
        stepSliceInfotmp.inputOmniPipeSliceStride.push_back(0);
        stepSliceInfotmp.outputOmniPipeSliceStride.push_back(0);
        dataSliceLevelz.insert(dataSliceLevelz.end(),stepSliceInfotmp);
    }
    //z的斜对角
    for(u64 osn=zConnerStep;osn<outerStepNum;osn++)
    {
        struct StepSliceInfo stepSliceInfotmp;
        struct BuffInfo bitmp;
        //给z轴上每卡发同xy轴的数据，所以从（0,0，z）的数据开始，发xRankSize*yRankSize-1片，，少的是自己那片，
        BuffInfoAssign(bitmp,processedDataEachRank+zAGOffect[osn],processedDataEachRank+zAGOffect[osn],0);
        StepSliceInfoAssign(stepSliceInfotmp,bitmp,zAGDataSize[osn]/dataTypeSize,zAGDataSize[osn],xRankSize*yRankSize*dataSize,xRankSize*yRankSize*dataSize);

        u64 slicestride=0;//计算发给同一个rank的多片数据的位置

        for(u64 connerDataSlice=0;connerDataSlice<xRankSize*yRankSize;connerDataSlice++)
        {
            u64 currentDataSliceId=zAxis*xRankSize*yRankSize+connerDataSlice;//斜对角先算算是哪一片
            if(currentDataSliceId==rankid)
            {
                slicestride=slicestride+dataSize;//斜对角数据不包含自己的数据。
                // zAGDataStartAddr[osn][connerDataSlice]=-1;
            }
            else
            {
                stepSliceInfotmp.inputOmniPipeSliceStride.insert(stepSliceInfotmp.inputOmniPipeSliceStride.end(),slicestride);
                stepSliceInfotmp.outputOmniPipeSliceStride.insert(stepSliceInfotmp.outputOmniPipeSliceStride.end(),slicestride);
                slicestride=slicestride+dataSize;
            }
        }
        dataSliceLevelz.insert(dataSliceLevelz.end(),stepSliceInfotmp);
    }
    //x轴y轴2d偏移，就正常2d
    for(u64 osn=0;osn<outerStepNum;osn++)
    {
        CalAllgather2DOffset(xAGOffect[osn],yAGOffect[osn],innerStepNum,xRankSize,yRankSize,xAGDataSize[osn],yAGDataSize[osn]);
    }
    //算x轴偏移
    //机内快，前1步只有同轴，一片数据2d
    std::vector<StepSliceInfo> dataSliceLevelx;
    for(u64 osn=0;osn<xyConnerStep;osn++)
    {
        //x轴比y轴慢，前n-1步只有同轴
        for(u64 isn=0;isn<xInCornerStep;isn++)
        {
            struct StepSliceInfo stepSliceInfotmp;
            struct BuffInfo bitmp;
            //同轴，所以自己那片数据加偏移
            BuffInfoAssign(bitmp,(zAxis*xRankSize*yRankSize+yAxis*xRankSize)*dataSize+processedDataEachRank+xyAGOffect[osn]+xAGOffect[osn][isn],(zAxis*xRankSize*yRankSize+yAxis*xRankSize)*dataSize+processedDataEachRank+xyAGOffect[osn]+xAGOffect[osn][isn],0);
            StepSliceInfoAssign(stepSliceInfotmp,bitmp,xAGDataSize[osn][isn]/dataTypeSize,xAGDataSize[osn][isn],dataSize,dataSize);
            stepSliceInfotmp.inputOmniPipeSliceStride.push_back(0);
            stepSliceInfotmp.outputOmniPipeSliceStride.push_back(0);

            dataSliceLevelx.insert(dataSliceLevelx.end(),stepSliceInfotmp);
        }

        //x轴比y轴慢，1步斜对角
        for(u64 isn=xInCornerStep;isn<innerStepNum;isn++)
        {
            struct StepSliceInfo stepSliceInfotmp;
            struct BuffInfo bitmp;
            //向x轴卡转发y轴的数据片
            BuffInfoAssign(bitmp,(zAxis*xRankSize*yRankSize)*dataSize+processedDataEachRank+xyAGOffect[osn]+xAGOffect[osn][isn],(zAxis*xRankSize*yRankSize)*dataSize+processedDataEachRank+xyAGOffect[osn]+xAGOffect[osn][isn],0);
            // BuffInfoAssign(bitmp,(zAxis*xRankSize*yRankSize+xAxis)*dataSize+xyAGOffect[osn]+xAGOffect[osn][isn],(zAxis*xRankSize*yRankSize+xAxis)*dataSize+xyAGOffect[osn]+xAGOffect[osn][isn],0);
            StepSliceInfoAssign(stepSliceInfotmp,bitmp,xAGDataSize[osn][isn]/dataTypeSize,xAGDataSize[osn][isn],dataSize,dataSize);

            u64 slicestride=0;//0卡从1开始，其他卡从0开始，所以要单独判断
            for(u64 connerDataSlice=0;connerDataSlice<yRankSize;connerDataSlice++)
            {
                u64 currentDataSliceId=zAxis*xRankSize*yRankSize+connerDataSlice*xRankSize+xAxis;//斜对角先算算是哪一片
                if(currentDataSliceId==rankid)
                {
                    slicestride=slicestride+xRankSize*dataSize;//自己那片不做
                }
                else
                {
                    stepSliceInfotmp.inputOmniPipeSliceStride.insert(stepSliceInfotmp.inputOmniPipeSliceStride.end(),slicestride);
                    stepSliceInfotmp.outputOmniPipeSliceStride.insert(stepSliceInfotmp.outputOmniPipeSliceStride.end(),slicestride);
                    slicestride=slicestride+xRankSize*dataSize;
                }
            }
            dataSliceLevelx.insert(dataSliceLevelx.end(),stepSliceInfotmp);
        }
    }
    //第n步斜对角，zRankSize-1片做2d
    for(u64 osn=xyConnerStep;osn<outerStepNum;osn++)
    {
        for(u64 isn=0;isn<xInCornerStep;isn++)
        {
            struct StepSliceInfo stepSliceInfotmp;
            struct BuffInfo bitmp;
            //这zRankSize-1片中的第一片在（x,y,0）的位置
            BuffInfoAssign(bitmp,(yAxis*xRankSize)*dataSize+processedDataEachRank+xyAGOffect[osn]+xAGOffect[osn][isn],(yAxis*xRankSize)*dataSize+processedDataEachRank+xyAGOffect[osn]+xAGOffect[osn][isn],0);
            // BuffInfoAssign(bitmp,(yAxis*xRankSize+xAxis)*dataSize+xyAGOffect[osn]+xAGOffect[osn][isn],(yAxis*xRankSize+xAxis)*dataSize+xyAGOffect[osn]+xAGOffect[osn][isn],0);
            StepSliceInfoAssign(stepSliceInfotmp,bitmp,xAGDataSize[osn][isn]/dataTypeSize,xAGDataSize[osn][isn],dataSize,dataSize);
            u64 slicestride=0;
            //zRankSize-1片做2d
            for(u64 outSliceNum=0;outSliceNum<zRankSize;outSliceNum++)
            {
                u64 currentDataSliceId=outSliceNum*xRankSize*yRankSize+yAxis*xRankSize+xAxis;//算算是zRankSize-1中的哪一片
                if(currentDataSliceId==rankid)//自己的不做，只做斜对角的
                {
                    slicestride=slicestride+xRankSize*yRankSize*dataSize;
                }
                else
                {
                    stepSliceInfotmp.inputOmniPipeSliceStride.insert(stepSliceInfotmp.inputOmniPipeSliceStride.end(),slicestride);
                    stepSliceInfotmp.outputOmniPipeSliceStride.insert(stepSliceInfotmp.outputOmniPipeSliceStride.end(),slicestride);
                    slicestride=slicestride+xRankSize*yRankSize*dataSize;
                }
            }
            dataSliceLevelx.insert(dataSliceLevelx.end(),stepSliceInfotmp);
        }
        for(u64 isn=xInCornerStep;isn<innerStepNum;isn++)
        {
            struct StepSliceInfo stepSliceInfotmp;
            struct BuffInfo bitmp;
            //
            BuffInfoAssign(bitmp,processedDataEachRank+xyAGOffect[osn]+xAGOffect[osn][isn],processedDataEachRank+xyAGOffect[osn]+xAGOffect[osn][isn],0);
            StepSliceInfoAssign(stepSliceInfotmp,bitmp,xAGDataSize[osn][isn]/dataTypeSize,xAGDataSize[osn][isn],dataSize,dataSize);
            u64 slicestride=0;
            for(u64 outSliceNum=0;outSliceNum<zRankSize;outSliceNum++)
            {
                u64 currentDataSliceId=outSliceNum*xRankSize*yRankSize+yAxis*xRankSize+xAxis;//算算是zRankSize-1中的哪一片
                if(currentDataSliceId==rankid)//自己的不做，只做斜对角的
                {
                    slicestride=slicestride+xRankSize*yRankSize*dataSize;
                }
                else
                {
                    for(u64 connerDataSlice=0;connerDataSlice<yRankSize;connerDataSlice++)
                    {
                        u64 currentInnerStepDataSliceId=outSliceNum*xRankSize*yRankSize+connerDataSlice*xRankSize+xAxis;//算算是机内斜对角中的哪一片
                        if(currentInnerStepDataSliceId==currentDataSliceId && yRankSize>1)//自己的不做，只做斜对角的
                        {
                            slicestride=slicestride+xRankSize*dataSize;
                        }
                        else
                        {
                            stepSliceInfotmp.inputOmniPipeSliceStride.insert(stepSliceInfotmp.inputOmniPipeSliceStride.end(),slicestride);
                            stepSliceInfotmp.outputOmniPipeSliceStride.insert(stepSliceInfotmp.outputOmniPipeSliceStride.end(),slicestride);
                            slicestride=slicestride+xRankSize*dataSize;
                        }
                    }
                }
            }
            dataSliceLevelx.insert(dataSliceLevelx.end(),stepSliceInfotmp);
        }
    }
    //算y轴偏移,和计算xy的相似
    std::vector<StepSliceInfo> dataSliceLevely;
    for(u64 osn=0;osn<xyConnerStep;osn++)
    {
        //前1步只有同轴，一片数据2d
        for(u64 isn=0;isn<yInCornerStep;isn++)
        {
            struct StepSliceInfo stepSliceInfotmp;
            struct BuffInfo bitmp;
            //
            int startOffset=(zAxis*xRankSize*yRankSize+xAxis)*dataSize+processedDataEachRank+xyAGOffect[osn]+yAGOffect[osn][isn];
            BuffInfoAssign(bitmp,startOffset,startOffset,0);
            StepSliceInfoAssign(stepSliceInfotmp,bitmp,yAGDataSize[osn][isn]/dataTypeSize,yAGDataSize[osn][isn],xRankSize*dataSize,xRankSize*dataSize);
            stepSliceInfotmp.inputOmniPipeSliceStride.push_back(0);
            stepSliceInfotmp.outputOmniPipeSliceStride.push_back(0);
            dataSliceLevely.insert(dataSliceLevely.end(),stepSliceInfotmp);
        }
        for(u64 isn=yInCornerStep;isn<innerStepNum;isn++)
        {
            struct StepSliceInfo stepSliceInfotmp;
            struct BuffInfo bitmp;

            int startOffset=(zAxis*xRankSize*yRankSize)*dataSize+processedDataEachRank+xyAGOffect[osn]+yAGOffect[osn][isn];
            BuffInfoAssign(bitmp,startOffset,startOffset,0);
            StepSliceInfoAssign(stepSliceInfotmp,bitmp,yAGDataSize[osn][isn]/dataTypeSize,yAGDataSize[osn][isn],xRankSize*dataSize,xRankSize*dataSize);
            u64 slicestride=0;
            // int firstSliceMark=0;
            for(u64 connerDataSlice=0;connerDataSlice<xRankSize;connerDataSlice++)
            {
                u64 currentDataSliceId=zAxis*xRankSize*yRankSize+yAxis*xRankSize+connerDataSlice;//斜对角先算算是哪一片
                if(currentDataSliceId==rankid)
                {
                    slicestride=slicestride+dataSize;//自己那片不做
                }
                else
                {
                    stepSliceInfotmp.inputOmniPipeSliceStride.insert(stepSliceInfotmp.inputOmniPipeSliceStride.end(),slicestride);
                    stepSliceInfotmp.outputOmniPipeSliceStride.insert(stepSliceInfotmp.outputOmniPipeSliceStride.end(),slicestride);
                    slicestride=slicestride+dataSize;
                }
            }
            dataSliceLevely.insert(dataSliceLevely.end(),stepSliceInfotmp);
        }
    }
    //n-1步斜对角，zRankSize-1片做2d
    for(u64 osn=xyConnerStep;osn<outerStepNum;osn++)
    {
        for(u64 isn=0;isn<yInCornerStep;isn++)
        {
            struct StepSliceInfo stepSliceInfotmp;
            struct BuffInfo bitmp;
            //

            int startOffset=(xAxis)*dataSize+processedDataEachRank+xyAGOffect[osn]+yAGOffect[osn][isn];
            BuffInfoAssign(bitmp,startOffset,startOffset,0);
            StepSliceInfoAssign(stepSliceInfotmp,bitmp,yAGDataSize[osn][isn]/dataTypeSize,yAGDataSize[osn][isn],xRankSize*dataSize,xRankSize*dataSize);
            u64 slicestride=0;
            for(u64 outSliceNum=0;outSliceNum<zRankSize;outSliceNum++)
            {
                u64 currentDataSliceId=outSliceNum*xRankSize*yRankSize+yAxis*xRankSize+xAxis;//算算是zRankSize-1中的哪一片
                if(currentDataSliceId==rankid)//自己的不做，只做斜对角的
                {
                    slicestride=slicestride+xRankSize*yRankSize*dataSize;
                }
                else
                {
                    stepSliceInfotmp.inputOmniPipeSliceStride.insert(stepSliceInfotmp.inputOmniPipeSliceStride.end(),slicestride);
                    stepSliceInfotmp.outputOmniPipeSliceStride.insert(stepSliceInfotmp.outputOmniPipeSliceStride.end(),slicestride);
                    slicestride=slicestride+xRankSize*yRankSize*dataSize;
                }
            }
            dataSliceLevely.insert(dataSliceLevely.end(),stepSliceInfotmp);
        }
        for(u64 isn=yInCornerStep;isn<innerStepNum;isn++)
        {
            struct StepSliceInfo stepSliceInfotmp;
            struct BuffInfo bitmp;
            //
            int startOffset=processedDataEachRank+xyAGOffect[osn]+yAGOffect[osn][isn];
            BuffInfoAssign(bitmp,startOffset,startOffset,0);
            StepSliceInfoAssign(stepSliceInfotmp,bitmp,yAGDataSize[osn][isn]/dataTypeSize,yAGDataSize[osn][isn],xRankSize*dataSize,xRankSize*dataSize);
            u64 slicestride=0;
            // int firstSliceMark=0;
            for(u64 outSliceNum=0;outSliceNum<zRankSize;outSliceNum++)
            {
                u64 currentDataSliceId=outSliceNum*xRankSize*yRankSize+yAxis*xRankSize+xAxis;//算算是zRankSize-1中的哪一片
                if(currentDataSliceId==rankid)//自己的不做，只做斜对角的
                {
                    slicestride=slicestride+xRankSize*yRankSize*dataSize;
                }
                else
                {
                    slicestride=outSliceNum*xRankSize*yRankSize*dataSize;
                    for(u64 connerDataSlice=0;connerDataSlice<xRankSize;connerDataSlice++)
                    {
                        u64 currentInnerStepDataSliceId=outSliceNum*xRankSize*yRankSize+yAxis*xRankSize+connerDataSlice;;//算算是机内斜对角中的哪一片
                        if(currentInnerStepDataSliceId==currentDataSliceId && xRankSize>1)//自己的不做，只做斜对角的
                        {
                            slicestride=slicestride+dataSize;
                        }
                        else
                        {
                            stepSliceInfotmp.inputOmniPipeSliceStride.insert(stepSliceInfotmp.inputOmniPipeSliceStride.end(),slicestride);
                            stepSliceInfotmp.outputOmniPipeSliceStride.insert(stepSliceInfotmp.outputOmniPipeSliceStride.end(),slicestride);
                            slicestride=slicestride+dataSize;
                        }
                    }
                }
            }
            dataSliceLevely.insert(dataSliceLevely.end(),stepSliceInfotmp);
        }
    }
    struct OmniPipeSliceInfo dataSliceInfoxyz;
    dataSliceInfoxyz.dataSliceLevel0=dataSliceLevelx;
    dataSliceInfoxyz.dataSliceLevel1=dataSliceLevely;
    dataSliceInfoxyz.dataSliceLevel2=dataSliceLevelz;

    return dataSliceInfoxyz;
}

//rs数据偏移计算。存疑，nhr轴间reduce是否需要单独列出buffer地址，
//levelRankSize三轴大小，dataSize单卡数据量，endpointAttrBw三轴带宽，levelRankId三轴坐标
OmniPipeSliceInfo CalcRSOmniPipeSliceInfo(OmniPipeSliceParam &omniPipeSliceParam)
{
    u64 processedDataEachRank=0;//预留偏移参数，现在填0
    //公共拓扑参数
    int maxStepNum=MAX_STEP_NUM+1;//rs多拆了一步出来所以是+1
    // uint32_t dataTypeSize=dataType;
    std::vector<u64> levelRankSize=omniPipeSliceParam.levelRankSize;
    u64 dataSize=omniPipeSliceParam.dataWholeSize;
    u64 dataSizePerLoop=omniPipeSliceParam.dataSizePerLoop;
    u64 dataTypeSize=omniPipeSliceParam.dataTypeSize;
    std::vector<EndpointAttrBwCoeff> endpointAttrBw=omniPipeSliceParam.endpointAttrBw;
    std::vector<u64> levelRankId=omniPipeSliceParam.levelRankId;
    u64 xRankSize=levelRankSize[0];//x轴卡数
    u64 yRankSize=levelRankSize[1];//y轴卡数
    u64 zRankSize=levelRankSize[2];//z轴卡数
    u64 scratchBaseOffSet=0;//scratch后面根据拓扑改
    double xB=endpointAttrBw[0];
    double yB=endpointAttrBw[1];
    double zB=endpointAttrBw[2];
    double xyB=CalcBandwidth2D(xB,yB,xRankSize,yRankSize,maxStepNum-1);//算的时候减去多拆的一步
    u64 xAxis=levelRankId[0];//当前卡x坐标
    u64 yAxis=levelRankId[1];//当前卡y坐标
    u64 zAxis=levelRankId[2];//当前卡z坐标
    u64 rankid;//当前卡rankid
    rankid=xAxis+yAxis*xRankSize+zAxis*xRankSize*yRankSize;//当前卡rankid计算
    u64 zRSDataSize[maxStepNum];
    u64 xyRSDataSize[maxStepNum];
    u64 xRSDataSize[maxStepNum][maxStepNum];
    u64 yRSDataSize[maxStepNum][maxStepNum];
    u64 outerStepNum;//机内机间步数
    u64 innerStepNum;//机内两轴步数
    u64 zRSOffect[maxStepNum];//z轴偏移
    u64 xRSOffect[maxStepNum][maxStepNum];//x轴偏移
    u64 yRSOffect[maxStepNum][maxStepNum];//y轴偏移
    u64 xyRSOffect[maxStepNum];//xy整体偏移
    std::vector <std::vector <std::vector<u64>>> axlesReduceDstAddr;
    //buffer分4块，第一块放自己的数据，2.3.4块放别人x.y.z发来的数据，起始地址为xCclBufferBaseOff,yCclBufferBaseOff,zCclBufferBaseOff补充buffer基础偏移计算
    u64 xCclBufferBaseOff=0;
    u64 yCclBufferBaseOff=0;
    u64 zCclBufferBaseOff=0;
    int zConnerStep=0;
    int xyConnerStep=0;
    int xInCornerStep=0;
    int yInCornerStep=0;
    std::vector <std::vector<u64>> xyzDataSizeStep;
    std::vector<u64> scratchSizexyz;
    //z的斜对角，斜对角是给同z轴的每一张卡转发同xy的数据。
    if(xyB>zB)
    {
        //z是慢轴，n-1步同轴+1步斜对角
        outerStepNum=CalReducescatterDataSize2D(zRSDataSize,xyRSDataSize,zB,xyB,zRankSize,xRankSize*yRankSize,dataSizePerLoop,maxStepNum);
        for(u64 i=0;i<outerStepNum;i++)
        {
            innerStepNum=CalReducescatterDataSize2D(xRSDataSize[i],yRSDataSize[i],xB,yB,xRankSize,yRankSize,xyRSDataSize[i],maxStepNum);
        }
        CalReducescatter2DOffset(zRSOffect,xyRSOffect,outerStepNum,zRankSize,xRankSize*yRankSize,zRSDataSize,xyRSDataSize);

        if(outerStepNum>2)
        {
            zConnerStep=1;
            xyConnerStep=outerStepNum-2;
        }
        if(innerStepNum>2)//这里判断下，步数为1的时候只进下面的循环，否则这里走一步
        {
            xInCornerStep=1;//步数为1的时候只走一步，否则走innerStepNum-1步
            yInCornerStep=innerStepNum-2;//步数为1的时候只走一步，否则走innerStepNum-2步
        }
        scratchSizexyz=CalScratchSize((u64*)xRSDataSize,(u64*)yRSDataSize,zRSDataSize,levelRankSize,zConnerStep,outerStepNum,innerStepNum,maxStepNum,omniPipeSliceParam.levelAlgType,omniPipeSliceParam.engine);
        xyzDataSizeStep=CalRSDataSizeStep((u64*)xRSDataSize,(u64*)yRSDataSize,zRSDataSize,levelRankSize,zConnerStep,outerStepNum,innerStepNum,maxStepNum);\
    }
    else
    {
        outerStepNum=CalReducescatterDataSize2D(xyRSDataSize,zRSDataSize,xyB,zB,xRankSize*yRankSize,zRankSize,dataSizePerLoop,maxStepNum);
        for(u64 i=0;i<outerStepNum;i++)
        {
            innerStepNum=CalReducescatterDataSize2D(xRSDataSize[i],yRSDataSize[i],xB,yB,xRankSize,yRankSize,xyRSDataSize[i],maxStepNum);
        }
        CalReducescatter2DOffset(xyRSOffect,zRSOffect,outerStepNum,xRankSize*yRankSize,zRankSize,xyRSDataSize,zRSDataSize);

        if(outerStepNum>2)
        {
            zConnerStep=outerStepNum-2;
            xyConnerStep=1;
        }
        if(innerStepNum>2)//这里判断下，步数为1的时候只进下面的循环，否则这里走一步
        {
            xInCornerStep=1;//步数为1的时候只走一步，否则走innerStepNum-1步
            yInCornerStep=innerStepNum-2;//步数为1的时候只走一步，否则走innerStepNum-2步
        }
        scratchSizexyz=CalScratchSize((u64*)xRSDataSize,(u64*)yRSDataSize,zRSDataSize,levelRankSize,zConnerStep,outerStepNum,innerStepNum,maxStepNum,omniPipeSliceParam.levelAlgType,omniPipeSliceParam.engine);
        xyzDataSizeStep=CalRSDataSizeStep((u64*)xRSDataSize,(u64*)yRSDataSize,zRSDataSize,levelRankSize,zConnerStep,outerStepNum,innerStepNum,maxStepNum);
    }
    if(omniPipeSliceParam.opMode==OpMode::OPBASE && omniPipeSliceParam.engine==CommEngine::COMM_ENGINE_AICPU_TS)
    {
        xCclBufferBaseOff=dataSizePerLoop*xRankSize*yRankSize*zRankSize;
        HCCL_INFO("xCclBufferBaseOff=%u, dataSizePerLoop=%u, xRankSize=%u, yRankSize=%u, zRankSize=%u", xCclBufferBaseOff, dataSizePerLoop, xRankSize, yRankSize, zRankSize);
    }
    yCclBufferBaseOff=xCclBufferBaseOff+scratchSizexyz[0];
    zCclBufferBaseOff=yCclBufferBaseOff+scratchSizexyz[1];

    std::vector<StepSliceInfo> dataSliceLevelz;
    for(u64 osn=0;osn<zConnerStep;osn++)
    {
        struct StepSliceInfo stepSliceInfotmp;
        struct BuffInfo bitmp;
        u64 inOutOffset=processedDataEachRank+zRSOffect[osn];
        BuffInfoAssign(bitmp,inOutOffset,inOutOffset,zCclBufferBaseOff);
        StepSliceInfoAssign(stepSliceInfotmp,bitmp,zRSDataSize[osn]/dataTypeSize,zRSDataSize[osn],xRankSize*yRankSize*dataSize,xyzDataSizeStep[2][osn]);

        u64 inputslicestride=0;
        u64 outputslicestride=0;
        for(u64 connerDataSlice=0;connerDataSlice<xRankSize*yRankSize;connerDataSlice++)
        {
            u64 currentDataSliceId=zAxis*xRankSize*yRankSize+connerDataSlice;//斜对角先算算是哪一片
            if(currentDataSliceId==rankid)
            {
                inputslicestride=inputslicestride+dataSize;
            }
            else
            {
                stepSliceInfotmp.inputOmniPipeSliceStride.insert(stepSliceInfotmp.inputOmniPipeSliceStride.end(),inputslicestride);
                stepSliceInfotmp.outputOmniPipeSliceStride.insert(stepSliceInfotmp.outputOmniPipeSliceStride.end(),outputslicestride);
                inputslicestride+=dataSize;
                outputslicestride+=zRSDataSize[osn];
            }
        }
        dataSliceLevelz.insert(dataSliceLevelz.end(),stepSliceInfotmp);
    }
    for(u64 osn=zConnerStep;osn<outerStepNum;osn++)
    {
        //z的同轴
        struct StepSliceInfo stepSliceInfotmp;
        struct BuffInfo bitmp;

        u64 inOutOffset=(yAxis*xRankSize+xAxis)*dataSize+processedDataEachRank+zRSOffect[osn];
        BuffInfoAssign(bitmp,inOutOffset,inOutOffset,zCclBufferBaseOff);
        StepSliceInfoAssign(stepSliceInfotmp,bitmp,zRSDataSize[osn]/dataTypeSize,zRSDataSize[osn],xRankSize*yRankSize*dataSize,xyzDataSizeStep[2][osn]);
        stepSliceInfotmp.inputOmniPipeSliceStride.push_back(0);
        stepSliceInfotmp.outputOmniPipeSliceStride.push_back(0);
        dataSliceLevelz.insert(dataSliceLevelz.end(),stepSliceInfotmp);
    }

    //x轴y轴2d偏移，就正常2d
    for(u64 osn=0;osn<outerStepNum;osn++)
    {
        CalReducescatter2DOffset(xRSOffect[osn],yRSOffect[osn],innerStepNum,xRankSize,yRankSize,xRSDataSize[osn],yRSDataSize[osn]);
    }
    //算x轴偏移
    //前1步只有斜对角，多片数据2d
    std::vector<StepSliceInfo> dataSliceLevelx;
    for(u64 osn=0;osn<xyConnerStep;osn++)
    {
        for(u64 isn=0;isn<xInCornerStep;isn++)
        {
            struct StepSliceInfo stepSliceInfotmp;
            struct BuffInfo bitmp;
            u64 inOutOffset=processedDataEachRank+xyRSOffect[osn]+xRSOffect[osn][isn];
            BuffInfoAssign(bitmp,inOutOffset,inOutOffset,xCclBufferBaseOff);
            StepSliceInfoAssign(stepSliceInfotmp,bitmp,xRSDataSize[osn][isn]/dataTypeSize,xRSDataSize[osn][isn],dataSize,xyzDataSizeStep[0][osn*innerStepNum+isn]);

            u64 inputslicestride=0;
            u64 outputslicestride=0;
            int firstSliceMark=0;//这里得单独看是第一片开始还是第0片开始
            // dataSliceLevelx.insert(stepSliceInfotmp);

            for(u64 outSliceNum=0;outSliceNum<zRankSize;outSliceNum++)
            {
                u64 currentDataSliceId=outSliceNum*xRankSize*yRankSize+yAxis*xRankSize+xAxis;//算算是zRankSize-1中的哪一片
                if(currentDataSliceId==rankid)//自己的不做，只做斜对角的
                {
                    inputslicestride=inputslicestride+xRankSize*yRankSize*dataSize;
                }
                else
                {
                    for(u64 connerDataSlice=0;connerDataSlice<yRankSize;connerDataSlice++)
                    {
                        u64 currentInnerStepDataSliceId=connerDataSlice*xRankSize+xAxis;//算算是机内斜对角中的哪一片
                        if(currentInnerStepDataSliceId==yAxis*xRankSize+xAxis && yRankSize>1)//自己的不做，只做斜对角的
                        {
                            inputslicestride=inputslicestride+xRankSize*dataSize;
                        }
                        else
                        {
                            stepSliceInfotmp.inputOmniPipeSliceStride.insert(stepSliceInfotmp.inputOmniPipeSliceStride.end(),inputslicestride);
                            stepSliceInfotmp.outputOmniPipeSliceStride.insert(stepSliceInfotmp.outputOmniPipeSliceStride.end(),outputslicestride);
                            inputslicestride+=xRankSize*dataSize;
                            outputslicestride+=xRSDataSize[osn][isn];
                        }
                    }
                }
            }
            dataSliceLevelx.insert(dataSliceLevelx.end(),stepSliceInfotmp);
        }
        for(u64 isn=xInCornerStep;isn<innerStepNum;isn++)
        {
            struct StepSliceInfo stepSliceInfotmp;
            struct BuffInfo bitmp;
            u64 inOutOffset=(yAxis*xRankSize)*dataSize+processedDataEachRank+xyRSOffect[osn]+xRSOffect[osn][isn];
            BuffInfoAssign(bitmp,inOutOffset,inOutOffset,xCclBufferBaseOff);
            StepSliceInfoAssign(stepSliceInfotmp,bitmp,xRSDataSize[osn][isn]/dataTypeSize,xRSDataSize[osn][isn],dataSize,xyzDataSizeStep[0][osn*innerStepNum+isn]);
            u64 inputslicestride=0;
            u64 outputslicestride=0;
            for(u64 outSliceNum=0;outSliceNum<zRankSize;outSliceNum++)
            {
                u64 currentDataSliceId=outSliceNum*xRankSize*yRankSize+yAxis*xRankSize+xAxis;//算算是zRankSize-1中的哪一片
                if(currentDataSliceId==rankid)//自己的不做，只做斜对角的
                {
                    inputslicestride=inputslicestride+xRankSize*yRankSize*dataSize;

                }
                else
                {

                    stepSliceInfotmp.inputOmniPipeSliceStride.insert(stepSliceInfotmp.inputOmniPipeSliceStride.end(),inputslicestride);
                    stepSliceInfotmp.outputOmniPipeSliceStride.insert(stepSliceInfotmp.outputOmniPipeSliceStride.end(),outputslicestride);
                    inputslicestride+=xRankSize*yRankSize*dataSize;
                    outputslicestride+=xRSDataSize[osn][isn];
                }
            }
            dataSliceLevelx.insert(dataSliceLevelx.end(),stepSliceInfotmp);
        }
    }
    for(u64 osn=xyConnerStep;osn<outerStepNum;osn++)
    {
        for(u64 isn=0;isn<xInCornerStep;isn++)
        {
            struct StepSliceInfo stepSliceInfotmp;
            struct BuffInfo bitmp;
            u64 inOutOffset=zAxis*xRankSize*yRankSize*dataSize+processedDataEachRank+xyRSOffect[osn]+xRSOffect[osn][isn];
            BuffInfoAssign(bitmp,inOutOffset,inOutOffset,xCclBufferBaseOff);
            StepSliceInfoAssign(stepSliceInfotmp,bitmp,xRSDataSize[osn][isn]/dataTypeSize,xRSDataSize[osn][isn],dataSize,xyzDataSizeStep[0][osn*innerStepNum+isn]);
            u64 inputslicestride=0;
            u64 outputslicestride=0;
            for(u64 connerDataSlice=0;connerDataSlice<yRankSize;connerDataSlice++)
            {
                u64 currentDataSliceId=zAxis*xRankSize*yRankSize+connerDataSlice*xRankSize+xAxis;//斜对角先算算是哪一片
                if(currentDataSliceId==rankid)
                {
                    inputslicestride=inputslicestride+xRankSize*dataSize;
                }
                else
                {
                    stepSliceInfotmp.inputOmniPipeSliceStride.insert(stepSliceInfotmp.inputOmniPipeSliceStride.end(),inputslicestride);
                    stepSliceInfotmp.outputOmniPipeSliceStride.insert(stepSliceInfotmp.outputOmniPipeSliceStride.end(),outputslicestride);
                    inputslicestride+=xRankSize*dataSize;
                    outputslicestride+=xRSDataSize[osn][isn];
                }
            }
            dataSliceLevelx.insert(dataSliceLevelx.end(),stepSliceInfotmp);
        }
        for(u64 isn=xInCornerStep;isn<innerStepNum;isn++)
        {
            struct StepSliceInfo stepSliceInfotmp;
            struct BuffInfo bitmp;

            u64 inOutOffset=(zAxis*xRankSize*yRankSize+yAxis*xRankSize)*dataSize+processedDataEachRank+xyRSOffect[osn]+xRSOffect[osn][isn];
            BuffInfoAssign(bitmp,inOutOffset,inOutOffset,xCclBufferBaseOff);
            StepSliceInfoAssign(stepSliceInfotmp,bitmp,xRSDataSize[osn][isn]/dataTypeSize,xRSDataSize[osn][isn],dataSize,xyzDataSizeStep[0][osn*innerStepNum+isn]);
            stepSliceInfotmp.inputOmniPipeSliceStride.push_back(0);
            stepSliceInfotmp.outputOmniPipeSliceStride.push_back(0);

            dataSliceLevelx.insert(dataSliceLevelx.end(),stepSliceInfotmp);
        }
    }
    //算y轴偏移,和计算xy的相似
    //前n-1步斜对角，多片数据2d
    std::vector<StepSliceInfo> dataSliceLevely;
    for(u64 osn=0;osn<xyConnerStep;osn++)
    {
        for(u64 isn=0;isn<yInCornerStep;isn++)
        {
            struct StepSliceInfo stepSliceInfotmp;
            struct BuffInfo bitmp;
            u64 inOutOffset=processedDataEachRank+xyRSOffect[osn]+yRSOffect[osn][isn];
            BuffInfoAssign(bitmp,inOutOffset,inOutOffset,yCclBufferBaseOff);
            StepSliceInfoAssign(stepSliceInfotmp,bitmp,yRSDataSize[osn][isn]/dataTypeSize,yRSDataSize[osn][isn],xRankSize*dataSize,xyzDataSizeStep[1][osn*innerStepNum+isn]);
            u64 inputslicestride=0;
            u64 outputslicestride=0;
            //这里得单独看是第一片开始还是第0片开始
            for(u64 outSliceNum=0;outSliceNum<zRankSize;outSliceNum++)
            {
                u64 currentDataSliceId=outSliceNum*xRankSize*yRankSize+yAxis*xRankSize+xAxis;//算算是zRankSize-1中的哪一片
                if(currentDataSliceId==rankid)//自己的不做，只做斜对角的
                {
                    inputslicestride=inputslicestride+xRankSize*yRankSize*dataSize;
                }
                else
                {
                    inputslicestride=outSliceNum*xRankSize*yRankSize*dataSize;
                    for(u64 connerDataSlice=0;connerDataSlice<xRankSize;connerDataSlice++)
                    {
                        u64 currentInnerStepDataSliceId=outSliceNum*xRankSize*yRankSize+yAxis*xRankSize+connerDataSlice;//算算是机内斜对角中的哪一片
                        if(currentInnerStepDataSliceId==currentDataSliceId && xRankSize>1)//自己的不做，只做斜对角的
                        {
                            inputslicestride=inputslicestride+dataSize;
                        }
                        else
                        {

                            stepSliceInfotmp.inputOmniPipeSliceStride.insert(stepSliceInfotmp.inputOmniPipeSliceStride.end(),inputslicestride);
                            stepSliceInfotmp.outputOmniPipeSliceStride.insert(stepSliceInfotmp.outputOmniPipeSliceStride.end(),outputslicestride);
                            inputslicestride+=dataSize;
                            outputslicestride+=yRSDataSize[osn][isn];
                        }
                    }
                }
            }
            dataSliceLevely.insert(dataSliceLevely.end(),stepSliceInfotmp);
        }
        for(u64 isn=yInCornerStep;isn<innerStepNum;isn++)
        {
            struct StepSliceInfo stepSliceInfotmp;
            struct BuffInfo bitmp;
            u64 inOutOffset=(xAxis)*dataSize+processedDataEachRank+xyRSOffect[osn]+yRSOffect[osn][isn];
            BuffInfoAssign(bitmp,inOutOffset,inOutOffset,yCclBufferBaseOff);
            StepSliceInfoAssign(stepSliceInfotmp,bitmp,yRSDataSize[osn][isn]/dataTypeSize,yRSDataSize[osn][isn],xRankSize*dataSize,xyzDataSizeStep[1][osn*innerStepNum+isn]);
            u64 inputslicestride=0;
            u64 outputslicestride=0;
            for(u64 outSliceNum=0;outSliceNum<zRankSize;outSliceNum++)
            {
                u64 currentDataSliceId=outSliceNum*xRankSize*yRankSize+yAxis*xRankSize+xAxis;//算算是zRankSize-1中的哪一片
                if(currentDataSliceId==rankid)//自己的不做，只做斜对角的
                {

                    inputslicestride=inputslicestride+xRankSize*yRankSize*dataSize;
                }
                else
                {

                    stepSliceInfotmp.inputOmniPipeSliceStride.insert(stepSliceInfotmp.inputOmniPipeSliceStride.end(),inputslicestride);
                    stepSliceInfotmp.outputOmniPipeSliceStride.insert(stepSliceInfotmp.outputOmniPipeSliceStride.end(),outputslicestride);
                    inputslicestride+=xRankSize*yRankSize*dataSize;
                    outputslicestride+=yRSDataSize[osn][isn];
                }
            }
            dataSliceLevely.insert(dataSliceLevely.end(),stepSliceInfotmp);
        }
    }
    for(u64 osn=xyConnerStep;osn<outerStepNum;osn++)
    {
        for(u64 isn=0;isn<yInCornerStep;isn++)
        {
            struct StepSliceInfo stepSliceInfotmp;
            struct BuffInfo bitmp;
            u64 inOutOffset=(zAxis*xRankSize*yRankSize)*dataSize+processedDataEachRank+xyRSOffect[osn]+yRSOffect[osn][isn];
            BuffInfoAssign(bitmp,inOutOffset,inOutOffset,yCclBufferBaseOff);
            StepSliceInfoAssign(stepSliceInfotmp,bitmp,yRSDataSize[osn][isn]/dataTypeSize,yRSDataSize[osn][isn],xRankSize*dataSize,xyzDataSizeStep[1][osn*innerStepNum+isn]);
            u64 inputslicestride=0;
            u64 outputslicestride=0;
            for(u64 connerDataSlice=0;connerDataSlice<xRankSize;connerDataSlice++)
            {
                u64 currentDataSliceId=zAxis*xRankSize*yRankSize+yAxis*xRankSize+connerDataSlice;//斜对角先算算是哪一片
                if(currentDataSliceId==rankid)
                {
                    inputslicestride=inputslicestride+dataSize;

                }
                else
                {

                    stepSliceInfotmp.inputOmniPipeSliceStride.insert(stepSliceInfotmp.inputOmniPipeSliceStride.end(),inputslicestride);
                    stepSliceInfotmp.outputOmniPipeSliceStride.insert(stepSliceInfotmp.outputOmniPipeSliceStride.end(),outputslicestride);
                    inputslicestride+=dataSize;
                    outputslicestride+=yRSDataSize[osn][isn];
                }
            }
            dataSliceLevely.insert(dataSliceLevely.end(),stepSliceInfotmp);
        }
        for(u64 isn=yInCornerStep;isn<innerStepNum;isn++)
        {
            struct StepSliceInfo stepSliceInfotmp;
            struct BuffInfo bitmp;

            u64 inOutOffset=(zAxis*xRankSize*yRankSize+xAxis)*dataSize+processedDataEachRank+xyRSOffect[osn]+yRSOffect[osn][isn];
            BuffInfoAssign(bitmp,inOutOffset,inOutOffset,yCclBufferBaseOff);
            StepSliceInfoAssign(stepSliceInfotmp,bitmp,yRSDataSize[osn][isn]/dataTypeSize,yRSDataSize[osn][isn],xRankSize*dataSize,xyzDataSizeStep[1][osn*innerStepNum+isn]);
            stepSliceInfotmp.inputOmniPipeSliceStride.push_back(0);
            stepSliceInfotmp.outputOmniPipeSliceStride.push_back(0);

            dataSliceLevely.insert(dataSliceLevely.end(),stepSliceInfotmp);
        }
    }

    struct OmniPipeSliceInfo dataSliceInfoxyz;
    dataSliceInfoxyz.dataSliceLevel0=dataSliceLevelx;
    dataSliceInfoxyz.dataSliceLevel1=dataSliceLevely;
    dataSliceInfoxyz.dataSliceLevel2=dataSliceLevelz;
    HCCL_INFO("dataSliceInfoxyz,001[%u]", dataSliceInfoxyz.dataSliceLevel0[0].buffInfo.hcclBuffBaseOff);
    return dataSliceInfoxyz;
}
}