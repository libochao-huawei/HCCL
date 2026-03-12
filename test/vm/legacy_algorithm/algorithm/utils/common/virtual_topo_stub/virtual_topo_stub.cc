#include "virtual_topo_stub.h"

using namespace Hccl;

shared_ptr<NetInstance::Peer> VirtualTopoStub::InitPeer(RankId rankId, LocalId localId, DeviceId deviceId)
{
    shared_ptr<NetInstance::Peer> rank = make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    return rank;
}

shared_ptr<NetInstance::Fabric> VirtualTopoStub::InitFabric(FabricId FabricId, PlaneId netPlaneId)
{
    shared_ptr<NetInstance::Fabric> fabric = make_shared<NetInstance::Fabric>(FabricId, netPlaneId);
    return fabric;
}

shared_ptr<NetInstance> VirtualTopoStub::InitNetInstance(u32 level, string id)
{
    shared_ptr<NetInstance> fabGroup;
    if (level == 0) {
        fabGroup = make_shared<InnerNetInstance>(level, id);
    } else {
        fabGroup = make_shared<ClosNetInstance>(level, id);
    }
    return fabGroup;
}

shared_ptr<NetInstance::ConnInterface> VirtualTopoStub::InitConnInterface(IpAddress addr)
{
    AddrPosition pos = AddrPosition::DEVICE;
    LinkType inputLinkType = LinkType::PEER2PEER;
    std::set<LinkProtocol> inputLinkProtocol = {LinkProtocol::UB_CTP};
    std::set<string> ports = {"0/1"};
    std::shared_ptr<NetInstance::ConnInterface> iface = std::make_shared<NetInstance::ConnInterface>(addr, ports, pos, inputLinkType, inputLinkProtocol);
    return iface;
}

shared_ptr<NetInstance::ConnInterface> VirtualTopoStub::InitConnInterface(
        IpAddress addr, std::set<string> ports, AddrPosition pos, LinkType inputLinkType, std::set<LinkProtocol> inputLinkProtocols)
{
    std::shared_ptr<NetInstance::ConnInterface> iface = std::make_shared<NetInstance::ConnInterface>(addr, ports, pos, inputLinkType, inputLinkProtocols);
    return iface;
}

void VirtualTopoStub::AddLinkStub(shared_ptr<NetInstance> fabGroup, shared_ptr<NetInstance::Node> srcPeer,
    shared_ptr<NetInstance::Node> dstPeer, shared_ptr<NetInstance::ConnInterface> srcIface, shared_ptr<NetInstance::ConnInterface> dstIface,
    LinkType type, std::set<LinkProtocol> protocals)
{
    std::shared_ptr<NetInstance::Link> link =
        std::make_shared<NetInstance::Link>(srcPeer, dstPeer, srcIface, dstIface, type, protocals);

    fabGroup->AddLink(link);
}

void VirtualTopoStub::AddLinkStub(shared_ptr<NetInstance> fabGroup, shared_ptr<NetInstance::Node> srcPeer,
    shared_ptr<NetInstance::Node> dstPeer, shared_ptr<NetInstance::ConnInterface> srcIface, shared_ptr<NetInstance::ConnInterface> dstIface)
{
    LinkType type = LinkType::PEER2PEER;
    std::set<LinkProtocol> protocals = {LinkProtocol::UB_CTP};

    std::shared_ptr<NetInstance::Link> link =
        std::make_shared<NetInstance::Link>(srcPeer, dstPeer, srcIface, dstIface, type, protocals);

    fabGroup->AddLink(link);
}

void VirtualTopoStub::AddLinkStub(shared_ptr<NetInstance> fabGroup, shared_ptr<NetInstance::Node> srcPeer,
    shared_ptr<NetInstance::Node> dstPeer, shared_ptr<NetInstance::ConnInterface> srcIface, shared_ptr<NetInstance::ConnInterface> dstIface, 
    LinkDirection direct, u32 hop)
{
    LinkType type = LinkType::PEER2PEER;
    std::set<LinkProtocol> protocals = {LinkProtocol::UB_CTP};

    std::shared_ptr<NetInstance::Link> link =
        std::make_shared<NetInstance::Link>(srcPeer, dstPeer, srcIface, dstIface, type, protocals, direct, hop);

    fabGroup->AddLink(link);
}

void VirtualTopoStub::TopoInit91095TwoTimesTwo(const string &rankTable)
{
    // 打桩virtual topo
    shared_ptr<NetInstance> fabGroup = InitNetInstance(0, "test");
    this->netInsts_[0].emplace("test", fabGroup);

    // 初始化rank0信息
    auto rank0 = InitPeer(0, 0);
    this->AddPeer(rank0);  // 存储rank0信息到peers
    fabGroup->AddNode(rank0);
    char rank0Address[] = "0.0.0.0";  // 打桩用sendIP地址
    IpAddress rank0Addr(rank0Address);
    auto iface0 = InitConnInterface(rank0Addr);
    rank0->AddConnInterface(0, iface0);
    rank0->AddNetInstance(fabGroup);
    fabGroup->AddRankId(0);

    // 初始化rank1信息
    auto rank1 = InitPeer(1, 1);
    this->AddPeer(rank1);  // 存储rank1信息到peers
    fabGroup->AddNode(rank1);
    char rank1Address[] = "1.0.0.0";  // 打桩用sendIP地址
    IpAddress rank1Addr(rank1Address);
    auto iface1 = InitConnInterface(rank1Addr);
    rank1->AddConnInterface(0, iface1);
    rank1->AddNetInstance(fabGroup);
    fabGroup->AddRankId(1);

    // 初始化rank2信息
    auto rank2 = InitPeer(2, 8);
    this->AddPeer(rank2);  // 存储rank1信息到peers
    fabGroup->AddNode(rank2);
    char rank2Address[] = "2.0.0.0";  // 打桩用sendIP地址
    IpAddress rank2Addr(rank2Address);
    auto iface2 = InitConnInterface(rank2Addr);
    rank2->AddConnInterface(0, iface2);
    rank2->AddNetInstance(fabGroup);
    fabGroup->AddRankId(2);

    // 初始化rank3信息
    auto rank3 = InitPeer(3, 9);
    this->AddPeer(rank3);  // 存储rank1信息到peers
    fabGroup->AddNode(rank3);
    char rank3Address[] = "3.0.0.0";  // 打桩用sendIP地址
    IpAddress rank3Addr(rank3Address);
    auto iface3 = InitConnInterface(rank3Addr);
    rank3->AddConnInterface(0, iface3);
    rank3->AddNetInstance(fabGroup);
    fabGroup->AddRankId(3);

    this->InitInnerRanks();

    // rank0: 连接rank1， 连接rank2
    AddLinkStub(fabGroup, rank0, rank1, iface0, iface1);
    AddLinkStub(fabGroup, rank0, rank2, iface0, iface2);

    // rank1: 连接rank0，连接rank3
    AddLinkStub(fabGroup, rank1, rank0, iface1, iface0);
    AddLinkStub(fabGroup, rank1, rank3, iface1, iface3);

    // rank2: 连接rank0，连接rank3
    AddLinkStub(fabGroup, rank2, rank0, iface2, iface0);
    AddLinkStub(fabGroup, rank2, rank3, iface2, iface3);

    // rank3: 连接rank1，连接rank2
    AddLinkStub(fabGroup, rank3, rank1, iface3, iface1);
    AddLinkStub(fabGroup, rank3, rank2, iface3, iface2);
}

void VirtualTopoStub::TopoInit91095TwoServerTimesTwo(const string &rankTable)
{
    // 打桩virtual topo
    shared_ptr<NetInstance> fabGroup0 = InitNetInstance(0, "server0");
    shared_ptr<NetInstance> fabGroup1 = InitNetInstance(0, "server1");
    shared_ptr<NetInstance> fabGroupLevel1 = InitNetInstance(1, "clos0");
    this->netInsts_[0].emplace("server0", fabGroup0);
    this->netInsts_[0].emplace("server1", fabGroup1);
    this->netInsts_[1].emplace("clos0", fabGroupLevel1);
    // 初始化fabric0信息
    auto fabric0 = InitFabric(0);
    fabGroupLevel1->AddNode(fabric0);
    char fabricAddress0[] = "5.0.0.0";  // 打桩用sendIP地址
    IpAddress fabricAddr0(fabricAddress0);
    auto ifaceNode0 = InitConnInterface(fabricAddr0);
    fabric0->AddConnInterface(1, ifaceNode0);
    // 初始化fabric1信息
    auto fabric1 = InitFabric(1);
    fabGroupLevel1->AddNode(fabric1);
    char fabricAddress1[] = "6.0.0.0";  // 打桩用sendIP地址
    IpAddress fabricAddr1(fabricAddress1);
    auto ifaceNode1 = InitConnInterface(fabricAddr1);
    fabric1->AddConnInterface(1, ifaceNode1);


    // 初始化rank0信息
    auto rank0 = InitPeer(0, 0);
    this->AddPeer(rank0);  // 存储rank0信息到peers
    fabGroup0->AddNode(rank0);
    fabGroupLevel1->AddNode(rank0);
    char rank0Address0[] = "0.0.0.0";  // 打桩用sendIP地址
    IpAddress rank0Addr0(rank0Address0);
    auto rank0iface0 = InitConnInterface(rank0Addr0);
    rank0->AddConnInterface(0, rank0iface0);
    char rank0Address1[] = "0.0.0.1";  // 打桩用sendIP地址
    IpAddress rank0Addr1(rank0Address1);
    auto rank0iface1 = InitConnInterface(rank0Addr1);
    rank0->AddConnInterface(1, rank0iface1);
    rank0->AddNetInstance(fabGroup0);
    rank0->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(0);
    fabGroupLevel1->AddRankId(0);
 
    // 初始化rank1信息
    auto rank1 = InitPeer(1, 1);
    this->AddPeer(rank1);  // 存储rank0信息到peers
    fabGroup0->AddNode(rank1);
    fabGroupLevel1->AddNode(rank1);
    char rank1Address0[] = "1.0.0.0";  // 打桩用sendIP地址
    IpAddress rank1Addr0(rank1Address0);
    auto rank1iface0 = InitConnInterface(rank1Addr0);
    rank1->AddConnInterface(0, rank1iface0);
    char rank1Address1[] = "1.0.0.1";  // 打桩用sendIP地址
    IpAddress rank1Addr1(rank1Address1);
    auto rank1iface1 = InitConnInterface(rank1Addr1);
    rank1->AddConnInterface(1, rank1iface1);
    rank1->AddNetInstance(fabGroup0);
    rank1->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(1);
    fabGroupLevel1->AddRankId(1);
 
    // 初始化rank2信息
    auto rank2 = InitPeer(2, 0);
    this->AddPeer(rank2);  // 存储rank0信息到peers
    fabGroup1->AddNode(rank2);
    fabGroupLevel1->AddNode(rank2);
    char rank2Address0[] = "2.0.0.0";  // 打桩用sendIP地址
    IpAddress rank2Addr0(rank2Address0);
    auto rank2iface0 = InitConnInterface(rank2Addr0);
    rank2->AddConnInterface(0, rank2iface0);
    char rank2Address1[] = "2.0.0.1";  // 打桩用sendIP地址
    IpAddress rank2Addr1(rank2Address1);
    auto rank2iface1 = InitConnInterface(rank2Addr1);
    rank2->AddConnInterface(1, rank2iface1);
    rank2->AddNetInstance(fabGroup1);
    rank2->AddNetInstance(fabGroupLevel1);
    fabGroup1->AddRankId(2);
    fabGroupLevel1->AddRankId(2);
 
    // 初始化rank3信息
    auto rank3 = InitPeer(3, 1);
    this->AddPeer(rank3);  // 存储rank0信息到peers
    fabGroup1->AddNode(rank3);
    fabGroupLevel1->AddNode(rank3);
    char rank3Address0[] = "3.0.0.0";  // 打桩用sendIP地址
    IpAddress rank3Addr0(rank3Address0);
    auto rank3iface0 = InitConnInterface(rank3Addr0);
    rank3->AddConnInterface(0, rank3iface0);
    char rank3Address1[] = "3.0.0.1";  // 打桩用sendIP地址
    IpAddress rank3Addr1(rank3Address1);
    auto rank3iface1 = InitConnInterface(rank3Addr1);
    rank3->AddConnInterface(1, rank3iface1);
    rank3->AddNetInstance(fabGroup1);
    rank3->AddNetInstance(fabGroupLevel1);
    fabGroup1->AddRankId(3);
    fabGroupLevel1->AddRankId(3);
    this->InitInnerRanks();
 
    AddLinkStub(fabGroup0, rank0, rank1, rank0iface0, rank1iface0);
    AddLinkStub(fabGroupLevel1, rank0, fabric0, rank0iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank0, fabric1, rank0iface1, ifaceNode1);
 
    AddLinkStub(fabGroup0, rank1, rank0, rank1iface0, rank0iface0);
    AddLinkStub(fabGroupLevel1, rank1, fabric0, rank1iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank1, fabric1, rank1iface1, ifaceNode1);

    AddLinkStub(fabGroup1, rank2, rank3, rank2iface0, rank3iface0);
    AddLinkStub(fabGroupLevel1, rank2, fabric0, rank2iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank2, fabric1, rank2iface1, ifaceNode1);

    AddLinkStub(fabGroup1, rank3, rank2, rank3iface0, rank2iface0);
    AddLinkStub(fabGroupLevel1, rank3, fabric0, rank3iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank3, fabric1, rank3iface1, ifaceNode1);

    AddLinkStub(fabGroupLevel1, fabric0, rank0, ifaceNode0, rank0iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank1, ifaceNode0, rank1iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank2, ifaceNode0, rank2iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank3, ifaceNode0, rank3iface0);
    AddLinkStub(fabGroupLevel1, fabric1, rank0, ifaceNode1, rank0iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank1, ifaceNode1, rank1iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank2, ifaceNode1, rank2iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank3, ifaceNode1, rank3iface1);
}

void VirtualTopoStub::TopoInit91095OneTimesN(const string &rankTable, int numRanks)
{
    // 创建NetInstance对象
    shared_ptr<NetInstance> fabGroup = InitNetInstance(0, "test");
    this->netInsts_[0].emplace("test", fabGroup);
    std::map<shared_ptr<NetInstance::Peer>, shared_ptr<NetInstance::ConnInterface>> peer2Interface;
    // 初始化每个rank
    for (u32 rankId = 0; rankId < numRanks; ++rankId) {
        // 创建Peer对象
        auto peer = InitPeer(rankId, rankId);
        this->AddPeer(peer);
        fabGroup->AddNode(peer);
        // 生成IP地址，格式为"rankId.0.0.0"
        char ipAddress[16];
        snprintf(ipAddress, sizeof(ipAddress), "%d.0.0.0", rankId);
        IpAddress addr(ipAddress);
        // 创建ConnInterface并添加到Peer
        auto iface = InitConnInterface(addr);
        peer->AddConnInterface(0, iface);
        peer2Interface[peer] = iface;
        peer->AddNetInstance(fabGroup);
        fabGroup->AddRankId(rankId);
    }

    // 初始化内部rank
    this->InitInnerRanks();
    // 生成全连接
    for (u32 i = 0; i < numRanks; ++i) {
        for (u32 j = i + 1; j < numRanks; ++j) {
            // 获取rank i和rank j的Peer对象
            auto srcPeer = this->peers_[i];
            auto dstPeer = this->peers_[j];
            // 获取它们的ConnInterface
            auto srcIface = peer2Interface[srcPeer];
            auto dstIface = peer2Interface[dstPeer];
            // 添加从i到j的连接
            AddLinkStub(fabGroup, srcPeer, dstPeer, srcIface, dstIface);
            // 添加从j到i的连接
            AddLinkStub(fabGroup, dstPeer, srcPeer, dstIface, srcIface);
        }
    }
}

void VirtualTopoStub::TopoInit91095OneTimesFour(const string &rankTable)
{
    // 打桩virtual topo
    shared_ptr<NetInstance> fabGroup = InitNetInstance(0, "test");
    this->netInsts_[0].emplace("test", fabGroup);
 
    // 初始化rank0信息
    auto rank0 = InitPeer(0, 0);
    this->AddPeer(rank0);  // 存储rank0信息到peers
    fabGroup->AddNode(rank0);
    char rank0Address[] = "0.0.0.0";  // 打桩用sendIP地址
    IpAddress rank0Addr(rank0Address);
    auto iface0 = InitConnInterface(rank0Addr);
    rank0->AddConnInterface(0, iface0);
    rank0->AddNetInstance(fabGroup);
    fabGroup->AddRankId(0);
 
    // 初始化rank1信息
    auto rank1 = InitPeer(1, 1);
    this->AddPeer(rank1);  // 存储rank1信息到peers
    fabGroup->AddNode(rank1);
    char rank1Address[] = "1.0.0.0";  // 打桩用sendIP地址
    IpAddress rank1Addr(rank1Address);
    auto iface1 = InitConnInterface(rank1Addr);
    rank1->AddConnInterface(0, iface1);
    rank1->AddNetInstance(fabGroup);
    fabGroup->AddRankId(1);
 
    // 初始化rank2信息
    auto rank2 = InitPeer(2, 2);
    this->AddPeer(rank2);  // 存储rank1信息到peers
    fabGroup->AddNode(rank2);
    char rank2Address[] = "2.0.0.0";  // 打桩用sendIP地址
    IpAddress rank2Addr(rank2Address);
    auto iface2 = InitConnInterface(rank2Addr);
    rank2->AddConnInterface(0, iface2);
    rank2->AddNetInstance(fabGroup);
    fabGroup->AddRankId(2);
 
    // 初始化rank3信息
    auto rank3 = InitPeer(3, 3);
    this->AddPeer(rank3);  // 存储rank1信息到peers
    fabGroup->AddNode(rank3);
    char rank3Address[] = "3.0.0.0";  // 打桩用sendIP地址
    IpAddress rank3Addr(rank3Address);
    auto iface3 = InitConnInterface(rank3Addr);
    rank3->AddConnInterface(0, iface3);
    rank3->AddNetInstance(fabGroup);
    fabGroup->AddRankId(3);
 
    this->InitInnerRanks();
 
    // rank0: 连接rank1， 连接rank2
    AddLinkStub(fabGroup, rank0, rank1, iface0, iface1);
    AddLinkStub(fabGroup, rank0, rank2, iface0, iface2);
    AddLinkStub(fabGroup, rank0, rank3, iface0, iface3);
 
 
    // rank1: 连接rank0，连接rank3
    AddLinkStub(fabGroup, rank1, rank0, iface1, iface0);
    AddLinkStub(fabGroup, rank1, rank3, iface1, iface3);
    AddLinkStub(fabGroup, rank1, rank2, iface1, iface2);
 
    // rank2: 连接rank0，连接rank3
    AddLinkStub(fabGroup, rank2, rank0, iface2, iface0);
    AddLinkStub(fabGroup, rank2, rank3, iface2, iface3);
    AddLinkStub(fabGroup, rank2, rank1, iface2, iface1);
 
    // rank3: 连接rank1，连接rank2
    AddLinkStub(fabGroup, rank3, rank1, iface3, iface1);
    AddLinkStub(fabGroup, rank3, rank2, iface3, iface2);
    AddLinkStub(fabGroup, rank3, rank0, iface3, iface0);
}

void VirtualTopoStub::TopoInit91095OneTimesTwoDetour(const string &rankTable)
{
    // 打桩virtual topo
    shared_ptr<NetInstance> fabGroup = InitNetInstance(0, "test");
    this->netInsts_[0].emplace("test", fabGroup);

    // 初始化rank0信息
    auto rank0 = InitPeer(0, 0);
    this->AddPeer(rank0);  // 存储rank0信息到peers
    fabGroup->AddNode(rank0);
    char rank0Address[] = "0.0.0.0";  // 打桩用sendIP地址
    IpAddress rank0Addr(rank0Address);
    auto iface0 = InitConnInterface(rank0Addr);
    rank0->AddConnInterface(0, iface0);
    rank0->AddNetInstance(fabGroup);
    fabGroup->AddRankId(0);

    // 初始化rank1信息
    auto rank1 = InitPeer(1, 1);
    this->AddPeer(rank1);  // 存储rank1信息到peers
    fabGroup->AddNode(rank1);
    char rank1Address[] = "1.0.0.0";  // 打桩用sendIP地址
    IpAddress rank1Addr(rank1Address);
    auto iface1 = InitConnInterface(rank1Addr);
    rank1->AddConnInterface(0, iface1);
    rank1->AddNetInstance(fabGroup);
    fabGroup->AddRankId(1);

    this->InitInnerRanks();

    // rank0: 连接rank1， 连接rank2
    AddLinkStub(fabGroup, rank0, rank1, iface0, iface1, LinkDirection::BOTH, 1);
    AddLinkStub(fabGroup, rank0, rank1, iface0, iface1, LinkDirection::SEND_ONLY, 2);
    AddLinkStub(fabGroup, rank0, rank1, iface0, iface1, LinkDirection::RECV_ONLY, 2);

    // rank1: 连接rank0，连接rank3
    AddLinkStub(fabGroup, rank1, rank0, iface1, iface0, LinkDirection::BOTH, 1);
    AddLinkStub(fabGroup, rank1, rank0, iface1, iface0, LinkDirection::SEND_ONLY, 2);
    AddLinkStub(fabGroup, rank1, rank0, iface1, iface0, LinkDirection::RECV_ONLY, 2);
}

void VirtualTopoStub::TopoInit91095TwoTimesThree(const string &rankTable)
{
// 打桩virtual topo
    shared_ptr<NetInstance> fabGroup = InitNetInstance(0, "test");
    this->netInsts_[0].emplace("test", fabGroup);

    // 初始化rank0信息
    auto rank0 = InitPeer(0, 0);
    this->AddPeer(rank0);  // 存储rank0信息到peers
    fabGroup->AddNode(rank0);
    char rank0Address[] = "0.0.0.0";  // 打桩用sendIP地址
    IpAddress rank0Addr(rank0Address);
    auto iface0 = InitConnInterface(rank0Addr);
    rank0->AddConnInterface(0, iface0);
    rank0->AddNetInstance(fabGroup);
    fabGroup->AddRankId(0);

    // 初始化rank1信息
    auto rank1 = InitPeer(1, 1);
    this->AddPeer(rank1);  // 存储rank1信息到peers
    fabGroup->AddNode(rank1);
    char rank1Address[] = "1.0.0.0";  // 打桩用sendIP地址
    IpAddress rank1Addr(rank1Address);
    auto iface1 = InitConnInterface(rank1Addr);
    rank1->AddConnInterface(0, iface1);
    rank1->AddNetInstance(fabGroup);
    fabGroup->AddRankId(1);

    // 初始化rank2信息
    auto rank2 = InitPeer(2, 2);
    this->AddPeer(rank2);  // 存储rank1信息到peers
    fabGroup->AddNode(rank2);
    char rank2Address[] = "2.0.0.0";  // 打桩用sendIP地址
    IpAddress rank2Addr(rank2Address);
    auto iface2 = InitConnInterface(rank2Addr);
    rank2->AddConnInterface(0, iface2);
    rank2->AddNetInstance(fabGroup);
    fabGroup->AddRankId(2);

    // 初始化rank3信息
    auto rank3 = InitPeer(3, 8);
    this->AddPeer(rank3);  // 存储rank1信息到peers
    fabGroup->AddNode(rank3);
    char rank3Address[] = "3.0.0.0";  // 打桩用sendIP地址
    IpAddress rank3Addr(rank3Address);
    auto iface3 = InitConnInterface(rank3Addr);
    rank3->AddConnInterface(0, iface3);
    rank3->AddNetInstance(fabGroup);
    fabGroup->AddRankId(3);

    // 初始化rank3信息
    auto rank4 = InitPeer(4, 9);
    this->AddPeer(rank4);  // 存储rank1信息到peers
    fabGroup->AddNode(rank4);
    char rank4Address[] = "4.0.0.0";  // 打桩用sendIP地址
    IpAddress rank4Addr(rank4Address);
    auto iface4 = InitConnInterface(rank4Addr);
    rank4->AddConnInterface(0, iface4);
    rank4->AddNetInstance(fabGroup);
    fabGroup->AddRankId(4);

    // 初始化rank3信息
    auto rank5 = InitPeer(5, 10);
    this->AddPeer(rank5);  // 存储rank1信息到peers
    fabGroup->AddNode(rank5);
    char rank5Address[] = "5.0.0.0";  // 打桩用sendIP地址
    IpAddress rank5Addr(rank5Address);
    auto iface5 = InitConnInterface(rank5Addr);
    rank5->AddConnInterface(0, iface5);
    rank5->AddNetInstance(fabGroup);
    fabGroup->AddRankId(5);

    this->InitInnerRanks();

    // rank0: 连接rank1， 连接rank2
    AddLinkStub(fabGroup, rank0, rank1, iface0, iface1);
    AddLinkStub(fabGroup, rank0, rank2, iface0, iface2);
    AddLinkStub(fabGroup, rank0, rank3, iface0, iface3);

    // rank1: 连接rank0，连接rank3
    AddLinkStub(fabGroup, rank1, rank0, iface1, iface0);
    AddLinkStub(fabGroup, rank1, rank2, iface1, iface2);
    AddLinkStub(fabGroup, rank1, rank4, iface1, iface4);

    // rank2: 连接rank0，连接rank3
    AddLinkStub(fabGroup, rank2, rank0, iface2, iface0);
    AddLinkStub(fabGroup, rank2, rank1, iface2, iface1);
    AddLinkStub(fabGroup, rank2, rank5, iface2, iface5);

    // rank3: 连接rank1，连接rank2
    AddLinkStub(fabGroup, rank3, rank0, iface3, iface0);
    AddLinkStub(fabGroup, rank3, rank4, iface3, iface4);
    AddLinkStub(fabGroup, rank3, rank5, iface3, iface5);

    // rank4: 连接rank1，连接rank2
    AddLinkStub(fabGroup, rank4, rank1, iface4, iface1);
    AddLinkStub(fabGroup, rank4, rank3, iface4, iface3);
    AddLinkStub(fabGroup, rank4, rank5, iface4, iface5);

    // rank5: 连接rank0，连接rank3
    AddLinkStub(fabGroup, rank5, rank4, iface5, iface4);
    AddLinkStub(fabGroup, rank5, rank2, iface5, iface2);
    AddLinkStub(fabGroup, rank5, rank3, iface5, iface3);
}

void VirtualTopoStub::TopoInit91095OneTimesThree(const string &rankTable)
{
    // 打桩virtual topo
    shared_ptr<NetInstance> fabGroup = InitNetInstance(0, "test");
    this->netInsts_[0].emplace("test", fabGroup);

    // 初始化rank0信息
    auto rank0 = InitPeer(0, 0);
    this->AddPeer(rank0);  // 存储rank0信息到peers
    fabGroup->AddNode(rank0);
    char rank0Address[] = "0.0.0.0";  // 打桩用sendIP地址
    IpAddress rank0Addr(rank0Address);
    auto iface0 = InitConnInterface(rank0Addr);
    rank0->AddConnInterface(0, iface0);
    rank0->AddNetInstance(fabGroup);
    fabGroup->AddRankId(0);

    // 初始化rank1信息
    auto rank1 = InitPeer(1, 1);
    this->AddPeer(rank1);  // 存储rank1信息到peers
    fabGroup->AddNode(rank1);
    char rank1Address[] = "1.0.0.0";  // 打桩用sendIP地址
    IpAddress rank1Addr(rank1Address);
    auto iface1 = InitConnInterface(rank1Addr);
    rank1->AddConnInterface(0, iface1);
    rank1->AddNetInstance(fabGroup);
    fabGroup->AddRankId(1);

    // 初始化rank2信息
    auto rank2 = InitPeer(2, 2);
    this->AddPeer(rank2);  // 存储rank1信息到peers
    fabGroup->AddNode(rank2);
    char rank2Address[] = "2.0.0.0";  // 打桩用sendIP地址
    IpAddress rank2Addr(rank2Address);
    auto iface2 = InitConnInterface(rank2Addr);
    rank2->AddConnInterface(0, iface2);
    rank2->AddNetInstance(fabGroup);
    fabGroup->AddRankId(2);

    this->InitInnerRanks();

    // rank0: 连接rank1， 连接rank2
    AddLinkStub(fabGroup, rank0, rank1, iface0, iface1);
    AddLinkStub(fabGroup, rank0, rank2, iface0, iface2);

    // rank1: 连接rank0，连接rank3
    AddLinkStub(fabGroup, rank1, rank0, iface1, iface0);
    AddLinkStub(fabGroup, rank1, rank2, iface1, iface2);

    // rank2: 连接rank0，连接rank3
    AddLinkStub(fabGroup, rank2, rank0, iface2, iface0);
    AddLinkStub(fabGroup, rank2, rank1, iface2, iface1);
}

void VirtualTopoStub::TopoInit91095OneTimesOne(const string &rankTable)
{
    // 打桩virtual topo
    shared_ptr<NetInstance> fabGroup = InitNetInstance(0, "test");
    this->netInsts_[0].emplace("test", fabGroup);

    // 初始化rank0信息
    auto rank0 = InitPeer(0, 0);
    this->AddPeer(rank0);  // 存储rank0信息到peers
    fabGroup->AddNode(rank0);
    char rank0Address[] = "0.0.0.0";  // 打桩用sendIP地址
    IpAddress rank0Addr(rank0Address);
    auto iface0 = InitConnInterface(rank0Addr);
    rank0->AddConnInterface(0, iface0);
    rank0->AddNetInstance(fabGroup);
    fabGroup->AddRankId(0);

    this->InitInnerRanks();
}

void VirtualTopoStub::TopoInit91095TwoPlusOnePlusOne(const string &rankTable)
{
    // 打桩virtual topo
    shared_ptr<NetInstance> fabGroup = InitNetInstance(0, "test");
    this->netInsts_[0].emplace("test", fabGroup);

    // 初始化rank0信息
    auto rank0 = InitPeer(0, 0);
    this->AddPeer(rank0);  // 存储rank0信息到peers
    fabGroup->AddNode(rank0);
    char rank0Address[] = "0.0.0.0";  // 打桩用sendIP地址
    IpAddress rank0Addr(rank0Address);
    auto iface0 = InitConnInterface(rank0Addr);
    rank0->AddConnInterface(0, iface0);
    rank0->AddNetInstance(fabGroup);
    fabGroup->AddRankId(0);

    // 初始化rank1信息
    auto rank1 = InitPeer(1, 1);
    this->AddPeer(rank1);  // 存储rank1信息到peers
    fabGroup->AddNode(rank1);
    char rank1Address[] = "1.0.0.0";  // 打桩用sendIP地址
    IpAddress rank1Addr(rank1Address);
    auto iface1 = InitConnInterface(rank1Addr);
    rank1->AddConnInterface(0, iface1);
    rank1->AddNetInstance(fabGroup);
    fabGroup->AddRankId(1);

    // 初始化rank2信息
    auto rank2 = InitPeer(2, 8);
    this->AddPeer(rank2);  // 存储rank1信息到peers
    fabGroup->AddNode(rank2);
    char rank2Address[] = "2.0.0.0";  // 打桩用sendIP地址
    IpAddress rank2Addr(rank2Address);
    auto iface2 = InitConnInterface(rank2Addr);
    rank2->AddConnInterface(0, iface2);
    rank2->AddNetInstance(fabGroup);
    fabGroup->AddRankId(2);

    // 初始化rank3信息
    auto rank3 = InitPeer(3, 16);
    this->AddPeer(rank3);  // 存储rank1信息到peers
    fabGroup->AddNode(rank3);
    char rank3Address[] = "3.0.0.0";  // 打桩用sendIP地址
    IpAddress rank3Addr(rank3Address);
    auto iface3 = InitConnInterface(rank3Addr);
    rank3->AddConnInterface(0, iface3);
    rank3->AddNetInstance(fabGroup);
    fabGroup->AddRankId(3);

    this->InitInnerRanks();

    // rank0: 连接rank1， 连接rank2
    AddLinkStub(fabGroup, rank0, rank1, iface0, iface1);
    AddLinkStub(fabGroup, rank0, rank2, iface0, iface2);

    // rank1: 连接rank0，连接rank3
    AddLinkStub(fabGroup, rank1, rank0, iface1, iface0);

    // rank2: 连接rank0，连接rank3
    AddLinkStub(fabGroup, rank2, rank0, iface2, iface0);
    AddLinkStub(fabGroup, rank2, rank3, iface2, iface3);

    // rank3: 连接rank1，连接rank2
    AddLinkStub(fabGroup, rank3, rank2, iface3, iface2);
}

void VirtualTopoStub::TopoInit2HCCSLink(const string &rankTable)
{
    // 打桩virtual topo
    shared_ptr<NetInstance> fabGroup = InitNetInstance(0, "test");
    this->netInsts_[0].emplace("test", fabGroup);

    // 初始化rank0信息
    auto rank0 = InitPeer(0, 0);
    this->AddPeer(rank0);  // 存储rank0信息到peers
    fabGroup->AddNode(rank0);
    char rank0Address[] = "0.0.0.0";  // 打桩用sendIP地址
    IpAddress rank0Addr(rank0Address);
    auto iface0 = InitConnInterface(rank0Addr);
    rank0->AddConnInterface(0, iface0);
    rank0->AddNetInstance(fabGroup);
    fabGroup->AddRankId(0);

    // 初始化rank1信息
    auto rank1 = InitPeer(1, 1);
    this->AddPeer(rank1);  // 存储rank1信息到peers
    fabGroup->AddNode(rank1);
    char rank1Address[] = "1.0.0.0";  // 打桩用sendIP地址
    IpAddress rank1Addr(rank1Address);
    auto iface1 = InitConnInterface(rank1Addr);
    rank1->AddConnInterface(0, iface1);
    rank1->AddNetInstance(fabGroup);
    fabGroup->AddRankId(1);

    // rank0: 连接rank1
    AddLinkStub(fabGroup, rank0, rank1, iface0, iface1);
    // rank1: 连接rank0
    AddLinkStub(fabGroup, rank1, rank0, iface1, iface0);
}

void VirtualTopoStub::TopoInit4RankRDMALink(const string &rankTable)
{
    // 打桩virtual topo
    shared_ptr<NetInstance> fabGroup = InitNetInstance(0, "test");
    this->netInsts_[0].emplace("test", fabGroup);

    // 初始化rank0信息
    auto rank0 = InitPeer(0, 0);
    this->AddPeer(rank0);  // 存储rank0信息到peers
    fabGroup->AddNode(rank0);
    char rank0Address[] = "0.0.0.0";  // 打桩用sendIP地址
    IpAddress rank0Addr(rank0Address);
    auto iface0 = InitConnInterface(rank0Addr);
    rank0->AddConnInterface(0, iface0);
    rank0->AddNetInstance(fabGroup);
    fabGroup->AddRankId(0);

    // 初始化rank1信息
    auto rank1 = InitPeer(1, 1);
    this->AddPeer(rank1);  // 存储rank1信息到peers
    fabGroup->AddNode(rank1);
    char rank1Address[] = "1.0.0.0";  // 打桩用sendIP地址
    IpAddress rank1Addr(rank1Address);
    auto iface1 = InitConnInterface(rank1Addr);
    rank1->AddConnInterface(0, iface1);
    rank1->AddNetInstance(fabGroup);
    fabGroup->AddRankId(1);

    // 初始化rank2信息
    auto rank2 = InitPeer(2, 2);
    this->AddPeer(rank2);  // 存储rank1信息到peers
    fabGroup->AddNode(rank2);
    char rank2Address[] = "2.0.0.0";  // 打桩用sendIP地址
    IpAddress rank2Addr(rank2Address);
    auto iface2 = InitConnInterface(rank2Addr);
    rank2->AddConnInterface(0, iface2);
    rank2->AddNetInstance(fabGroup);
    fabGroup->AddRankId(2);

    // 初始化rank3信息
    auto rank3 = InitPeer(3, 3);
    this->AddPeer(rank3);  // 存储rank1信息到peers
    fabGroup->AddNode(rank3);
    char rank3Address[] = "3.0.0.0";  // 打桩用sendIP地址
    IpAddress rank3Addr(rank3Address);
    auto iface3 = InitConnInterface(rank3Addr);
    rank3->AddConnInterface(0, iface3);
    rank3->AddNetInstance(fabGroup);
    fabGroup->AddRankId(3);

    this->InitInnerRanks();

    // rank0: 连接rank1， 连接rank2
    AddLinkStub(fabGroup, rank0, rank1, iface0, iface1, LinkType::PEER2PEER, {LinkProtocol::ROCE});
    AddLinkStub(fabGroup, rank0, rank2, iface0, iface2, LinkType::PEER2PEER, {LinkProtocol::ROCE});
    AddLinkStub(fabGroup, rank0, rank3, iface0, iface3, LinkType::PEER2PEER, {LinkProtocol::ROCE});
    AddLinkStub(fabGroup, rank0, rank1, iface0, iface1, LinkType::PEER2PEER, {LinkProtocol::ROCE});
    AddLinkStub(fabGroup, rank0, rank2, iface0, iface2, LinkType::PEER2PEER, {LinkProtocol::ROCE});
    AddLinkStub(fabGroup, rank0, rank3, iface0, iface3, LinkType::PEER2PEER, {LinkProtocol::ROCE});

    // rank1: 连接rank0，连接rank3
    AddLinkStub(fabGroup, rank1, rank0, iface1, iface0, LinkType::PEER2PEER, {LinkProtocol::ROCE});
    AddLinkStub(fabGroup, rank1, rank3, iface1, iface3, LinkType::PEER2PEER, {LinkProtocol::ROCE});
    AddLinkStub(fabGroup, rank1, rank2, iface1, iface2, LinkType::PEER2PEER, {LinkProtocol::ROCE});

    // rank2: 连接rank0，连接rank3
    AddLinkStub(fabGroup, rank2, rank0, iface2, iface0, LinkType::PEER2PEER, {LinkProtocol::ROCE});
    AddLinkStub(fabGroup, rank2, rank3, iface2, iface3, LinkType::PEER2PEER, {LinkProtocol::ROCE});
    AddLinkStub(fabGroup, rank2, rank1, iface2, iface1, LinkType::PEER2PEER, {LinkProtocol::ROCE});

    // rank3: 连接rank1，连接rank2
    AddLinkStub(fabGroup, rank3, rank1, iface3, iface1, LinkType::PEER2PEER, {LinkProtocol::ROCE});
    AddLinkStub(fabGroup, rank3, rank2, iface3, iface2, LinkType::PEER2PEER, {LinkProtocol::ROCE});
    AddLinkStub(fabGroup, rank3, rank0, iface3, iface0, LinkType::PEER2PEER, {LinkProtocol::ROCE});
}

void VirtualTopoStub::TopoInit91095TwoPodTwoTwoAndTwoTwo(const string &rankTable)// {{0,1},{8,9}},{{3,5},{19,21}}, 对称2D
{
    // 打桩virtual topo
    shared_ptr<NetInstance> fabGroup0 = InitNetInstance(0, "server0");
    shared_ptr<NetInstance> fabGroup1 = InitNetInstance(0, "server1");
    shared_ptr<NetInstance> fabGroupLevel1 = InitNetInstance(1, "clos0");
    this->netInsts_[0].emplace("server0", fabGroup0);
    this->netInsts_[0].emplace("server1", fabGroup1);
    this->netInsts_[1].emplace("clos0", fabGroupLevel1);
    // 初始化fabric0信息
    auto fabric0 = InitFabric(0);
    fabGroupLevel1->AddNode(fabric0);
    char fabricAddress0[] = "13.0.0.0";  // 打桩用sendIP地址
    IpAddress fabricAddr0(fabricAddress0);
    auto ifaceNode0 = InitConnInterface(fabricAddr0);
    fabric0->AddConnInterface(0, ifaceNode0);
    // 初始化fabric1信息
    auto fabric1 = InitFabric(1);
    fabGroupLevel1->AddNode(fabric1);
    char fabricAddress1[] = "14.0.0.0";  // 打桩用sendIP地址
    IpAddress fabricAddr1(fabricAddress1);
    auto ifaceNode1 = InitConnInterface(fabricAddr1);
    fabric1->AddConnInterface(1, ifaceNode1);

    // 初始化rank0信息
    auto rank0 = InitPeer(0, 0);
    this->AddPeer(rank0);  // 存储rank0信息到peers
    fabGroup0->AddNode(rank0);
    fabGroupLevel1->AddNode(rank0);
    char rank0Address0[] = "0.0.0.0";  // 打桩用sendIP地址
    IpAddress rank0Addr0(rank0Address0);
    auto rank0iface0 = InitConnInterface(rank0Addr0);
    rank0->AddConnInterface(0, rank0iface0);
    char rank0Address1[] = "0.0.0.1";  // 打桩用sendIP地址
    IpAddress rank0Addr1(rank0Address1);
    auto rank0iface1 = InitConnInterface(rank0Addr1);
    rank0->AddConnInterface(1, rank0iface1);
    rank0->AddNetInstance(fabGroup0);
    rank0->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(0);
    fabGroupLevel1->AddRankId(0);
 
    // 初始化rank1信息
    auto rank1 = InitPeer(1, 1);
    this->AddPeer(rank1);  // 存储rank0信息到peers
    fabGroup0->AddNode(rank1);
    fabGroupLevel1->AddNode(rank1);
    char rank1Address0[] = "1.0.0.0";  // 打桩用sendIP地址
    IpAddress rank1Addr0(rank1Address0);
    auto rank1iface0 = InitConnInterface(rank1Addr0);
    rank1->AddConnInterface(0, rank1iface0);
    char rank1Address1[] = "1.0.0.1";  // 打桩用sendIP地址
    IpAddress rank1Addr1(rank1Address1);
    auto rank1iface1 = InitConnInterface(rank1Addr1);
    rank1->AddConnInterface(1, rank1iface1);
    rank1->AddNetInstance(fabGroup0);
    rank1->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(1);
    fabGroupLevel1->AddRankId(1);

    // 初始化rank2信息
    auto rank2 = InitPeer(2, 8);
    this->AddPeer(rank2);  // 存储rank2信息到peers
    fabGroup0->AddNode(rank2);
    fabGroupLevel1->AddNode(rank2);
    char rank2Address0[] = "2.0.0.0";  // 打桩用sendIP地址
    IpAddress rank2Addr0(rank2Address0);
    auto rank2iface0 = InitConnInterface(rank2Addr0);
    rank2->AddConnInterface(0, rank2iface0);
    char rank2Address1[] = "2.0.0.1";  // 打桩用sendIP地址
    IpAddress rank2Addr1(rank2Address1);
    auto rank2iface1 = InitConnInterface(rank2Addr1);
    rank2->AddConnInterface(1, rank2iface1);
    rank2->AddNetInstance(fabGroup0);
    rank2->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(2);
    fabGroupLevel1->AddRankId(2);
 
    // 初始化rank3信息
    auto rank3 = InitPeer(3, 9);
    this->AddPeer(rank3);  // 存储rank3信息到peers
    fabGroup0->AddNode(rank3);
    fabGroupLevel1->AddNode(rank3);
    char rank3Address0[] = "3.0.0.0";  // 打桩用sendIP地址
    IpAddress rank3Addr0(rank3Address0);
    auto rank3iface0 = InitConnInterface(rank3Addr0);
    rank3->AddConnInterface(0, rank3iface0);
    char rank3Address1[] = "3.0.0.1";  // 打桩用sendIP地址
    IpAddress rank3Addr1(rank3Address1);
    auto rank3iface1 = InitConnInterface(rank3Addr1);
    rank3->AddConnInterface(1, rank3iface1);
    rank3->AddNetInstance(fabGroup0);
    rank3->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(3);
    fabGroupLevel1->AddRankId(3);

    // 初始化rank8信息（到第二个server了）
    auto rank8 = InitPeer(8, 3);
    this->AddPeer(rank8);  // 存储rank0信息到peers
    fabGroup1->AddNode(rank8);
    fabGroupLevel1->AddNode(rank8);
    char rank8Address0[] = "8.0.0.0";  // 打桩用sendIP地址
    IpAddress rank8Addr0(rank8Address0);
    auto rank8iface0 = InitConnInterface(rank8Addr0);
    rank8->AddConnInterface(0, rank8iface0);
    char rank8Address1[] = "8.0.0.1";  // 打桩用sendIP地址
    IpAddress rank8Addr1(rank8Address1);
    auto rank8iface1 = InitConnInterface(rank8Addr1);
    rank8->AddConnInterface(1, rank8iface1);
    rank8->AddNetInstance(fabGroup1);
    rank8->AddNetInstance(fabGroupLevel1);
    fabGroup1->AddRankId(8);
    fabGroupLevel1->AddRankId(8);
 
    // 初始化rank9信息
    auto rank9 = InitPeer(9, 5);
    this->AddPeer(rank9);  // 存储rank0信息到peers
    fabGroup1->AddNode(rank9);
    fabGroupLevel1->AddNode(rank9);
    char rank9Address0[] = "9.0.0.0";  // 打桩用sendIP地址
    IpAddress rank9Addr0(rank9Address0);
    auto rank9iface0 = InitConnInterface(rank9Addr0);
    rank9->AddConnInterface(0, rank9iface0);
    char rank9Address1[] = "9.0.0.1";  // 打桩用sendIP地址
    IpAddress rank9Addr1(rank9Address1);
    auto rank9iface1 = InitConnInterface(rank9Addr1);
    rank9->AddConnInterface(1, rank9iface1);
    rank9->AddNetInstance(fabGroup1);
    rank9->AddNetInstance(fabGroupLevel1);
    fabGroup1->AddRankId(9);
    fabGroupLevel1->AddRankId(9);

    // 初始化rank10信息
    auto rank10 = InitPeer(10, 19);
    this->AddPeer(rank10);  // 存储rank0信息到peers
    fabGroup1->AddNode(rank10);
    fabGroupLevel1->AddNode(rank10);
    char rank10Address0[] = "10.0.0.0";  // 打桩用sendIP地址
    IpAddress rank10Addr0(rank10Address0);
    auto rank10iface0 = InitConnInterface(rank10Addr0);
    rank10->AddConnInterface(0, rank10iface0);
    char rank10Address1[] = "10.0.0.1";  // 打桩用sendIP地址
    IpAddress rank10Addr1(rank10Address1);
    auto rank10iface1 = InitConnInterface(rank10Addr1);
    rank10->AddConnInterface(1, rank10iface1);
    rank10->AddNetInstance(fabGroup1);
    rank10->AddNetInstance(fabGroupLevel1);
    fabGroup1->AddRankId(10);
    fabGroupLevel1->AddRankId(10);

    // 初始化rank11信息
    auto rank11 = InitPeer(11, 21);
    this->AddPeer(rank11);  // 存储rank0信息到peers
    fabGroup1->AddNode(rank11);
    fabGroupLevel1->AddNode(rank11);
    char rank11Address0[] = "11.0.0.0";  // 打桩用sendIP地址
    IpAddress rank11Addr0(rank11Address0);
    auto rank11iface0 = InitConnInterface(rank11Addr0);
    rank11->AddConnInterface(0, rank11iface0);
    char rank11Address1[] = "11.0.0.1";  // 打桩用sendIP地址
    IpAddress rank11Addr1(rank11Address1);
    auto rank11iface1 = InitConnInterface(rank11Addr1);
    rank11->AddConnInterface(1, rank11iface1);
    rank11->AddNetInstance(fabGroup1);
    rank11->AddNetInstance(fabGroupLevel1);
    fabGroup1->AddRankId(11);
    fabGroupLevel1->AddRankId(11);

    this->InitInnerRanks();
    
    //rank0，连接rank1，2，4，6
    AddLinkStub(fabGroup0, rank0, rank1, rank0iface0, rank1iface0);
    AddLinkStub(fabGroup0, rank0, rank2, rank0iface0, rank2iface0);
    AddLinkStub(fabGroupLevel1, rank0, fabric0, rank0iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank0, fabric1, rank0iface1, ifaceNode1);
    
    //rank1，连接rank0，3，5，7
    AddLinkStub(fabGroup0, rank1, rank0, rank1iface0, rank0iface0);
    AddLinkStub(fabGroup0, rank1, rank3, rank1iface0, rank3iface0);
    AddLinkStub(fabGroupLevel1, rank1, fabric0, rank1iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank1, fabric1, rank1iface1, ifaceNode1);
    
    //rank2，连接rank0，3，4，6
    AddLinkStub(fabGroup1, rank2, rank0, rank2iface0, rank0iface0);
    AddLinkStub(fabGroup1, rank2, rank3, rank2iface0, rank3iface0);
    AddLinkStub(fabGroupLevel1, rank2, fabric0, rank2iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank2, fabric1, rank2iface1, ifaceNode1);

    //rank3，连接rank1，2，5，7
    AddLinkStub(fabGroup1, rank3, rank1, rank3iface0, rank1iface0);
    AddLinkStub(fabGroup1, rank3, rank2, rank3iface0, rank2iface0);
    AddLinkStub(fabGroupLevel1, rank3, fabric0, rank3iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank3, fabric1, rank3iface1, ifaceNode1);

    //rank8，连接rank9，10
    AddLinkStub(fabGroup1, rank8, rank9, rank8iface0, rank9iface0);
    AddLinkStub(fabGroup1, rank8, rank10, rank8iface0, rank10iface0);
    AddLinkStub(fabGroupLevel1, rank8, fabric0, rank8iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank8, fabric1, rank8iface1, ifaceNode1);

    //rank9，连接rank8，11
    AddLinkStub(fabGroup1, rank9, rank8, rank9iface0, rank8iface0);
    AddLinkStub(fabGroup1, rank9, rank11, rank9iface0, rank11iface0);
    AddLinkStub(fabGroupLevel1, rank9, fabric0, rank9iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank9, fabric1, rank9iface1, ifaceNode1);

    //rank10，连接rank8，11
    AddLinkStub(fabGroup1, rank10, rank8, rank10iface0, rank8iface0);
    AddLinkStub(fabGroup1, rank10, rank11, rank10iface0, rank11iface0);
    AddLinkStub(fabGroupLevel1, rank10, fabric0, rank10iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank10, fabric1, rank10iface1, ifaceNode1);

    //rank11，连接rank9,10
    AddLinkStub(fabGroup1, rank11, rank9, rank11iface0, rank9iface0);
    AddLinkStub(fabGroup1, rank11, rank10, rank11iface0, rank10iface0);
    AddLinkStub(fabGroupLevel1, rank11, fabric0, rank11iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank11, fabric1, rank11iface1, ifaceNode1);


    AddLinkStub(fabGroupLevel1, fabric0, rank0, ifaceNode0, rank0iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank1, ifaceNode0, rank1iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank2, ifaceNode0, rank2iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank3, ifaceNode0, rank3iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank8, ifaceNode0, rank8iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank9, ifaceNode0, rank9iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank10, ifaceNode0, rank10iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank11, ifaceNode0, rank11iface0);

    AddLinkStub(fabGroupLevel1, fabric1, rank0, ifaceNode1, rank0iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank1, ifaceNode1, rank1iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank2, ifaceNode1, rank2iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank3, ifaceNode1, rank3iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank8, ifaceNode1, rank8iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank9, ifaceNode1, rank9iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank10, ifaceNode1, rank10iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank11, ifaceNode1, rank11iface1);
}

void VirtualTopoStub::TopoInit91095TwoPodFourTwoAndTwoTwo(const string &rankTable)//{{0,1},{8,9},{24,25},{32,33}},{{3,5},{19,21}},非对称2D
{
    // 打桩virtual topo
    shared_ptr<NetInstance> fabGroup0 = InitNetInstance(0, "server0");
    shared_ptr<NetInstance> fabGroup1 = InitNetInstance(0, "server1");
    shared_ptr<NetInstance> fabGroupLevel1 = InitNetInstance(1, "clos0");
    this->netInsts_[0].emplace("server0", fabGroup0);
    this->netInsts_[0].emplace("server1", fabGroup1);
    this->netInsts_[1].emplace("clos0", fabGroupLevel1);
    // 初始化fabric0信息
    auto fabric0 = InitFabric(0);
    fabGroupLevel1->AddNode(fabric0);
    char fabricAddress0[] = "13.0.0.0";  // 打桩用sendIP地址
    IpAddress fabricAddr0(fabricAddress0);
    auto ifaceNode0 = InitConnInterface(fabricAddr0);
    fabric0->AddConnInterface(0, ifaceNode0);
    // 初始化fabric1信息
    auto fabric1 = InitFabric(1);
    fabGroupLevel1->AddNode(fabric1);
    char fabricAddress1[] = "14.0.0.0";  // 打桩用sendIP地址
    IpAddress fabricAddr1(fabricAddress1);
    auto ifaceNode1 = InitConnInterface(fabricAddr1);
    fabric1->AddConnInterface(1, ifaceNode1);


    // 初始化rank0信息
    auto rank0 = InitPeer(0, 0);
    this->AddPeer(rank0);  // 存储rank0信息到peers
    fabGroup0->AddNode(rank0);
    fabGroupLevel1->AddNode(rank0);
    char rank0Address0[] = "0.0.0.0";  // 打桩用sendIP地址
    IpAddress rank0Addr0(rank0Address0);
    auto rank0iface0 = InitConnInterface(rank0Addr0);
    rank0->AddConnInterface(0, rank0iface0);
    char rank0Address1[] = "0.0.0.1";  // 打桩用sendIP地址
    IpAddress rank0Addr1(rank0Address1);
    auto rank0iface1 = InitConnInterface(rank0Addr1);
    rank0->AddConnInterface(1, rank0iface1);
    rank0->AddNetInstance(fabGroup0);
    rank0->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(0);
    fabGroupLevel1->AddRankId(0);
 
    // 初始化rank1信息
    auto rank1 = InitPeer(1, 1);
    this->AddPeer(rank1);  // 存储rank0信息到peers
    fabGroup0->AddNode(rank1);
    fabGroupLevel1->AddNode(rank1);
    char rank1Address0[] = "1.0.0.0";  // 打桩用sendIP地址
    IpAddress rank1Addr0(rank1Address0);
    auto rank1iface0 = InitConnInterface(rank1Addr0);
    rank1->AddConnInterface(0, rank1iface0);
    char rank1Address1[] = "1.0.0.1";  // 打桩用sendIP地址
    IpAddress rank1Addr1(rank1Address1);
    auto rank1iface1 = InitConnInterface(rank1Addr1);
    rank1->AddConnInterface(1, rank1iface1);
    rank1->AddNetInstance(fabGroup0);
    rank1->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(1);
    fabGroupLevel1->AddRankId(1);

    // 初始化rank2信息
    auto rank2 = InitPeer(2, 8);
    this->AddPeer(rank2);  // 存储rank2信息到peers
    fabGroup0->AddNode(rank2);
    fabGroupLevel1->AddNode(rank2);
    char rank2Address0[] = "2.0.0.0";  // 打桩用sendIP地址
    IpAddress rank2Addr0(rank2Address0);
    auto rank2iface0 = InitConnInterface(rank2Addr0);
    rank2->AddConnInterface(0, rank2iface0);
    char rank2Address1[] = "2.0.0.1";  // 打桩用sendIP地址
    IpAddress rank2Addr1(rank2Address1);
    auto rank2iface1 = InitConnInterface(rank2Addr1);
    rank2->AddConnInterface(1, rank2iface1);
    rank2->AddNetInstance(fabGroup0);
    rank2->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(2);
    fabGroupLevel1->AddRankId(2);
 
    // 初始化rank3信息
    auto rank3 = InitPeer(3, 9);
    this->AddPeer(rank3);  // 存储rank3信息到peers
    fabGroup0->AddNode(rank3);
    fabGroupLevel1->AddNode(rank3);
    char rank3Address0[] = "3.0.0.0";  // 打桩用sendIP地址
    IpAddress rank3Addr0(rank3Address0);
    auto rank3iface0 = InitConnInterface(rank3Addr0);
    rank3->AddConnInterface(0, rank3iface0);
    char rank3Address1[] = "3.0.0.1";  // 打桩用sendIP地址
    IpAddress rank3Addr1(rank3Address1);
    auto rank3iface1 = InitConnInterface(rank3Addr1);
    rank3->AddConnInterface(1, rank3iface1);
    rank3->AddNetInstance(fabGroup0);
    rank3->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(3);
    fabGroupLevel1->AddRankId(3);

    // 初始化rank4信息
    auto rank4 = InitPeer(4, 24);
    this->AddPeer(rank4);  // 存储rank0信息到peers
    fabGroup0->AddNode(rank4);
    fabGroupLevel1->AddNode(rank4);
    char rank4Address0[] = "4.0.0.0";  // 打桩用sendIP地址
    IpAddress rank4Addr0(rank4Address0);
    auto rank4iface0 = InitConnInterface(rank4Addr0);
    rank4->AddConnInterface(0, rank4iface0);
    char rank4Address1[] = "4.0.0.1";  // 打桩用sendIP地址
    IpAddress rank4Addr1(rank4Address1);
    auto rank4iface1 = InitConnInterface(rank4Addr1);
    rank4->AddConnInterface(1, rank4iface1);
    rank4->AddNetInstance(fabGroup0);
    rank4->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(4);
    fabGroupLevel1->AddRankId(4);
 
    // 初始化rank5信息
    auto rank5 = InitPeer(5, 25);
    this->AddPeer(rank5);  // 存储rank5信息到peers
    fabGroup0->AddNode(rank5);
    fabGroupLevel1->AddNode(rank5);
    char rank5Address0[] = "5.0.0.0";  // 打桩用sendIP地址
    IpAddress rank5Addr0(rank5Address0);
    auto rank5iface0 = InitConnInterface(rank5Addr0);
    rank5->AddConnInterface(0, rank5iface0);
    char rank5Address1[] = "5.0.0.1";  // 打桩用sendIP地址
    IpAddress rank5Addr1(rank5Address1);
    auto rank5iface1 = InitConnInterface(rank5Addr1);
    rank5->AddConnInterface(1, rank5iface1);
    rank5->AddNetInstance(fabGroup0);
    rank5->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(5);
    fabGroupLevel1->AddRankId(5);

    // 初始化rank6信息
    auto rank6 = InitPeer(6, 32);
    this->AddPeer(rank6);  // 存储rank6信息到peers
    fabGroup0->AddNode(rank6);
    fabGroupLevel1->AddNode(rank6);
    char rank6Address0[] = "6.0.0.0";  // 打桩用sendIP地址
    IpAddress rank6Addr0(rank6Address0);
    auto rank6iface0 = InitConnInterface(rank6Addr0);
    rank6->AddConnInterface(0, rank6iface0);
    char rank6Address1[] = "6.0.0.1";  // 打桩用sendIP地址
    IpAddress rank6Addr1(rank6Address1);
    auto rank6iface1 = InitConnInterface(rank6Addr1);
    rank6->AddConnInterface(1, rank6iface1);
    rank6->AddNetInstance(fabGroup0);
    rank6->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(6);
    fabGroupLevel1->AddRankId(6);
 
    // 初始化rank7信息
    auto rank7 = InitPeer(7, 33);
    this->AddPeer(rank7);  // 存储rank7信息到peers
    fabGroup0->AddNode(rank7);
    fabGroupLevel1->AddNode(rank7);
    char rank7Address0[] = "7.0.0.0";  // 打桩用sendIP地址
    IpAddress rank7Addr0(rank7Address0);
    auto rank7iface0 = InitConnInterface(rank7Addr0);
    rank7->AddConnInterface(0, rank7iface0);
    char rank7Address1[] = "7.0.0.1";  // 打桩用sendIP地址
    IpAddress rank7Addr1(rank7Address1);
    auto rank7iface1 = InitConnInterface(rank7Addr1);
    rank7->AddConnInterface(1, rank7iface1);
    rank7->AddNetInstance(fabGroup0);
    rank7->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(7);
    fabGroupLevel1->AddRankId(7);
 
    // 初始化rank8信息（到第二个server了）
    auto rank8 = InitPeer(8, 3);
    this->AddPeer(rank8);  // 存储rank0信息到peers
    fabGroup1->AddNode(rank8);
    fabGroupLevel1->AddNode(rank8);
    char rank8Address0[] = "8.0.0.0";  // 打桩用sendIP地址
    IpAddress rank8Addr0(rank8Address0);
    auto rank8iface0 = InitConnInterface(rank8Addr0);
    rank8->AddConnInterface(0, rank8iface0);
    char rank8Address1[] = "8.0.0.1";  // 打桩用sendIP地址
    IpAddress rank8Addr1(rank8Address1);
    auto rank8iface1 = InitConnInterface(rank8Addr1);
    rank8->AddConnInterface(1, rank8iface1);
    rank8->AddNetInstance(fabGroup1);
    rank8->AddNetInstance(fabGroupLevel1);
    fabGroup1->AddRankId(8);
    fabGroupLevel1->AddRankId(8);
 
    // 初始化rank9信息
    auto rank9 = InitPeer(9, 5);
    this->AddPeer(rank9);  // 存储rank0信息到peers
    fabGroup1->AddNode(rank9);
    fabGroupLevel1->AddNode(rank9);
    char rank9Address0[] = "9.0.0.0";  // 打桩用sendIP地址
    IpAddress rank9Addr0(rank9Address0);
    auto rank9iface0 = InitConnInterface(rank9Addr0);
    rank9->AddConnInterface(0, rank9iface0);
    char rank9Address1[] = "9.0.0.1";  // 打桩用sendIP地址
    IpAddress rank9Addr1(rank9Address1);
    auto rank9iface1 = InitConnInterface(rank9Addr1);
    rank9->AddConnInterface(1, rank9iface1);
    rank9->AddNetInstance(fabGroup1);
    rank9->AddNetInstance(fabGroupLevel1);
    fabGroup1->AddRankId(9);
    fabGroupLevel1->AddRankId(9);

    // 初始化rank10信息
    auto rank10 = InitPeer(10, 19);
    this->AddPeer(rank10);  // 存储rank0信息到peers
    fabGroup1->AddNode(rank10);
    fabGroupLevel1->AddNode(rank10);
    char rank10Address0[] = "10.0.0.0";  // 打桩用sendIP地址
    IpAddress rank10Addr0(rank10Address0);
    auto rank10iface0 = InitConnInterface(rank10Addr0);
    rank10->AddConnInterface(0, rank10iface0);
    char rank10Address1[] = "10.0.0.1";  // 打桩用sendIP地址
    IpAddress rank10Addr1(rank10Address1);
    auto rank10iface1 = InitConnInterface(rank10Addr1);
    rank10->AddConnInterface(1, rank10iface1);
    rank10->AddNetInstance(fabGroup1);
    rank10->AddNetInstance(fabGroupLevel1);
    fabGroup1->AddRankId(10);
    fabGroupLevel1->AddRankId(10);

    // 初始化rank11信息
    auto rank11 = InitPeer(11, 21);
    this->AddPeer(rank11);  // 存储rank0信息到peers
    fabGroup1->AddNode(rank11);
    fabGroupLevel1->AddNode(rank11);
    char rank11Address0[] = "11.0.0.0";  // 打桩用sendIP地址
    IpAddress rank11Addr0(rank11Address0);
    auto rank11iface0 = InitConnInterface(rank11Addr0);
    rank11->AddConnInterface(0, rank11iface0);
    char rank11Address1[] = "11.0.0.1";  // 打桩用sendIP地址
    IpAddress rank11Addr1(rank11Address1);
    auto rank11iface1 = InitConnInterface(rank11Addr1);
    rank11->AddConnInterface(1, rank11iface1);
    rank11->AddNetInstance(fabGroup1);
    rank11->AddNetInstance(fabGroupLevel1);
    fabGroup1->AddRankId(11);
    fabGroupLevel1->AddRankId(11);

    this->InitInnerRanks();
    
    //rank0，连接rank1，2，4，6
    AddLinkStub(fabGroup0, rank0, rank1, rank0iface0, rank1iface0);
    AddLinkStub(fabGroup0, rank0, rank2, rank0iface0, rank2iface0);
    AddLinkStub(fabGroup0, rank0, rank4, rank0iface0, rank4iface0);
    AddLinkStub(fabGroup0, rank0, rank6, rank0iface0, rank6iface0);
    AddLinkStub(fabGroupLevel1, rank0, fabric0, rank0iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank0, fabric1, rank0iface1, ifaceNode1);
    
    //rank1，连接rank0，3，5，7
    AddLinkStub(fabGroup0, rank1, rank0, rank1iface0, rank0iface0);
    AddLinkStub(fabGroup0, rank1, rank3, rank1iface0, rank3iface0);
    AddLinkStub(fabGroup0, rank1, rank5, rank1iface0, rank5iface0);
    AddLinkStub(fabGroup0, rank1, rank7, rank1iface0, rank7iface0);
    AddLinkStub(fabGroupLevel1, rank1, fabric0, rank1iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank1, fabric1, rank1iface1, ifaceNode1);
    
    //rank2，连接rank0，3，4，6
    AddLinkStub(fabGroup1, rank2, rank0, rank2iface0, rank0iface0);
    AddLinkStub(fabGroup1, rank2, rank3, rank2iface0, rank3iface0);
    AddLinkStub(fabGroup1, rank2, rank4, rank2iface0, rank4iface0);
    AddLinkStub(fabGroup1, rank2, rank6, rank2iface0, rank6iface0);
    AddLinkStub(fabGroupLevel1, rank2, fabric0, rank2iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank2, fabric1, rank2iface1, ifaceNode1);

    //rank3，连接rank1，2，5，7
    AddLinkStub(fabGroup1, rank3, rank1, rank3iface0, rank1iface0);
    AddLinkStub(fabGroup1, rank3, rank2, rank3iface0, rank2iface0);
    AddLinkStub(fabGroup1, rank3, rank5, rank3iface0, rank5iface0);
    AddLinkStub(fabGroup1, rank3, rank7, rank3iface0, rank7iface0);
    AddLinkStub(fabGroupLevel1, rank3, fabric0, rank3iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank3, fabric1, rank3iface1, ifaceNode1);

    //rank4，连接rank0，2，5，6
    AddLinkStub(fabGroup1, rank4, rank0, rank4iface0, rank0iface0);
    AddLinkStub(fabGroup1, rank4, rank2, rank4iface0, rank2iface0);
    AddLinkStub(fabGroup1, rank4, rank5, rank4iface0, rank5iface0);
    AddLinkStub(fabGroup1, rank4, rank6, rank4iface0, rank6iface0);
    AddLinkStub(fabGroupLevel1, rank4, fabric0, rank4iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank4, fabric1, rank4iface1, ifaceNode1);

    //rank5，连接rank1，3，4，7
    AddLinkStub(fabGroup1, rank5, rank1, rank5iface0, rank1iface0);
    AddLinkStub(fabGroup1, rank5, rank3, rank5iface0, rank3iface0);
    AddLinkStub(fabGroup1, rank5, rank4, rank5iface0, rank4iface0);
    AddLinkStub(fabGroup1, rank5, rank7, rank5iface0, rank7iface0);
    AddLinkStub(fabGroupLevel1, rank5, fabric0, rank5iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank5, fabric1, rank5iface1, ifaceNode1);

    //rank6，连接rank0，2，4，7
    AddLinkStub(fabGroup1, rank6, rank0, rank6iface0, rank0iface0);
    AddLinkStub(fabGroup1, rank6, rank2, rank6iface0, rank2iface0);
    AddLinkStub(fabGroup1, rank6, rank4, rank6iface0, rank4iface0);
    AddLinkStub(fabGroup1, rank6, rank7, rank6iface0, rank7iface0);
    AddLinkStub(fabGroupLevel1, rank6, fabric0, rank6iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank6, fabric1, rank6iface1, ifaceNode1);

    //rank7，连接rank1，3，5，6
    AddLinkStub(fabGroup1, rank7, rank1, rank7iface0, rank1iface0);
    AddLinkStub(fabGroup1, rank7, rank3, rank7iface0, rank3iface0);
    AddLinkStub(fabGroup1, rank7, rank5, rank7iface0, rank5iface0);
    AddLinkStub(fabGroup1, rank7, rank6, rank7iface0, rank6iface0);
    AddLinkStub(fabGroupLevel1, rank7, fabric0, rank7iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank7, fabric1, rank7iface1, ifaceNode1);


    //rank8，连接rank9，10
    AddLinkStub(fabGroup1, rank8, rank9, rank8iface0, rank9iface0);
    AddLinkStub(fabGroup1, rank8, rank10, rank8iface0, rank10iface0);
    AddLinkStub(fabGroupLevel1, rank8, fabric0, rank8iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank8, fabric1, rank8iface1, ifaceNode1);

    //rank9，连接rank8，11
    AddLinkStub(fabGroup1, rank9, rank8, rank9iface0, rank8iface0);
    AddLinkStub(fabGroup1, rank9, rank11, rank9iface0, rank11iface0);
    AddLinkStub(fabGroupLevel1, rank9, fabric0, rank9iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank9, fabric1, rank9iface1, ifaceNode1);

    //rank10，连接rank8，11
    AddLinkStub(fabGroup1, rank10, rank8, rank10iface0, rank8iface0);
    AddLinkStub(fabGroup1, rank10, rank11, rank10iface0, rank11iface0);
    AddLinkStub(fabGroupLevel1, rank10, fabric0, rank10iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank10, fabric1, rank10iface1, ifaceNode1);

    //rank11，连接rank9,10
    AddLinkStub(fabGroup1, rank11, rank9, rank11iface0, rank9iface0);
    AddLinkStub(fabGroup1, rank11, rank10, rank11iface0, rank10iface0);
    AddLinkStub(fabGroupLevel1, rank11, fabric0, rank11iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank11, fabric1, rank11iface1, ifaceNode1);


    AddLinkStub(fabGroupLevel1, fabric0, rank0, ifaceNode0, rank0iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank1, ifaceNode0, rank1iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank2, ifaceNode0, rank2iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank3, ifaceNode0, rank3iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank4, ifaceNode0, rank4iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank5, ifaceNode0, rank5iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank6, ifaceNode0, rank6iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank7, ifaceNode0, rank7iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank8, ifaceNode0, rank8iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank9, ifaceNode0, rank9iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank10, ifaceNode0, rank10iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank11, ifaceNode0, rank11iface0);

    AddLinkStub(fabGroupLevel1, fabric1, rank0, ifaceNode1, rank0iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank1, ifaceNode1, rank1iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank2, ifaceNode1, rank2iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank3, ifaceNode1, rank3iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank4, ifaceNode1, rank4iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank5, ifaceNode1, rank5iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank6, ifaceNode1, rank6iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank7, ifaceNode1, rank7iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank8, ifaceNode1, rank8iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank9, ifaceNode1, rank9iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank10, ifaceNode1, rank10iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank11, ifaceNode1, rank11iface1);
}

void VirtualTopoStub::TopoInit91095TwoPodIrregularEightAndIrregularFour(const string &rankTable)//{{0,2},{9,11},{16,18},{25,27}},{{4,6,7},{12}}
{
    // 打桩virtual topo
    shared_ptr<NetInstance> fabGroup0 = InitNetInstance(0, "server0");
    shared_ptr<NetInstance> fabGroup1 = InitNetInstance(0, "server1");
    shared_ptr<NetInstance> fabGroupLevel1 = InitNetInstance(1, "clos0");
    this->netInsts_[0].emplace("server0", fabGroup0);
    this->netInsts_[0].emplace("server1", fabGroup1);
    this->netInsts_[1].emplace("clos0", fabGroupLevel1);
    // 初始化fabric0信息
    auto fabric0 = InitFabric(0);
    fabGroupLevel1->AddNode(fabric0);
    char fabricAddress0[] = "13.0.0.0";  // 打桩用sendIP地址
    IpAddress fabricAddr0(fabricAddress0);
    auto ifaceNode0 = InitConnInterface(fabricAddr0);
    fabric0->AddConnInterface(0, ifaceNode0);
    // 初始化fabric1信息
    auto fabric1 = InitFabric(1);
    fabGroupLevel1->AddNode(fabric1);
    char fabricAddress1[] = "14.0.0.0";  // 打桩用sendIP地址
    IpAddress fabricAddr1(fabricAddress1);
    auto ifaceNode1 = InitConnInterface(fabricAddr1);
    fabric1->AddConnInterface(1, ifaceNode1);


    // 初始化rank0信息
    auto rank0 = InitPeer(0, 0);
    this->AddPeer(rank0);  // 存储rank0信息到peers
    fabGroup0->AddNode(rank0);
    fabGroupLevel1->AddNode(rank0);
    char rank0Address0[] = "0.0.0.0";  // 打桩用sendIP地址
    IpAddress rank0Addr0(rank0Address0);
    auto rank0iface0 = InitConnInterface(rank0Addr0);
    rank0->AddConnInterface(0, rank0iface0);
    char rank0Address1[] = "0.0.0.1";  // 打桩用sendIP地址
    IpAddress rank0Addr1(rank0Address1);
    auto rank0iface1 = InitConnInterface(rank0Addr1);
    rank0->AddConnInterface(1, rank0iface1);
    rank0->AddNetInstance(fabGroup0);
    rank0->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(0);
    fabGroupLevel1->AddRankId(0);
 
    // 初始化rank1信息
    auto rank1 = InitPeer(1, 2);
    this->AddPeer(rank1);  // 存储rank0信息到peers
    fabGroup0->AddNode(rank1);
    fabGroupLevel1->AddNode(rank1);
    char rank1Address0[] = "1.0.0.0";  // 打桩用sendIP地址
    IpAddress rank1Addr0(rank1Address0);
    auto rank1iface0 = InitConnInterface(rank1Addr0);
    rank1->AddConnInterface(0, rank1iface0);
    char rank1Address1[] = "1.0.0.1";  // 打桩用sendIP地址
    IpAddress rank1Addr1(rank1Address1);
    auto rank1iface1 = InitConnInterface(rank1Addr1);
    rank1->AddConnInterface(1, rank1iface1);
    rank1->AddNetInstance(fabGroup0);
    rank1->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(1);
    fabGroupLevel1->AddRankId(1);

    // 初始化rank2信息
    auto rank2 = InitPeer(2, 9);
    this->AddPeer(rank2);  // 存储rank2信息到peers
    fabGroup0->AddNode(rank2);
    fabGroupLevel1->AddNode(rank2);
    char rank2Address0[] = "2.0.0.0";  // 打桩用sendIP地址
    IpAddress rank2Addr0(rank2Address0);
    auto rank2iface0 = InitConnInterface(rank2Addr0);
    rank2->AddConnInterface(0, rank2iface0);
    char rank2Address1[] = "2.0.0.1";  // 打桩用sendIP地址
    IpAddress rank2Addr1(rank2Address1);
    auto rank2iface1 = InitConnInterface(rank2Addr1);
    rank2->AddConnInterface(1, rank2iface1);
    rank2->AddNetInstance(fabGroup0);
    rank2->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(2);
    fabGroupLevel1->AddRankId(2);
 
    // 初始化rank3信息
    auto rank3 = InitPeer(3, 11);
    this->AddPeer(rank3);  // 存储rank3信息到peers
    fabGroup0->AddNode(rank3);
    fabGroupLevel1->AddNode(rank3);
    char rank3Address0[] = "3.0.0.0";  // 打桩用sendIP地址
    IpAddress rank3Addr0(rank3Address0);
    auto rank3iface0 = InitConnInterface(rank3Addr0);
    rank3->AddConnInterface(0, rank3iface0);
    char rank3Address1[] = "3.0.0.1";  // 打桩用sendIP地址
    IpAddress rank3Addr1(rank3Address1);
    auto rank3iface1 = InitConnInterface(rank3Addr1);
    rank3->AddConnInterface(1, rank3iface1);
    rank3->AddNetInstance(fabGroup0);
    rank3->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(3);
    fabGroupLevel1->AddRankId(3);

    // 初始化rank4信息
    auto rank4 = InitPeer(4, 16);
    this->AddPeer(rank4);  // 存储rank4信息到peers
    fabGroup0->AddNode(rank4);
    fabGroupLevel1->AddNode(rank4);
    char rank4Address0[] = "4.0.0.0";  // 打桩用sendIP地址
    IpAddress rank4Addr0(rank4Address0);
    auto rank4iface0 = InitConnInterface(rank4Addr0);
    rank4->AddConnInterface(0, rank4iface0);
    char rank4Address1[] = "4.0.0.1";  // 打桩用sendIP地址
    IpAddress rank4Addr1(rank4Address1);
    auto rank4iface1 = InitConnInterface(rank4Addr1);
    rank4->AddConnInterface(1, rank4iface1);
    rank4->AddNetInstance(fabGroup0);
    rank4->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(4);
    fabGroupLevel1->AddRankId(4);
 
    // 初始化rank5信息
    auto rank5 = InitPeer(5, 18);
    this->AddPeer(rank5);  // 存储rank5信息到peers
    fabGroup0->AddNode(rank5);
    fabGroupLevel1->AddNode(rank5);
    char rank5Address0[] = "5.0.0.0";  // 打桩用sendIP地址
    IpAddress rank5Addr0(rank5Address0);
    auto rank5iface0 = InitConnInterface(rank5Addr0);
    rank5->AddConnInterface(0, rank5iface0);
    char rank5Address1[] = "5.0.0.1";  // 打桩用sendIP地址
    IpAddress rank5Addr1(rank5Address1);
    auto rank5iface1 = InitConnInterface(rank5Addr1);
    rank5->AddConnInterface(1, rank5iface1);
    rank5->AddNetInstance(fabGroup0);
    rank5->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(5);
    fabGroupLevel1->AddRankId(5);

    // 初始化rank6信息
    auto rank6 = InitPeer(6, 25);
    this->AddPeer(rank6);  // 存储rank6信息到peers
    fabGroup0->AddNode(rank6);
    fabGroupLevel1->AddNode(rank6);
    char rank6Address0[] = "6.0.0.0";  // 打桩用sendIP地址
    IpAddress rank6Addr0(rank6Address0);
    auto rank6iface0 = InitConnInterface(rank6Addr0);
    rank6->AddConnInterface(0, rank6iface0);
    char rank6Address1[] = "6.0.0.1";  // 打桩用sendIP地址
    IpAddress rank6Addr1(rank6Address1);
    auto rank6iface1 = InitConnInterface(rank6Addr1);
    rank6->AddConnInterface(1, rank6iface1);
    rank6->AddNetInstance(fabGroup0);
    rank6->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(6);
    fabGroupLevel1->AddRankId(6);
 
    // 初始化rank7信息
    auto rank7 = InitPeer(7, 27);
    this->AddPeer(rank7);  // 存储rank7信息到peers
    fabGroup0->AddNode(rank7);
    fabGroupLevel1->AddNode(rank7);
    char rank7Address0[] = "7.0.0.0";  // 打桩用sendIP地址
    IpAddress rank7Addr0(rank7Address0);
    auto rank7iface0 = InitConnInterface(rank7Addr0);
    rank7->AddConnInterface(0, rank7iface0);
    char rank7Address1[] = "7.0.0.1";  // 打桩用sendIP地址
    IpAddress rank7Addr1(rank7Address1);
    auto rank7iface1 = InitConnInterface(rank7Addr1);
    rank7->AddConnInterface(1, rank7iface1);
    rank7->AddNetInstance(fabGroup0);
    rank7->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(7);
    fabGroupLevel1->AddRankId(7);
 
    // 初始化rank8信息（到第二个server了）
    auto rank8 = InitPeer(8, 4);
    this->AddPeer(rank8);  // 存储rank0信息到peers
    fabGroup1->AddNode(rank8);
    fabGroupLevel1->AddNode(rank8);
    char rank8Address0[] = "8.0.0.0";  // 打桩用sendIP地址
    IpAddress rank8Addr0(rank8Address0);
    auto rank8iface0 = InitConnInterface(rank8Addr0);
    rank8->AddConnInterface(0, rank8iface0);
    char rank8Address1[] = "8.0.0.1";  // 打桩用sendIP地址
    IpAddress rank8Addr1(rank8Address1);
    auto rank8iface1 = InitConnInterface(rank8Addr1);
    rank8->AddConnInterface(1, rank8iface1);
    rank8->AddNetInstance(fabGroup1);
    rank8->AddNetInstance(fabGroupLevel1);
    fabGroup1->AddRankId(8);
    fabGroupLevel1->AddRankId(8);
 
    // 初始化rank9信息
    auto rank9 = InitPeer(9, 6);
    this->AddPeer(rank9);  // 存储rank0信息到peers
    fabGroup1->AddNode(rank9);
    fabGroupLevel1->AddNode(rank9);
    char rank9Address0[] = "9.0.0.0";  // 打桩用sendIP地址
    IpAddress rank9Addr0(rank9Address0);
    auto rank9iface0 = InitConnInterface(rank9Addr0);
    rank9->AddConnInterface(0, rank9iface0);
    char rank9Address1[] = "9.0.0.1";  // 打桩用sendIP地址
    IpAddress rank9Addr1(rank9Address1);
    auto rank9iface1 = InitConnInterface(rank9Addr1);
    rank9->AddConnInterface(1, rank9iface1);
    rank9->AddNetInstance(fabGroup1);
    rank9->AddNetInstance(fabGroupLevel1);
    fabGroup1->AddRankId(9);
    fabGroupLevel1->AddRankId(9);

    // 初始化rank10信息
    auto rank10 = InitPeer(10, 7);
    this->AddPeer(rank10);  // 存储rank0信息到peers
    fabGroup1->AddNode(rank10);
    fabGroupLevel1->AddNode(rank10);
    char rank10Address0[] = "10.0.0.0";  // 打桩用sendIP地址
    IpAddress rank10Addr0(rank10Address0);
    auto rank10iface0 = InitConnInterface(rank10Addr0);
    rank10->AddConnInterface(0, rank10iface0);
    char rank10Address1[] = "10.0.0.1";  // 打桩用sendIP地址
    IpAddress rank10Addr1(rank10Address1);
    auto rank10iface1 = InitConnInterface(rank10Addr1);
    rank10->AddConnInterface(1, rank10iface1);
    rank10->AddNetInstance(fabGroup1);
    rank10->AddNetInstance(fabGroupLevel1);
    fabGroup1->AddRankId(10);
    fabGroupLevel1->AddRankId(10);

    // 初始化rank11信息
    auto rank11 = InitPeer(11, 12);
    this->AddPeer(rank11);  // 存储rank0信息到peers
    fabGroup1->AddNode(rank11);
    fabGroupLevel1->AddNode(rank11);
    char rank11Address0[] = "11.0.0.0";  // 打桩用sendIP地址
    IpAddress rank11Addr0(rank11Address0);
    auto rank11iface0 = InitConnInterface(rank11Addr0);
    rank11->AddConnInterface(0, rank11iface0);
    char rank11Address1[] = "11.0.0.1";  // 打桩用sendIP地址
    IpAddress rank11Addr1(rank11Address1);
    auto rank11iface1 = InitConnInterface(rank11Addr1);
    rank11->AddConnInterface(1, rank11iface1);
    rank11->AddNetInstance(fabGroup1);
    rank11->AddNetInstance(fabGroupLevel1);
    fabGroup1->AddRankId(11);
    fabGroupLevel1->AddRankId(11);

    this->InitInnerRanks();
    
    //rank0，连接rank1,rank4
    AddLinkStub(fabGroup0, rank0, rank1, rank0iface0, rank1iface0);
    AddLinkStub(fabGroup0, rank0, rank4, rank0iface0, rank4iface0);
    AddLinkStub(fabGroupLevel1, rank0, fabric0, rank0iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank0, fabric1, rank0iface1, ifaceNode1);
    
    //rank1，连接rank0,5
    AddLinkStub(fabGroup0, rank1, rank0, rank1iface0, rank0iface0);
    AddLinkStub(fabGroup0, rank1, rank5, rank1iface0, rank5iface0);
    AddLinkStub(fabGroupLevel1, rank1, fabric0, rank1iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank1, fabric1, rank1iface1, ifaceNode1);
    
    //rank2，连接rank3,6
    AddLinkStub(fabGroup1, rank2, rank3, rank2iface0, rank3iface0);
    AddLinkStub(fabGroup1, rank2, rank6, rank2iface0, rank6iface0);
    AddLinkStub(fabGroupLevel1, rank2, fabric0, rank2iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank2, fabric1, rank2iface1, ifaceNode1);

    //rank3，连接rank2,7
    AddLinkStub(fabGroup1, rank3, rank2, rank3iface0, rank2iface0);
    AddLinkStub(fabGroup1, rank3, rank7, rank3iface0, rank7iface0);
    AddLinkStub(fabGroupLevel1, rank3, fabric0, rank3iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank3, fabric1, rank3iface1, ifaceNode1);

    //rank4，连接rank0,5
    AddLinkStub(fabGroup1, rank4, rank0, rank4iface0, rank0iface0);
    AddLinkStub(fabGroup1, rank4, rank5, rank4iface0, rank5iface0);
    AddLinkStub(fabGroupLevel1, rank4, fabric0, rank4iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank4, fabric1, rank4iface1, ifaceNode1);

    //rank5，连接rank1,4
    AddLinkStub(fabGroup1, rank5, rank1, rank5iface0, rank1iface0);
    AddLinkStub(fabGroup1, rank5, rank4, rank5iface0, rank4iface0);
    AddLinkStub(fabGroupLevel1, rank5, fabric0, rank5iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank5, fabric1, rank5iface1, ifaceNode1);

    //rank6，连接rank2，7
    AddLinkStub(fabGroup1, rank6, rank2, rank6iface0, rank2iface0);
    AddLinkStub(fabGroup1, rank6, rank7, rank6iface0, rank7iface0);
    AddLinkStub(fabGroupLevel1, rank6, fabric0, rank6iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank6, fabric1, rank6iface1, ifaceNode1);

    //rank7，连接rank3，6
    AddLinkStub(fabGroup1, rank7, rank3, rank7iface0, rank3iface0);
    AddLinkStub(fabGroup1, rank7, rank6, rank7iface0, rank6iface0);
    AddLinkStub(fabGroupLevel1, rank7, fabric0, rank7iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank7, fabric1, rank7iface1, ifaceNode1);

    //rank8，连接rank9,10,11
    AddLinkStub(fabGroup1, rank8, rank9, rank8iface0, rank9iface0);
    AddLinkStub(fabGroup1, rank8, rank10, rank8iface0, rank10iface0);
    AddLinkStub(fabGroup1, rank8, rank11, rank8iface0, rank11iface0);
    AddLinkStub(fabGroupLevel1, rank8, fabric0, rank8iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank8, fabric1, rank8iface1, ifaceNode1);

    //rank9，连接rank8,10
    AddLinkStub(fabGroup1, rank9, rank8, rank9iface0, rank8iface0);
    AddLinkStub(fabGroup1, rank9, rank10, rank9iface0, rank10iface0);
    AddLinkStub(fabGroupLevel1, rank9, fabric0, rank9iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank9, fabric1, rank9iface1, ifaceNode1);

    //rank10，连接rank8,9
    AddLinkStub(fabGroup1, rank10, rank8, rank10iface0, rank8iface0);
    AddLinkStub(fabGroup1, rank10, rank9, rank10iface0, rank9iface0);
    AddLinkStub(fabGroupLevel1, rank10, fabric0, rank10iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank10, fabric1, rank10iface1, ifaceNode1);

    //rank11，连接rank8
    AddLinkStub(fabGroup1, rank11, rank8, rank11iface0, rank8iface0);
    AddLinkStub(fabGroupLevel1, rank11, fabric0, rank11iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank11, fabric1, rank11iface1, ifaceNode1);


    AddLinkStub(fabGroupLevel1, fabric0, rank0, ifaceNode0, rank0iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank1, ifaceNode0, rank1iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank2, ifaceNode0, rank2iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank3, ifaceNode0, rank3iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank4, ifaceNode0, rank4iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank5, ifaceNode0, rank5iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank6, ifaceNode0, rank6iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank7, ifaceNode0, rank7iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank8, ifaceNode0, rank8iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank9, ifaceNode0, rank9iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank10, ifaceNode0, rank10iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank11, ifaceNode0, rank11iface0);

    AddLinkStub(fabGroupLevel1, fabric1, rank0, ifaceNode1, rank0iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank1, ifaceNode1, rank1iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank2, ifaceNode1, rank2iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank3, ifaceNode1, rank3iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank4, ifaceNode1, rank4iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank5, ifaceNode1, rank5iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank6, ifaceNode1, rank6iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank7, ifaceNode1, rank7iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank8, ifaceNode1, rank8iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank9, ifaceNode1, rank9iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank10, ifaceNode1, rank10iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank11, ifaceNode1, rank11iface1);
}

void VirtualTopoStub::TopoInit91095TwoPodFourTwoAndThree(const string &rankTable)//{{0,1},{8,9},{24,25},{32,33}},{{3,5,6}},非对称2D,1D,gcd为1
{
    // 打桩virtual topo
    shared_ptr<NetInstance> fabGroup0 = InitNetInstance(0, "server0");
    shared_ptr<NetInstance> fabGroup1 = InitNetInstance(0, "server1");
    shared_ptr<NetInstance> fabGroupLevel1 = InitNetInstance(1, "clos0");
    this->netInsts_[0].emplace("server0", fabGroup0);
    this->netInsts_[0].emplace("server1", fabGroup1);
    this->netInsts_[1].emplace("clos0", fabGroupLevel1);
    // 初始化fabric0信息
    auto fabric0 = InitFabric(0);
    fabGroupLevel1->AddNode(fabric0);
    char fabricAddress0[] = "13.0.0.0";  // 打桩用sendIP地址
    IpAddress fabricAddr0(fabricAddress0);
    auto ifaceNode0 = InitConnInterface(fabricAddr0);
    fabric0->AddConnInterface(0, ifaceNode0);
    // 初始化fabric1信息
    auto fabric1 = InitFabric(1);
    fabGroupLevel1->AddNode(fabric1);
    char fabricAddress1[] = "14.0.0.0";  // 打桩用sendIP地址
    IpAddress fabricAddr1(fabricAddress1);
    auto ifaceNode1 = InitConnInterface(fabricAddr1);
    fabric1->AddConnInterface(1, ifaceNode1);


    // 初始化rank0信息
    auto rank0 = InitPeer(0, 0);
    this->AddPeer(rank0);  // 存储rank0信息到peers
    fabGroup0->AddNode(rank0);
    fabGroupLevel1->AddNode(rank0);
    char rank0Address0[] = "0.0.0.0";  // 打桩用sendIP地址
    IpAddress rank0Addr0(rank0Address0);
    auto rank0iface0 = InitConnInterface(rank0Addr0);
    rank0->AddConnInterface(0, rank0iface0);
    char rank0Address1[] = "0.0.0.1";  // 打桩用sendIP地址
    IpAddress rank0Addr1(rank0Address1);
    auto rank0iface1 = InitConnInterface(rank0Addr1);
    rank0->AddConnInterface(1, rank0iface1);
    rank0->AddNetInstance(fabGroup0);
    rank0->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(0);
    fabGroupLevel1->AddRankId(0);
 
    // 初始化rank1信息
    auto rank1 = InitPeer(1, 1);
    this->AddPeer(rank1);  // 存储rank0信息到peers
    fabGroup0->AddNode(rank1);
    fabGroupLevel1->AddNode(rank1);
    char rank1Address0[] = "1.0.0.0";  // 打桩用sendIP地址
    IpAddress rank1Addr0(rank1Address0);
    auto rank1iface0 = InitConnInterface(rank1Addr0);
    rank1->AddConnInterface(0, rank1iface0);
    char rank1Address1[] = "1.0.0.1";  // 打桩用sendIP地址
    IpAddress rank1Addr1(rank1Address1);
    auto rank1iface1 = InitConnInterface(rank1Addr1);
    rank1->AddConnInterface(1, rank1iface1);
    rank1->AddNetInstance(fabGroup0);
    rank1->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(1);
    fabGroupLevel1->AddRankId(1);

    // 初始化rank2信息
    auto rank2 = InitPeer(2, 8);
    this->AddPeer(rank2);  // 存储rank2信息到peers
    fabGroup0->AddNode(rank2);
    fabGroupLevel1->AddNode(rank2);
    char rank2Address0[] = "2.0.0.0";  // 打桩用sendIP地址
    IpAddress rank2Addr0(rank2Address0);
    auto rank2iface0 = InitConnInterface(rank2Addr0);
    rank2->AddConnInterface(0, rank2iface0);
    char rank2Address1[] = "2.0.0.1";  // 打桩用sendIP地址
    IpAddress rank2Addr1(rank2Address1);
    auto rank2iface1 = InitConnInterface(rank2Addr1);
    rank2->AddConnInterface(1, rank2iface1);
    rank2->AddNetInstance(fabGroup0);
    rank2->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(2);
    fabGroupLevel1->AddRankId(2);
 
    // 初始化rank3信息
    auto rank3 = InitPeer(3, 9);
    this->AddPeer(rank3);  // 存储rank3信息到peers
    fabGroup0->AddNode(rank3);
    fabGroupLevel1->AddNode(rank3);
    char rank3Address0[] = "3.0.0.0";  // 打桩用sendIP地址
    IpAddress rank3Addr0(rank3Address0);
    auto rank3iface0 = InitConnInterface(rank3Addr0);
    rank3->AddConnInterface(0, rank3iface0);
    char rank3Address1[] = "3.0.0.1";  // 打桩用sendIP地址
    IpAddress rank3Addr1(rank3Address1);
    auto rank3iface1 = InitConnInterface(rank3Addr1);
    rank3->AddConnInterface(1, rank3iface1);
    rank3->AddNetInstance(fabGroup0);
    rank3->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(3);
    fabGroupLevel1->AddRankId(3);

    // 初始化rank4信息
    auto rank4 = InitPeer(4, 24);
    this->AddPeer(rank4);  // 存储rank0信息到peers
    fabGroup0->AddNode(rank4);
    fabGroupLevel1->AddNode(rank4);
    char rank4Address0[] = "4.0.0.0";  // 打桩用sendIP地址
    IpAddress rank4Addr0(rank4Address0);
    auto rank4iface0 = InitConnInterface(rank4Addr0);
    rank4->AddConnInterface(0, rank4iface0);
    char rank4Address1[] = "4.0.0.1";  // 打桩用sendIP地址
    IpAddress rank4Addr1(rank4Address1);
    auto rank4iface1 = InitConnInterface(rank4Addr1);
    rank4->AddConnInterface(1, rank4iface1);
    rank4->AddNetInstance(fabGroup0);
    rank4->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(4);
    fabGroupLevel1->AddRankId(4);
 
    // 初始化rank5信息
    auto rank5 = InitPeer(5, 25);
    this->AddPeer(rank5);  // 存储rank5信息到peers
    fabGroup0->AddNode(rank5);
    fabGroupLevel1->AddNode(rank5);
    char rank5Address0[] = "5.0.0.0";  // 打桩用sendIP地址
    IpAddress rank5Addr0(rank5Address0);
    auto rank5iface0 = InitConnInterface(rank5Addr0);
    rank5->AddConnInterface(0, rank5iface0);
    char rank5Address1[] = "5.0.0.1";  // 打桩用sendIP地址
    IpAddress rank5Addr1(rank5Address1);
    auto rank5iface1 = InitConnInterface(rank5Addr1);
    rank5->AddConnInterface(1, rank5iface1);
    rank5->AddNetInstance(fabGroup0);
    rank5->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(5);
    fabGroupLevel1->AddRankId(5);

    // 初始化rank6信息
    auto rank6 = InitPeer(6, 32);
    this->AddPeer(rank6);  // 存储rank6信息到peers
    fabGroup0->AddNode(rank6);
    fabGroupLevel1->AddNode(rank6);
    char rank6Address0[] = "6.0.0.0";  // 打桩用sendIP地址
    IpAddress rank6Addr0(rank6Address0);
    auto rank6iface0 = InitConnInterface(rank6Addr0);
    rank6->AddConnInterface(0, rank6iface0);
    char rank6Address1[] = "6.0.0.1";  // 打桩用sendIP地址
    IpAddress rank6Addr1(rank6Address1);
    auto rank6iface1 = InitConnInterface(rank6Addr1);
    rank6->AddConnInterface(1, rank6iface1);
    rank6->AddNetInstance(fabGroup0);
    rank6->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(6);
    fabGroupLevel1->AddRankId(6);
 
    // 初始化rank7信息
    auto rank7 = InitPeer(7, 33);
    this->AddPeer(rank7);  // 存储rank7信息到peers
    fabGroup0->AddNode(rank7);
    fabGroupLevel1->AddNode(rank7);
    char rank7Address0[] = "7.0.0.0";  // 打桩用sendIP地址
    IpAddress rank7Addr0(rank7Address0);
    auto rank7iface0 = InitConnInterface(rank7Addr0);
    rank7->AddConnInterface(0, rank7iface0);
    char rank7Address1[] = "7.0.0.1";  // 打桩用sendIP地址
    IpAddress rank7Addr1(rank7Address1);
    auto rank7iface1 = InitConnInterface(rank7Addr1);
    rank7->AddConnInterface(1, rank7iface1);
    rank7->AddNetInstance(fabGroup0);
    rank7->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(7);
    fabGroupLevel1->AddRankId(7);
 
    // 初始化rank8信息（到第二个server了）
    auto rank8 = InitPeer(8, 3);
    this->AddPeer(rank8);  // 存储rank0信息到peers
    fabGroup1->AddNode(rank8);
    fabGroupLevel1->AddNode(rank8);
    char rank8Address0[] = "8.0.0.0";  // 打桩用sendIP地址
    IpAddress rank8Addr0(rank8Address0);
    auto rank8iface0 = InitConnInterface(rank8Addr0);
    rank8->AddConnInterface(0, rank8iface0);
    char rank8Address1[] = "8.0.0.1";  // 打桩用sendIP地址
    IpAddress rank8Addr1(rank8Address1);
    auto rank8iface1 = InitConnInterface(rank8Addr1);
    rank8->AddConnInterface(1, rank8iface1);
    rank8->AddNetInstance(fabGroup1);
    rank8->AddNetInstance(fabGroupLevel1);
    fabGroup1->AddRankId(8);
    fabGroupLevel1->AddRankId(8);
 
    // 初始化rank9信息
    auto rank9 = InitPeer(9, 5);
    this->AddPeer(rank9);  // 存储rank0信息到peers
    fabGroup1->AddNode(rank9);
    fabGroupLevel1->AddNode(rank9);
    char rank9Address0[] = "9.0.0.0";  // 打桩用sendIP地址
    IpAddress rank9Addr0(rank9Address0);
    auto rank9iface0 = InitConnInterface(rank9Addr0);
    rank9->AddConnInterface(0, rank9iface0);
    char rank9Address1[] = "9.0.0.1";  // 打桩用sendIP地址
    IpAddress rank9Addr1(rank9Address1);
    auto rank9iface1 = InitConnInterface(rank9Addr1);
    rank9->AddConnInterface(1, rank9iface1);
    rank9->AddNetInstance(fabGroup1);
    rank9->AddNetInstance(fabGroupLevel1);
    fabGroup1->AddRankId(9);
    fabGroupLevel1->AddRankId(9);

    // 初始化rank10信息
    auto rank10 = InitPeer(10, 6);
    this->AddPeer(rank10);  // 存储rank0信息到peers
    fabGroup1->AddNode(rank10);
    fabGroupLevel1->AddNode(rank10);
    char rank10Address0[] = "10.0.0.0";  // 打桩用sendIP地址
    IpAddress rank10Addr0(rank10Address0);
    auto rank10iface0 = InitConnInterface(rank10Addr0);
    rank10->AddConnInterface(0, rank10iface0);
    char rank10Address1[] = "10.0.0.1";  // 打桩用sendIP地址
    IpAddress rank10Addr1(rank10Address1);
    auto rank10iface1 = InitConnInterface(rank10Addr1);
    rank10->AddConnInterface(1, rank10iface1);
    rank10->AddNetInstance(fabGroup1);
    rank10->AddNetInstance(fabGroupLevel1);
    fabGroup1->AddRankId(10);
    fabGroupLevel1->AddRankId(10);

    this->InitInnerRanks();
    
    //rank0，连接rank1，2，4，6
    AddLinkStub(fabGroup0, rank0, rank1, rank0iface0, rank1iface0);
    AddLinkStub(fabGroup0, rank0, rank2, rank0iface0, rank2iface0);
    AddLinkStub(fabGroup0, rank0, rank4, rank0iface0, rank4iface0);
    AddLinkStub(fabGroup0, rank0, rank6, rank0iface0, rank6iface0);
    AddLinkStub(fabGroupLevel1, rank0, fabric0, rank0iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank0, fabric1, rank0iface1, ifaceNode1);
    
    //rank1，连接rank0，3，5，7
    AddLinkStub(fabGroup0, rank1, rank0, rank1iface0, rank0iface0);
    AddLinkStub(fabGroup0, rank1, rank3, rank1iface0, rank3iface0);
    AddLinkStub(fabGroup0, rank1, rank5, rank1iface0, rank5iface0);
    AddLinkStub(fabGroup0, rank1, rank7, rank1iface0, rank7iface0);
    AddLinkStub(fabGroupLevel1, rank1, fabric0, rank1iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank1, fabric1, rank1iface1, ifaceNode1);
    
    //rank2，连接rank0，3，4，6
    AddLinkStub(fabGroup1, rank2, rank0, rank2iface0, rank0iface0);
    AddLinkStub(fabGroup1, rank2, rank3, rank2iface0, rank3iface0);
    AddLinkStub(fabGroup1, rank2, rank4, rank2iface0, rank4iface0);
    AddLinkStub(fabGroup1, rank2, rank6, rank2iface0, rank6iface0);
    AddLinkStub(fabGroupLevel1, rank2, fabric0, rank2iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank2, fabric1, rank2iface1, ifaceNode1);

    //rank3，连接rank1，2，5，7
    AddLinkStub(fabGroup1, rank3, rank1, rank3iface0, rank1iface0);
    AddLinkStub(fabGroup1, rank3, rank2, rank3iface0, rank2iface0);
    AddLinkStub(fabGroup1, rank3, rank5, rank3iface0, rank5iface0);
    AddLinkStub(fabGroup1, rank3, rank7, rank3iface0, rank7iface0);
    AddLinkStub(fabGroupLevel1, rank3, fabric0, rank3iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank3, fabric1, rank3iface1, ifaceNode1);

    //rank4，连接rank0，2，5，6
    AddLinkStub(fabGroup1, rank4, rank0, rank4iface0, rank0iface0);
    AddLinkStub(fabGroup1, rank4, rank2, rank4iface0, rank2iface0);
    AddLinkStub(fabGroup1, rank4, rank5, rank4iface0, rank5iface0);
    AddLinkStub(fabGroup1, rank4, rank6, rank4iface0, rank6iface0);
    AddLinkStub(fabGroupLevel1, rank4, fabric0, rank4iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank4, fabric1, rank4iface1, ifaceNode1);

    //rank5，连接rank1，3，4，7
    AddLinkStub(fabGroup1, rank5, rank1, rank5iface0, rank1iface0);
    AddLinkStub(fabGroup1, rank5, rank3, rank5iface0, rank3iface0);
    AddLinkStub(fabGroup1, rank5, rank4, rank5iface0, rank4iface0);
    AddLinkStub(fabGroup1, rank5, rank7, rank5iface0, rank7iface0);
    AddLinkStub(fabGroupLevel1, rank5, fabric0, rank5iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank5, fabric1, rank5iface1, ifaceNode1);

    //rank6，连接rank0，2，4，7
    AddLinkStub(fabGroup1, rank6, rank0, rank6iface0, rank0iface0);
    AddLinkStub(fabGroup1, rank6, rank2, rank6iface0, rank2iface0);
    AddLinkStub(fabGroup1, rank6, rank4, rank6iface0, rank4iface0);
    AddLinkStub(fabGroup1, rank6, rank7, rank6iface0, rank7iface0);
    AddLinkStub(fabGroupLevel1, rank6, fabric0, rank6iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank6, fabric1, rank6iface1, ifaceNode1);

    //rank7，连接rank1，3，5，6
    AddLinkStub(fabGroup1, rank7, rank1, rank7iface0, rank1iface0);
    AddLinkStub(fabGroup1, rank7, rank3, rank7iface0, rank3iface0);
    AddLinkStub(fabGroup1, rank7, rank5, rank7iface0, rank5iface0);
    AddLinkStub(fabGroup1, rank7, rank6, rank7iface0, rank6iface0);
    AddLinkStub(fabGroupLevel1, rank7, fabric0, rank7iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank7, fabric1, rank7iface1, ifaceNode1);


    //rank8，连接rank9，10
    AddLinkStub(fabGroup1, rank8, rank9, rank8iface0, rank9iface0);
    AddLinkStub(fabGroup1, rank8, rank10, rank8iface0, rank10iface0);
    AddLinkStub(fabGroupLevel1, rank8, fabric0, rank8iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank8, fabric1, rank8iface1, ifaceNode1);

    //rank9，连接rank8，10
    AddLinkStub(fabGroup1, rank9, rank8, rank9iface0, rank8iface0);
    AddLinkStub(fabGroup1, rank9, rank10, rank9iface0, rank10iface0);
    AddLinkStub(fabGroupLevel1, rank9, fabric0, rank9iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank9, fabric1, rank9iface1, ifaceNode1);

    //rank10，连接rank8，9
    AddLinkStub(fabGroup1, rank10, rank8, rank10iface0, rank8iface0);
    AddLinkStub(fabGroup1, rank10, rank9, rank10iface0, rank9iface0);
    AddLinkStub(fabGroupLevel1, rank10, fabric0, rank10iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank10, fabric1, rank10iface1, ifaceNode1);


    AddLinkStub(fabGroupLevel1, fabric0, rank0, ifaceNode0, rank0iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank1, ifaceNode0, rank1iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank2, ifaceNode0, rank2iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank3, ifaceNode0, rank3iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank4, ifaceNode0, rank4iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank5, ifaceNode0, rank5iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank6, ifaceNode0, rank6iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank7, ifaceNode0, rank7iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank8, ifaceNode0, rank8iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank9, ifaceNode0, rank9iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank10, ifaceNode0, rank10iface0);

    AddLinkStub(fabGroupLevel1, fabric1, rank0, ifaceNode1, rank0iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank1, ifaceNode1, rank1iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank2, ifaceNode1, rank2iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank3, ifaceNode1, rank3iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank4, ifaceNode1, rank4iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank5, ifaceNode1, rank5iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank6, ifaceNode1, rank6iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank7, ifaceNode1, rank7iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank8, ifaceNode1, rank8iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank9, ifaceNode1, rank9iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank10, ifaceNode1, rank10iface1);
}

void VirtualTopoStub::TopoInit91095TwoPodThreeTwoAndThree(const string &rankTable)//{{0,1},{8,9},{24,25}},{{3,5,6}},非对称2D,1D，走Level0Nhr和1D
{
    // 打桩virtual topo
    shared_ptr<NetInstance> fabGroup0 = InitNetInstance(0, "server0");
    shared_ptr<NetInstance> fabGroup1 = InitNetInstance(0, "server1");
    shared_ptr<NetInstance> fabGroupLevel1 = InitNetInstance(1, "clos0");
    this->netInsts_[0].emplace("server0", fabGroup0);
    this->netInsts_[0].emplace("server1", fabGroup1);
    this->netInsts_[1].emplace("clos0", fabGroupLevel1);
    // 初始化fabric0信息
    auto fabric0 = InitFabric(0);
    fabGroupLevel1->AddNode(fabric0);
    char fabricAddress0[] = "10.0.0.0";  // 打桩用sendIP地址
    IpAddress fabricAddr0(fabricAddress0);
    auto ifaceNode0 = InitConnInterface(fabricAddr0);
    fabric0->AddConnInterface(0, ifaceNode0);
    // 初始化fabric1信息
    auto fabric1 = InitFabric(1);
    fabGroupLevel1->AddNode(fabric1);
    char fabricAddress1[] = "11.0.0.0";  // 打桩用sendIP地址
    IpAddress fabricAddr1(fabricAddress1);
    auto ifaceNode1 = InitConnInterface(fabricAddr1);
    fabric1->AddConnInterface(1, ifaceNode1);


    // 初始化rank0信息
    auto rank0 = InitPeer(0, 0);
    this->AddPeer(rank0);  // 存储rank0信息到peers
    fabGroup0->AddNode(rank0);
    fabGroupLevel1->AddNode(rank0);
    char rank0Address0[] = "0.0.0.0";  // 打桩用sendIP地址
    IpAddress rank0Addr0(rank0Address0);
    auto rank0iface0 = InitConnInterface(rank0Addr0);
    rank0->AddConnInterface(0, rank0iface0);
    char rank0Address1[] = "0.0.0.1";  // 打桩用sendIP地址
    IpAddress rank0Addr1(rank0Address1);
    auto rank0iface1 = InitConnInterface(rank0Addr1);
    rank0->AddConnInterface(1, rank0iface1);
    rank0->AddNetInstance(fabGroup0);
    rank0->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(0);
    fabGroupLevel1->AddRankId(0);
 
    // 初始化rank1信息
    auto rank1 = InitPeer(1, 1);
    this->AddPeer(rank1);  // 存储rank0信息到peers
    fabGroup0->AddNode(rank1);
    fabGroupLevel1->AddNode(rank1);
    char rank1Address0[] = "1.0.0.0";  // 打桩用sendIP地址
    IpAddress rank1Addr0(rank1Address0);
    auto rank1iface0 = InitConnInterface(rank1Addr0);
    rank1->AddConnInterface(0, rank1iface0);
    char rank1Address1[] = "1.0.0.1";  // 打桩用sendIP地址
    IpAddress rank1Addr1(rank1Address1);
    auto rank1iface1 = InitConnInterface(rank1Addr1);
    rank1->AddConnInterface(1, rank1iface1);
    rank1->AddNetInstance(fabGroup0);
    rank1->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(1);
    fabGroupLevel1->AddRankId(1);

    // 初始化rank2信息
    auto rank2 = InitPeer(2, 8);
    this->AddPeer(rank2);  // 存储rank2信息到peers
    fabGroup0->AddNode(rank2);
    fabGroupLevel1->AddNode(rank2);
    char rank2Address0[] = "2.0.0.0";  // 打桩用sendIP地址
    IpAddress rank2Addr0(rank2Address0);
    auto rank2iface0 = InitConnInterface(rank2Addr0);
    rank2->AddConnInterface(0, rank2iface0);
    char rank2Address1[] = "2.0.0.1";  // 打桩用sendIP地址
    IpAddress rank2Addr1(rank2Address1);
    auto rank2iface1 = InitConnInterface(rank2Addr1);
    rank2->AddConnInterface(1, rank2iface1);
    rank2->AddNetInstance(fabGroup0);
    rank2->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(2);
    fabGroupLevel1->AddRankId(2);
 
    // 初始化rank3信息
    auto rank3 = InitPeer(3, 9);
    this->AddPeer(rank3);  // 存储rank3信息到peers
    fabGroup0->AddNode(rank3);
    fabGroupLevel1->AddNode(rank3);
    char rank3Address0[] = "3.0.0.0";  // 打桩用sendIP地址
    IpAddress rank3Addr0(rank3Address0);
    auto rank3iface0 = InitConnInterface(rank3Addr0);
    rank3->AddConnInterface(0, rank3iface0);
    char rank3Address1[] = "3.0.0.1";  // 打桩用sendIP地址
    IpAddress rank3Addr1(rank3Address1);
    auto rank3iface1 = InitConnInterface(rank3Addr1);
    rank3->AddConnInterface(1, rank3iface1);
    rank3->AddNetInstance(fabGroup0);
    rank3->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(3);
    fabGroupLevel1->AddRankId(3);

    // 初始化rank4信息
    auto rank4 = InitPeer(4, 24);
    this->AddPeer(rank4);  // 存储rank0信息到peers
    fabGroup0->AddNode(rank4);
    fabGroupLevel1->AddNode(rank4);
    char rank4Address0[] = "4.0.0.0";  // 打桩用sendIP地址
    IpAddress rank4Addr0(rank4Address0);
    auto rank4iface0 = InitConnInterface(rank4Addr0);
    rank4->AddConnInterface(0, rank4iface0);
    char rank4Address1[] = "4.0.0.1";  // 打桩用sendIP地址
    IpAddress rank4Addr1(rank4Address1);
    auto rank4iface1 = InitConnInterface(rank4Addr1);
    rank4->AddConnInterface(1, rank4iface1);
    rank4->AddNetInstance(fabGroup0);
    rank4->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(4);
    fabGroupLevel1->AddRankId(4);
 
    // 初始化rank5信息
    auto rank5 = InitPeer(5, 25);
    this->AddPeer(rank5);  // 存储rank5信息到peers
    fabGroup0->AddNode(rank5);
    fabGroupLevel1->AddNode(rank5);
    char rank5Address0[] = "5.0.0.0";  // 打桩用sendIP地址
    IpAddress rank5Addr0(rank5Address0);
    auto rank5iface0 = InitConnInterface(rank5Addr0);
    rank5->AddConnInterface(0, rank5iface0);
    char rank5Address1[] = "5.0.0.1";  // 打桩用sendIP地址
    IpAddress rank5Addr1(rank5Address1);
    auto rank5iface1 = InitConnInterface(rank5Addr1);
    rank5->AddConnInterface(1, rank5iface1);
    rank5->AddNetInstance(fabGroup0);
    rank5->AddNetInstance(fabGroupLevel1);
    fabGroup0->AddRankId(5);
    fabGroupLevel1->AddRankId(5);

 
    // 初始化rank6信息（到第二个server了）
    auto rank6 = InitPeer(6, 3);
    this->AddPeer(rank6);  // 存储rank0信息到peers
    fabGroup1->AddNode(rank6);
    fabGroupLevel1->AddNode(rank6);
    char rank6Address0[] = "6.0.0.0";  // 打桩用sendIP地址
    IpAddress rank6Addr0(rank6Address0);
    auto rank6iface0 = InitConnInterface(rank6Addr0);
    rank6->AddConnInterface(0, rank6iface0);
    char rank6Address1[] = "6.0.0.1";  // 打桩用sendIP地址
    IpAddress rank6Addr1(rank6Address1);
    auto rank6iface1 = InitConnInterface(rank6Addr1);
    rank6->AddConnInterface(1, rank6iface1);
    rank6->AddNetInstance(fabGroup1);
    rank6->AddNetInstance(fabGroupLevel1);
    fabGroup1->AddRankId(6);
    fabGroupLevel1->AddRankId(6);
 
    // 初始化rank7信息
    auto rank7 = InitPeer(7, 5);
    this->AddPeer(rank7);  // 存储rank0信息到peers
    fabGroup1->AddNode(rank7);
    fabGroupLevel1->AddNode(rank7);
    char rank7Address0[] = "7.0.0.0";  // 打桩用sendIP地址
    IpAddress rank7Addr0(rank7Address0);
    auto rank7iface0 = InitConnInterface(rank7Addr0);
    rank7->AddConnInterface(0, rank7iface0);
    char rank7Address1[] = "7.0.0.1";  // 打桩用sendIP地址
    IpAddress rank7Addr1(rank7Address1);
    auto rank7iface1 = InitConnInterface(rank7Addr1);
    rank7->AddConnInterface(1, rank7iface1);
    rank7->AddNetInstance(fabGroup1);
    rank7->AddNetInstance(fabGroupLevel1);
    fabGroup1->AddRankId(7);
    fabGroupLevel1->AddRankId(7);

    // 初始化rank8信息
    auto rank8 = InitPeer(8, 6);
    this->AddPeer(rank8);  // 存储rank0信息到peers
    fabGroup1->AddNode(rank8);
    fabGroupLevel1->AddNode(rank8);
    char rank8Address0[] = "8.0.0.0";  // 打桩用sendIP地址
    IpAddress rank8Addr0(rank8Address0);
    auto rank8iface0 = InitConnInterface(rank8Addr0);
    rank8->AddConnInterface(0, rank8iface0);
    char rank8Address1[] = "8.0.0.1";  // 打桩用sendIP地址
    IpAddress rank8Addr1(rank8Address1);
    auto rank8iface1 = InitConnInterface(rank8Addr1);
    rank8->AddConnInterface(1, rank8iface1);
    rank8->AddNetInstance(fabGroup1);
    rank8->AddNetInstance(fabGroupLevel1);
    fabGroup1->AddRankId(8);
    fabGroupLevel1->AddRankId(8);

    this->InitInnerRanks();
    
    //rank0，连接rank1，2，4
    AddLinkStub(fabGroup0, rank0, rank1, rank0iface0, rank1iface0);
    AddLinkStub(fabGroup0, rank0, rank2, rank0iface0, rank2iface0);
    AddLinkStub(fabGroup0, rank0, rank4, rank0iface0, rank4iface0);
    AddLinkStub(fabGroupLevel1, rank0, fabric0, rank0iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank0, fabric1, rank0iface1, ifaceNode1);
    
    //rank1，连接rank0，3，5
    AddLinkStub(fabGroup0, rank1, rank0, rank1iface0, rank0iface0);
    AddLinkStub(fabGroup0, rank1, rank3, rank1iface0, rank3iface0);
    AddLinkStub(fabGroup0, rank1, rank5, rank1iface0, rank5iface0);
    AddLinkStub(fabGroupLevel1, rank1, fabric0, rank1iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank1, fabric1, rank1iface1, ifaceNode1);
    
    //rank2，连接rank0，3，4
    AddLinkStub(fabGroup1, rank2, rank0, rank2iface0, rank0iface0);
    AddLinkStub(fabGroup1, rank2, rank3, rank2iface0, rank3iface0);
    AddLinkStub(fabGroup1, rank2, rank4, rank2iface0, rank4iface0);
    AddLinkStub(fabGroupLevel1, rank2, fabric0, rank2iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank2, fabric1, rank2iface1, ifaceNode1);

    //rank3，连接rank1，2，5
    AddLinkStub(fabGroup1, rank3, rank1, rank3iface0, rank1iface0);
    AddLinkStub(fabGroup1, rank3, rank2, rank3iface0, rank2iface0);
    AddLinkStub(fabGroup1, rank3, rank5, rank3iface0, rank5iface0);
    AddLinkStub(fabGroupLevel1, rank3, fabric0, rank3iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank3, fabric1, rank3iface1, ifaceNode1);

    //rank4，连接rank0，2，5
    AddLinkStub(fabGroup1, rank4, rank0, rank4iface0, rank0iface0);
    AddLinkStub(fabGroup1, rank4, rank2, rank4iface0, rank2iface0);
    AddLinkStub(fabGroup1, rank4, rank5, rank4iface0, rank5iface0);
    AddLinkStub(fabGroupLevel1, rank4, fabric0, rank4iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank4, fabric1, rank4iface1, ifaceNode1);

    //rank5，连接rank1，3，4
    AddLinkStub(fabGroup1, rank5, rank1, rank5iface0, rank1iface0);
    AddLinkStub(fabGroup1, rank5, rank3, rank5iface0, rank3iface0);
    AddLinkStub(fabGroup1, rank5, rank4, rank5iface0, rank4iface0);
    AddLinkStub(fabGroupLevel1, rank5, fabric0, rank5iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank5, fabric1, rank5iface1, ifaceNode1);


    //rank6，连接rank7，8
    AddLinkStub(fabGroup1, rank6, rank7, rank6iface0, rank7iface0);
    AddLinkStub(fabGroup1, rank6, rank8, rank6iface0, rank8iface0);
    AddLinkStub(fabGroupLevel1, rank6, fabric0, rank6iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank6, fabric1, rank6iface1, ifaceNode1);

    //rank7，连接rank6，8
    AddLinkStub(fabGroup1, rank7, rank6, rank7iface0, rank6iface0);
    AddLinkStub(fabGroup1, rank7, rank8, rank7iface0, rank8iface0);
    AddLinkStub(fabGroupLevel1, rank7, fabric0, rank7iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank7, fabric1, rank7iface1, ifaceNode1);

    //rank8，连接rank6，7
    AddLinkStub(fabGroup1, rank8, rank6, rank8iface0, rank6iface0);
    AddLinkStub(fabGroup1, rank8, rank7, rank8iface0, rank7iface0);
    AddLinkStub(fabGroupLevel1, rank8, fabric0, rank8iface0, ifaceNode0);
    AddLinkStub(fabGroupLevel1, rank8, fabric1, rank8iface1, ifaceNode1);


    AddLinkStub(fabGroupLevel1, fabric0, rank0, ifaceNode0, rank0iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank1, ifaceNode0, rank1iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank2, ifaceNode0, rank2iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank3, ifaceNode0, rank3iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank4, ifaceNode0, rank4iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank5, ifaceNode0, rank5iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank6, ifaceNode0, rank6iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank7, ifaceNode0, rank7iface0);
    AddLinkStub(fabGroupLevel1, fabric0, rank8, ifaceNode0, rank8iface0);

    AddLinkStub(fabGroupLevel1, fabric1, rank0, ifaceNode1, rank0iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank1, ifaceNode1, rank1iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank2, ifaceNode1, rank2iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank3, ifaceNode1, rank3iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank4, ifaceNode1, rank4iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank5, ifaceNode1, rank5iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank6, ifaceNode1, rank6iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank7, ifaceNode1, rank7iface1);
    AddLinkStub(fabGroupLevel1, fabric1, rank8, ifaceNode1, rank8iface1);
}

