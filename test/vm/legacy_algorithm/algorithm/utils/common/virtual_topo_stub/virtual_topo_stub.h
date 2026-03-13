#ifndef HCCLV2_VIRTUAL_TOPO_STUB_H
#define HCCLV2_VIRTUAL_TOPO_STUB_H

#include "rank_gph.h"

namespace Hccl {

// 打桩用常量，為了解決clean code問題
constexpr int LinkBandWidth10 = 20;
constexpr int LinkBandWidth20 = 20;
constexpr int LinkBandWidth30 = 30;
constexpr int LinkDelay30 = 30;
constexpr int LinkDelay60 = 60;

class VirtualTopoStub : public RankGraph {
public:
    VirtualTopoStub(RankId rankId) : RankGraph(rankId)
    {
    }

    void TopoInit91095TwoTimesTwo(const string &rankTable);
    void TopoInit91095OneTimesFour(const string &rankTable);
    void TopoInit91095OneTimesTwoDetour(const string &rankTable);
    void TopoInit91095TwoTimesThree(const string &rankTable);
    void TopoInit91095OneTimesThree(const string &rankTable);
    void TopoInit91095TwoPlusOnePlusOne(const string &rankTable);
    void TopoInit4RankRDMALink(const string &rankTable);
    void TopoInit91095OneTimesOne(const string &rankTable);
    void TopoInit91095OneTimesN(const string &rankTable, int numRanks);
    void TopoInit91095TwoPodTwoTwoAndTwoTwo(const string &rankTable);
    void TopoInit91095TwoServerTimesTwo(const string &rankTable);
    void TopoInit91095TwoPodFourTwoAndTwoTwo(const string &rankTable);
    void TopoInit91095TwoPodIrregularEightAndIrregularFour(const string &rankTable);
    void TopoInit91095TwoPodFourTwoAndThree(const string &rankTable);
    void TopoInit91095TwoPodThreeTwoAndThree(const string &rankTable);

    void TopoInit2HCCSLink(const string &rankTable);

   //绕路 // void TopoInit91095OneTimesTwoDetour(const string &rankTable);

private:
    shared_ptr<NetInstance::Peer> InitPeer(RankId rankId, LocalId localId, DeviceId deviceId = 0);
    shared_ptr<NetInstance::Fabric> InitFabric(FabricId FabricId, PlaneId netPlaneId = "0");
    shared_ptr<NetInstance> InitNetInstance(u32 level, string id);
    shared_ptr<NetInstance::ConnInterface> InitConnInterface(IpAddress addr);
    shared_ptr<NetInstance::ConnInterface> InitConnInterface(
        IpAddress addr, std::set<string> ports, AddrPosition pos, LinkType inputLinkType, std::set<LinkProtocol> inputLinkProtocols);
    void AddLinkStub(shared_ptr<NetInstance> fabGroup, shared_ptr<NetInstance::Node> srcPeer,
        shared_ptr<NetInstance::Node> dstPeer, shared_ptr<NetInstance::ConnInterface> srcIface, shared_ptr<NetInstance::ConnInterface> dstIface,
        LinkType type, std::set<LinkProtocol> protocals);
    void AddLinkStub(shared_ptr<NetInstance> fabGroup, shared_ptr<NetInstance::Node> srcPeer,
        shared_ptr<NetInstance::Node> dstPeer, shared_ptr<NetInstance::ConnInterface> srcIface, shared_ptr<NetInstance::ConnInterface> dstIface);
    void AddLinkStub(shared_ptr<NetInstance> fabGroup, shared_ptr<NetInstance::Node> srcPeer,
        shared_ptr<NetInstance::Node> dstPeer, shared_ptr<NetInstance::ConnInterface> srcIface, shared_ptr<NetInstance::ConnInterface> dstIface, 
        LinkDirection direct, u32 hop);
};

}  // namespace Hccl

#endif