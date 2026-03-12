#ifndef __LLT_HCCL_STUB_GE_H__
#define __LLT_HCCL_STUB_GE_H__

#include "graph/compute_graph.h"
#include <unordered_map>
#include "graph/ascend_string.h"
#include "common/ge_common/util.h"
#include "graph/debug/ge_log.h"
#include "graph/ge_error_codes.h"

namespace ge {
class ComputeGraphImpl {
public:
    using ConstComputeGraphPtr = std::shared_ptr<ConstComputeGraph>;
    template <class T>
    using Vistor = RangeVistor<T, std::shared_ptr<ConstComputeGraph>>;
    explicit ComputeGraphImpl(const std::string &name);

    string GetName() const;
    std::vector<std::shared_ptr<ComputeGraph>> GetAllSubgraphs() const;
    std::shared_ptr<ComputeGraph> AddSubGraph(const std::shared_ptr<ComputeGraph> &sub_graph);
    Vistor<NodePtr> GetDirectNode(const ConstComputeGraphPtr &compute_graph) const;
    void SetName(const string &name);
    Vistor<NodePtr> GetAllNodes(const ConstComputeGraphPtr &compute_graph) const;
    graphStatus RemoveNode(const NodePtr &node);
    NodePtr AddNode(OpDescPtr op, const ComputeGraphPtr &compute_graph);

    void SetGraphID(uint32_t graph_id)
    {
        graph_id_ = graph_id;
    }
    uint32_t GetGraphID() const
    {
        return graph_id_;
    }
    void SetSessionID(const uint64_t session_id)
    {
        session_id_ = session_id;
    }
    uint64_t GetSessionID() const
    {
        return session_id_;
    }

private:
    std::string name_;
    std::list<NodePtr> nodes_;
    uint32_t graph_id_ = 0;
    uint64_t session_id_ = 0;
    ProtoAttrMap attrs_;
    std::vector<NodePtr> input_nodes_;
    std::vector<std::shared_ptr<ComputeGraph>> sub_graph_;
    bool is_valid_flag_;
    bool need_iteration_ = false;
};

class AnchorImpl {
public:
    AnchorImpl(const NodePtr &owner_node, int idx);
    ~AnchorImpl() = default;
    vector<std::weak_ptr<Anchor>> peer_anchors_;
    std::weak_ptr<Node> owner_node_;
    int idx_;
};

class Node::NodeImpl {
public:
    NodeImpl() = default;
    NodeImpl(const OpDescPtr &op, const ComputeGraphPtr &owner_graph);
    ~NodeImpl() = default;
    graphStatus Init(const NodePtr &node);
    std::string GetName() const;
    std::string GetType() const;
    uint32_t GetAllOutDataAnchorsSize() const;
    InDataAnchorPtr GetInDataAnchor(int idx) const;
    OutDataAnchorPtr GetOutDataAnchor(int idx) const;
    Node::Vistor<NodePtr> GetInDataNodes(const ConstNodePtr &owner_node) const;
    OpDescPtr GetOpDesc() const;
    Node::Vistor<std::pair<NodePtr, OutDataAnchorPtr>> GetInDataNodesAndAnchors(const ConstNodePtr &owner_node) const;

    OpDescPtr op_;
    std::weak_ptr<ComputeGraph> owner_graph_;
    vector<InDataAnchorPtr> in_data_anchors_;
    vector<OutDataAnchorPtr> out_data_anchors_;
    InControlAnchorPtr in_control_anchor_;
    OutControlAnchorPtr out_control_anchor_;
    map<string, GeAttrValue> attrs_;  // lint !e1073
    bool has_init_{false};
    bool host_node_{false};
    bool anchor_status_updated_{false};
};

class IRMetaData {
public:
    explicit IRMetaData(const std::string &op_name) : op_name_(op_name){};
    IRMetaData() = default;
    const std::vector<std::pair<std::string, IrInputType>> &GetIrInputs() const;

    std::string op_name_;
    std::vector<std::pair<std::string, IrInputType>> ir_inputs_;
};

class OpMetadata {
public:
    int64_t id_{0};
    IRMetaData ir_meta_;
};

class OpDescImpl {
public:
    OpDescImpl();
    OpDescImpl(const std::string &name, const std::string &type);
    OpDescImpl(const ProtoMsgOwner &proto_msg_owner, ge::proto::OpDef *op_def);
    ~OpDescImpl() = default;
    void SetId(int64_t id);
    const IRMetaData &GetIRMeta() const;

    GeIrProtoHelper<ge::proto::OpDef> op_def_;
    vector<GeTensorDescPtr> inputs_desc_{};
    map<string, uint32_t> input_name_idx_{};
    OpMetadata meta_data_;
};

class GeTensorDescImpl {
public:
    GeTensorDescImpl();
    GeTensorDescImpl(GeShape shape, Format format, DataType dt);
    GeTensorDescImpl(const GeTensorDescImpl &desc);
    GeTensorDescImpl(GeTensorDescImpl &&desc);
    GeTensorDescImpl(const ProtoMsgOwner &proto_owner, proto::TensorDescriptor *proto_msg);
    ~GeTensorDescImpl() = default;
};

using OpCreatorV2 = std::function<Operator(const AscendString &)>;
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OperatorCreatorRegister {
public:
    OperatorCreatorRegister(const char_t *const operator_type, OpCreatorV2 const &op_creator);
    ~OperatorCreatorRegister() = default;
};

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<Anchor>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<DataAnchor>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<ControlAnchor>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<InDataAnchor>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<OutDataAnchor>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<InControlAnchor>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<OutControlAnchor>();
}  // namespace ge
#endif  // __LLT_HCCL_STUB_GE_H__