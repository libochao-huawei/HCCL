#include "llt_hccl_stub.h"
#define private public
#define protected public
#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/lowering/buffer_pool.h"
#include "register/kernel_registry.h"
#include "register/kernel_registry_impl.h"
#include "register/op_impl_registry.h"
#include "external/register/op_tiling_info.h"
#include "register/op_impl_kernel_registry.h"
#include "exe_graph/lowering/tiling_parse_context_builder.h"
#include "exe_graph/lowering/tiling_context_builder.h"
#include "exe_graph/lowering/shape_utils.h"
#include "register/op_tiling_attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "register/op_impl_registry_base.h"
#include "base/context_builder/context_holder_builder.h"
#include "common/ge_common/util.h"
#include "graph/debug/ge_util.h"
#include "base/context_builder/op_tiling_parse_context_builder.h"
#include "base/context_builder/op_tiling_context_builder.h"
#include "graph/operator_reg.h"
#include "exe_graph/runtime/storage_shape.h"
#include "base/context_builder/op_context_builder_impl.h"

namespace gert {
template <typename T>
struct ComGraphMakeUniq {
    using unique_object = std::unique_ptr<T>;
};

template <typename T>
struct ComGraphMakeUniq<T[]> {
    using unique_array = std::unique_ptr<T[]>;
};

template <typename T, size_t B>
struct ComGraphMakeUniq<T[B]> {
    struct invalid_type {};
};

template <typename T, typename... Args>
static inline typename ComGraphMakeUniq<T>::unique_object ComGraphMakeUnique(Args &&...args)
{
    using T_nc = typename std::remove_const<T>::type;
    return std::unique_ptr<T>(new (std::nothrow) T_nc(std::forward<Args>(args)...));
}

template <typename T>
static inline typename ComGraphMakeUniq<T>::unique_array ComGraphMakeUnique(const size_t num)
{
    return std::unique_ptr<T>(new (std::nothrow) typename std::remove_extent<T>::type[num]());
}

namespace {
ge::graphStatus InitInputInstanceInfo(const ge::NodePtr &node, ComputeNodeInfo &compute_node_info)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InitCompileTimeTD(const ge::NodePtr &node, ComputeNodeInfo &compute_node_info)
{
    return ge::GRAPH_SUCCESS;
}

uint32_t GeMemcpy(uint8_t *dst_ptr, size_t dst_size, const uint8_t *src_ptr, const size_t src_size)
{
    return 0;
}

std::unique_ptr<uint8_t[]> CreateComputeNodeInfoImpl(const std::unique_ptr<uint8_t[]> &attr_buf, const size_t attr_size,
    const ge::NodePtr &node, gert::bg::BufferPool &buffer_pool, size_t &total_size)
{
    const auto ir_input_num = node->GetOpDesc()->GetIrInputs().size();
    const auto input_num = node->GetInDataNodesAndAnchors().size();
    const auto output_num = node->GetAllOutDataAnchorsSize();
    HCCL_INFO("node: %s(%s), ir_input_num:%zu, input_num:%zu, output_num:%u.",
        node->GetName().c_str(),
        node->GetType().c_str(),
        ir_input_num,
        input_num,
        output_num);
    ComputeNodeInfo::CalcSize(ir_input_num, input_num, output_num, total_size);
    GE_ASSERT_TRUE(!ge::AddOverflow(total_size, attr_size, total_size));
    auto compute_node_info_holder = ComGraphMakeUnique<uint8_t[]>(total_size);

    auto node_name = buffer_pool.AddStr(node->GetName().c_str());
    auto node_type = buffer_pool.AddStr(node->GetType().c_str());
    auto compute_node_info = ge::PtrToPtr<uint8_t, ComputeNodeInfo>(compute_node_info_holder.get());
    compute_node_info->Init(ir_input_num,
        input_num,
        output_num,
        ge::PtrToPtr<void, ge::char_t>(ge::ValueToPtr(node_name)),
        ge::PtrToPtr<void, ge::char_t>(ge::ValueToPtr(node_type)));

    auto ret = InitInputInstanceInfo(node, *compute_node_info);
    ret = InitCompileTimeTD(node, *compute_node_info);
    auto attr = compute_node_info->MutableAttrs();
    const auto offset = ge::PtrToPtr<RuntimeAttrs, uint8_t>(attr) - compute_node_info_holder.get();
    if (static_cast<size_t>(offset) > total_size) {
        return nullptr;
    }
    ret = GeMemcpy(ge::PtrToPtr<RuntimeAttrs, uint8_t>(attr), (total_size - offset), attr_buf.get(), attr_size);
    return compute_node_info_holder;
}

std::unique_ptr<uint8_t[]> CreateAttrBuffer(const std::vector<std::vector<uint8_t>> &attrs, size_t &total_size)
{
    typedef struct {
        size_t attr_num;
        uint8_t reserved_[40];  // Reserved field, 32+8, do not directly use when only 8-byte left
        size_t offset[0];
    } RuntimeAttrsDef;
    total_size = sizeof(RuntimeAttrsDef);
    size_t offset_size = 0U;
    if (ge::MulOverflow(sizeof(size_t), attrs.size(), offset_size)) {
        return nullptr;
    }
    if (ge::AddOverflow(total_size, offset_size, total_size)) {
        return nullptr;
    }
    for (const auto &attr : attrs) {
        if (ge::AddOverflow(total_size, attr.size(), total_size)) {
            return nullptr;
        }
    }
    auto attr_holder = ComGraphMakeUnique<uint8_t[]>(total_size);
    auto attr_def = ge::PtrToPtr<uint8_t, RuntimeAttrsDef>(attr_holder.get());
    attr_def->attr_num = attrs.size();
    memset(attr_def->reserved_, 0, sizeof(attr_def->reserved_));
    size_t current_offset = sizeof(RuntimeAttrsDef) + sizeof(size_t) * attr_def->attr_num;
    auto attr_pos = attr_holder.get();
    for (size_t i = 0; i < attrs.size(); ++i) {
        attr_def->offset[i] = current_offset;
        const auto ret =
            GeMemcpy(attr_pos + current_offset, total_size - current_offset, attrs[i].data(), attrs[i].size());
        current_offset += attrs[i].size();
    }
    return attr_holder;
}
}  // namespace

// No.3 KernelRegistryImpl
KernelRegistryImpl &KernelRegistryImpl::GetInstance()
{
    static KernelRegistryImpl registry;
    return registry;
}

void KernelRegistryImpl::RegisterKernel(std::string kernel_type, KernelInfo kernel_infos)
{
    kernel_infos_[std::move(kernel_type)] = std::move(kernel_infos);
}

const KernelRegistryImpl::KernelFuncs *KernelRegistryImpl::FindKernelFuncs(const std::string &kernel_type) const
{
    auto iter = kernel_infos_.find(kernel_type);
    if (iter == kernel_infos_.end()) {
        return nullptr;
    }
    return &iter->second.func;
}

const KernelRegistry::KernelInfo *KernelRegistryImpl::FindKernelInfo(const std::string &kernel_type) const
{
    auto iter = kernel_infos_.find(kernel_type);
    if (iter == kernel_infos_.end()) {
        return nullptr;
    }
    return &iter->second;
}

TilingParseContextBuilder &TilingParseContextBuilder::CompileJson(const ge::char_t *compile_json)
{
    compile_json_ = const_cast<ge::char_t *>(compile_json);
    return *this;
}

TilingParseContextBuilder &TilingParseContextBuilder::PlatformInfo(void *platform_info)
{
    platform_info_ = platform_info;
    return *this;
}

TilingParseContextBuilder &TilingParseContextBuilder::CompileInfoCreatorFunc(
    OpImplKernelRegistry::CompileInfoCreatorFunc create_func)
{
    create_func_ = create_func;
    return *this;
}

TilingParseContextBuilder &TilingParseContextBuilder::CompileInfoDeleterFunc(
    OpImplKernelRegistry::CompileInfoDeleterFunc delete_func)
{
    delete_func_ = delete_func;
    return *this;
}

ge::NodePtr KernelRunContextBuilder::MakeNode(const ge::OpDescPtr &op_desc)
{
    const auto node_id = op_desc->GetId();
    graph_ = std::make_shared<ge::ComputeGraph>("tmp");
    auto fake_node = graph_->AddNode(op_desc);
    for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); ++i) {
        const auto input_desc = op_desc->GetInputDesc(i);
        auto op_data = ge::OpDescBuilder(std::to_string(i), "Data").AddInput("x").AddOutput("y").Build();
        auto data_node = graph_->AddNode(op_data);
        ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), fake_node->GetInDataAnchor(i));
    }
    op_desc->SetId(node_id);
    return fake_node;
}

KernelContextHolder KernelRunContextBuilder::Build(const ge::OpDescPtr &op_desc)
{
    KernelContextHolder holder;
    size_t size = sizeof(KernelRunContext) + sizeof(Chain *) * (inputs_.size() + outputs_.size());
    holder.context_holder_ = ComGraphMakeUnique<uint8_t[]>(size);
    if (holder.context_holder_ == nullptr) {
        return holder;
    }
    size_t extend_info_size;
    holder.compute_node_extend_holder_ =
        bg::CreateComputeNodeInfo(MakeNode(op_desc), holder.buffer_pool_, extend_info_size);

    if (holder.compute_node_extend_holder_ == nullptr) {
        return holder;
    }
    auto compute_node_info = ge::PtrToPtr<uint8_t, ComputeNodeInfo>(holder.compute_node_extend_holder_.get());
    compute_node_info->SetNodeName(
        holder.buffer_pool_.GetBufById(reinterpret_cast<size_t>(compute_node_info->GetNodeName())));
    compute_node_info->SetNodeType(
        holder.buffer_pool_.GetBufById(reinterpret_cast<size_t>(compute_node_info->GetNodeType())));
    holder.context_ = ge::PtrToPtr<uint8_t, KernelContext>(holder.context_holder_.get());
    auto kernel_run_context = holder.context_->GetContext();
    kernel_run_context->input_size = inputs_.size();
    kernel_run_context->output_size = outputs_.size();
    kernel_run_context->compute_node_info = compute_node_info;
    kernel_run_context->output_start = &(kernel_run_context->values[kernel_run_context->input_size]);
    holder.value_holder_.resize(inputs_.size() + outputs_.size());
    for (size_t i = 0UL; i < holder.value_holder_.size(); ++i) {
        kernel_run_context->values[i] = ge::PtrToPtr<Chain, AsyncAnyValue>(&holder.value_holder_[i]);
    }
    for (size_t i = 0UL; i < inputs_.size(); ++i) {
        holder.value_holder_[i].Set(inputs_[i].first, inputs_[i].second);
    }
    for (size_t i = 0UL; i < outputs_.size(); ++i) {
        holder.value_holder_[inputs_.size() + i].Set(outputs_[i].first, outputs_[i].second);
    }
    return holder;
}

KernelContextHolder TilingParseContextBuilder::Build(const ge::Operator &op)
{
    KernelContextHolder holder;

    const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    std::vector<std::pair<void *, gert::Chain::Deleter>> tiling_parse_outputs(1, std::make_pair(nullptr, nullptr));
    return gert::KernelRunContextBuilder()
        .Inputs({compile_json_})
        .Inputs({platform_info_})
        .Inputs({const_cast<ge::char_t *>(op_desc->GetType().c_str())})
        .Outputs(tiling_parse_outputs)
        .Build(op_desc);
}

template class OpContextBuilderBase<OpTilingContextBuilder>;
template class OpContextBuilderBase<OpTilingParseContextBuilder>;

template <typename T>
OpContextBuilderBase<T>::OpContextBuilderBase() : impl_()
{}
template <typename T>
OpContextBuilderBase<T>::~OpContextBuilderBase() = default;

template <typename T>
T &OpContextBuilderBase<T>::OpType(const AscendString &op_type)
{
    return static_cast<T &>(*this);
}

template <typename T>
T &OpContextBuilderBase<T>::OpName(const AscendString &op_name)
{
    return static_cast<T &>(*this);
}

template <typename T>
T &OpContextBuilderBase<T>::IONum(size_t input_ir_num, size_t output_ir_num)
{
    return static_cast<T &>(*this);
}

TilingContextBuilder &TilingContextBuilder::CompileInfo(void *compile_info)
{
    compile_info_ = compile_info;
    return *this;
}

TilingContextBuilder &TilingContextBuilder::PlatformInfo(void *platform_info)
{
    platform_info_ = platform_info;
    return *this;
}

TilingContextBuilder &TilingContextBuilder::TilingData(void *tiling_data)
{
    outputs_[TilingContext::kOutputTilingData] = tiling_data;
    return *this;
}

TilingContextBuilder &TilingContextBuilder::Workspace(ContinuousVector *workspace)
{
    outputs_[TilingContext::kOutputWorkspace] = workspace;
    return *this;
}

TilingContextBuilder &TilingContextBuilder::SpaceRegistry(const gert::OpImplSpaceRegistryPtr &space_registry)
{
    (void)space_registry;
    return *this;
}

ge::graphStatus TilingContextBuilder::BuildRTInputTensors(const ge::Operator &op)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingContextBuilder::BuildRTOutputShapes(const ge::Operator &op)
{
    return ge::GRAPH_SUCCESS;
}

KernelContextHolder TilingContextBuilder::Build(const ge::Operator &op)
{
    KernelContextHolder holder;
    auto node = ge::OpDescUtils::GetOpDescFromOperator(op);
    std::vector<void *> context_inputs;
    auto ret = BuildRTInputTensors(op);
    ret = BuildRTOutputShapes(op);
    for (const auto &input_holder : rt_tensor_holders_) {
        context_inputs.emplace_back(input_holder.get());
    }
    context_inputs.emplace_back(compile_info_);
    context_inputs.emplace_back(platform_info_);
    context_inputs.emplace_back(nullptr);

    return base_builder_.Inputs(context_inputs).Outputs(outputs_).Build(node);
}

ge::graphStatus TilingData::AppendConvertedAttrVal(
    const RuntimeAttrs *attrs, const size_t attr_index, const AttrDataType src_type, const AttrDataType dst_type)
{
    return ge::GRAPH_SUCCESS;
}

const Shape g_vec_1_shape;
int64_t GetInputCDim(TilingContext *kernel_context, const size_t index)
{
    return 0;
}

class OpImplRegisterV2Impl {
public:
    OpImplRegistry::OpType op_type;
    OpImplRegistry::OpImplFunctions functions;
    bool is_private_attr_registered = false;
};

namespace {
void RegisterOpImplToRegistry(const OpImplRegisterV2Impl *rd)
{
    if (rd == nullptr) {
        return;
    }
    auto &funcs = OpImplRegistry::GetInstance().CreateOrGetOpImpl(rd->op_type.GetString());
    if (rd->functions.infer_shape != nullptr) {
        funcs.infer_shape = rd->functions.infer_shape;
    }
    if (rd->functions.infer_shape_range != nullptr) {
        funcs.infer_shape_range = rd->functions.infer_shape_range;
    }
    if (rd->functions.infer_datatype != nullptr) {
        funcs.infer_datatype = rd->functions.infer_datatype;
    }
    if (rd->functions.tiling != nullptr) {
        funcs.tiling = rd->functions.tiling;
        funcs.max_tiling_data_size = rd->functions.max_tiling_data_size;
    }
    if (rd->functions.inputs_dependency != 0U) {
        funcs.inputs_dependency = rd->functions.inputs_dependency;
    }
    if (rd->functions.tiling_parse != nullptr) {
        funcs.tiling_parse = rd->functions.tiling_parse;
        funcs.compile_info_creator = rd->functions.compile_info_creator;
        funcs.compile_info_deleter = rd->functions.compile_info_deleter;
    }
    if (rd->is_private_attr_registered) {
        funcs.private_attrs = rd->functions.private_attrs;
        funcs.unique_private_attrs = rd->functions.unique_private_attrs;
    }
}
}  // namespace

OpImplRegistry &OpImplRegistry::GetInstance()
{
    static OpImplRegistry instance;
    return instance;
}

OpImplRegistry::OpImplFunctionsV2 &OpImplRegistry::CreateOrGetOpImpl(const ge::char_t *op_type)
{
    (void)reserved_;
    return types_to_impl_[op_type];
}

const OpImplRegistry::OpImplFunctionsV2 *OpImplRegistry::GetOpImpl(const ge::char_t *op_type) const
{
    const auto iter = types_to_impl_.find(op_type);
    if (iter == types_to_impl_.end()) {
        return nullptr;
    }
    return &iter->second;
}

const OpImplRegistry::PrivateAttrList &OpImplRegistry::GetPrivateAttrs(const ge::char_t *op_type) const
{
    const auto op_impl_ptr = GetOpImpl(op_type);
    if (op_impl_ptr == nullptr) {
        static OpImplRegistry::PrivateAttrList emptyPrivateAttr;
        return emptyPrivateAttr;
    }
    return op_impl_ptr->private_attrs;
}

const std::map<OpImplRegisterV2::OpType, OpImplRegistry::OpImplFunctionsV2> &OpImplRegistry::GetAllTypesToImpl() const
{
    return types_to_impl_;
}

std::map<OpImplRegisterV2::OpType, OpImplRegistry::OpImplFunctionsV2> &OpImplRegistry::GetAllTypesToImpl()
{
    return types_to_impl_;
}

OpImplRegisterV2::OpImplRegisterV2(const ge::char_t *op_type) : impl_(new(std::nothrow) OpImplRegisterV2Impl)
{
    if (impl_ != nullptr) {
        impl_->op_type = op_type;
        impl_->functions.infer_shape = nullptr;
        impl_->functions.infer_shape_range = nullptr;
        impl_->functions.infer_datatype = nullptr;
        impl_->functions.inputs_dependency = 0U;

        // two fields controlled by tiling func
        impl_->functions.tiling = nullptr;
        impl_->functions.max_tiling_data_size = std::numeric_limits<size_t>::max();

        // 3 fields controlled by tiling_parse func
        impl_->functions.tiling_parse = nullptr;
        impl_->functions.compile_info_creator = nullptr;
        impl_->functions.compile_info_deleter = nullptr;

        (void)OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type);
    }
}

OpImplRegisterV2 &OpImplRegisterV2::Tiling(
    OpImplKernelRegistry::TilingKernelFunc tiling_func, size_t max_tiling_data_size)
{
    if (impl_ != nullptr) {
        impl_->functions.tiling = tiling_func;
        impl_->functions.max_tiling_data_size = max_tiling_data_size;
    }
    return *this;
}

OpImplRegisterV2 &OpImplRegisterV2::TilingParse(KernelRegistry::KernelFunc tiling_parse_func,
    OpImplKernelRegistry::CompileInfoCreatorFunc creator_func,
    OpImplKernelRegistry::CompileInfoDeleterFunc deleter_func)
{
    if (impl_ != nullptr) {
        impl_->functions.tiling_parse = tiling_parse_func;
        impl_->functions.compile_info_creator = creator_func;
        impl_->functions.compile_info_deleter = deleter_func;
    }
    return *this;
}

OpImplRegisterV2::~OpImplRegisterV2() = default;
OpImplRegisterV2::OpImplRegisterV2(const OpImplRegisterV2 &register_data)
{
    RegisterOpImplToRegistry(register_data.impl_.get());
}

OpImplRegisterV2::OpImplRegisterV2(OpImplRegisterV2 &&register_data) noexcept
{
    RegisterOpImplToRegistry(register_data.impl_.get());
}

ge::graphStatus ComputeNodeInfo::CalcSize(
    const size_t ir_inputs_num, const size_t inputs_num, const size_t outputs_num, size_t &total_size)
{
    size_t ir_inputs_size;
    size_t inputs_size;
    size_t outputs_size;

    GE_ASSERT_TRUE(!ge::MulOverflow(sizeof(gert::AnchorInstanceInfo), ir_inputs_num, ir_inputs_size));
    GE_ASSERT_TRUE(!ge::MulOverflow(sizeof(CompileTimeTensorDesc), inputs_num, inputs_size));
    GE_ASSERT_TRUE(!ge::MulOverflow(sizeof(CompileTimeTensorDesc), outputs_num, outputs_size));

    total_size = sizeof(ComputeNodeInfo);
    GE_ASSERT_TRUE(!ge::AddOverflow(total_size, ir_inputs_size, total_size));
    GE_ASSERT_TRUE(!ge::AddOverflow(total_size, inputs_size, total_size));
    GE_ASSERT_TRUE(!ge::AddOverflow(total_size, outputs_size, total_size));
    return ge::GRAPH_SUCCESS;
}

void ComputeNodeInfo::Init(const size_t ir_inputs_num, const size_t inputs_num, const size_t outputs_num,
    const char *node_name, const char *node_type)
{
    ir_inputs_num_ = ir_inputs_num;
    inputs_num_ = inputs_num;
    outputs_num_ = outputs_num;
    node_name_ = node_name;
    node_type_ = node_type;
    memset(reserved_, 0, sizeof(reserved_));
}

RuntimeAttrs *ComputeNodeInfo::MutableAttrs() const
{
    return const_cast<RuntimeAttrs *>(GetAttrs());
}

class OpTilingContextBuilderImpl : public ContextBuilderImpl {
public:
    OpTilingContextBuilderImpl() : ContextBuilderImpl()
    {}
    ~OpTilingContextBuilderImpl() override = default;

    std::unique_ptr<ContextHolderImpl> BuildTilingContext()
    {
        std::unique_ptr<ContextHolderImpl> holder = std::make_unique<ContextHolderImpl>();
        return holder;
    }
};
OpTilingContextBuilder::OpTilingContextBuilder()
{}

OpTilingContextBuilder::~OpTilingContextBuilder()
{}

OpTilingContextBuilder &OpTilingContextBuilder::CompileInfo(const void *compile_info)
{
    return *this;
}

OpTilingContextBuilder &OpTilingContextBuilder::PlatformInfo(const void *platform_info)
{
    return *this;
}

OpTilingContextBuilder &OpTilingContextBuilder::Deterministic(int32_t deterministic)
{
    return *this;
}

OpTilingContextBuilder &OpTilingContextBuilder::TilingData(
    const gert::TilingData *tiling_data, gert::Chain::Deleter deleter)
{
    return *this;
}

OpTilingContextBuilder &OpTilingContextBuilder::Workspace(const gert::ContinuousVector *workspace)
{
    return *this;
}

OpTilingContextBuilder &OpTilingContextBuilder::InputTensors(const std::vector<gert::Tensor *> &inputs)
{
    return *this;
}

OpTilingContextBuilder &OpTilingContextBuilder::OutputTensors(const std::vector<gert::Tensor *> &outputs)
{
    return *this;
}

ContextHolder<TilingContext> OpTilingContextBuilder::Build()
{
    return ContextHolder<TilingContext>();
}

OpTilingParseContextBuilder::OpTilingParseContextBuilder()
{}

OpTilingParseContextBuilder::~OpTilingParseContextBuilder()
{}

OpTilingParseContextBuilder &OpTilingParseContextBuilder::InputTensorDesc(size_t index, ge::DataType dtype,
    ge::Format origin_format, ge::Format storage_format, const gert::ExpandDimsType &expand_dims_type)
{
    return *this;
}

OpTilingParseContextBuilder &OpTilingParseContextBuilder::OutputTensorDesc(size_t index, ge::DataType dtype,
    ge::Format origin_format, ge::Format storage_format, const gert::ExpandDimsType &expand_dims_type)
{
    return *this;
}

OpTilingParseContextBuilder &OpTilingParseContextBuilder::CompiledJson(const ge::char_t *compiled_json)
{
    return *this;
}

OpTilingParseContextBuilder &OpTilingParseContextBuilder::CompiledInfo(const void *compile_info)
{
    return *this;
}

OpTilingParseContextBuilder &OpTilingParseContextBuilder::PlatformInfo(const void *platform_info)
{
    return *this;
}

ContextHolder<TilingParseContext> OpTilingParseContextBuilder::Build()
{
    return ContextHolder<TilingParseContext>();
}

void *ContextHolderVoid::GetContext() const
{
    static int i = 0;
    return &i;
}

ContextHolderVoid::ContextHolderVoid() = default;
ContextHolderVoid::~ContextHolderVoid() = default;

namespace bg {
// namespace
std::atomic<int64_t> ValueHolder::id_generator_{0};
ValueHolder::~ValueHolder() = default;

ValueHolder::ValueHolder() : id_(id_generator_++), type_(ValueHolderType::kValueHolderTypeEnd), index_(0), placement_(0)
{

}

ValueHolderPtr ValueHolder::CreateMateFromNode(ge::FastNode *node, int32_t index, ValueHolderType type)
{
    auto holder = std::shared_ptr<ValueHolder>(new (std::nothrow) ValueHolder());
    return holder;
}

const char *BufferPool::GetBufById(const BufId id) const
{
    for (const auto &buf_and_id : bufs_to_id_) {
        if (buf_and_id.second == id) {
            return buf_and_id.first.c_str();
        }
    }
    for (const auto &buf_and_id : large_bufs_to_id_) {
        if (buf_and_id.second == id) {
            return buf_and_id.first.c_str();
        }
    }
    return nullptr;
}

bool AppendAttr(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs)
{
    return true;
}

bool GetAllIrAttrs(const ge::NodePtr &node, std::vector<std::vector<uint8_t>> &runtime_attrs)
{
    return true;
}

std::unique_ptr<uint8_t[]> CreateComputeNodeInfo(const ge::NodePtr &node, BufferPool &buffer_pool, size_t &total_size)
{
    size_t attr_size;
    std::vector<std::vector<uint8_t>> runtime_attrs;
    std::vector<ge::AnyValue> runtime_attrs_list;
    GetAllIrAttrs(node, runtime_attrs);
    for (auto &runtime_attr : runtime_attrs_list) {
        AppendAttr(runtime_attr, runtime_attrs);
    }
    const auto attr_buf = CreateAttrBuffer(runtime_attrs, attr_size);
    return CreateComputeNodeInfoImpl(attr_buf, attr_size, node, buffer_pool, total_size);
}

BufferPool::BufId BufferPool::AddLargeBuf(std::string &&str)
{
    auto id = id_generator_++;
    large_bufs_to_id_.emplace_back(std::move(str), id);
    return id;
}

BufferPool::BufId BufferPool::AddBuf(std::string &&str)
{
    auto res = bufs_to_id_.emplace(std::move(str), id_generator_);
    if (res.second) {
        ++id_generator_;
    }
    return res.first->second;
}

BufferPool::BufId BufferPool::AddStr(const char *data)
{
    size_t len = strlen(data) + 1;
    if (len >= 1024U * 1024U) {
        return AddLargeBuf(std::string(data, len));
    }
    return AddBuf(std::string(data, len));
}
}  // namespace bg
}  // namespace gert

namespace optiling {
namespace utils {
class OpRunInfoImpl {
public:
    OpRunInfoImpl() = default;
    ~OpRunInfoImpl() = default;

    OpRunInfoImpl(const uint32_t &block_dim, const bool &clear_atomic, const uint64_t &tiling_key)
        : block_dim_(block_dim), clear_atomic_(clear_atomic), tiling_key_(tiling_key), addr_base_(nullptr),
          max_size_(0), offset_(0), tiling_cond_(-1)
    {

    }

private:
    uint32_t block_dim_;
    bool clear_atomic_;
    uint64_t tiling_key_;
    ByteBuffer tiling_data_;
    std::vector<int64_t> workspaces_;
    void *addr_base_;
    uint64_t max_size_;
    uint64_t offset_;
    int32_t tiling_cond_;
};

void OpRunInfo::AddTilingData(const ge::char_t *value, const size_t size)
{
    (void)impl_->tiling_data_.write(value, static_cast<std::streamsize>(size));
    (void)impl_->tiling_data_.flush();
    return;
}

void OpRunInfo::SetBlockDim(const uint32_t &block_dim)
{
    return;
}

void OpRunInfo::SetTilingKey(const uint64_t &new_tiling_key)
{
    return;
}
}  // namespace utils
ge::graphStatus GetOperatorAttrValue(const ge::Operator &op, const char *attr_name, const char *attr_dtype,
    optiling::AttrDataPtr &attr_data_ptr, const char *target_dtype)
{
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
