#ifndef HCCL_SIM_DATA_DEFS_H
#define HCCL_SIM_DATA_DEFS_H

#include <iostream>
#include <sstream>
#include <cstdint>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include "hccl_common_defs.h"

#define MAX_DEV_NUM 1024
#define MAX_NOTIFY_NUM 8192
#define MAX_STREAM_NUM 1024
#define MAX_MEM_BLOCK_NUM 128
#define MAX_MOCK_MEM_BLOCK_NUM 65536
#define MIN_MOCK_MEM 0x10000000000
#define MAX_MOCK_MEM 0x4000000000000
#define MOCK_MEM_GRAN 0x200000

namespace ipc = boost::interprocess;


/** -------------------------- 共享内存容器别名 -------------------------- */
// 共享内存段管理器类型别名
using ShmSegmentManager = ipc::managed_shared_memory::segment_manager;
// 共享字符串分配器
using ShmStringAllocator = ipc::allocator<char, ShmSegmentManager>;
// 共享内存字符串
using ShmString = ipc::basic_string<char, std::char_traits<char>, ShmStringAllocator>;
// 共享内存分配器
template <typename T>
using ShmAllocator = ipc::allocator<T, ShmSegmentManager>;
// 共享内存Vector
template <typename T>
using ShmVector = ipc::vector<T, ShmAllocator<T>>;
// 共享内存Map Allocator
template<typename K, typename V>
using ShmMapAllocator = ipc::allocator<std::pair<const K, V>, ShmSegmentManager>;
// 共享内存Map
template<typename K, typename V>
using ShmMap = boost::container::map<K, V, std::less<K>, ShmMapAllocator<K, V>>;


// CheckOpParam 结构体
struct CheckOpParam {
    ipc::interprocess_mutex mutex;
    int op_type;
    uint64_t op_id;
    bool is_completed;
    ShmString op_desc;

    // 构造函数声明
    CheckOpParam(ShmSegmentManager* seg_mgr);

    // 禁止默认构造/拷贝
    CheckOpParam() = delete;
    CheckOpParam(const CheckOpParam&) = delete;
    CheckOpParam& operator=(const CheckOpParam&) = delete;
};

union ShmNpuPos {
	struct {
		uint32_t podId : 8;
		uint32_t serId : 16;
		uint32_t phyId : 8;
	} field;
	uint32_t value{UINT32_MAX};

	ShmNpuPos() = default;
	explicit ShmNpuPos(uint32_t v) : value(v) {}
	ShmNpuPos(uint32_t pod, uint32_t ser, uint32_t phy) : field({pod, ser, phy}) {}

	bool operator<(const ShmNpuPos& other) const { return value < other.value; }
	bool operator>(const ShmNpuPos& other) const { return value > other.value; }
	bool operator==(const ShmNpuPos& other) const { return value == other.value; }
	bool operator!=(const ShmNpuPos& other) const { return value != other.value; }

	bool IsValid() const { return value != UINT32_MAX; }

	std::string ToString() const {
		std::stringstream ss;
		ss << "(" << field.podId << "," << field.serId << "," << field.phyId << ")";
		return ss.str();
	}
};

union ShmNpuResId {
	struct {
		uint64_t podId : 8;
		uint64_t serId : 16;
		uint64_t phyId : 8;
		uint64_t resId : 32;
	} field;
	uint64_t value{UINT64_MAX};

	ShmNpuResId() = default;
	explicit ShmNpuResId(uint64_t v) : value(v) {}
	ShmNpuResId(uint64_t pod, uint64_t ser, uint64_t phy, uint64_t res) : field({pod, ser, phy, res}) {}

	bool operator<(const ShmNpuResId& other) const { return value < other.value; }
	bool operator>(const ShmNpuResId& other) const { return value > other.value; }
	bool operator==(const ShmNpuResId& other) const { return value == other.value; }
	bool operator!=(const ShmNpuResId& other) const { return value != other.value; }

	bool IsValid() const { return value != UINT64_MAX; }
	ShmNpuPos GetNpuPos() const { return ShmNpuPos(field.podId, field.serId, field.phyId); }

	std::string ToString() const {
		std::stringstream ss;
		ss << "(" << field.podId << "," << field.serId << "," << field.phyId << "," << field.resId << ")";
		return ss.str();
	}
};

struct ShmSimNotify {
	ShmNpuResId notifyId{UINT64_MAX};
	bool isUsed{false};
	bool value{false};

	ShmSimNotify() = default;
	ShmSimNotify(const ShmSimNotify&) = delete;
	ShmSimNotify& operator=(const ShmSimNotify&) = delete;
};

struct ShmSimStream {
	ShmNpuResId streamId{UINT64_MAX};
	bool isUsed{false};

	ShmSimStream() = default;
	ShmSimStream(const ShmSimNotify&) = delete;
	ShmSimStream& operator=(const ShmSimStream&) = delete;
};

struct ShmNpuMemBlock {
	ipc::offset_ptr<void> addr{nullptr};
	uint64_t size{0};
	uint8_t bufferType{0};
	uintptr_t mockAddr{MIN_MOCK_MEM};

	ShmNpuMemBlock() = default;
	ShmNpuMemBlock(void* memAddr, uint64_t memSize) : addr(memAddr), size(memSize) {}
	ShmNpuMemBlock(const ShmNpuMemBlock&) = delete;
	ShmNpuMemBlock& operator=(const ShmNpuMemBlock&) = delete;

	bool IsValid() const { return addr != nullptr && size != 0; }
	void Reset() {
		addr = nullptr;
		size = 0;
	}
};

struct ProxyBuffer {
	ipc::interprocess_mutex mutex{};

	size_t totalSize{0};
	size_t allocatedSize{0};
	ipc::offset_ptr<void> bufferPtr{nullptr};

	ProxyBuffer() = default;
	ProxyBuffer(const ProxyBuffer&) = delete;
	ProxyBuffer& operator=(const ProxyBuffer&) = delete;
};

struct MockBufferBlock {
	uint64_t addr{0};
	size_t size{0};
	uint32_t npuPos{0};
	uint8_t bufferType{BufferType::RESERVED};
	uint64_t end() { return addr + size; };
};

struct MockProxyBuffer {
	ipc::interprocess_mutex mutex{};

	MockBufferBlock mockMem[MAX_MOCK_MEM_BLOCK_NUM];

	const uint64_t mockBase{MIN_MOCK_MEM};
    uint64_t nextAddr{MIN_MOCK_MEM};             // 下一个可用地址
	uint64_t mockMemCnt{0};

	MockProxyBuffer() {};
	MockProxyBuffer(const MockProxyBuffer&) = delete;
    MockProxyBuffer& operator=(const MockProxyBuffer&) = delete;
};

struct ShmSimNpu {
	ipc::interprocess_mutex mutex{};

	ShmNpuPos npuPos{UINT32_MAX};
	uint32_t memCount{0};
	size_t cclIdx{MAX_MEM_BLOCK_NUM};
	uint32_t devType_;

	// 硬件资源
	ShmSimNotify notify[MAX_NOTIFY_NUM];
	ShmSimStream stream[MAX_STREAM_NUM];
	ShmNpuMemBlock memory[MAX_MEM_BLOCK_NUM];

	ShmSimNpu() = default;
	ShmSimNpu(const ShmSimNpu&) = delete;
	ShmSimNpu& operator=(const ShmSimNpu&) = delete;
};

struct ShmSimWorld {
	ipc::interprocess_mutex mutex{};

	// NPU仿真建模
	uint32_t devNum{0};
	ShmSimNpu npu[MAX_DEV_NUM];

	ShmSimWorld() = default;
	ShmSimWorld(const ShmSimWorld&) = delete;
    ShmSimWorld& operator=(const ShmSimWorld&) = delete;
};

struct ShmCommDomain {
	ipc::interprocess_mutex mutex{};

	uint32_t rankNum{0};
	ShmNpuPos rankId2NpuPos[MAX_DEV_NUM];

	ShmCommDomain() = default;
	ShmCommDomain(const ShmCommDomain&) = delete;
	ShmCommDomain& operator=(const ShmCommDomain&) = delete;
};

struct TaskCollection {
    uint32_t indexGen;       // 任务索引生成器
    uint32_t maxSize;        // 最大任务数
    HcclTaskMetaData tasks[]; // 柔性数组存储任务（实际大小由 maxSize 决定）

    // 构造函数：初始化成员变量（柔性数组无需手动分配，由外部计算总大小）
    TaskCollection(uint32_t max_task_count) 
        : indexGen(0), maxSize(max_task_count) {
        // 柔性数组 tasks 会在对象创建时，随总大小一起分配
    }

    // 禁用默认构造和拷贝（避免错误使用）
    TaskCollection() = delete;
    TaskCollection(const TaskCollection&) = delete;
    TaskCollection& operator=(const TaskCollection&) = delete;
};

struct DumpMemBlock {
	uint32_t rankId;
    uint64_t startAddr;    // 物理起始地址
    uint64_t size;         // 块大小
    uint8_t  bufferType;         // 类型
};

struct ProxyConfig {
	ipc::interprocess_mutex mutex{};

	int consoleLogLevel{1};
	int fileLogLevel{1};

	ProxyConfig() = default;
	ProxyConfig(const ProxyConfig&) = delete;
	ProxyConfig& operator=(const ProxyConfig&) = delete;
};

#endif