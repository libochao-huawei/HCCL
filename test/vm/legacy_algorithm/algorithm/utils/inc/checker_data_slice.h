#ifndef LLT_DATA_SLICE_H
#define LLT_DATA_SLICE_H

#include "checker_buffer_type.h"

#include "llt_common.h"

#include <string>
#include <iostream>
#include <sstream>
#include <memory>

#include "checker_string_util.h"

namespace checker {

class  DataSlice {
public:
    explicit DataSlice() : type(BufferType::INPUT), offset(0), size(0)
    {
    }

    explicit DataSlice(u64 size) : type(BufferType::INPUT), offset(0), size(size)
    {
    }

    DataSlice(BufferType bufferType, u64 offset, u64 size) : type(bufferType), offset(offset), size(size)
    {
    }

    bool operator==(const DataSlice &other) const
    {
        return (type == other.type && offset == other.offset && size == other.size);
    }

    bool operator!=(const DataSlice &other) const
    {
        return (type != other.type || offset != other.offset || size != other.size);
    }

    std::string Describe() const
    {
        return StringFormat("DataSlice[%s, offset=%llu, size=%llu]", type.Describe().c_str(), offset, size);
    }

    inline BufferType GetType() const
    {
        return type;
    }

    inline u64 GetOffset() const
    {
        return offset;
    }

    inline u64 GetSize() const
    {
        return size;
    }

    void SetBufferType(const BufferType bufferType)
    {
        type = bufferType;
    }

    void SetOffset(u64 off)
    {
        offset = off;
    }

    void SetSize(u64 sizeOfSlice)
    {
        size = sizeOfSlice;
    }

private:
    BufferType type;
    u64        offset;
    u64        size;
};

bool DataSliceSizeIsEqual(std::unique_ptr<DataSlice> &a, std::unique_ptr<DataSlice> &b);
bool DataSliceSizeIsEqual(std::unique_ptr<DataSlice> &a, std::unique_ptr<DataSlice> &b, std::unique_ptr<DataSlice> &c);

}
#endif