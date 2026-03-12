#ifndef LLT_BUFFER_TYPE_H
#define LLT_BUFFER_TYPE_H

#include <map>
#include <string>

#include "enum_factory.h"

namespace checker {
MAKE_ENUM(BufferType, INPUT, OUTPUT, INPUT_CCL, OUTPUT_CCL, SCRATCH, INPUT_AIV, OUTPUT_AIV, AIV_COMMINFO, USERBUF_AIV, MS, RESERVED)
}

#endif