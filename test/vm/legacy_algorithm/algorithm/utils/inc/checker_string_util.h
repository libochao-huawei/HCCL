#ifndef LLT_STRING_UTIL_H
#define LLT_STRING_UTIL_H

#include <string>
#include <vector>
#include <sstream>
#include "securec.h"

namespace checker {

template <typename... Args> inline std::string StringFormat(const char *format, Args... args)
{
    using namespace std;
    constexpr size_t bufSize = BUFSIZ;
    char             buffer[bufSize];
    size_t           actualSize = snprintf_s(&buffer[0], bufSize, bufSize, format, args...);
    actualSize++;

    if (actualSize > bufSize) {
        std::vector<char> newbuffer(actualSize);
        snprintf_s(newbuffer.data(), actualSize, actualSize, format, args...);
        return newbuffer.data();
    }
    return buffer;
}

template <typename I> std::string Dec2hex(I i)
{
    static_assert(std::is_integral<I>::value, "type I is not a integral");
    std::stringstream ss;
    ss << std::hex << "0x" << i;
    return ss.str();
}

std::vector<std::string> SplitString(const std::string &str, const char c);

template <class T> T String2T(const std::string &s)
{
    // T must support >>
    T                  t;
    std::istringstream ist(s);
    ist >> t;
    return t;
}

}

#endif
