#ifndef LLT_ENUM_FACTORY_H
#define LLT_ENUM_FACTORY_H

#include "checker_string_util.h"

#include <string>
#include <sstream>

#define MAKE_ENUM(enumClass, ...)                                                                                      \
    class enumClass {                                                                                                  \
    public:                                                                                                            \
        enum Value : uint8_t { __VA_ARGS__, __COUNT__, INVALID };                                                      \
                                                                                                                       \
        enumClass() = default;                                                                                         \
                                                                                                                       \
        constexpr enumClass(Value v) : value(v)                                                                        \
        {                                                                                                              \
        }                                                                                                              \
                                                                                                                       \
        constexpr operator Value() const                                                                               \
        {                                                                                                              \
            return value;                                                                                              \
        }                                                                                                              \
                                                                                                                       \
        constexpr bool operator==(enumClass a) const                                                                   \
        {                                                                                                              \
            return value == a.value;                                                                                   \
        }                                                                                                              \
                                                                                                                       \
        constexpr bool operator!=(enumClass a) const                                                                   \
        {                                                                                                              \
            return value != a.value;                                                                                   \
        }                                                                                                              \
                                                                                                                       \
        constexpr bool operator<(enumClass a) const                                                                    \
        {                                                                                                              \
            return value < a.value;                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        constexpr bool operator==(Value v) const                                                                       \
        {                                                                                                              \
            return value == v;                                                                                         \
        }                                                                                                              \
                                                                                                                       \
        constexpr bool operator!=(Value v) const                                                                       \
        {                                                                                                              \
            return value != v;                                                                                         \
        }                                                                                                              \
                                                                                                                       \
        constexpr bool operator<(Value v) const                                                                        \
        {                                                                                                              \
            return value < v;                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        std::string Describe() const                                                                                   \
        {                                                                                                              \
            static std::vector<std::string> m = InitStrs();                                                            \
            if (value > m.size())                                                                                      \
                return std::string(#enumClass) + "::Invalid";                                                          \
            return m[value];                                                                                           \
        }                                                                                                              \
                                                                                                                       \
        friend std::ostream &operator<<(std::ostream &stream, const enumClass &v)                                      \
        {                                                                                                              \
            return stream << v.Describe();                                                                             \
        }                                                                                                              \
                                                                                                                       \
    private:                                                                                                           \
        Value value;                                                                                                   \
                                                                                                                       \
        static std::vector<std::string> InitStrs()                                                                     \
        {                                                                                                              \
            std::vector<std::string> strings;                                                                          \
            std::string              s = #__VA_ARGS__;                                                                 \
            std::string              token;                                                                            \
            for (char c : s) {                                                                                         \
                if (c == ' ' || c == ',') {                                                                            \
                    if (!token.empty()) {                                                                              \
                        strings.push_back({std::string(#enumClass) + "::" + token});                                   \
                        token.clear();                                                                                 \
                    }                                                                                                  \
                } else {                                                                                               \
                    token += c;                                                                                        \
                }                                                                                                      \
            }                                                                                                          \
            if (!token.empty())                                                                                        \
                strings.push_back({std::string(#enumClass) + "::" + token});                                           \
            return strings;                                                                                            \
        }                                                                                                              \
    };

#endif
