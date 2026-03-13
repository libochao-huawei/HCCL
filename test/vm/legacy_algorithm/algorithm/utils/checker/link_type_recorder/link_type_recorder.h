#ifndef HCCLV1_LINK_TYPE_RECORDER_H
#define HCCLV1_LINK_TYPE_RECORDER_H


#include "llt_common.h"
#include <map>
#include "hccl_common.h"
#include "checker_def.h"
namespace checker {

class LinkTypeRecorder {
public:
    static LinkTypeRecorder* Global();
    std::map<CheckerDevType, std::map<u32, std::map<u32, LinkTypeInServer>>> devLinkTypeMap_;

    void SetLinkTypeMap(std::vector<CheckerDevType> &devTypes);
    void SetLinkTypeMapOf910A();
    void SetLinkTypeMapOf910B();
    void SetLinkTypeMapOf310P3V();
    void SetLinkTypeMapOf310P3Dou();
    void SetLinkTypeMapOf910_93();
    void SetIs310P3V(bool is310P3V);

    bool is310P3V_ = false;
};

}

#endif