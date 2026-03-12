#include "link_type_recorder.h"

namespace checker {
LinkTypeRecorder* LinkTypeRecorder::Global()
{
    static LinkTypeRecorder* globalLinkTypeRecorder = new LinkTypeRecorder;
    return globalLinkTypeRecorder;
}

void LinkTypeRecorder::SetIs310P3V(bool is310P3V)
{
    is310P3V_ = is310P3V;
    return;
}

void LinkTypeRecorder::SetLinkTypeMap(std::vector<CheckerDevType> &devTypes)
{
    devLinkTypeMap_.clear();

    for (std::vector<CheckerDevType>::iterator it = devTypes.begin(); it != devTypes.end(); it++) {
        switch (*it) {
            case CheckerDevType::DEV_TYPE_910:
                this->SetLinkTypeMapOf910A();
                break;
            case CheckerDevType::DEV_TYPE_910B:
                this->SetLinkTypeMapOf910B();
                break;
            case CheckerDevType::DEV_TYPE_310P3:
                if (is310P3V_) {
                    this->SetLinkTypeMapOf310P3V();
                } else {
                    this->SetLinkTypeMapOf310P3Dou();
                }
                break;
            case CheckerDevType::DEV_TYPE_910_93:
                this->SetLinkTypeMapOf910_93();
                break;
            default:
                HCCL_ERROR("the devType [%d] is not support", *it);
                break;
        }
    }
}

void LinkTypeRecorder::SetLinkTypeMapOf910A()
{
    auto &linkTypeMap_ = devLinkTypeMap_[CheckerDevType::DEV_TYPE_910];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            linkTypeMap_[i][j] = LinkTypeInServer::RESERVED_LINK_TYPE;
        }
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i != j) {
                linkTypeMap_[i][j] = LinkTypeInServer::HCCS_TYPE;
            }
        }
    }
    for (int i = 4; i < 8; i++) {
        for (int j = 4; j < 8; j++) {
            if (i != j) {
                linkTypeMap_[i][j] = LinkTypeInServer::HCCS_TYPE;
            }
        }
    }
    linkTypeMap_[0][4] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[1][5] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[2][6] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[3][7] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[4][0] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[5][1] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[6][2] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[7][3] = LinkTypeInServer::PXI_TYPE;
}

void LinkTypeRecorder::SetLinkTypeMapOf910B()
{
    auto &linkTypeMap_ = devLinkTypeMap_[CheckerDevType::DEV_TYPE_910B];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            linkTypeMap_[i][j] = LinkTypeInServer::RESERVED_LINK_TYPE;
        }
    }
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (i != j) {
                linkTypeMap_[i][j] = LinkTypeInServer::HCCS_TYPE;
            }
        }
    }
    for (int i = 8; i < 16; i++) {
        for (int j = 8; j < 16; j++) {
            if (i != j) {
                linkTypeMap_[i][j] = LinkTypeInServer::HCCS_TYPE;
            }
        }
    }
    linkTypeMap_[0][8] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[1][9] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[2][10] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[3][11] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[4][12] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[5][13] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[6][14] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[7][15] = LinkTypeInServer::PXI_TYPE;

    linkTypeMap_[8][0] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[9][1] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[10][2] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[11][3] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[12][4] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[13][5] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[14][6] = LinkTypeInServer::PXI_TYPE;
    linkTypeMap_[15][7] = LinkTypeInServer::PXI_TYPE;
}

void LinkTypeRecorder::SetLinkTypeMapOf310P3V()
{
    auto &linkTypeMap_ = devLinkTypeMap_[CheckerDevType::DEV_TYPE_310P3];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            linkTypeMap_[i][j] = LinkTypeInServer::RESERVED_LINK_TYPE;
        }
    }

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (i != j) {
                linkTypeMap_[i][j] = LinkTypeInServer::PXI_TYPE;
            }
        }
    }
    return;
}

void LinkTypeRecorder::SetLinkTypeMapOf310P3Dou()
{
    auto &linkTypeMap_ = devLinkTypeMap_[CheckerDevType::DEV_TYPE_310P3];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            linkTypeMap_[i][j] = LinkTypeInServer::RESERVED_LINK_TYPE;
        }
    }

    linkTypeMap_[0][1] = LinkTypeInServer::HCCS_TYPE;
    linkTypeMap_[1][0] = LinkTypeInServer::HCCS_TYPE;

    linkTypeMap_[2][3] = LinkTypeInServer::HCCS_TYPE;
    linkTypeMap_[3][2] = LinkTypeInServer::HCCS_TYPE;

    linkTypeMap_[4][5] = LinkTypeInServer::HCCS_TYPE;
    linkTypeMap_[5][4] = LinkTypeInServer::HCCS_TYPE;

    linkTypeMap_[6][7] = LinkTypeInServer::HCCS_TYPE;
    linkTypeMap_[7][6] = LinkTypeInServer::HCCS_TYPE;

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if ((i % 2 == 1) || (j % 2 == 1)) {
                continue;
            }

            if (i != j) {
                linkTypeMap_[i][j] = LinkTypeInServer::PXI_TYPE;
            }
        }
    }

    return;
}

void LinkTypeRecorder::SetLinkTypeMapOf910_93()
{
    auto &linkTypeMap_ = devLinkTypeMap_[CheckerDevType::DEV_TYPE_910_93];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            linkTypeMap_[i][j] = LinkTypeInServer::RESERVED_LINK_TYPE;
        }
    }

    linkTypeMap_[0][1] = LinkTypeInServer::SIO_TYPE;
    linkTypeMap_[1][0] = LinkTypeInServer::SIO_TYPE;

    linkTypeMap_[2][3] = LinkTypeInServer::SIO_TYPE;
    linkTypeMap_[3][2] = LinkTypeInServer::SIO_TYPE;

    linkTypeMap_[4][5] = LinkTypeInServer::SIO_TYPE;
    linkTypeMap_[5][4] = LinkTypeInServer::SIO_TYPE;

    linkTypeMap_[6][7] = LinkTypeInServer::SIO_TYPE;
    linkTypeMap_[7][6] = LinkTypeInServer::SIO_TYPE;

    linkTypeMap_[8][9] = LinkTypeInServer::SIO_TYPE;
    linkTypeMap_[9][8] = LinkTypeInServer::SIO_TYPE;

    linkTypeMap_[10][11] = LinkTypeInServer::SIO_TYPE;
    linkTypeMap_[11][10] = LinkTypeInServer::SIO_TYPE;

    linkTypeMap_[12][13] = LinkTypeInServer::SIO_TYPE;
    linkTypeMap_[13][12] = LinkTypeInServer::SIO_TYPE;

    linkTypeMap_[14][15] = LinkTypeInServer::SIO_TYPE;
    linkTypeMap_[15][14] = LinkTypeInServer::SIO_TYPE;

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            // 同封装的两个die内已经有了SIO链路，不需要重复赋值
            if ((i / 2) == (j / 2)) {
                continue;
            }
            linkTypeMap_[i][j] = LinkTypeInServer::HCCS_SW_TYPE;
        }
    }
    return;
}

}