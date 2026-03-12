#ifndef __LLT_HCCL_STUB_PUB_H__
#define __LLT_HCCL_STUB_PUB_H__
#include "hccl/hccl_types.h"
#include "comm.h"

void log_level_set_stub(s32 log_level);

#ifdef __cplusplus
extern "C" {
#endif
int ra_cq_create(void *rdev_handle, struct cq_attr *attr);
int ra_cq_destroy(void *rdev_handle, struct cq_attr *attr);
int ra_is_first_used(int ins_id);
int ra_is_last_used(int ins_id);
int ra_normal_qp_create(void *rdev_handle, struct ibv_qp_init_attr *qp_init_attr, void **qp_handle, void** qp);
int ra_normal_qp_destroy(void *qp_handle);
int ra_set_qp_attr_qos(void *qpHandle, struct qos_attr *attr);
int ra_set_qp_attr_timeout(void *qpHandle, u32 *timeout);
int ra_set_qp_attr_retry_cnt(void *qpHandle, u32 *retry_cnt);
int ra_create_comp_channel(const void *rdma_handle, void **comp_channel);
int ra_destroy_comp_channel(const void *rdma_handle, void *comp_channel);
int ra_get_cqe_err_info(unsigned int phy_id, struct cqe_err_info *info);
int ra_get_qp_attr(void *qp_handle, struct qp_attr *attr);
int ra_create_srq(const void*, struct srq_attr *);
int ra_destroy_srq(const void*, struct srq_attr *);
int ra_qp_create_with_attrs(void *rdev_handle, struct qp_ext_attrs *ext_attrs, void **qp_handle);
int ra_ai_qp_create(void *rdma_handle, struct qp_ext_attrs *qp_attrs, struct ai_qp_info *info, void **qpHandle);
int ra_rdev_get_support_lite(void *rdma_handle, int *support_lite);
int ra_typical_qp_create(void *rdev_handle, int flag, int qp_mode, struct typical_qp *qp_info, void **qp_handle);
int ra_typical_qp_modify(void *rdev_handle, struct typical_qp *local_qp_info, struct typical_qp *remote_qp_info);
int ra_typical_send_wr(void *qp_handle, struct send_wr *wr, struct send_wr_rsp *op_rsp);
int ra_socket_accept_credit_add(struct socket_listen_info_t conn[], unsigned int num, unsigned int creditLimit);
int ra_remap_mr(const void *rdmaHandle, struct mem_remap_info info[], unsigned int num);
int ra_get_tls_enable(struct ra_info *info, bool *tls_enable);
#ifdef __cplusplus
}
#endif
#endif /* __LLT_HCCL_STUB_PUB_H__ */

