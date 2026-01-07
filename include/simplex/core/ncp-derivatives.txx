#ifndef __simplex_core_ncp_derivatives_txx__
#define __simplex_core_ncp_derivatives_txx__

#include "simplex/core/ncp-derivatives.hpp"

namespace simplex
{

    extern template struct ContactSolverDerivativesTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl>;

    extern template void ContactSolverDerivativesTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl>::jvp(
        const Eigen::MatrixBase<Eigen::Ref<const context::MatrixXs>> &);

} // namespace simplex

#endif // __simplex_core_ncp_derivatives_txx__