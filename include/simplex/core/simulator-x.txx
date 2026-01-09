#ifndef __simplex_core_simulator_x_txx__
#define __simplex_core_simulator_x_txx__

#include "simplex/core/simulator-x.hpp"

namespace simplex
{

    extern template struct SimulatorXTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>;

    extern template void
    SimulatorXTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::step<::simplex::ADMMContactSolverTpl>(
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        context::Scalar);

    extern template void
    SimulatorXTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::step<::simplex::ADMMContactSolverTpl>(
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const ::pinocchio::container::aligned_vector<::pinocchio::ForceTpl<context::Scalar>> &,
        context::Scalar);

    extern template void
    SimulatorXTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::step<::pinocchio::PGSContactSolverTpl>(
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        context::Scalar);

    extern template void
    SimulatorXTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::step<::pinocchio::PGSContactSolverTpl>(
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const ::pinocchio::container::aligned_vector<::pinocchio::ForceTpl<context::Scalar>> &,
        context::Scalar);

    extern template void SimulatorXTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::resolveConstraints<
        ::simplex::ADMMContactSolverTpl>();

    extern template void SimulatorXTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::resolveConstraints<
        ::pinocchio::PGSContactSolverTpl>();

#ifdef SIMPLEX_WITH_CLARABEL_SUPPORT
    extern template void SimulatorXTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::resolveConstraints<
        ::simplex::ClarabelContactSolverTpl>();

    extern template void
    SimulatorXTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::step<::simplex::ClarabelContactSolverTpl>(
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        context::Scalar);

    extern template void
    SimulatorXTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::step<::simplex::ClarabelContactSolverTpl>(
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const ::pinocchio::container::aligned_vector<::pinocchio::ForceTpl<context::Scalar>> &,
        context::Scalar);
#endif // SIMPLEX_WITH_CLARABEL_SUPPORT

} // namespace simplex

#endif // __simplex_core_simulator_x_txx__
