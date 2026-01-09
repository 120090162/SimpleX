#include "simplex/core/simulator-x.hpp"

namespace simplex
{

    template struct SimulatorXTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>;

    template void
    SimulatorXTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::step<::simplex::ADMMContactSolverTpl>(
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        context::Scalar);

    template void
    SimulatorXTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::step<::simplex::ADMMContactSolverTpl>(
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const ::pinocchio::container::aligned_vector<::pinocchio::ForceTpl<context::Scalar>> &,
        context::Scalar);

    template void
    SimulatorXTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::step<::pinocchio::PGSContactSolverTpl>(
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        context::Scalar);

    template void
    SimulatorXTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::step<::pinocchio::PGSContactSolverTpl>(
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const ::pinocchio::container::aligned_vector<::pinocchio::ForceTpl<context::Scalar>> &,
        context::Scalar);

    template void SimulatorXTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::resolveConstraints<
        ::simplex::ADMMContactSolverTpl>();

    template void SimulatorXTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::resolveConstraints<
        ::pinocchio::PGSContactSolverTpl>();

#ifdef SIMPLEX_WITH_CLARABEL_SUPPORT
    template void SimulatorXTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::resolveConstraints<
        ::simplex::ClarabelContactSolverTpl>();

    template void
    SimulatorXTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::step<::simplex::ClarabelContactSolverTpl>(
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        context::Scalar);

    template void
    SimulatorXTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::step<::simplex::ClarabelContactSolverTpl>(
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const ::pinocchio::container::aligned_vector<::pinocchio::ForceTpl<context::Scalar>> &,
        context::Scalar);
#endif // SIMPLEX_WITH_CLARABEL_SUPPORT

} // namespace simplex
