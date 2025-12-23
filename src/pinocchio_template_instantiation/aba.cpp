#include "simplex/fwd.hpp"
#include <pinocchio/algorithm/aba.hpp>

template PINOCCHIO_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI const ::simplex::context::VectorXs & ::pinocchio::aba<
    ::simplex::context::Scalar,
    ::simplex::context::Options,
    ::pinocchio::JointCollectionDefaultTpl,
    ::simplex::context::VectorXs,
    ::simplex::context::VectorXs,
    ::simplex::context::VectorXs>(
    const ::simplex::context::Model &,
    ::simplex::context::Data &,
    const Eigen::MatrixBase<::simplex::context::VectorXs> &,
    const Eigen::MatrixBase<::simplex::context::VectorXs> &,
    const Eigen::MatrixBase<::simplex::context::VectorXs> &,
    const Convention);

template PINOCCHIO_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI const ::simplex::context::VectorXs & ::pinocchio::aba<
    ::simplex::context::Scalar,
    ::simplex::context::Options,
    ::pinocchio::JointCollectionDefaultTpl,
    ::simplex::context::VectorXs,
    ::simplex::context::VectorXs,
    ::simplex::context::VectorXs,
    ::simplex::context::Force,
    Eigen::aligned_allocator<::simplex::context::Force>>(
    const ::simplex::context::Model &,
    ::simplex::context::Data &,
    const Eigen::MatrixBase<::simplex::context::VectorXs> &,
    const Eigen::MatrixBase<::simplex::context::VectorXs> &,
    const Eigen::MatrixBase<::simplex::context::VectorXs> &,
    const std::vector<::simplex::context::Force, Eigen::aligned_allocator<::simplex::context::Force>> &,
    const Convention);
