#ifndef __simplex_pinocchio_template_instantiation_aba_txx__
#define __simplex_pinocchio_template_instantiation_aba_txx__

extern template PINOCCHIO_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI const ::simplex::context::VectorXs & ::pinocchio::aba<
    ::simplex::context::Scalar,
    ::simplex::context::Options,
    ::pinocchio::JointCollectionDefaultTpl,
    ::simplex::context::VectorXs,
    ::simplex::context::VectorXs,
    ::simplex::context::VectorXs>(
    const ::simplex::context::Model &,
    ::simplex::context::Data &,
    const Eigen::MatrixBase<context::VectorXs> &,
    const Eigen::MatrixBase<context::VectorXs> &,
    const Eigen::MatrixBase<context::VectorXs> &,
    const Convention);

extern template PINOCCHIO_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI const ::simplex::context::VectorXs & ::pinocchio::aba<
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
    const Eigen::MatrixBase<context::VectorXs> &,
    const Eigen::MatrixBase<context::VectorXs> &,
    const Eigen::MatrixBase<context::VectorXs> &,
    const std::vector<::simplex::context::Force, Eigen::aligned_allocator<::simplex::context::Force>> &,
    const Convention);

#endif // __simplex_pinocchio_template_instantiation_aba_txx__
