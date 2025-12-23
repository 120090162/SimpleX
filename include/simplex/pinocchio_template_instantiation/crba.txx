#ifndef __simplex_pinocchio_template_instantiation_crba_txx__
#define __simplex_pinocchio_template_instantiation_crba_txx__

extern template PINOCCHIO_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI const ::simplex::context::MatrixXs & ::pinocchio::
    crba<::simplex::context::Scalar, ::simplex::context::Options, ::pinocchio::JointCollectionDefaultTpl, ::simplex::context::VectorXs>(
        const context::Model &, context::Data &, const Eigen::MatrixBase<context::VectorXs> &, const Convention convention);

#endif // __simplex_pinocchio_template_instantiation_crba_txx__