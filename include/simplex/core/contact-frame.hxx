#ifndef __simplex_core_contact_frame_hxx__
#define __simplex_core_contact_frame_hxx__

namespace simplex
{

    // ==========================================================================
    template<typename S, int O>
    template<typename Vector3Normal, typename Vector3Position>
    void PlacementFromNormalAndPositionTpl<S, O>::calc(
        const Eigen::MatrixBase<Vector3Normal> & normal,     // [input]
        const Eigen::MatrixBase<Vector3Position> & position, // [input]
        SE3 & M                                              // [output]
    )
    {
        // Static asserts on input sizes
        EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Vector3Normal, 3);
        EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Vector3Position, 3);

        M.translation() = position;

        // Ensure normal is normalized
        assert(normal.norm() - 1 <= 1e-12);

        // The normal will serve as the z-axis of M's rotation.
        const Vector3s normal_ = normal.normalized();
        const Vector3s e_ref(PlacementFromNormalAndPosition::getReferenceVector(normal_));

        M.rotation().col(2) = normal_;                                          // z-axis
        M.rotation().col(0) = (e_ref.cross(normal)).normalized();               // x-axis
        M.rotation().col(1) = (M.rotation().col(2)).cross(M.rotation().col(0)); // y-axis
    }

    // ========================================================================
    template<typename S, int O>
    template<typename Matrix6x3Type1, typename Matrix6x3Type2>
    void PlacementFromNormalAndPositionTpl<S, O>::calcDiff(
        const SE3 & M,                                          // [input]
        const Eigen::MatrixBase<Matrix6x3Type1> & dM_dnormal_,  // [output]
        const Eigen::MatrixBase<Matrix6x3Type2> & dM_dposition_ // [output]
    )
    {
        EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Matrix6x3Type1, 6, 3);
        EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Matrix6x3Type1, 6, 3);

        // Remove constness
        Matrix6x3Type1 & dM_dnormal = const_cast<Matrix6x3Type1 &>(dM_dnormal_.derived());
        Matrix6x3Type2 & dM_dposition = const_cast<Matrix6x3Type2 &>(dM_dposition_.derived());

        // TODO(louis): exploit sparsity, don't use 6x3 matrices...
        //   -> dM_dn linear part is always 0
        //   -> dM_dp angular part is always 0

        ///
        dM_dnormal.setZero();
        Block3 dM_dnormal_angular = dM_dnormal.template bottomRows<3>();
        dM_dnormal_angular.row(0) = -M.rotation().col(1);
        dM_dnormal_angular.row(1) = M.rotation().col(0);
        const Vector3s e_ref(PlacementFromNormalAndPosition::getReferenceVector(M.rotation().col(2)));
        dM_dnormal_angular.row(2) = (M.rotation().col(1)).cross(e_ref) / (e_ref.cross(M.rotation().col(2))).norm();

        ///
        dM_dposition.setZero();
        dM_dposition.template topRows<3>() = M.rotation().transpose(); // dp = R^T dposition
    }

} // namespace simplex

#endif // __simplex_core_contact_frame_hxx__
