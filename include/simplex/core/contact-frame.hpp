#ifndef __simplex_core_contact_frame_hpp__
#define __simplex_core_contact_frame_hpp__

#include "simplex/core/fwd.hpp"
#include "simplex/macros.hpp"

#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/multibody/geometry-object.hpp>

#ifndef SIMPLEX_SKIP_COLLISION_DERIVATIVES_CONTRIBUTIONS
    #include <diffcoal/contact_patch_derivative.hpp>
#endif

namespace simplex
{

    /// @brief Operator to construct a placement from a given normal and position.
    template<typename _Scalar, int _Options>
    struct PlacementFromNormalAndPositionTpl
    {
        using Scalar = _Scalar;
        enum
        {
            Options = _Options
        };

        using SE3 = ::pinocchio::SE3Tpl<Scalar, Options>;
        using Matrix63s = Eigen::Matrix<Scalar, 6, 3, Options>;
        using Vector3s = Eigen::Matrix<Scalar, 3, 1, Options>;
        using Block3 = Eigen::Block<Matrix6x3Type1, 3, 3>;

        /// \brief Computes a placement M from a normal and position.
        /// \param[in] normal is the z-axis of the rotation part of M.
        /// \param[in] position is the translation part of M.
        /// \param[out] M the placement.
        template<typename Vector3Normal, typename Vector3Position>
        static void calc(
            const Eigen::MatrixBase<Vector3Normal> & normal,     // [input]
            const Eigen::MatrixBase<Vector3Position> & position, // [input]
            SE3 & M                                              // [output]
        );

        /// \brief Computes the derivative of M w.r.t normal and position used to construct it.
        /// \param[in] M
        /// \param[out] dM_dnormal derivative w.r.t normal
        /// \param[out] dM_dposition derivative w.r.t position
        template<typename Matrix6x3Type1, typename Matrix6x3Type2>
        static void calcDiff(
            const SE3 & M,                                         // [input]
            const Eigen::MatrixBase<Matrix6x3Type1> & dM_dnormal,  // [output]
            const Eigen::MatrixBase<Matrix6x3Type2> & dM_dposition // [output]
        );

        SIMPLEX_PROTECTED
        /// \brief Given a normal, returns the associated non-coliear reference vector.
        /// This vector is then used to compute the rotation part of the placement.
        template<typename Vector3>
        static Vector3s getReferenceVector(const Eigen::MatrixBase<Vector3> & normal)
        {
            EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Vector3, 3);

            Vector3s e_ref(PlacementFromNormalAndPosition::reference_vector());
            static constexpr Scalar eps = 1e-6;
            if ((e_ref.cross(normal)).norm() <= eps)
            {
                e_ref = PlacementFromNormalAndPosition::other_reference_vector();
            }

            return e_ref;
        }

        /// \brief Reference vector to compute the rotation part of a placement.
        /// If the normal is colinear to this reference vector, `e_ref2` is used instead.
        static Vector3s reference_vector()
        {
            return {1, 0, 0};
        }

        /// \brief Other reference vector to compute the rotation part of a placement.
        static Vector3s other_reference_vector()
        {
            return {0, 1, 0};
        }
    };

#ifndef SIMPLEX_SKIP_COLLISION_DERIVATIVES_CONTRIBUTIONS
    /// \brief Functor to compute the derivatives of a contact patch.
    struct ComputeContactPatchDerivative : ::diffcoal::ComputeContactPatchDerivative
    {
        using Base = ::diffcoal::ComputeContactPatchDerivative;
        using GeometryObject = ::pinocchio::GeometryObject;

        ComputeContactPatchDerivative(const GeometryObject & go1, const GeometryObject & go2)
        : Base(go1.geometry.get(), go2.geometry.get())
        , go1_ptr(&go1)
        , go2_ptr(&go2)
        {
        }

        virtual ~ComputeContactPatchDerivative() {};

        virtual void
        run(const ::coal::Transform3f & M1,
            const ::coal::Transform3f & M2,
            const ::coal::ContactPatch & patch,
            const ::diffcoal::ContactPatchDerivativeRequest & drequest,
            ::diffcoal::ContactPatchDerivative & dpatch) const override
        {
            typedef ::coal::CollisionGeometry const * Pointer;
            const_cast<Pointer &>(Base::geom1) = go1_ptr->geometry.get();
            const_cast<Pointer &>(Base::geom2) = go2_ptr->geometry.get();
            return Base::run(M1, M2, patch, drequest, dpatch);
        }

        bool operator==(const ComputeContactPatchDerivative & other) const
        {
            return Base::operator==(other) && go1_ptr == other.go1_ptr && go2_ptr == other.go2_ptr; // Maybe, it would be better to just
                                                                                                    // check *go2_ptr == *other.go2_ptr
        }

        bool operator!=(const ComputeContactPatchDerivative & other) const
        {
            return !(*this == other);
        }

        const GeometryObject & getGeometryObject1() const
        {
            return *go1_ptr;
        }
        const GeometryObject & getGeometryObject2() const
        {
            return *go2_ptr;
        }

        SIMPLEX_PROTECTED
        const GeometryObject * go1_ptr;
        const GeometryObject * go2_ptr;
    };
#endif

} // namespace simplex

#include "simplex/core/contact-frame.hxx"

#endif // __simplex_core_contact_frame_hpp__
