#ifndef __simplex_core_fwd_hpp__
#define __simplex_core_fwd_hpp__

#include "simplex/fwd.hpp"
#include <pinocchio/utils/reference.hpp>

namespace simplex
{
    /**
     * @brief Helper namespace to bring Pinocchio's reference utilities into Simplex.
     *
     * These utilities help in managing pointers and references consistently.
     */
    namespace helper
    {
        using pinocchio::helper::get_pointer;
        using pinocchio::helper::get_ref;
    } // namespace helper

    // --- Forward Declarations of Template Classes ---

    /**
     * @brief Core structure representing the constrained dynamics problem with derivatives.
     */
    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl = ::pinocchio::JointCollectionDefaultTpl>
    struct ConstraintsProblemDerivativesTpl;

    /**
     * @brief Core structure representing the constrained dynamics problem.
     */
    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl = ::pinocchio::JointCollectionDefaultTpl>
    struct ConstraintsProblemTpl;

    /**
     * @brief Main simple simulator class for forward dynamics and integration.
     */
    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl = ::pinocchio::JointCollectionDefaultTpl>
    struct SimulatorTpl;

    /**
     * @brief Main simplex simulator class for forward dynamics and integration.
     */
    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl = ::pinocchio::JointCollectionDefaultTpl>
    struct SimulatorXTpl;

    /**
     * @brief Container for derivatives of the contact solver (Jacobians of the LCP/MCP).
     */
    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl = ::pinocchio::JointCollectionDefaultTpl>
    struct ContactSolverDerivativesTpl;

    /**
     * @brief Main class for computing and storing simulation sensitivities (Differentiable Physics).
     */
    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl = ::pinocchio::JointCollectionDefaultTpl>
    struct SimulatorDerivativesTpl;

    /**
     * @brief Utility to compute coordinate frames from surface normals and positions.
     */
    template<typename Scalar, int Options>
    struct PlacementFromNormalAndPositionTpl;

    // --- Default Typedefs (using the global context) ---

    /** @brief Default ConstraintsProblemDerivatives using context types. */
    typedef ConstraintsProblemDerivativesTpl<context::Scalar, context::Options> ConstraintsProblemDerivatives;

    /** @brief Default ConstraintsProblem using context types. */
    typedef ConstraintsProblemTpl<context::Scalar, context::Options> ConstraintsProblem;

    /** @brief Default Simulator using context types. */
    typedef SimulatorTpl<context::Scalar, context::Options> Simulator;

    /** @brief Default SimulatorX using context types. */
    typedef SimulatorXTpl<context::Scalar, context::Options> SimulatorX;

    /** @brief Default ContactSolverDerivatives using context types. */
    typedef ContactSolverDerivativesTpl<context::Scalar, context::Options> ContactSolverDerivatives;

    /** @brief Default SimulatorDerivatives using context types. */
    typedef SimulatorDerivativesTpl<context::Scalar, context::Options> SimulatorDerivatives;

    /** @brief Default PlacementFromNormalAndPosition using context types. */
    typedef PlacementFromNormalAndPositionTpl<context::Scalar, context::Options> PlacementFromNormalAndPosition;

} // namespace simplex

#endif // ifndef __simplex_core_fwd_hpp__