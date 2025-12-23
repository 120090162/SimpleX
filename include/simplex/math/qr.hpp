#ifndef __simplex_math_qr_hpp__
#define __simplex_math_qr_hpp__

#include "simplex/math/fwd.hpp"
#include <Eigen/QR>

namespace simplex
{
    namespace math
    {
        /**
         * @brief A wrapper for Eigen solvers to provide a unified in-place solve interface.
         *
         * In-place solving overwrites the input RHS (Right-Hand Side) matrix with the
         * solution matrix, minimizing memory reallocations.
         */
        template<typename _SolverType>
        struct SolveInPlaceWrapper : _SolverType
        {
            typedef _SolverType SolverType;

            /**
             * @brief Generic fallback for in-place solving.
             *
             * Note: This version still performs an internal copy because the underlying
             * solver does not support direct in-place operations.
             */
            template<typename MatrixType>
            void solveInPlace(const Eigen::MatrixBase<MatrixType> & mat) const
            {
                // Use Pinocchio macro to determine the plain matrix type for temporary storage
                typename PINOCCHIO_EIGEN_PLAIN_TYPE(MatrixType) res(mat);
                res = this->solve(mat);
                // Overwrite the input matrix with the computed solution
                mat.const_cast_derived() = res; // const -> non-const
            }

        }; // struct SolveInPlaceWrapper

        /**
         * @brief Optimized specialization for Eigen::HouseholderQR.
         *
         * This version exploits the Householder decomposition structure to perform
         * the solve truly in-place without forming the full Q matrix.
         */
        template<typename _MatrixType>
        struct SolveInPlaceWrapper<Eigen::HouseholderQR<_MatrixType>> : Eigen::HouseholderQR<_MatrixType>
        {
            typedef Eigen::HouseholderQR<_MatrixType> SolverType;

            /**
             * @brief Performs an optimized in-place solve.
             *
             * Mathematically: x = R^-1 * Q* * b
             * 1. Applies Householder transformations (Q*) to the input.
             * 2. Solves the upper triangular system (R) via back-substitution.
             */
            template<typename MatrixType>
            void solveInPlace(const Eigen::MatrixBase<MatrixType> & mat_) const
            {
                // Calculate the effective rank for the QR decomposition
                const Eigen::Index rank = (std::min)(this->rows(), this->cols());
                MatrixType & mat = mat_.const_cast_derived();

                // Step 1: Apply Q* (adjoint of Q) to the input matrix from the left.
                // setLength(rank) ensures we only apply necessary transformations.
                mat.applyOnTheLeft(this->householderQ().setLength(rank).adjoint());

                // Step 2: Solve R * x = (Q* * b) using the upper triangular part of the QR matrix.
                // TriangularView::solveInPlace modifies the top rows of 'mat' directly.
                this->m_qr.topLeftCorner(rank, rank).template triangularView<Eigen::Upper>().solveInPlace(mat.topRows(rank));

                // Step 3: For over-determined systems, zero out the redundant rows.
                if (this->cols() > rank)
                {
                    mat.bottomRows(this->cols() - rank).setZero();
                }
            }
        };

    } // namespace math
} // namespace simplex

#endif // ifndef __simplex_math_qr_hpp__