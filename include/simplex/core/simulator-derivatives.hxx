#ifndef __simplex_core_simulator_derivatives_hxx__
#define __simplex_core_simulator_derivatives_hxx__

#include "simplex/core/simulator-derivatives.hpp"
#include "simplex/tracy.hpp"

#include <pinocchio/algorithm/aba-derivatives.hpp>

namespace simplex
{
    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    SimulatorDerivativesTpl<S, O, JointCollectionTpl>::SimulatorDerivativesTpl(const SimulatorX & simulator)
    : measure_timings(false)
    , contact_solver_derivatives(simulator.workspace.getConstraintProblemHandle())
#ifndef SIMPLEX_SKIP_COLLISION_DERIVATIVES_CONTRIBUTIONS
    , patch_derivative_requests(simulator.geom_model().collisionPairs.size())
    , patch_derivatives(simulator.geom_model().collisionPairs.size())
#endif
    , timer_(false)
    {
        allocate(simulator);
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    void SimulatorDerivativesTpl<S, O, JointCollectionTpl>::allocate(const SimulatorX & simulator)
    {
        const int nv = simulator.model().nv;
        const int max_nc = (int)simulator.workspace.constraint_problem().getMaxNumberOfContacts();
        dvnew_dq.resize(nv, nv);
        dvnew_dv.resize(nv, nv);
        dvnew_dtau.resize(nv, nv);
        danew_dq.resize(nv, nv);
        danew_dv.resize(nv, nv);
        danew_dtau.resize(nv, nv);
        dsigma1_dq_.resize(6, nv);
        dsigma1_dv_.resize(6, nv);
        dsigma2_dq_.resize(6, nv);
        dsigma2_dv_.resize(6, nv);
        dGlamg_dvnew_.resize(3, nv);
        MinvJT_.resize(nv, 3 * max_nc);
        dGlamg_dtheta_.resize(3 * max_nc, 3 * nv);

        dual_collision_correction_.resize(nv, nv);
        dJcTlam_dq_.resize(nv, nv);
        J1_.resize(6, nv);
        J2_.resize(6, nv);

#ifndef SIMPLEX_SKIP_COLLISION_DERIVATIVES_CONTRIBUTIONS
        // Primal/dual collision detection corrective terms
        primal_collision_correction_.resize(3 * max_nc, nv);

        Jc_.resize(6, nv);
        doMc_dq_.resize(6, nv);
        dJcv_dq_.resize(6, nv);
        dual_swap_doMc_dq_.resize(6, nv);

        const pinocchio::GeometryModel & gm = simulator.geom_model();
        patch_derivative_functors_.reserve(gm.collisionPairs.size());
        for (std::size_t cp_index = 0; cp_index < gm.collisionPairs.size(); ++cp_index)
        {
            const ::pinocchio::CollisionPair cp = gm.collisionPairs[cp_index];
            const ::pinocchio::GeometryObject & obj_1 = gm.geometryObjects[cp.first];
            const ::pinocchio::GeometryObject & obj_2 = gm.geometryObjects[cp.second];
            patch_derivative_functors_.emplace_back(obj_1, obj_2);
        }
#endif
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    template<typename ConfigVectorType, typename VelocityVectorType, typename TorqueVectorType>
    void SimulatorDerivativesTpl<S, O, JointCollectionTpl>::stepDerivatives(
        SimulatorX & simulator,
        const Eigen::MatrixBase<ConfigVectorType> & q,
        const Eigen::MatrixBase<VelocityVectorType> & v,
        const Eigen::MatrixBase<TorqueVectorType> & tau,
        Scalar dt)
    {
        SIMPLEX_TRACY_ZONE_SCOPED_N("SimulatorDerivatives::stepDerivatives");
        PINOCCHIO_UNUSED_VARIABLE(tau);

        if (measure_timings)
        {
            timer_.start();
        }

        PINOCCHIO_EIGEN_MALLOC_NOT_ALLOWED();

        danew_dq.setZero();
        danew_dv.setZero();
        danew_dtau.setZero();
        {
            SIMPLEX_TRACY_ZONE_SCOPED_N("SimulatorDerivatives::stepDerivatives - aba derivatives ");

            pinocchio::computeABADerivatives(
                simulator.model(), simulator.data(), q, v, simulator.state.tau_total, simulator.state.fext, ::pinocchio::make_ref(danew_dq),
                ::pinocchio::make_ref(danew_dv), ::pinocchio::make_ref(danew_dtau));
        }
        nc = simulator.workspace.constraint_problem().getNumberOfContacts();
        if (nc > 0)
        {
            {
                SIMPLEX_TRACY_ZONE_SCOPED_N("SimulatorDerivatives::stepDerivatives - FK derivatives");
                pinocchio::computeForwardKinematicsDerivatives(simulator.model(), simulator.data(), q, simulator.state.vnew);
            }

            /// Notation: vnew = simulator.step(q, v, tau).
            /// Usefull equations (we don't write fext for sake of simplicity):
            ///    vnew = vfree + dt * Minv * JT * lam
            ///         = v + dt * ABA(q, v, tau, fcontact) (fcontact = Xc * lam)
            ///         = FD(q, v, tau, lam(q, v, tau))
            /// where lam = NCP(q, v, tau).
            /// in what follows we denote anew = ABA(q, v, tau, fcontact) = (vnew - v) / dt
            ///
            /// So if theta = (q, v, tau):
            /// dvnew/dtheta = dFD/dtheta + dFD/dlam * dlam/dtheta.
            ///   --> dFD/dtheta are the derivatives of ABA.
            ///   --> dFD/dlam = Minv * JT
            ///   --> dlam/dtheta = solution to Ax=b, where A and b are obtained by derivating the
            ///   conditions of the NCP.
            ///   Notably, we have b = d(Gx + g)/dtheta with x = lam, considered as fixed.

            /// Computing d(G * x + g)/dtheta with x = lam (contact forces) fixed.
            /// Note: for the moment, theta = (q, v, tau).
            /// We denote sigma = G * lam + g. We have sigma = J * vnew.

#ifndef SIMPLEX_SKIP_COLLISION_DERIVATIVES_CONTRIBUTIONS
            // First we compute the collision detection correction terms.
            // Note: it's important that this function gets called **after** computeABADerivatives,
            // because we need ddq_dtau (Minv).
            computePrimalDualCollisionCorrection(simulator, simulator.state.vnew);
// #else
//             computeDualCorrection(simulator, simulator.state.vnew);
#endif

            for (ContactIndex i = 0; i < nc; ++i)
            {
                /// Computing dsigma/dq and dsigma/dv

                const ConstraintModel & cmodel = simulator.workspace.constraint_problem().constraint_models[i];
                // TODO(quentinll) for now we only support contact constraints
                if (const FrictionalPointConstraintModel * fpcmodel = boost::get<const FrictionalPointConstraintModel>(&cmodel))
                {

                    const JointIndex joint1_id = fpcmodel->joint1_id;
                    const SE3 & placement1 = fpcmodel->joint1_placement;
                    dsigma1_dq_.setZero();
                    dsigma1_dv_.setZero();
                    pinocchio::getFrameVelocityDerivatives(
                        simulator.model(), simulator.data(), joint1_id, placement1, pinocchio::LOCAL, dsigma1_dq_, dsigma1_dv_);

                    const JointIndex joint2_id = fpcmodel->joint2_id;
                    const SE3 & placement2 = fpcmodel->joint2_placement;
                    dsigma2_dq_.setZero();
                    dsigma2_dv_.setZero();
                    pinocchio::getFrameVelocityDerivatives(
                        simulator.model(), simulator.data(), joint2_id, placement2, pinocchio::LOCAL, dsigma2_dq_, dsigma2_dv_);
                }
                else
                {
                    // TODO: support other constraint types
                }

                dGlamg_dvnew_ = dsigma2_dv_.template topRows<3>() - dsigma1_dv_.template topRows<3>();

                /// dsigma/dtheta = dsigma/dvnew * dvnew/dtheta, with vnew = v + dt * ABA(q, v, tau, ftot).
                /// Note: ftot is considered fixed when constructing dGlamg_dtheta.
                /// dsigma/dtheta = dsigma/dq * dq/dtheta (dq/dtheta = Id if theta = q, 0 otherwise).
                auto dGlamg_dq = dGlamg_dtheta().template middleRows<3>(int(3 * i)).leftCols(simulator.model().nv);
                dGlamg_dq = dsigma2_dq_.template topRows<3>() - dsigma1_dq_.template topRows<3>();
                dGlamg_dq.noalias() += dt * dGlamg_dvnew_ * danew_dq;
#ifndef SIMPLEX_SKIP_COLLISION_DERIVATIVES_CONTRIBUTIONS
                dGlamg_dq.noalias() += primal_collision_correction().template middleRows<3>(int(3 * i));
                dGlamg_dq.noalias() += dt * dGlamg_dvnew_ * dual_collision_correction();
#endif

                auto dGlamg_dv = dGlamg_dtheta().template middleRows<3>(int(3 * i)).middleCols(simulator.model().nv, simulator.model().nv);
                dGlamg_dv.noalias() = dGlamg_dvnew_;
                dGlamg_dv.noalias() += dt * dGlamg_dvnew_ * danew_dv;

                auto dGlamg_dtau = dGlamg_dtheta().template middleRows<3>(int(3 * i)).rightCols(simulator.model().nv);
                dGlamg_dtau.noalias() = dt * dGlamg_dvnew_ * danew_dtau.transpose();

                /// We divide the right-hand side by dt to work in acceleration units
                /// so that we directly compute the Jacobian of the contact forces (and not the impulses).
                dGlamg_dtheta() /= dt;
            }
            // TODO(quentinll): add gradient terms from Baumgarte stabilization and other corrections.
#ifndef NDEBUG
            std::cout << simplex::logging::DEBUG << "SimulatorDerivatives::stepDerivatives: nc = " << nc << std::endl;
            std::cout << simplex::logging::DEBUG << "SimulatorDerivatives::stepDerivatives: dGlamg_dtheta: " << dGlamg_dtheta()
                      << std::endl;
            std::cout << simplex::logging::DEBUG
                      << "SimulatorDerivatives::stepDerivatives: dual_collision_correction: " << dual_collision_correction() << std::endl;
#endif // NDEBUG

            /// NCP derivatives
            contact_solver_derivatives.compute();
            contact_solver_derivatives.jvp(::pinocchio::make_const_ref(dGlamg_dtheta()));
            const auto dlam_dq = contact_solver_derivatives.dlam_dtheta().leftCols(simulator.model().nv);
            const auto dlam_dv = contact_solver_derivatives.dlam_dtheta().middleCols(simulator.model().nv, simulator.model().nv);
            const auto dlam_dtau = contact_solver_derivatives.dlam_dtheta().rightCols(simulator.model().nv);

            /// MinvJT * dNCP
            const ConstraintCholeskyDecomposition & chol = simulator.workspace.constraint_problem().constraint_cholesky_decomposition;
            chol.getJMinv(MinvJT().transpose());

            /// ABA correction using dNCP
            danew_dq.noalias() += MinvJT() * dlam_dq;

#ifndef NDEBUG
            std::cout << simplex::logging::DEBUG << "SimulatorDerivatives::stepDerivatives: danew_dq = " << danew_dq << std::endl;
#endif // NDEBUG

            // danew_dq.noalias() += dual_collision_correction();

#ifndef NDEBUG
            std::cout << simplex::logging::DEBUG << "SimulatorDerivatives::stepDerivatives: danew_dq_corr = " << danew_dq << std::endl;
#endif // NDEBUG

#ifndef SIMPLEX_SKIP_COLLISION_DERIVATIVES_CONTRIBUTIONS
            danew_dq.noalias() += dual_collision_correction();
#endif
            danew_dv.noalias() += MinvJT() * dlam_dv;
            danew_dtau.noalias() += MinvJT() * dlam_dtau;
        }

        dvnew_dq = dt * danew_dq;
        dvnew_dv = dt * danew_dv;
        dvnew_dv.diagonal().array() += Scalar(1);
        dvnew_dtau = dt * danew_dtau;

        PINOCCHIO_EIGEN_MALLOC_ALLOWED();

        if (measure_timings)
        {
            timer_.stop();
            timings_ = timer_.elapsed();
        }
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    template<typename VelocityVectorType>
    void SimulatorDerivativesTpl<S, O, JointCollectionTpl>::computeDualCorrection(
        const SimulatorX & simulator, const Eigen::MatrixBase<VelocityVectorType> & v)
    {
        PINOCCHIO_UNUSED_VARIABLE(v);

        const ConstraintsProblemDerivatives & constraint_problem = simulator.workspace.constraint_problem();

        // 初始化累加器，用于存储 d(J^T * lambda) / dq
        dJcTlam_dq_.setZero();

        // 遍历所有发生碰撞的对
        for (std::size_t i = 0; i < constraint_problem.pairs_in_collision.size(); ++i)
        {
            const std::size_t col_pair_id = constraint_problem.pairs_in_collision[i];
            const ContactMapper & contact_mapper = constraint_problem.contact_mappers[col_pair_id];

            // 遍历该碰撞对中的所有接触点
            for (std::size_t j = 0; j < contact_mapper.count; ++j)
            {
                const std::size_t idc = contact_mapper.begin + j;
                const ConstraintModel & cmodel = constraint_problem.constraint_models[idc];

                if (const FrictionalPointConstraintModel * fpcmodel = boost::get<const FrictionalPointConstraintModel>(&cmodel))
                {
                    const JointIndex joint1_id = fpcmodel->joint1_id;
                    const JointIndex joint2_id = fpcmodel->joint2_id;

                    // 获取 Pinocchio Data (去除 const)
                    Data & data = const_cast<Data &>(simulator.data());

                    // 1. 获取当前接触约束的力 (Impulse/Force)
                    // --------------------------------------------------------
                    // 注意：这里的力通常是在 "Contact Frame" (CENTERED/LOCAL_WORLD_ALIGNED) 下表示的
                    const Vector3s f_linear =
                        simulator.workspace.constraint_problem().frictional_point_constraints_forces().template segment<3>(int(3 * idc));
                    // 点接触只有线性力，没有力矩
                    const Force wrench_contact(f_linear, Vector3s::Zero());

                    // 2. 计算 Dual Correction: d(Jc^T * lambda) / dq
                    // --------------------------------------------------------
                    // 我们分别计算物体 1 和物体 2 的贡献。
                    // 公式原理：对于刚体上的固定点，d(J^T * F)/dq = J^T * (F_local x_dual) * J
                    // 其中 (F_local x_dual) 是力在局部系下的空间叉乘矩阵的对偶形式。

                    // --- Body 1 (Joint 1) ---
                    if (joint1_id > 0) // 如果不是宇宙/基座
                    {
                        // 计算 Jacobian (LOCAL 坐标系)
                        J1_.setZero();
                        ::pinocchio::getFrameJacobian(
                            simulator.model(), data, joint1_id, fpcmodel->joint1_placement, ::pinocchio::LOCAL, J1_);

                        // 将接触力变换到 Joint 1 的局部坐标系
                        // i1Mc 是从 Contact Frame 到 Joint 1 Frame 的变换 (即 fpcmodel->joint1_placement)
                        const SE3 & i1Mc = fpcmodel->joint1_placement;
                        const Force wrench1 = i1Mc.act(wrench_contact);

                        // 计算 wrench1 对应的对偶叉乘矩阵 (6x6)
                        // 注意符号：Jc = J2 - J1，所以 Body 1 受到的力是 -lambda
                        // 但是 d(-J1^T * lambda) = - (dJ1^T * lambda)
                        Matrix6s coswap1;
                        dualSmallAdSwap(wrench1, coswap1);

                        // 累加项: - J1^T * coswap1 * J1
                        dJcTlam_dq_.noalias() -= J1_.transpose() * coswap1 * J1_;
                    }

                    // --- Body 2 (Joint 2) ---
                    if (joint2_id > 0)
                    {
                        // 计算 Jacobian (LOCAL 坐标系)
                        J2_.setZero();
                        ::pinocchio::getFrameJacobian(
                            simulator.model(), data, joint2_id, fpcmodel->joint2_placement, ::pinocchio::LOCAL, J2_);

                        // 将接触力变换到 Joint 2 的局部坐标系
                        const SE3 & i2Mc = fpcmodel->joint2_placement;
                        const Force wrench2 = i2Mc.act(wrench_contact);

                        Matrix6s coswap2;
                        dualSmallAdSwap(wrench2, coswap2);

                        // 累加项: + J2^T * coswap2 * J2
                        dJcTlam_dq_.noalias() += J2_.transpose() * coswap2 * J2_;
                    }

                    // --------------------------------------------------------
                    // Primal Correction (d(Jc*v)/dq)
                    // --------------------------------------------------------
                    // 如果还需要计算 Primal 项 (用于 dGlamg_dq)，在刚体假设下，
                    // 这就是经典的 "Frame Acceleration Drift" (gamma = Jdot * v)。
                    // 之前的 primal_collision_correction_ 存储的就是这一项。

                    // 简化计算：直接利用 Pinocchio 的 getFrameAcceleration 计算 drift
                    // 注意：这需要 getFrameAcceleration 被调用过，或者手动计算 v_frame x v_joint
                    // 这里为了聚焦 Dual 项，暂时略过复杂的 Primal 简化实现，
                    // 但逻辑上它不再包含 doMc_dq 项。
                }
            }
        }

        // 3. 最终组装到加速度导数中
        // 对应公式 (26) 中的 M^{-1} * (dJ^T/dq * lambda)
        // danew_dtau 存储了 M^{-1} (ABA 关于 tau 的导数)
        dual_collision_correction().noalias() = danew_dtau * dJcTlam_dq_;
    }

#ifndef SIMPLEX_SKIP_COLLISION_DERIVATIVES_CONTRIBUTIONS
    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    template<typename VelocityVectorType>
    void SimulatorDerivativesTpl<S, O, JointCollectionTpl>::computePrimalDualCollisionCorrection(
        const SimulatorX & simulator, const Eigen::MatrixBase<VelocityVectorType> & v)
    {
        const ConstraintsProblemDerivatives & constraint_problem = simulator.workspace.constraint_problem();
        assert(constraint_problem.getNumberOfContacts() > 0 && "Should not call this function with no contact points.");

        // Since Jc^T * lam = sum_i Jc_i^T * lam_i, we accumulate the dual collision correction terms
        // for each contact point.
        dJcTlam_dq_.setZero();
        for (std::size_t i = 0; i < constraint_problem.pairs_in_collision.size(); ++i)
        {
            const std::size_t col_pair_id = constraint_problem.pairs_in_collision[i];

            const pinocchio::GeometryModel & gm = simulator.geom_model();
            const GeomIndex geom1_id = gm.collisionPairs[col_pair_id].first;
            const GeometryObject & geom1 = gm.geometryObjects[geom1_id];
            const GeomIndex geom2_id = gm.collisionPairs[col_pair_id].second;
            const GeometryObject & geom2 = gm.geometryObjects[geom2_id];

            // Compute collision detection derivatives.
            const CollisionResult & cres = simulator.geom_data().collisionResults[col_pair_id];
            PINOCCHIO_UNUSED_VARIABLE(cres);
            assert(cres.isCollision() && "There should be a collision between the shapes of this collision pair.");
            const ContactPatchResult & cpatch_res = simulator.geom_data().contactPatchResults[col_pair_id];
            const ContactPatch & patch = cpatch_res.getContactPatch(0);

            const ::coal::Transform3f oMg1(::pinocchio::toFclTransform3f(simulator.geom_data().oMg[geom1_id]));
            const ::coal::Transform3f oMg2(::pinocchio::toFclTransform3f(simulator.geom_data().oMg[geom2_id]));

            const ContactPatchDerivativeRequest & drequest = patch_derivative_requests[col_pair_id];
            ContactPatchDerivative & dpatch = patch_derivatives[col_pair_id];
            ComputeContactPatchDerivative & calc = patch_derivative_functors_[col_pair_id];
            dpatch.clear();
            calc(oMg1, oMg2, patch, drequest, dpatch);

            // Iterate through all the points of the contact patch
            const ContactMapper & contact_mapper = constraint_problem.contact_mappers[col_pair_id];
            for (std::size_t j = 0; j < contact_mapper.count; ++j)
            {
                const std::size_t idc = contact_mapper.begin + j;
                const ConstraintModel & cmodel = constraint_problem.constraint_models[idc];
                // TODO(quentinll) for now we only support contact constraints
                if (const FrictionalPointConstraintModel * fpcmodel = boost::get<const FrictionalPointConstraintModel>(&cmodel))
                {
                    const JointIndex joint1_id = fpcmodel->joint1_id;
                    const JointIndex joint2_id = fpcmodel->joint2_id;
                    const SE3 & i1Mc = fpcmodel->joint1_placement;
                    const SE3 & i2Mc = fpcmodel->joint2_placement;
                    // TODO: it's now i1Mc1 and i2Mc2
                    // Jacobian goes as X2 J2 - X1J1
                    const SE3 & oMc = constraint_problem.point_contact_constraint_placements[idc];
                    assert(geom1.parentJoint == fpcmodel->joint1_id && "Invalid constraint joint1_id");
                    assert(geom2.parentJoint == fpcmodel->joint2_id && "Invalid constraint joint2_id");

                    // ----------------------------------------------------------------
                    // First we compute doMc/dq
                    //
                    // Compute doMc/dn and doMc/dp
                    Matrix63s doMc_dn, doMc_dp;
                    PlacementFromNormalAndPosition::calcDiff(oMc, doMc_dn, doMc_dp);

                    // Get derivative of contact patch point to have dn/dM1, dn/dM2, dp/dM1 and dp/dM2
                    const auto & p = oMc.translation();
                    Matrix36s dp_dM1, dp_dM2;
                    dpatch.getDerivativeOfContactPatchPoint(patch, oMg1, oMg2, p, dp_dM1, dp_dM2);

                    // Get jacobian of geoms frames
                    Data & data = const_cast<Data &>(simulator.data());

                    // Note: it's mandatory to set it to zero before calling any `get<...>Jacobian`!
                    J1_.setZero();
                    ::pinocchio::getFrameJacobian(simulator.model(), data, joint1_id, geom1.placement, ::pinocchio::LOCAL, J1_);
                    J2_.setZero();
                    ::pinocchio::getFrameJacobian(simulator.model(), data, joint2_id, geom2.placement, ::pinocchio::LOCAL, J2_);

                    // Chaining the derivatives to get doMc/dq
                    // TODO(jcarpent) optmize
                    doMc_dq_.template topRows<3>() =
                        doMc_dp.template topRows<3>() * dp_dM1 * J1_ + doMc_dp.template topRows<3>() * dp_dM2 * J2_;

                    doMc_dq_.template bottomRows<3>() = doMc_dn.template bottomRows<3>() * dpatch.dnormal_dM1() * J1_
                                                        + doMc_dn.template bottomRows<3>() * dpatch.dnormal_dM2() * J2_;

                    // ----------------------------------------------------------------
                    // Compute Primal Corrective term
                    J1_.setZero();
                    ::pinocchio::getFrameJacobian(simulator.model(), data, joint1_id, i1Mc, ::pinocchio::LOCAL, J1_);
                    J2_.setZero();
                    ::pinocchio::getFrameJacobian(simulator.model(), data, joint2_id, i2Mc, ::pinocchio::LOCAL, J2_);

                    // Contact jacobian for the current contact point.
                    Jc_ = J2_ - J1_;

                    J1_.setZero();
                    ::pinocchio::getJointJacobian(simulator.model(), data, joint1_id, ::pinocchio::LOCAL, J1_);
                    J2_.setZero();
                    ::pinocchio::getJointJacobian(simulator.model(), data, joint2_id, ::pinocchio::LOCAL, J2_);

                    const Motion cvel(Jc_ * v);
                    const Motion c1vel(J1_ * v);
                    const Motion c2vel(J2_ * v);
                    // TODO(louis): malloc?
                    dJcv_dq_.noalias() = cvel.toActionMatrix() * doMc_dq_;
                    dJcv_dq_.noalias() += i1Mc.toActionMatrixInverse() * c1vel.toActionMatrix() * J1_;
                    dJcv_dq_.noalias() -= i2Mc.toActionMatrixInverse() * c2vel.toActionMatrix() * J2_;

                    // TODO(louis): for 3D contact points, exploit the sparsity by extracting only the linear
                    // part in the previous computations.
                    primal_collision_correction().template middleRows<3>(int(3 * idc)) = dJcv_dq_.template topRows<3>();

                    // ----------------------------------------------------------------
                    // Compute Dual Corrective term
                    const Force wrench(
                        simulator.workspace.constraint_problem().point_contact_constraint_forces().template segment<3>(int(3 * idc)), //
                        Vector3s::Zero());
                    Matrix6s coswap;
                    dualSmallAdSwap(wrench, coswap);
                    dual_swap_doMc_dq_.noalias() = coswap * doMc_dq_;

                    const Force wrench1(i1Mc.act(wrench));
                    Matrix6s coswap1;
                    dualSmallAdSwap(wrench1, coswap1);

                    const Force wrench2(i2Mc.act(wrench));
                    Matrix6s coswap2;
                    dualSmallAdSwap(wrench2, coswap2);

                    // TODO(louis): malloc?
                    dJcTlam_dq_.noalias() -= Jc_.transpose() * dual_swap_doMc_dq_;
                    dJcTlam_dq_.noalias() -= J1_.transpose() * coswap1 * J1_;
                    dJcTlam_dq_.noalias() += J2_.transpose() * coswap2 * J2_;
                }
                else
                {
                    // TODO: support other constraint types
                }
            }
        }
        // Once all the contact point dual contributions have been accumulated, we can construct the
        // dual correction term.
        dual_collision_correction().noalias() = danew_dtau * dJcTlam_dq_;
    }
#endif

} // namespace simplex

#endif // ifndef __simplex_core_simulator_derivatives_hxx__
