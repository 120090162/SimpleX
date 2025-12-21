//
// Copyright (c) 2024 INRIA
//

#ifndef __simple_core_constraints_problem_hxx__
#define __simple_core_constraints_problem_hxx__

#include "simple/core/constraints-problem.hpp"
#include "simple/tracy.hpp"
#include "simple/utils/visitors.hpp"

#include <pinocchio/algorithm/contact-cholesky.hpp>
#include <pinocchio/algorithm/contact-info.hpp>
#include <pinocchio/algorithm/contact-jacobian.hpp>
#include <pinocchio/algorithm/contact-solver-utils.hpp>
#include <pinocchio/alloca.hpp>

#include <hpp/fcl/collision_data.h>
#include <boost/variant.hpp>

namespace simple
{

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  ConstraintsProblemTpl<S, O, JointCollectionTpl>::ConstraintsProblemTpl(
    const ModelHandle & model_handle,
    const DataHandle & data_handle,
    const GeometryModelHandle & geom_model_handle,
    const GeometryDataHandle & geom_data_handle,
    const BilateralPointConstraintModelVector & bilateral_point_constraint_models,
    const WeldConstraintModelVector & weld_constraint_models)
  : m_model(model_handle)
  , m_data(data_handle)
  , m_geom_model(geom_model_handle)
  , m_geom_data(geom_data_handle)
  , m_current_constraints_pb_id(0)
  , m_max_num_contact_per_collision_pair(4)
  , m_num_contacts(0)
  , m_previous_num_contacts(0)
  , m_previous_joint_limits_constraint_forces_size(0)
  {
    for (std::size_t i = 0; i < bilateral_point_constraint_models.size(); ++i)
    {
      this->bilateral_point_constraint_models.emplace_back(bilateral_point_constraint_models[i]);
      this->bilateral_point_constraint_datas.emplace_back(bilateral_point_constraint_models[i].createData());
    }
    for (std::size_t i = 0; i < weld_constraint_models.size(); ++i)
    {
      this->weld_constraint_models.emplace_back(weld_constraint_models[i]);
      this->weld_constraint_datas.emplace_back(weld_constraint_models[i].createData());
    }
    this->allocate();
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  ConstraintsProblemTpl<S, O, JointCollectionTpl>::ConstraintsProblemTpl(
    const ModelHandle & model_handle,
    const DataHandle & data_handle,
    const GeometryModelHandle & geom_model_handle,
    const GeometryDataHandle & geom_data_handle)
  : ConstraintsProblemTpl(
      model_handle,                          //
      data_handle,                           //
      geom_model_handle,                     //
      geom_data_handle,                      //
      BilateralPointConstraintModelVector(), //
      WeldConstraintModelVector())
  {
  }

  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  ConstraintsProblemTpl<S, O, JointCollectionTpl>::ConstraintsProblemTpl(
    const ModelHandle & model_handle,
    const DataHandle & data_handle,
    const GeometryModelHandle & geom_model_handle,
    const GeometryDataHandle & geom_data_handle,
    const BilateralPointConstraintModelVector & bilateral_point_constraint_models)
  : ConstraintsProblemTpl(
      model_handle,                      //
      data_handle,                       //
      geom_model_handle,                 //
      geom_data_handle,                  //
      bilateral_point_constraint_models, //
      WeldConstraintModelVector())
  {
  }

  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  ConstraintsProblemTpl<S, O, JointCollectionTpl>::ConstraintsProblemTpl(
    const ModelHandle & model_handle,
    const DataHandle & data_handle,
    const GeometryModelHandle & geom_model_handle,
    const GeometryDataHandle & geom_data_handle,
    const WeldConstraintModelVector & weld_constraint_models)
  : ConstraintsProblemTpl(
      model_handle,                          //
      data_handle,                           //
      geom_model_handle,                     //
      geom_data_handle,                      //
      BilateralPointConstraintModelVector(), //
      weld_constraint_models)
  {
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  void ConstraintsProblemTpl<S, O, JointCollectionTpl>::clear()
  {
    // Clearing joint limits
    this->m_previous_joint_limits_constraint_forces_size = this->joint_limit_constraint_size();
    JointLimitConstraintModel * jlcm = boost::get<JointLimitConstraintModel>(&this->joint_limit_constraint_model);
    assert(jlcm != nullptr);
    if (jlcm != nullptr)
    {
      this->m_previous_joint_limit_active_set = jlcm->getActiveSetIndexes();
    }

    // Clearing contacts
    this->pairs_in_collision.clear();
    this->contact_id_to_collision_pair.clear();
    this->frictional_point_constraints_placements().clear();
    this->m_previous_num_contacts = this->m_num_contacts;
    this->m_num_contacts = 0;

    // Clearing constraint models corresponding to joint limits and contacts
    std::size_t begin_idx_to_erase = this->getNumberOfJointFrictionConstraints();
    begin_idx_to_erase += this->getNumberOfBilateralConstraints();
    begin_idx_to_erase += this->getNumberOfWeldConstraints();
    this->constraint_models.erase(this->constraint_models.begin() + (int)begin_idx_to_erase, this->constraint_models.end());
    this->constraint_datas.erase(this->constraint_datas.begin() + (int)begin_idx_to_erase, this->constraint_datas.end());
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  void ConstraintsProblemTpl<S, O, JointCollectionTpl>::allocate()
  {
    SIMPLE_TRACY_ZONE_SCOPED_N("ConstraintsProblem::allocate");

    //
    // We first allocate memory for the constraints by setting their holders.
    //

    // -- Joint friction
    // first we check which joint actually has friction:
    int size_dry_friction = 0;
    std::vector<JointIndex> active_friction_joints_ids;
    std::vector<int> dry_friction_joint_idx_v;
    for (std::size_t i = 1; i < (std::size_t)this->model().njoints; ++i)
    {
      const pinocchio::JointModel & joint = this->model().joints[i];
      if (
        !(this->model().upperDryFrictionLimit.segment(joint.idx_v(), joint.nv()).isZero())
        || !(this->model().lowerDryFrictionLimit.segment(joint.idx_v(), joint.nv()).isZero()))
      {
        dry_friction_joint_idx_v.push_back(size_dry_friction);
        active_friction_joints_ids.push_back(joint.id());
        size_dry_friction += joint.nv();
      }
    }
    // now we can build the holder for the frictional joint constraint
    VectorXs lowerDryFrictionLimit = VectorXs::Zero(size_dry_friction);
    VectorXs upperDryFrictionLimit = VectorXs::Zero(size_dry_friction);
    for (std::size_t i = 0; i < active_friction_joints_ids.size(); ++i)
    {
      const JointIndex joint_id = active_friction_joints_ids[i];
      const pinocchio::JointModel & joint = this->model().joints[joint_id];
      lowerDryFrictionLimit.segment(dry_friction_joint_idx_v[i], joint.nv()) =
        this->model().lowerDryFrictionLimit.segment(joint.idx_v(), joint.nv());
      upperDryFrictionLimit.segment(dry_friction_joint_idx_v[i], joint.nv()) =
        this->model().upperDryFrictionLimit.segment(joint.idx_v(), joint.nv());
    }

    FrictionalJointConstraintModel fjcm(this->model(), active_friction_joints_ids);
    fjcm.set() = BoxSet(lowerDryFrictionLimit, upperDryFrictionLimit);
    this->joint_friction_constraint_model = fjcm;
    this->joint_friction_constraint_data = fjcm.createData();

    // -- Bilateral
    // they have been filled in the constructor

    // -- Weld
    // they have been filled in the constructor

    // -- Joint limits
    // first we check which joint actually has limits:
    std::vector<JointIndex> active_limit_joints_ids;
    for (std::size_t i = 1; i < (std::size_t)this->model().njoints; ++i)
    {
      const pinocchio::JointModel & joint = this->model().joints[i];
      if (joint.nq() == 1 && joint.hasConfigurationLimit()[0])
      {
        active_limit_joints_ids.push_back(joint.id());
      }
      // TODO: understand why we need to force nq==1
      // for (std::size_t j = 0; j < (std::size_t)joint.nq(); ++j)
      // {
      //   bool has_limit = joint.hasConfigurationLimit()[j];
      //   if (has_limit)
      //   {
      //     active_limit_joints_ids.push_back(joint.id());
      //     break;
      //   }
      // }
    }
    // now we can build the holder for the joint limits constraints
    // TODO check that we use the margins contained in the model
    JointLimitConstraintModel jlcm(this->model(), active_limit_joints_ids);
    this->joint_limit_constraint_model = jlcm;
    this->joint_limit_constraint_data = jlcm.createData();
    this->m_previous_joint_limit_active_set.reserve(std::size_t(this->joint_limit_constraint_max_size()));

    // -- Frictional contacts
    // For now, it's assumed that there is only one `hpp::fcl::Contact` per collision pair, meaning
    // that although there can be multiple contact points between two shapes, there can only be one
    // normal (and thus one contact patch for each collision pair). Note: each `Contact` has its own
    // `ContactPatch`. In the future, if we support multiple contact normals for a single collision
    // pair (i.e. for two non-convex objects), we will have to recompute `m_max_num_contacts`, by
    // factoring the number of `Contact` per collision pair.
    assert(this->geom_data().collisionRequests.size() == this->geom_model().collisionPairs.size());
    assert(this->geom_data().collisionRequests.size() == this->geom_data().collisionResults.size());
    std::size_t max_num_contacts = this->geom_model().collisionPairs.size() * this->m_max_num_contact_per_collision_pair;
    const int frictional_contact_max_size = (int)(3 * max_num_contacts);

    this->pairs_in_collision.reserve(this->geom_model().collisionPairs.size());
    this->contact_id_to_collision_pair.reserve(max_num_contacts);
    this->frictional_point_constraints_placements().reserve(max_num_contacts);
    this->previous_frictional_point_constraints_placements().reserve(max_num_contacts);
    this->previous_colliding_collision_pairs.assign(this->geom_model().collisionPairs.size(), false);
    this->contact_mappers.assign(this->geom_model().collisionPairs.size(), ContactMapper(0, 0));
    this->breaking_contacts.reserve(max_num_contacts);
    this->sticking_contacts.reserve(max_num_contacts);
    this->sliding_contacts.reserve(max_num_contacts);
    this->contact_modes.reserve(max_num_contacts);

    this->frictional_point_constraint_models.reserve(max_num_contacts);
    this->frictional_point_constraint_datas.reserve(max_num_contacts);
    this->frictional_point_constraint_models.clear();
    this->frictional_point_constraint_datas.clear();
    const ::pinocchio::FrictionCoefficientMatrix & friction_coefficients = ::pinocchio::getFrictionCoefficientMatrix();
    for (::pinocchio::PairIndex col_pair_id = 0; col_pair_id < this->geom_model().collisionPairs.size(); ++col_pair_id)
    {
      // add contact models/datas
      const GeomIndex geom_id1 = this->geom_model().collisionPairs[col_pair_id].first;
      const GeomIndex geom_id2 = this->geom_model().collisionPairs[col_pair_id].second;

      const ::pinocchio::GeometryObject & geom1 = this->geom_model().geometryObjects[geom_id1];
      const ::pinocchio::GeometryObject & geom2 = this->geom_model().geometryObjects[geom_id2];

      const ::pinocchio::JointIndex joint_id1 = geom1.parentJoint;
      const ::pinocchio::JointIndex joint_id2 = geom2.parentJoint;

      // fill collision pairs frictions and elasticities
      const ::pinocchio::PhysicsMaterial & material1 = geom1.physicsMaterial;
      const ::pinocchio::PhysicsMaterial & material2 = geom2.physicsMaterial;

      const Scalar friction =
        static_cast<Scalar>(friction_coefficients.getFrictionFromMaterialPair(material1.materialType, material2.materialType));
      const Scalar compliance = static_cast<Scalar>(Scalar(0.5) * (material1.compliance + material2.compliance));
      // const Scalar elasticity = static_cast<Scalar>(0.5 * (material1.elasticity + material2.elasticity));

      for (ContactIndex contact_id = 0; contact_id < this->m_max_num_contact_per_collision_pair; ++contact_id)
      {
        FrictionalPointConstraintModel fpcm(this->model(), joint_id1, SE3::Identity(), joint_id2, SE3::Identity());
        const std::string name = "FrictionalPointConstraintModel_" + std::to_string(geom_id1) + "_" + std::to_string(geom_id2);
        fpcm.name = name;
        fpcm.compliance().setConstant(compliance);
        fpcm.desired_constraint_offset = Vector3s(0, 0, this->geom_data().collisionRequests[col_pair_id].security_margin);
        fpcm.set() = CoulombFrictionCone(friction);
        this->frictional_point_constraint_models.emplace_back(fpcm);
        this->frictional_point_constraint_models.back().name = name;
        this->frictional_point_constraint_datas.emplace_back(fpcm.createData());
      }
    }
    assert(this->frictional_point_constraint_models.size() == max_num_contacts);
    assert(this->frictional_point_constraint_datas.size() == max_num_contacts);

    //
    // Now that all constraints have been listed, we can allocate the maximum memory for the constraint models and datas references.
    // These references will point to the data stored in the constraints models/data that are stored in the holders.
    // Certain constraints like frictional point constraints are dynamic, meaning that at a certain
    // time step they can be active or inactive.
    // Hence, after these constraints have been added to constraint_models/datas and that memory
    // has been allocated, they are removed from constraint_models/data.
    // It's then the job of the `update` method to deal with dynamic constraints.
    //
    std::size_t max_num_constraints = max_num_contacts + this->bilateral_point_constraint_models.size();
    max_num_constraints += this->weld_constraint_models.size();
    // TODO this is wrong, we can have more than one joint friction constraint or one joint limit constraint
    max_num_constraints += this->joint_friction_constraint_size() > 0 ? 1 : 0;
    max_num_constraints += this->joint_limit_constraint_size() > 0 ? 1 : 0;
    this->constraint_models.reserve(max_num_constraints);
    this->constraint_datas.reserve(max_num_constraints);
    this->constraint_models.clear();
    this->constraint_datas.clear();

    // variable used to make sure we allocate enough data to hold any constraint jacobian
    int max_constraint_size = this->geom_model().collisionPairs.size() > 0 ? 3 : 0;
    if (this->joint_friction_constraint_model.size() > 0)
    {
      this->constraint_models.emplace_back(this->joint_friction_constraint_model);
      this->constraint_datas.emplace_back(this->joint_friction_constraint_data);
      max_constraint_size = std::max(this->joint_friction_constraint_size(), max_constraint_size);
    }
    if (this->bilateral_point_constraint_models.size() > 0)
    {
      for (std::size_t i = 0; i < this->bilateral_point_constraint_models.size(); ++i)
      {
        this->constraint_models.emplace_back(this->bilateral_point_constraint_models[i]);
        this->constraint_datas.emplace_back(this->bilateral_point_constraint_datas[i]);
      }
      max_constraint_size = std::max(3, max_constraint_size);
    }
    if (this->weld_constraint_models.size() > 0)
    {
      for (std::size_t i = 0; i < this->weld_constraint_models.size(); ++i)
      {
        this->constraint_models.emplace_back(this->weld_constraint_models[i]);
        this->constraint_datas.emplace_back(this->weld_constraint_datas[i]);
      }
      max_constraint_size = std::max(6, max_constraint_size);
    }
    if (this->joint_limit_constraint_model.size() > 0)
    {
      this->constraint_models.emplace_back(this->joint_limit_constraint_model);
      this->constraint_datas.emplace_back(this->joint_limit_constraint_data);
      max_constraint_size = std::max(this->joint_limit_constraint_max_size(), max_constraint_size);
    }
    if (this->frictional_point_constraint_models.size() > 0)
    {
      for (std::size_t i = 0; i < this->frictional_point_constraint_models.size(); ++i)
      {
        this->constraint_models.emplace_back(this->frictional_point_constraint_models[i]);
        this->constraint_datas.emplace_back(this->frictional_point_constraint_datas[i]);
      }
      max_constraint_size = std::max(3, max_constraint_size);
    }

    const int constraints_problem_max_size = this->joint_friction_constraint_size()    //
                                             + this->bilateral_constraints_size()      //
                                             + this->weld_constraints_size()           //
                                             + this->joint_limit_constraint_max_size() //
                                             + frictional_contact_max_size;            //

    // allocate the rest of the ConstraintsProblem
    this->constraint_cholesky_decomposition.resize(this->model(), this->constraint_models);
    this->m_g.resize(constraints_problem_max_size);
    this->m_preconditioner.resize(constraints_problem_max_size);
    this->m_time_scaling_acc_to_constraints.resize(constraints_problem_max_size);
    this->m_constraints_forces_holder[0].resize(constraints_problem_max_size);
    this->m_constraints_forces_holder[1].resize(constraints_problem_max_size);
    this->m_constraints_velocities_warmstarts.resize(constraints_problem_max_size);
    this->m_constraints_velocities.resize(constraints_problem_max_size);

    // initialize constraints forces
    this->constraints_forces().setZero();
    this->previous_constraints_forces().setZero();

    this->clear(); // remove frictional contact models from constraints vector
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  void ConstraintsProblemTpl<S, O, JointCollectionTpl>::update(const bool compute_warm_start)
  {
    SIMPLE_TRACY_ZONE_SCOPED_N("ConstraintsProblem::update");

    this->updateConstraintsHolders();
    this->clear();
    assert(this->m_num_contacts == 0 && "The number of contacts should be 0.");
    assert(this->geom_data().collisionResults.size() == geom_model().collisionPairs.size());
    assert(this->checkConstraintsConsistencyWithGeometryModel());

    // Process joint limit constraints to check which ones are active
    ConstraintModel & constraint_model = this->joint_limit_constraint_model;
    ConstraintData & constraint_data = this->joint_limit_constraint_data;
    JointLimitConstraintModel * jlcm = boost::get<JointLimitConstraintModel>(&constraint_model);
    assert(jlcm != nullptr);
    if (jlcm != nullptr)
    {
      JointLimitConstraintData * jlcd = boost::get<JointLimitConstraintData>(&constraint_data);
      assert(jlcd != nullptr);
      jlcm->resize(this->model(), this->data(), *jlcd);
    }
    constraint_model.calc(this->model(), this->data(), constraint_data);
    int active_size = constraint_model.activeSize();
    if (active_size > 0)
    {
      this->joint_limit_constraint_forces().setZero();
      if (compute_warm_start && this->previous_joint_limit_constraint_size() > 0)
      {
        // Warm-start the joint limits constraint forces.
        // We assume that the active set indexes are sorted.
        // We search in the current active set which index belongs to the previous active set.
        const std::vector<std::size_t> & prev_active_set = this->m_previous_joint_limit_active_set;
        assert(prev_active_set.empty() == false);
        const std::vector<std::size_t> & active_set = jlcm->getActiveSetIndexes();

        std::size_t j = 0;
        for (std::size_t i = 0; i < active_set.size(); ++i)
        {
          const std::size_t idx_active = active_set[i];
          for (; j < prev_active_set.size(); ++j)
          {
            const std::size_t idx_prev_active = prev_active_set[j];
            if (idx_prev_active > idx_active)
            {
              break;
            }
            if (idx_prev_active == idx_active)
            {
              this->joint_limit_constraint_forces()[int(i)] = this->previous_joint_limit_constraint_forces()[int(j)];
              break;
            }
          }
        }
      }
      this->constraint_models.emplace_back(constraint_model);
      this->constraint_datas.emplace_back(constraint_data);
    }

    ContactIndex col_pair_contact_model_begin_id = 0; // Tracks the id of the first constraint model of the current collision pair
    for (GeomIndex col_pair_id = 0; col_pair_id < this->geom_model().collisionPairs.size(); col_pair_id++)
    {
      // TODO(louis): check if pair is active
      const hpp::fcl::CollisionResult & col_res = this->geom_data().collisionResults[col_pair_id];
      const hpp::fcl::ContactPatchResult & patch_res = this->geom_data().contactPatchResults[col_pair_id];
      // TODO(louis): extend to deal with multiple contact patches per collision pair.
      if (col_res.isCollision() && patch_res.numContactPatches() > 0)
      {
        this->pairs_in_collision.emplace_back(col_pair_id);

        const GeomIndex geom_id1 = this->geom_model().collisionPairs[col_pair_id].first;
        const GeomIndex geom_id2 = this->geom_model().collisionPairs[col_pair_id].second;

        const ::pinocchio::GeometryObject & geom1 = this->geom_model().geometryObjects[geom_id1];
        const ::pinocchio::GeometryObject & geom2 = this->geom_model().geometryObjects[geom_id2];

        const JointIndex joint_id1 = geom1.parentJoint;
        const JointIndex joint_id2 = geom2.parentJoint;

        const SE3 & oMi1 = this->data().oMi[joint_id1];
        const SE3 & oMi2 = this->data().oMi[joint_id2];

        // For the moment, we consider only one contact patch per collision pair.
        // All the contact points of the patch share the same normal.
        const hpp::fcl::ContactPatch & patch = patch_res.getContactPatch(0);
        const Vector3s & contact_normal = patch.getNormal();
        const std::size_t num_contacts_colpair = std::min(patch.size(), this->m_max_num_contact_per_collision_pair);
        // All the points of the patch share the same normal; the rotation part of the contact frame
        // does not need to be recomputed for every point of the patch.
        SE3 oMc;
        PlacementFromNormalAndPosition::calc(contact_normal, Vector3s::Zero(), oMc);
        for (ContactIndex contact_id = 0; contact_id < num_contacts_colpair; ++contact_id)
        {
          this->contact_id_to_collision_pair.push_back(col_pair_id);

          //
          oMc.translation() = patch.getPoint(contact_id);
          this->frictional_point_constraints_placements().emplace_back(oMc);
          const SE3 i1Mc = oMi1.actInv(oMc);
          //
          SE3 & oMc1 = oMc;
          oMc1.translation() = patch.getPointShape1(contact_id);
          const SE3 i1Mc1 = oMi1.actInv(oMc1);
          //
          SE3 & oMc2 = oMc;
          oMc2.translation() = patch.getPointShape2(contact_id);
          const SE3 i2Mc2 = oMi2.actInv(oMc2);

          // Update constraint models and datas
          const std::size_t constraint_model_id = col_pair_contact_model_begin_id + contact_id;
          FrictionalPointConstraintModel * fpcmodel =
            boost::get<FrictionalPointConstraintModel>(&(this->frictional_point_constraint_models[constraint_model_id]));
          if (fpcmodel == nullptr)
          {
            PINOCCHIO_THROW_PRETTY(std::runtime_error, "Invalid constraint model type.");
          }
          else
          {
            fpcmodel->joint1_placement = i1Mc1;
            fpcmodel->joint2_placement = i2Mc2;
          }
          // Note: friction is set in `allocate` when constructing the frictional point constraints.
          // The friction is retrieved from the materials of the geometries.
          // The same can be done for elasticity.
          const ConstraintModel & constraint_model = this->frictional_point_constraint_models[constraint_model_id];
          ConstraintData & constraint_data = this->frictional_point_constraint_datas[constraint_model_id];
          this->constraint_models.emplace_back(constraint_model);
          this->constraint_datas.emplace_back(constraint_data);

          constraint_model.calc(this->model(), this->data(), constraint_data);
#ifndef NDEBUG
          // sanity check
          const ::coal::CollisionRequest & col_req = this->geom_data().collisionRequests[col_pair_id];
          const FrictionalPointConstraintData * fpcdata = boost::get<const FrictionalPointConstraintData>(&constraint_data);
          assert(fpcdata != nullptr);
          assert(
            std::abs((fpcdata->constraint_position_error - Vector3s(0, 0, patch.penetration_depth - col_req.security_margin)).norm())
            < std::sqrt(Eigen::NumTraits<Scalar>::dummy_precision()));
#endif // NDEBUG

          // Warm-start the contact forces
          const auto cindex = static_cast<Eigen::Index>(3 * this->m_num_contacts);
          Vector3s warm_start_force = Vector3s::Zero();
          if (compute_warm_start && this->previous_colliding_collision_pairs[col_pair_id])
          {
            Scalar closest_previous_contact_distance = std::numeric_limits<Scalar>::infinity();
            ContactIndex closest_previous_contact_id = this->contact_mappers[col_pair_id].begin;
            SE3 cprevMc;
            for (ContactIndex previous_contact_id = this->contact_mappers[col_pair_id].begin;
                 previous_contact_id < this->contact_mappers[col_pair_id].begin + this->contact_mappers[col_pair_id].count;
                 ++previous_contact_id)
            {
              // TODO(quentin) search which previous contact corresponds to the current one in a more clever way
              const SE3 & i1Mc_previous = this->previous_frictional_point_constraints_placements()[previous_contact_id];
              const SE3 cprevMc_tmp = i1Mc_previous.actInv(i1Mc);
              Scalar previous_contact_distance = (i1Mc.translation() - i1Mc_previous.translation()).norm();
              if (previous_contact_distance < closest_previous_contact_distance)
              {
                closest_previous_contact_distance = previous_contact_distance;
                closest_previous_contact_id = previous_contact_id;
                cprevMc = cprevMc_tmp;
              }
            }
            const auto cprev_index = static_cast<Eigen::Index>(3 * closest_previous_contact_id);
            warm_start_force.noalias() = cprevMc.rotation()
                                         * this->previous_frictional_point_constraints_forces().template segment<3>(
                                           cprev_index); // TODO(quentin) check this is properly done
          }
          // Important: first update the number of contact, then set the eigen vectors like
          // frictional_point_constraints_forces, in this order.
          ++(this->m_num_contacts);
          this->frictional_point_constraints_forces().template segment<3>(cindex) = warm_start_force;
        }
        // update previous_colliding_collision_pairs and m_contact_mappers with newly created contacts
        this->previous_colliding_collision_pairs[col_pair_id] = true;

        // The role of the contact mapper is to map a collision pair to its number of contact points.
        this->contact_mappers[col_pair_id].begin = this->m_num_contacts - num_contacts_colpair;
        this->contact_mappers[col_pair_id].count = num_contacts_colpair;
      }
      else
      {
        this->contact_mappers[col_pair_id].begin = 0;
        this->contact_mappers[col_pair_id].count = 0; // nothing to map.
        this->previous_colliding_collision_pairs[col_pair_id] = false;
      }
      col_pair_contact_model_begin_id += this->m_max_num_contact_per_collision_pair;
    }

#ifndef NDEBUG
    const std::size_t num_constraints = this->getNumberOfJointFrictionConstraints() //
                                        + this->getNumberOfBilateralConstraints()   //
                                        + this->getNumberOfWeldConstraints()        //
                                        + this->getNumberOfJointLimitConstraints()  //
                                        + this->getNumberOfContacts();
    assert(this->constraint_models.size() == num_constraints);
    assert(this->constraint_datas.size() == num_constraints);
#endif

    // Call calc on the remaining constraint models (it has already been called for joint limits  and frictional contacts)
    for (std::size_t i = 0;
         i < this->getNumberOfJointFrictionConstraints() + this->getNumberOfBilateralConstraints() + this->getNumberOfWeldConstraints();
         ++i)
    {
      const ConstraintModel & constraint_model = this->constraint_models[i];
      ConstraintData & constraint_data = this->constraint_datas[i];
      constraint_model.calc(this->model(), this->data(), constraint_data);
    }

    // Update all constraints but frictional points constraints warmstarts
    if (this->joint_friction_constraint_size() > 0)
    {
      this->joint_friction_constraint_forces() = this->previous_joint_friction_constraint_forces();
    }

    if (this->bilateral_constraints_size() > 0)
    {
      this->bilateral_constraints_forces() = this->previous_bilateral_constraints_forces();
    }

    if (this->weld_constraints_size() > 0)
    {
      this->weld_constraints_forces() = this->previous_weld_constraints_forces();
    }
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  template<typename FreeVelocityVectorType, typename VelocityVectorType, typename VelocityWarmStartVectorType>
  void ConstraintsProblemTpl<S, O, JointCollectionTpl>::build(
    const Eigen::MatrixBase<FreeVelocityVectorType> & vfree,
    const Eigen::MatrixBase<VelocityVectorType> & v,
    const Eigen::MatrixBase<VelocityWarmStartVectorType> & v_warmstart,
    S dt)
  {
    SIMPLE_TRACY_ZONE_SCOPED_N("ConstraintsProblem::build");
    PINOCCHIO_UNUSED_VARIABLE(v_warmstart);

    PINOCCHIO_CHECK_ARGUMENT_SIZE(vfree.size(), v.size(), "The free velocity and the velocity should have the same size.");
    PINOCCHIO_CHECK_INPUT_ARGUMENT(dt >= 0, "The time step should be positive or null.");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(
      this->constraint_models.size(), this->constraint_datas.size(), "The number of constraint models and datas should be the same.");
    assert(this->check());

    // Compute `G`, the Delassus operator
    {
      SIMPLE_TRACY_ZONE_SCOPED_N("ConstraintsProblem::build - resize Delassus cholesky");
      this->constraint_cholesky_decomposition.resize(this->model(), this->constraint_models);
    }

    {
      SIMPLE_TRACY_ZONE_SCOPED_N("ConstraintsProblem::build - compute Delassus cholesky");
      // TODO(quentin): replace rho
      const Scalar rho = 1e-6;
      this->constraint_cholesky_decomposition.compute(this->model(), this->data(), this->constraint_models, this->constraint_datas, rho);
    }

    {
      SIMPLE_TRACY_ZONE_SCOPED_N("ConstraintsProblem::build - compute g");
      this->computeConstraintsDrift(vfree, v, dt);
    }

    this->preconditioner().setOnes();
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  template<typename FreeVelocityVectorType, typename VelocityVectorType>
  void ConstraintsProblemTpl<S, O, JointCollectionTpl>::computeConstraintsDrift(
    const Eigen::MatrixBase<FreeVelocityVectorType> & vfree, const Eigen::MatrixBase<VelocityVectorType> & v, const Scalar dt)
  {
    // TODO: express everything in velocity and use pinocchio helper to go to formulation level

    int cindex = 0;
    for (std::size_t i = 0; i < this->constraint_models.size(); ++i)
    {
      const ConstraintModel & cmodel = this->constraint_models[i];
      const ConstraintData & cdata = this->constraint_datas[i];
      const int cdim = cmodel.activeSize();
      auto gc = this->g().segment(cindex, cdim);

      using MapVectorXs = Eigen::Map<VectorXs>;
      auto drift_visitor = ::simple::visitors::make_lambda_visitor(
        //
        // Visitor for FrictionalJointConstraintModel - velocity formulation
        [&](const FrictionalJointConstraintModel & cmodel) {
          const FrictionalJointConstraintData & cdata_ = boost::get<const FrictionalJointConstraintData>(cdata);
          cmodel.jacobianMatrixProduct(this->model(), this->data(), cdata_, vfree, gc);
        },
        //
        // Visitor for BilateralConstraintModel - velocity formulation
        [&](const BilateralPointConstraintModel & cmodel) {
          const BilateralPointConstraintData & cdata_ = boost::get<const BilateralPointConstraintData>(cdata);
          const Scalar kp = cmodel.baumgarte_corrector_parameters().Kp;
          const Scalar kd = cmodel.baumgarte_corrector_parameters().Kd;

          // -- Jc * (vfree - kd * v)
          MapVectorXs vtmp(PINOCCHIO_EIGEN_MAP_ALLOCA(Scalar, this->model().nv, 1));
          vtmp = vfree - kd * v;
          cmodel.jacobianMatrixProduct(this->model(), this->data(), cdata_, vtmp, gc);

          // -- baumgarte kp correction
          const Vector3s correction = kp * cdata_.constraint_position_error;
          gc += correction / dt;
        },
        //
        // Visitor for WeldConstraintModel - velocity formulation
        [&](const WeldConstraintModel & cmodel) {
          const WeldConstraintData & cdata_ = boost::get<const WeldConstraintData>(cdata);
          const Scalar kp = cmodel.baumgarte_corrector_parameters().Kp;
          const Scalar kd = cmodel.baumgarte_corrector_parameters().Kd;

          // -- Jc * (vfree - kd * v)
          MapVectorXs vtmp(PINOCCHIO_EIGEN_MAP_ALLOCA(Scalar, this->model().nv, 1));
          vtmp = vfree - kd * v;
          cmodel.jacobianMatrixProduct(this->model(), this->data(), cdata_, vtmp, gc);

          // -- baumgaarte kp correction
          const Vector6s correction = kp * cdata_.constraint_position_error;
          gc += correction / dt;
        },
        //
        // Visitor for JointLimitConstraintModel - position formulation
        [&](const JointLimitConstraintModel & cmodel) {
          const JointLimitConstraintData & cdata_ = boost::get<const JointLimitConstraintData>(cdata);
          const Scalar kp = cmodel.baumgarte_corrector_parameters().Kp;
          const Scalar kd = cmodel.baumgarte_corrector_parameters().Kd;

          // -- Jc * (vfree - kd * v) * dt
          MapVectorXs vtmpdt(PINOCCHIO_EIGEN_MAP_ALLOCA(Scalar, this->model().nv, 1));
          vtmpdt = (vfree - kd * v) * dt;
          cmodel.jacobianMatrixProduct(this->model(), this->data(), cdata_, vtmpdt, gc);

          // -- baumgarte kp correction
          MapVectorXs joint_limit_correction(PINOCCHIO_EIGEN_MAP_ALLOCA(Scalar, cmodel.activeSize(), 1));
          joint_limit_correction = cdata_.constraint_residual; // copy

          const auto lower_bound_size = cmodel.set().getNegativeOrthant().size();
          auto joint_limit_correction_lower = joint_limit_correction.head(lower_bound_size);
          joint_limit_correction_lower = (joint_limit_correction_lower.array() > 0)     // if
                                           .select(                                     //
                                             joint_limit_correction_lower.array() * kp, // then
                                             joint_limit_correction_lower.array());     // else

          const auto upper_bound_size = cmodel.set().getPositiveOrthant().size();
          auto joint_limit_correction_upper = joint_limit_correction.tail(upper_bound_size);
          joint_limit_correction_upper = (joint_limit_correction_upper.array() < 0)     // if
                                           .select(                                     //
                                             joint_limit_correction_upper.array() * kp, // then
                                             joint_limit_correction_upper.array());     // else

          gc += joint_limit_correction;
        },
        //
        // Visitor for FrictionalPointConstraintModel - velocity formulation
        [&](const FrictionalPointConstraintModel & cmodel) {
          const FrictionalPointConstraintData & cdata_ = boost::get<const FrictionalPointConstraintData>(cdata);
          const Scalar kp = cmodel.baumgarte_corrector_parameters().Kp;
          const Scalar kd = cmodel.baumgarte_corrector_parameters().Kd;

          // TODO(quentin): no magic forces ?
          // TODO(quentin): take security_margin into account
          // TODO(louis): deal with elasticity. For that, we need to register which contact pairs were
          // in contact at the previous time step.
          // Vector3s correction(0, 0, std::min(0, Kb * penetration_depth / dt + elasticity * Jcv.coeff(2)));
          // -- Jc * (vfree - kd * v)
          MapVectorXs vtmp(PINOCCHIO_EIGEN_MAP_ALLOCA(Scalar, this->model().nv, 1));
          vtmp = vfree - kd * v;
          cmodel.jacobianMatrixProduct(this->model(), this->data(), cdata_, vtmp, gc);

          // -- baumgarte kp correction
          Vector3s correction = cdata_.constraint_position_error;
          correction = (correction.array() < 0)     // if
                         .select(                   //
                           correction.array() * kp, // then
                           0.);                     // else
          gc += correction / dt;
        });

      boost::apply_visitor(drift_visitor, cmodel);

      cindex += cdim;
    }
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  void ConstraintsProblemTpl<S, O, JointCollectionTpl>::updateConstraintsHolders()
  {
    this->m_current_constraints_pb_id = 1 - this->m_current_constraints_pb_id;
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  bool ConstraintsProblemTpl<S, O, JointCollectionTpl>::checkConstraintsConsistencyWithGeometryModel() const
  {
    assert(
      this->geom_data().collisionRequests.size() == this->geom_model().collisionPairs.size() //
      && "geom_data and geom_model do not coincide.");
    assert(
      this->geom_data().collisionRequests.size() == this->geom_data().collisionResults.size() //
      && "geom_data is inconsistent.");

    std::size_t max_num_contacts = this->geom_model().collisionPairs.size() * this->m_max_num_contact_per_collision_pair;

    return (
      this->frictional_point_constraint_models.size() == max_num_contacts     //
      && this->frictional_point_constraint_datas.size() == max_num_contacts); //
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  void ConstraintsProblemTpl<S, O, JointCollectionTpl>::collectActiveSet(Scalar epsilon)
  {
    // TODO we should add the active set for the other constraint types.
    this->breaking_contacts.clear();
    this->sticking_contacts.clear();
    this->sliding_contacts.clear();
    this->contact_modes.clear();

    const Vector3s e_z(0, 0, Scalar(1));
    const auto nc = static_cast<Eigen::Index>(this->getNumberOfContacts());
    for (Eigen::Index contact_id = 0; contact_id < nc; ++contact_id)
    {
      const Vector3s lambda_i = this->frictional_point_constraints_forces().template segment<3>(3 * contact_id);
      // TODO(quentinll) check if the compliance is taken into account in sigma_i. It should be.
      const Vector3s sigma_i = this->frictional_point_constraints_velocities().template segment<3>(3 * contact_id);
      if (lambda_i.norm() < epsilon) // breaking
      {
        this->breaking_contacts.push_back((ContactIndex)contact_id);
        this->contact_modes.push_back(ContactMode::BREAKING);
      }
      else if (sigma_i.norm() < epsilon) // sticking
      {
        this->sticking_contacts.push_back((ContactIndex)contact_id);
        this->contact_modes.push_back(ContactMode::STICKING);
      }
      else // sliding
      {
        this->sliding_contacts.push_back((ContactIndex)contact_id);
        this->contact_modes.push_back(ContactMode::SLIDING);
      }
    }

    assert(this->contact_modes.size() == (std::size_t)nc && "Wrong size of contact modes vector.");
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  bool ConstraintsProblemTpl<S, O, JointCollectionTpl>::check() const
  {
    const auto problem_size = static_cast<Eigen::Index>(this->constraints_problem_size());
    assert(this->m_model != nullptr && "The model handle points to nullptr.");
    assert(this->m_data != nullptr && "The data handle points to nullptr.");
    assert(this->m_geom_model != nullptr && "The geometry model handle points to nullptr.");
    assert(this->m_geom_data != nullptr && "The geometry data handle points to nullptr.");
    const std::size_t num_constraints = this->getNumberOfJointFrictionConstraints() //
                                        + this->getNumberOfBilateralConstraints()   //
                                        + this->getNumberOfWeldConstraints()        //
                                        + this->getNumberOfJointLimitConstraints()  //
                                        + this->getNumberOfContacts();
    assert(this->constraint_models.size() == num_constraints);
    assert(this->constraint_datas.size() == num_constraints);
    assert(
      (this->m_g.size() == this->m_constraints_velocities_warmstarts.size()) //
      && "g and constraint should always have the same (maximum) size.");
    assert((this->m_g.size() >= problem_size) && "The size of m_g is too small to contain the contact problem.");
    assert((this->g().size() == problem_size) && "The size of g does not match the size of the contact problem.");
    assert(
      (this->frictional_point_constraints_warmstart().size() == this->frictional_point_constraints_size())
      && "The size of the contact velocity warm-start does not match the size of the contact "
         "problem.");
    assert(
      (this->checkConstraintsConsistencyWithGeometryModel())
      && "The constraints model and data vectors are not consistent with the geom_model and the "
         "geom_data.");

    return static_cast<bool>(
      this->m_model != nullptr                                                                              //
      && this->m_data != nullptr                                                                            //
      && this->m_geom_model != nullptr                                                                      //
      && this->m_geom_data != nullptr                                                                       //
      && this->constraint_models.size() == num_constraints                                                  //
      && this->constraint_datas.size() == num_constraints                                                   //
      && this->m_g.size() == this->m_constraints_velocities_warmstarts.size()                               //
      && this->m_g.size() >= problem_size                                                                   //
      && this->g().size() == problem_size                                                                   //
      && this->frictional_point_constraints_warmstart().size() == this->frictional_point_constraints_size() //
      && this->checkConstraintsConsistencyWithGeometryModel());
  }

} // namespace simple

#endif // __simple_core_constraints_problem_hxx__
