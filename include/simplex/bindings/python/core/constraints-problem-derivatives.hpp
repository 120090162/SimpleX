#ifndef __simplex_python_core_constraint_problem_derivatives_hpp__
#define __simplex_python_core_constraint_problem_derivatives_hpp__

#include "simplex/core/constraints-problem-derivatives.hpp"
#include "simplex/bindings/python/fwd.hpp"

#include <pinocchio/bindings/python/utils/std-vector.hpp>
#include <pinocchio/bindings/python/utils/copyable.hpp>

namespace simplex
{
    namespace python
    {
        namespace bp = boost::python;

        // template<typename T>
        // struct StdVectorOfReferenceWrappersPythonVisitor : public bp::def_visitor<StdVectorOfReferenceWrappersPythonVisitor<T>>
        // {
        //   typedef std::vector<std::reference_wrapper<T>> Self;
        //
        // public:
        //   template<class PyClass>
        //   void visit(PyClass & cl) const
        //   {
        //     cl.def(
        //         "__getitem__", +[](const Self & self, std::size_t i) -> const T & { return self[i].get(); },
        //         bp::return_value_policy<bp::reference_existing_object>())
        //       .def("__len__", +[](const Self & self) -> std::size_t { return self.size(); });
        //   }
        //
        //   static void expose(const std::string & name)
        //   {
        //     bp::class_<Self>(name.c_str(), "Vector of reference wrappers.",
        //     bp::no_init).def(StdVectorOfReferenceWrappersPythonVisitor());
        //   }
        // };

        template<typename ConstraintsProblemDerivatives>
        struct ConstraintsProblemDerivativesPythonVisitor
        : public bp::def_visitor<ConstraintsProblemDerivativesPythonVisitor<ConstraintsProblemDerivatives>>
        {
            typedef typename ConstraintsProblemDerivatives::Scalar Scalar;
            typedef ConstraintsProblemDerivatives Self;

            using ModelHandle = typename Self::ModelHandle;
            using DataHandle = typename Self::DataHandle;
            using VectorXs = typename Self::VectorXs;
            using GeometryModelHandle = typename Self::GeometryModelHandle;
            using GeometryDataHandle = typename Self::GeometryDataHandle;
            using PlacementVector = typename Self::PlacementVector;
            using ConstraintModel = typename Self::ConstraintModel;
            using ConstraintData = typename Self::ConstraintData;
            using BilateralPointConstraintModel = typename Self::BilateralPointConstraintModel;
            using WeldConstraintModel = typename Self::WeldConstraintModel;
            using DelassusType = typename Self::DelassusType;

            using VectorStorageMapType = typename Self::VectorStorageMapType;
            using VectorView = typename Self::VectorView;

            template<class PyClass>
            void visit(PyClass & cl) const
            {
                cl
                    // ----------------------------------
                    // CONSTRUCTORS
                    .def(
                        bp::init<ModelHandle, DataHandle, GeometryModelHandle, GeometryDataHandle>(
                            bp::args("self", "model", "data", "geom_model", "geom_data"), "Constructor"))

                    // ----------------------------------
                    // ATTRIBUTES/METHODS

                    // ----------------------------------
                    // -- general
                    .def_readwrite("delassus_type", &Self::delassus_type, "Type of delassus. CHOLESKY or RIGID_BODY.")
                    .def_readonly(
                        "constraint_cholesky_decomposition", &Self::constraint_cholesky_decomposition,
                        "Cholesky decomposition of the constraints problem. In particular, it contains the "
                        "Cholesky decomposition of the Delassus' operator `G`.")
                    .def_readwrite(
                        "is_ncp", &Self::is_ncp,
                        "Type of constraints problem. If set to true, the constraints problem is a NCP, else it is a CCP.")

                    // TODO(louis): add no-copy getters
                    .add_property(
                        "constraints_problem_size", bp::make_function(+[](Self & self) -> int { return self.constraints_problem_size(); }),
                        "Size of the constraint problem.")
                    .add_property(
                        "g", bp::make_function(+[](Self & self) -> VectorXs { return VectorXs(self.g); }),
                        "Returns a copy of the drift term (g) of the constraint problem.")
                    .add_property(
                        "preconditioner", bp::make_function(+[](Self & self) -> VectorXs { return VectorXs(self.preconditioner); }),
                        "Returns a copy of the preconditioner of the constraint problem.")
                    .add_property(
                        "constraint_forces", bp::make_function(+[](Self & self) -> VectorXs { return VectorXs(self.constraint_forces); }),
                        "Returns a copy of the constraint forces.")
                    .add_property(
                        "constraint_velocities",
                        bp::make_function(+[](Self & self) -> VectorXs { return VectorXs(self.constraint_velocities); }),
                        "Returns a copy of the Constraint velocities.")

                    .def("update", &Self::update, bp::args("self"), "Update constraints with current model, data, geom_model, geom_data.")
                    .def("clear", &Self::clear, bp::args("self"), "Clear currrent contact quantities.")

                    // ----------------------------------
                    // -- joint friction
                    .def_readwrite(
                        "joint_friction_constraint_models", &Self::joint_friction_constraint_model, "Joint friction constraint model.")
                    .def_readonly(
                        "joint_friction_constraint_datas", &Self::joint_friction_constraint_data, "Joint friction constraint data.")
                    .add_property(
                        "joint_friction_constraint_size",
                        bp::make_function(+[](Self & self) -> int { return self.joint_friction_constraint_size(); }),
                        "Size of the vectors of dry friction constraints' forces/velocities.")
                    .def(
                        "getNumberOfJointFrictionConstraints", &Self::getNumberOfJointFrictionConstraints, bp::args("self"),
                        "Returns the number of joint friction constraints.")
                    .add_property(
                        "joint_friction_constraint_forces",
                        bp::make_function(+[](Self & self) -> VectorXs { return VectorXs(self.joint_friction_constraint_forces()); }),
                        "Returns a copy of the joint friction constraints' forces.")
                    .add_property(
                        "joint_friction_constraint_velocities",
                        bp::make_function(+[](Self & self) -> VectorXs { return VectorXs(self.joint_friction_constraint_velocities()); }),
                        "Returns a copy of the joint friction constraints' velocities.")

                    // ----------------------------------
                    // -- bilateral constraints
                    .def_readwrite(
                        "bilateral_point_constraint_models", &Self::bilateral_point_constraint_models,
                        "Vector of bilateral constraint models.")
                    .def_readonly(
                        "bilateral_point_constraint_datas", &Self::bilateral_point_constraint_datas,
                        "Vector of bilateral constraint datas.")
                    .add_property(
                        "bilateral_constraints_size",
                        bp::make_function(+[](Self & self) -> int { return self.bilateral_constraints_size(); }),
                        "Size of the vectors of bilateral constraints' forces/velocities.")
                    .def(
                        "getNumberOfBilateralConstraints", &Self::getNumberOfBilateralConstraints, bp::args("self"),
                        "Returns the number of bilateral constraints.")
                    .add_property(
                        "bilateral_constraints_forces",
                        bp::make_function(+[](Self & self) -> VectorXs { return VectorXs(self.bilateral_constraints_forces()); }),
                        "Returns a copy of the bilateral constraints' forces.")
                    .add_property(
                        "bilateral_constraints_velocities",
                        bp::make_function(+[](Self & self) -> VectorXs { return VectorXs(self.bilateral_constraints_velocities()); }),
                        "Return a copy of the bilateral constraints' velocities.")

                    // ----------------------------------
                    // -- weld constraints
                    .def_readwrite("weld_constraint_models", &Self::weld_constraint_models, "Vector of weld constraint models.")
                    .def_readonly("weld_constraint_datas", &Self::weld_constraint_datas, "Vector of weld constraint datas.")
                    .add_property(
                        "weld_constraints_size", bp::make_function(+[](Self & self) -> int { return self.weld_constraints_size(); }),
                        "Size of the vectors of weld constraints' forces/velocities.")
                    .def(
                        "getNumberOfWeldConstraints", &Self::getNumberOfWeldConstraints, bp::args("self"),
                        "Returns the number of weld constraints.")
                    .add_property(
                        "weld_constraints_forces",
                        bp::make_function(+[](Self & self) -> VectorXs { return VectorXs(self.weld_constraints_forces()); }),
                        "Returns a copy of the weld constraints' forces.")
                    .add_property(
                        "weld_constraints_velocities",
                        bp::make_function(+[](Self & self) -> VectorXs { return VectorXs(self.weld_constraints_velocities()); }),
                        "Returns a copy of the weld constraints' velocities.")

                    // ----------------------------------
                    // -- joint limits
                    .def_readwrite("joint_limit_constraint_model", &Self::joint_limit_constraint_model, "Joint limit constraint model.")
                    .def_readonly("joint_limit_constraint_data", &Self::joint_limit_constraint_data, "Joint limit constraint data.")
                    .add_property(
                        "joint_limit_constraint_size",
                        bp::make_function(+[](Self & self) -> int { return self.joint_limit_constraint_size(); }),
                        "Size of the vectors of joint limit constraints' forces/velocities.")
                    .add_property(
                        "joint_limit_constraint_max_size",
                        bp::make_function(+[](Self & self) -> int { return self.joint_limit_constraint_max_size(); }),
                        "Maximum size of the vectors of joint limit constraints' forces/velocities.")
                    .def(
                        "getNumberOfJointLimitConstraints", &Self::getNumberOfJointLimitConstraints, bp::args("self"),
                        "Returns the number of joint limit constraints.")
                    .add_property(
                        "joint_limit_constraint_forces",
                        bp::make_function(+[](Self & self) -> VectorXs { return VectorXs(self.joint_limit_constraint_forces()); }),
                        "Returns a copy of the joint limit constraints' forces.")
                    .add_property(
                        "joint_limit_constraint_velocities",
                        bp::make_function(+[](Self & self) -> VectorXs { return VectorXs(self.joint_limit_constraint_velocities()); }),
                        "Returns a copy of the joint limit constraints' velocities.")

                    // ----------------------------------
                    // -- frictional point constraints
                    .def_readwrite(
                        "frictional_point_constraint_models", &Self::frictional_point_constraint_models,
                        "Vector of frictional point contact constraint models.")
                    .def_readonly(
                        "frictional_point_constraint_datas", &Self::frictional_point_constraint_datas,
                        "Vector of frictional point contact constraint datas.")
                    .def(
                        "setMaxNumberOfContactsPerCollisionPair", &Self::setMaxNumberOfContactsPerCollisionPair, bp::args("self"),
                        "Set maximum number of contacts for each collision pair.")
                    .def(
                        "getMaxNumberOfContactsPerCollisionPair", &Self::getMaxNumberOfContactsPerCollisionPair, bp::args("self"),
                        "Get the maximum number of contacts for each collision pair.")
                    .def(
                        "getMaxNumberOfContacts", &Self::getMaxNumberOfContacts, bp::args("self"),
                        "Maximum number of contacts this `ConstraintsProblemDerivatives` can handle.")

                    .def("getNumberOfContacts", &Self::getNumberOfContacts, bp::args("self"), "Number of contacts.")

                    .add_property(
                        "frictional_point_constraints_size",
                        bp::make_function(+[](Self & self) -> int { return self.frictional_point_constraints_size(); }),
                        "Size of the vectors of frictional point contact constraints' forces/velocities.")
                    .add_property(
                        "frictional_point_constraints_forces",
                        bp::make_function(+[](Self & self) -> VectorXs { return VectorXs(self.frictional_point_constraints_forces()); }),
                        "Returns a copy of the frictional point contact constraints' forces.")
                    .add_property(
                        "frictional_point_constraints_velocities", bp::make_function(+[](Self & self) -> VectorXs {
                            return VectorXs(self.frictional_point_constraints_velocities());
                        }),
                        "Returns a copy of the point contact constraints' velocities.")

                    .def_readonly(
                        "point_contact_constraint_placements", &Self::point_contact_constraint_placements,
                        "Contact placements (oMc) related to contact constraints of the constraint problem.")

                    .def_readonly("pairs_in_collision", &Self::pairs_in_collision, "Ids of collision pairs which are in collision.")
                    .def_readonly(
                        "contact_id_to_collision_pair", &Self::contact_id_to_collision_pair,
                        "Vector that maps the id of the contact to the collision pair. Therefore "
                        "`contact_id_to_collision_pair[i]` is the id of the collision pair corresponding to "
                        "the i-th contact. Note: since collision pairs can have multiple contacts, the same id "
                        "can be found multiple times inside `contact_id_to_collision_pair`.")
                    .def_readonly("contact_mappers", &Self::contact_mappers, "Vector of contact mappers of the contact problem.")

                    .def_readonly(
                        "contact_modes", &Self::contact_modes,
                        "Contact modes associated to point contact constraints (breadking, sliding or sticking).")
                    .def(
                        "collectActiveSet", &Self::collectActiveSet, (bp::arg("self"), bp::arg("epsilon") = 1e-6),
                        "Collect active set of of the solution of the contact problem.");

                // StdVectorOfReferenceWrappersPythonVisitor<const ConstraintModel>::expose("StdRefVec_ConstraintModel");
                // StdVectorOfReferenceWrappersPythonVisitor<ConstraintData>::expose("StdRefVec_ConstraintData");

                cl.def_readonly("constraint_models", &Self::constraint_models, "Active constraint models.")
                    .def_readonly("constraint_datas", &Self::constraint_datas, "Active constraint datas.");
            }

            static void expose()
            {
                ::pinocchio::python::
                    StdVectorPythonVisitor<typename ConstraintsProblemDerivatives::BilateralPointConstraintModelVector, true>::expose(
                        "StdVec_BilateralPointConstraintModel");

                ::pinocchio::python::StdVectorPythonVisitor<
                    typename ConstraintsProblemDerivatives::WeldConstraintModelVector, true>::expose("StdVec_WeldConstraintModel");

                ::pinocchio::python::StdVectorPythonVisitor<std::vector<double>, true>::expose("StdVec_double");

                ::pinocchio::python::StdVectorPythonVisitor<
                    std::vector<typename ConstraintsProblemDerivatives::ContactMapper>, true>::expose("StdVec_ContactMapper");

                ::pinocchio::python::StdVectorPythonVisitor<std::vector<typename ConstraintsProblemDerivatives::ContactMode>, true>::expose(
                    "StdVec_ContactMode");

                bp::class_<typename ConstraintsProblemDerivatives::WrappedConstraintModel>("WrappedConstraintModel", bp::no_init)
                    .def(
                        "__getattr__",
                        +[](bp::object self, std::string const & name) {
                            using ConstraintModel = typename ConstraintsProblemDerivatives::ConstraintModel;
                            const ConstraintModel & obj = bp::extract<ConstraintModel>(self());
                            return bp::getattr(bp::object(bp::ptr(&obj)), name.c_str());
                        })
                    .def(
                        "__call__",
                        +[](const typename ConstraintsProblemDerivatives::WrappedConstraintModel & cmodel)
                            -> const typename ConstraintsProblemDerivatives::ConstraintModel & { return cmodel.get(); },
                        bp::return_internal_reference<>());

                bp::class_<typename ConstraintsProblemDerivatives::WrappedConstraintData>("WrappedConstraintData", bp::no_init)
                    .def(
                        "__getattr__",
                        +[](bp::object self, std::string const & name) {
                            using ConstraintData = typename ConstraintsProblemDerivatives::ConstraintData;
                            const ConstraintData & obj = bp::extract<ConstraintData>(self());
                            return bp::getattr(bp::object(bp::ptr(&obj)), name.c_str());
                        })
                    .def(
                        "__call__",
                        +[](const typename ConstraintsProblemDerivatives::WrappedConstraintData & cdata)
                            -> const typename ConstraintsProblemDerivatives::ConstraintData & { return cdata.get(); },
                        bp::return_internal_reference<>());

                ::pinocchio::python::StdVectorPythonVisitor<
                    typename ConstraintsProblemDerivatives::WrappedConstraintModelVector, true>::expose("StdVec_WrappedConstraintModel");
                ::pinocchio::python::StdVectorPythonVisitor<
                    typename ConstraintsProblemDerivatives::WrappedConstraintDataVector, true>::expose("StdVec_WrappedConstraintData");

                bp::enum_<DelassusType>("DelassusType") //
                    .value("DENSE", DelassusType::DENSE)
                    .value("CHOLESKY", DelassusType::CHOLESKY)
                    .value("RIGID_BODY", DelassusType::RIGID_BODY);

                bp::class_<ConstraintsProblemDerivatives>("ConstraintsProblemDerivatives", "Contact problem.\n", bp::no_init)
                    .def(ConstraintsProblemDerivativesPythonVisitor<ConstraintsProblemDerivatives>())
                    .def(::pinocchio::python::CopyableVisitor<ConstraintsProblemDerivatives>());
            }
        };

    } // namespace python
} // namespace simplex

#endif // ifndef __simplex_python_core_constraint_problem_derivatives_hpp__
