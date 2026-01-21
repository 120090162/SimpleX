#ifndef __simplex_python_core_ncp_derivatives_hpp__
#define __simplex_python_core_ncp_derivatives_hpp__

#include "simplex/core/ncp-derivatives.hpp"
#include "simplex/bindings/python/fwd.hpp"

#include <pinocchio/bindings/python/utils/copyable.hpp>

namespace simplex
{
    namespace python
    {
        namespace bp = boost::python;

        template<typename ContactSolverDerivatives>
        struct ContactSolverDerivativesPythonVisitor
        : public boost::python::def_visitor<ContactSolverDerivativesPythonVisitor<ContactSolverDerivatives>>
        {
            using Self = ContactSolverDerivatives;
            using Scalar = typename Self::Scalar;
            using VectorXs = typename Self::VectorXs;
            using MatrixXs = typename Self::MatrixXs;

            using ConstraintsProblemDerivativesHandle = typename Self::ConstraintsProblemDerivativesHandle;

            template<class PyClass>
            void visit(PyClass & cl) const
            {
                cl.def(bp::init<ConstraintsProblemDerivativesHandle>(bp::args("self", "constraint_problem"), "Constructor"))
                    .def_readwrite("measure_timings", &Self::measure_timings, "Measure timings of the `step` method.")

                    .def_readwrite(
                        "implicit_gradient_solver_type", &Self::implicit_gradient_solver_type,
                        "Which solver to use to solve the system of implicit gradients.")

                    .def(
                        "jvp", +[](Self & self, const MatrixXs dGlamgdtheta) { self.jvp(::pinocchio::make_const_ref(dGlamgdtheta)); },
                        bp::args("self", "dGlamgdtheta"),
                        "Computes the Jacobian vector product of the contact solver derivatives. The `compute` "
                        "method should be called before calling this method.")

                    .def(
                        "dlam_dtheta",
                        +[](const Self & self) -> MatrixXs {
                            MatrixXs rhs(self.dlam_dtheta());
                            return rhs;
                        },
                        bp::args("self"), "Gradients of lambda wrt theta.")

                    .def(
                        "G",
                        +[](const Self & self) -> MatrixXs {
                            MatrixXs G(self.G());
                            return G;
                        },
                        bp::args("self"), "Matrix of the system of implicit gradients.")

                    .def(
                        "rhs",
                        +[](const Self & self) -> MatrixXs {
                            MatrixXs rhs(self.rhs());
                            return rhs;
                        },
                        bp::args("self"), "Right hand side of the system of implicit gradients.")

                    .def(
                        "getContactSolverDerivativesCPUTimes", &Self::getContactSolverDerivativesCPUTimes, bp::args("self"),
                        "Get timings of the call to the contact solver in the last call to the `step` method. "
                        "These timings can be 0 if no contacts occured.")

                    .def(
                        "compute", &Self::compute, bp::args("self"),
                        "Performs the internal computation to formulate the linear system associated with the "
                        "implicit differentiation and computes the decomposition of the resulting matrix.");
            }

            static void expose()
            {
                bp::class_<ContactSolverDerivatives>("ContactSolverDerivatives", "Class computing derivatives of the NCP.", bp::no_init)
                    .def(ContactSolverDerivativesPythonVisitor<ContactSolverDerivatives>())
                    .def(::pinocchio::python::CopyableVisitor<ContactSolverDerivatives>());
            }
        };

    } // namespace python
} // namespace simplex

#endif // ifndef __simplex_python_core_ncp_derivatives_hpp__
