#ifndef __simplex_python_solver_clarabel_solver_hpp__
#define __simplex_python_solver_clarabel_solver_hpp__

#include "simplex/bindings/python/fwd.hpp"
#include "simplex/solver/clarabel-solver.hpp"

#include <boost/python.hpp>

namespace simplex
{
    namespace python
    {
        namespace bp = boost::python;

        template<typename Solver>
        struct ClarabelConstraintSolverPythonVisitor : public bp::def_visitor<ClarabelConstraintSolverPythonVisitor<Solver>>
        {
            using Self = Solver;
            using VectorXs = typename Self::VectorXs;

            template<class PyClass>
            void visit(PyClass & cl) const
            {
                cl.def(bp::init<>(bp::args("self"), "Default constructor."))
                    .def(bp::init<int>(bp::args("self", "problem_size"), "Construct solver for a given problem size."))

                    .add_property(
                        "problem_size",
                        bp::make_function(
                            +[](const Self & self) { return self.problem_size_; }, bp::return_value_policy<bp::return_by_value>()),
                        "Current size of the constraint problem handled by the solver.")

                    .def("setMaxIterations", &Self::setMaxIterations, bp::args("self", "max_iter"), "Set maximum solver iterations.")
                    .def(
                        "setAbsolutePrecision", &Self::setAbsolutePrecision, bp::args("self", "absolute_precision"),
                        "Set absolute precision tolerance.")
                    .def(
                        "setRelativePrecision", &Self::setRelativePrecision, bp::args("self", "relative_precision"),
                        "Set relative precision tolerance.")

                    .def(
                        "getPrimalSolution",
                        bp::make_function(
                            +[](const Self & self) { return VectorXs(self.getPrimalSolution()); },
                            bp::return_value_policy<bp::return_by_value>()),
                        "Return a copy of the last primal solution (constraint impulses).")
                    .def(
                        "getDualSolution",
                        bp::make_function(
                            +[](const Self & self) { return VectorXs(self.getDualSolution()); },
                            bp::return_value_policy<bp::return_by_value>()),
                        "Return a copy of the last dual solution (constraint velocities).")
                    .def("getIterationCount", &Self::getIterationCount, bp::args("self"), "Number of iterations used by the last solve.")
                    .def("isInitialized", &Self::isInitialized, bp::args("self"), "Return whether the solver has been initialized.")
                    .def("reset", &Self::reset, bp::args("self"), "Reset internal state and warm-start data.");
            }

            static void expose()
            {
                bp::class_<Self>("ClarabelConstraintSolver", "Clarabel based constraint solver wrapper for contact problems.")
                    .def(ClarabelConstraintSolverPythonVisitor<Self>());
            }
        };

    } // namespace python
} // namespace simplex

#endif // ifndef __simplex_python_solver_clarabel_solver_hpp__
