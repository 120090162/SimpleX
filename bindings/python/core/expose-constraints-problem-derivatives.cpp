#include "simplex/bindings/python/core/constraints-problem-derivatives.hpp"

namespace bp = boost::python;

namespace simplex
{
    namespace python
    {

        void exposeConstraintsProblemDerivatives()
        {
            bp::class_<ConstraintsProblemDerivatives::ContactMapper>(
                "ContactMapper",
                "Maps a single collision pair to contact points information (contact position, contact "
                "normal, contact placements, constraint models/datas, friction cones, elasticities, "
                "penetration depths).",
                bp::no_init)
                .def_readwrite(
                    "begin", &ConstraintsProblemDerivatives::ContactMapper::begin,
                    "The id of the first contact point for the considered collision pair.")
                .def_readwrite(
                    "count", &ConstraintsProblemDerivatives::ContactMapper::count,
                    "Number of contact points for the considered collision pair");

            using ContactMode = ConstraintsProblemDerivatives::ContactMode;
            bp::enum_<ContactMode>("ContactMode")
                .value("BREAKING", ContactMode::BREAKING)
                .value("STICKING", ContactMode::STICKING)
                .value("SLIDING", ContactMode::SLIDING)
                .export_values();

            // Register a handle to ConstraintsProblemDerivatives
            using ConstraintsProblemDerivativesHandle = std::shared_ptr<ConstraintsProblemDerivatives>;
            bp::register_ptr_to_python<ConstraintsProblemDerivativesHandle>();

            ConstraintsProblemDerivativesPythonVisitor<ConstraintsProblemDerivatives>::expose();
        }

    } // namespace python
} // namespace simplex
