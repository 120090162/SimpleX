#ifndef __simplex__test_utils_hpp__
#define __simplex__test_utils_hpp__

#define EIGEN_VECTOR_IS_APPROX(Va, Vb, precision)                                                                                          \
    BOOST_CHECK_MESSAGE(                                                                                                                   \
        ((Va) - (Vb)).isZero(precision), "check " #Va ".isApprox(" #Vb ") failed at precision "                                            \
                                             << precision << ". (" #Va " - " #Vb ").norm() = " << ((Va) - (Vb)).norm() << " [\n"           \
                                             << (Va).transpose() << "\n!=\n"                                                               \
                                             << (Vb).transpose() << "\n]")

#define INDEX_EQUALITY_CHECK(i1, i2) BOOST_CHECK_MESSAGE(i1 == i2, "check " #i1 "==" #i2 " failed. [" << i1 << " != " << i2 << "]")
#define INDEX_INEQUALITY_CHECK(i1, i2) BOOST_CHECK_MESSAGE(i1 <= i2, "check " #i1 "==" #i2 " failed. [" << i1 << " != " << i2 << "]")

#define REAL_IS_APPROX(a, b, precision)                                                                                                    \
    BOOST_CHECK_MESSAGE(                                                                                                                   \
        std::abs((a) - (b)) < precision,                                                                                                   \
        "check std::abs(" #a " - " #b ") = " << std::abs((a) - (b)) << " < " << precision << " failed. [" << a << " != " << b << "]")

#define FMT_HEADER_ONLY
#include <fmt/format.h>

#include "simplex/utils/logger.hpp"

#include <pinocchio/parsers/mjcf.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/geometry.hpp>

#include <clarabel.hpp>
#include <iomanip>
#include <iostream>
#include <string>

inline std::string findTestResource(const std::string & simple_test_resource_path)
{
    std::cout << simplex::logging::DEBUG << "Finding test resource: " << simple_test_resource_path << std::endl;
    std::cout << simplex::logging::DEBUG << "SIMPLEX_TEST_DIR: " << SIMPLEX_TEST_DIR << std::endl;
    if (simple_test_resource_path.substr(0, 8) != "SIMPLEX/")
    {
        throw std::runtime_error(fmt::format("Resource path '{}' must start with 'SIMPLEX/' ", simple_test_resource_path));
    }
    // N.B. SIMPLEX_TEST_DIR set in top-level CMakeLists.txt
    return std::string(SIMPLEX_TEST_DIR) + simple_test_resource_path.substr(7, simple_test_resource_path.size() - 7);
}

template<typename T>
inline void printArray(Eigen::Map<Eigen::VectorX<T>> & vec)
{
    std::cout << simplex::logging::DEBUG << "[";
    size_t n = vec.size();
    for (size_t i = 0; i < n; i++)
    {
        std::cout << std::fixed << std::setprecision(10) << vec.data()[i];
        if (i < n - 1)
        {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

template<typename T>
inline void printSolution(clarabel::DefaultSolution<T> & solution)
{
    std::cout << simplex::logging::DEBUG << "Solution (x)\t = ";
    printArray(solution.x);
    std::cout << simplex::logging::DEBUG << "Multipliers (z)\t = ";
    printArray(solution.z);
    std::cout << simplex::logging::DEBUG << "Slacks (s)\t = ";
    printArray(solution.s);
}

namespace simplex
{
    void addFloor(::pinocchio::GeometryModel & geom_model)
    {
        using namespace pinocchio;

        using CollisionGeometryPtr = GeometryObject::CollisionGeometryPtr;

        CollisionGeometryPtr floor_collision_shape = CollisionGeometryPtr(new coal::Halfspace(0.0, 0.0, 1.0, 0.0));

        const SE3 M = SE3::Identity();
        GeometryObject floor("floor", 0, 0, M, floor_collision_shape);
        geom_model.addGeometryObject(floor);
    }

    void load(::pinocchio::Model & model, ::pinocchio::GeometryModel & geom_model, bool load_floor = true, bool verbose = false)
    {
        auto mjcf_path = findTestResource("SIMPLEX/tests/resources/mujoco_humanoid.xml");
        ::pinocchio::mjcf::buildModel(mjcf_path, model, verbose);
        ::pinocchio::mjcf::buildGeom(model, mjcf_path, pinocchio::COLLISION, geom_model);
        if (load_floor)
            addFloor(geom_model);
    }

    void addCollisionPairs(const ::pinocchio::Model & model, ::pinocchio::GeometryModel & geom_model, const Eigen::VectorXd & qref)
    {
        using namespace pinocchio;
        Data data(model);
        GeometryData geom_data(geom_model);
        // TI this function to gain compilation speed on this test
        ::pinocchio::updateGeometryPlacements(model, data, geom_model, geom_data, qref);
        geom_model.removeAllCollisionPairs();
        for (std::size_t i = 0; i < geom_model.geometryObjects.size(); ++i)
        {
            for (std::size_t j = i; j < geom_model.geometryObjects.size(); ++j)
            {
                if (i == j)
                {
                    continue; // don't add collision pair if same object
                }
                const GeometryObject & gobj_i = geom_model.geometryObjects[i];
                const GeometryObject & gobj_j = geom_model.geometryObjects[j];
                if (gobj_i.name == "floor" || gobj_j.name == "floor")
                { // if floor, always add a collision pair
                    geom_model.addCollisionPair(CollisionPair(i, j));
                }
                else
                {
                    if (gobj_i.parentJoint == gobj_j.parentJoint)
                    { // don't add collision pair if same parent
                        continue;
                    }

                    // run collision detection -- add collision pair if shapes are not colliding
                    const SE3 M1 = geom_data.oMg[i];
                    const SE3 M2 = geom_data.oMg[j];

                    coal::CollisionRequest colreq;
                    colreq.security_margin = 1e-2; // 1cm of clearance
                    coal::CollisionResult colres;
                    coal::collide(
                        gobj_i.geometry.get(), ::pinocchio::toFclTransform3f(M1), //
                        gobj_j.geometry.get(), ::pinocchio::toFclTransform3f(M2), //
                        colreq, colres);
                    if (!colres.isCollision())
                    {
                        geom_model.addCollisionPair(CollisionPair(i, j));
                    }
                }
            }
        }
    }
} // namespace simplex

#endif // __simplex__test_utils_hpp__
