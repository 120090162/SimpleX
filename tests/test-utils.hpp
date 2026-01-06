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

#include <clarabel.hpp>
#include <iomanip>
#include <iostream>
#include <string>

inline std::string findTestResource(const std::string & simple_test_resource_path)
{
    std::cout << simplex::logging::INFO << "Finding test resource: " << simple_test_resource_path << std::endl;
    std::cout << simplex::logging::INFO << "SIMPLEX_TEST_DIR: " << SIMPLEX_TEST_DIR << std::endl;
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
    std::cout << simplex::logging::INFO << "[";
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
    std::cout << simplex::logging::INFO << "Solution (x)\t = ";
    printArray(solution.x);
    std::cout << simplex::logging::INFO << "Multipliers (z)\t = ";
    printArray(solution.z);
    std::cout << simplex::logging::INFO << "Slacks (s)\t = ";
    printArray(solution.s);
}

#endif // __simplex__test_utils_hpp__
