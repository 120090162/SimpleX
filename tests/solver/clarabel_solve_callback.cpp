#include "../test-utils.hpp"

#include <clarabel.hpp>
#include <Eigen/Eigen>
#include <vector>

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>

using namespace clarabel;
using namespace std;
using namespace Eigen;

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

typedef struct
{
    int count;
} CallbackData;

int custom_callback(DefaultInfo<double> & info, void * userdata)
{
    // This function is called at each iteration of the solver.
    // You can use it to monitor the progress of the solver or
    // to implement custom stopping criteria.
    // For example, we can print the current iteration number:
    std::cout << simplex::logging::DEBUG << "Custom callback at iteration" << info.iterations << ": ";
    // Cast the userdata pointer back to our struct type
    int count = 0;
    if (userdata != nullptr)
    {
        CallbackData * data = (CallbackData *)userdata;

        // Access and modify the state
        data->count++;
        count = data->count;
    }
    else
    {
        count = info.iterations;
    }

    // Return 0 to continue. Anything else to stop.
    if (count < 3)
    {
        std::cout << "tick" << std::endl;
        return 0; // continue
    }
    else
    {
        std::cout << "BOOM!" << std::endl;
        return 1; // stop
    }
}

BOOST_AUTO_TEST_CASE(example_callback_with_state)
{
    MatrixXd P_dense = MatrixXd::Zero(2, 2);
    SparseMatrix<double> P = P_dense.sparseView();
    P.makeCompressed();

    Vector<double, 2> q = {1.0, -1.0};

    // a 2-d box constraint, separated into 4 inequalities.
    // A = [I; -I]
    MatrixXd A_dense(4, 2);
    A_dense << 1., 0., 0., 1., -1., 0., 0., -1.;

    SparseMatrix<double> A = A_dense.sparseView();
    A.makeCompressed();

    Vector<double, 4> b = {1.0, 1.0, 1.0, 1.0};

    vector<SupportedConeT<double>> cones{
        NonnegativeConeT<double>(4),
        // {.tag = SupportedConeT<double>::Tag::NonnegativeConeT, .nonnegative_cone_t = {._0 = 4 }}
    };

    // Settings
    DefaultSettings<double> settings =
        DefaultSettingsBuilder<double>::default_settings().equilibrate_enable(true).equilibrate_max_iter(50).build();

    // Build solver
    DefaultSolver<double> solver(P, q, A, b, cones, settings);

    // configure a custom callback function
    CallbackData userdata = {-1};
    solver.set_termination_callback(custom_callback, &userdata);

    // Solve
    solver.solve();

    // turn off the callback
    solver.unset_termination_callback();

    // Solve again
    solver.solve();

    // Get solution
    DefaultSolution<double> solution = solver.solution();
    printSolution(solution);
}

BOOST_AUTO_TEST_CASE(example_callback)
{
    MatrixXd P_dense = MatrixXd::Zero(2, 2);
    SparseMatrix<double> P = P_dense.sparseView();
    P.makeCompressed();

    Vector<double, 2> q = {1.0, -1.0};

    // a 2-d box constraint, separated into 4 inequalities.
    // A = [I; -I]
    MatrixXd A_dense(4, 2);
    A_dense << 1., 0., 0., 1., -1., 0., 0., -1.;

    SparseMatrix<double> A = A_dense.sparseView();
    A.makeCompressed();

    Vector<double, 4> b = {1.0, 1.0, 1.0, 1.0};

    vector<SupportedConeT<double>> cones{
        NonnegativeConeT<double>(4),
        // {.tag = SupportedConeT<double>::Tag::NonnegativeConeT, .nonnegative_cone_t = {._0 = 4 }}
    };

    // Settings
    DefaultSettings<double> settings =
        DefaultSettingsBuilder<double>::default_settings().equilibrate_enable(true).equilibrate_max_iter(50).build();

    // Build solver
    DefaultSolver<double> solver(P, q, A, b, cones, settings);

    // configure a custom callback function
    solver.set_termination_callback(custom_callback, nullptr);

    // Solve
    solver.solve();

    // turn off the callback
    solver.unset_termination_callback();

    // Solve again
    solver.solve();

    // Get solution
    DefaultSolution<double> solution = solver.solution();
    printSolution(solution);
}

BOOST_AUTO_TEST_SUITE_END()
