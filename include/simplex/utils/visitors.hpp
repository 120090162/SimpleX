#ifndef __simplex_utils_visitors_hpp__
#define __simplex_utils_visitors_hpp__

#include <boost/variant.hpp>
#include <pinocchio/macros.hpp>
#include <pinocchio/algorithm/constraints/visitors/constraint-model-visitor.hpp>

namespace simplex
{
    namespace visitors
    {
        /**
         * @brief A helper structure that accumulates operator() overloads using variadic templates.
         *
         * It uses recursive inheritance to combine multiple lambda functions into a single
         * visitor object, making it compatible with boost::variant's visitation mechanism.
         */
        template<typename ReturnType, typename... Lambdas>
        struct lambda_visitor_helper;

        /**
         * @brief Base case for lambda_visitor_helper recursion.
         *
         * This specialization handles the scenario with no lambdas and provides a safe
         * handler for uninitialized or empty variant types (boost::blank).
         */
        template<typename ReturnType>
        struct lambda_visitor_helper<ReturnType>
        {
            lambda_visitor_helper()
            {
            }

            // Fallback operator to catch boost::blank (empty variants)
            ReturnType operator()(const boost::blank &) const
            {
                PINOCCHIO_THROW_PRETTY(std::invalid_argument, "SimpleX Error: Attempted to visit an empty boost::blank variant.");
                return ::pinocchio::visitors::internal::NoRun<ReturnType>::run();
            }
        };

        /**
         * @brief Recursive step for lambda_visitor_helper.
         *
         * It inherits from the first Lambda and recursively from the rest of the Lambdas.
         * Each level brings the operator() of the current Lambda into the derived scope.
         */
        template<typename ReturnType, typename Lambda, typename... Lambdas>
        struct lambda_visitor_helper<ReturnType, Lambda, Lambdas...>
        : Lambda
        , lambda_visitor_helper<ReturnType, Lambdas...>
        {
            typedef lambda_visitor_helper<ReturnType, Lambdas...> RecursiveHelper;

            // Initialize the current lambda and continue the recursion chain
            lambda_visitor_helper(Lambda lambda, Lambdas... lambdas)
            : Lambda(lambda)
            , RecursiveHelper(lambdas...)
            {
            }

            // Explicitly pull operator() from both current Lambda and the recursive parent
            // to resolve overload sets.
            using Lambda::operator();
            using RecursiveHelper::operator();
        };

        /**
         * @brief The main visitor class that bridges Boost's static_visitor with our helper.
         *
         * By inheriting from boost::static_visitor, this class can be passed to boost::apply_visitor.
         */
        template<typename ReturnType, typename... Lambdas>
        struct lambda_visitor
        : boost::static_visitor<ReturnType>
        , lambda_visitor_helper<ReturnType, Lambdas...>
        {
            typedef lambda_visitor_helper<ReturnType, Lambdas...> Helper;

            lambda_visitor(Lambdas... lambdas)
            : Helper(lambdas...)
            {
            }

            // Expose the accumulated operator() overloads
            using Helper::operator();
        };

        /**
         * @brief Factory function to create a lambda_visitor without explicit template arguments.
         *
         * Usage Example:
         *   auto my_visitor = make_lambda_visitor<double>(
         *     [](int x) { return (double)x; },
         *     [](const std::string& s) { return 0.0; }
         *   );
         *   boost::apply_visitor(my_visitor, my_variant);
         */
        template<typename ReturnType = void, typename... Lambdas>
        lambda_visitor<ReturnType, Lambdas...> make_lambda_visitor(Lambdas... lambdas)
        {
            return lambda_visitor<ReturnType, Lambdas...>(lambdas...);
        }

    } // namespace visitors
} // namespace simplex

#endif // __simplex_utils_visitors_hpp__