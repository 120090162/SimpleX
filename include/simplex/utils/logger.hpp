#ifndef __simplex_utils_logger_hpp__
#define __simplex_utils_logger_hpp__

namespace simplex
{
    namespace logging
    {
        const char * const INFO = "\033[0;37m[INFO]\033[0m ";
        const char * const WARNING = "\033[0;33m[WARNING]\033[0m ";
        const char * const ERROR = "\033[0;31m[ERROR]\033[0m ";
        const char * const DEBUG = "\033[0;32m[DEBUG]\033[0m ";
    } // namespace logging
} // namespace simplex

#endif // __simplex_utils_logger_hpp__