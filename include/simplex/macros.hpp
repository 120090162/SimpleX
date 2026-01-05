#ifndef __simplex_macros_hpp__
#define __simplex_macros_hpp__

#define SIMPLEX_PUBLIC public:
#ifdef SIMPLEX_ENABLE_PRIVATE_INTROSPECTION
    #define SIMPLEX_PROTECTED public:
    #define SIMPLEX_PRIVATE public:
#else
    #define SIMPLEX_PROTECTED protected:
    #define SIMPLEX_PRIVATE private:
#endif

#endif // ifndef __simplex_macros_hpp__
