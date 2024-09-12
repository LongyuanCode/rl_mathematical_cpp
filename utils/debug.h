// debug.h
#ifndef DEBUG_H
#define DEBUG_H

#include <iostream>
#include <cstdlib>  // For std::abort()

#ifdef DEBUGSTR
    #define DEBUG_ASSERT(condition, message) \
        do { \
            if (!(condition)) { \
                std::cerr << "Assertion failed: (" #condition "), function " \
                          << __FUNCTION__ << ", file " << __FILE__ \
                          << ", line " << __LINE__ << ": " << message << std::endl; \
                std::abort(); \
            } \
        } while (false)
#else
    #define DEBUG_ASSERT(condition, message) do {} while (false)
#endif

#endif // DEBUG_H
