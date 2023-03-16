#ifndef RF_ASSERT_H
#define RF_ASSERT_H
#pragma once

#include <stdexcept>

#define RF_ASSERT(tcond)                        \
    {                                           \
        if(!(tcond))                            \
        {                                       \
            throw std::runtime_error(__FILE__); \
        };                                      \
    }

#endif
