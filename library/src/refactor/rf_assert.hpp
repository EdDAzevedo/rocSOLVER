#ifndef RF_ASSERT_HPP
#define RF_ASSERT_HPP
#pragma once

#include <stdexcept>

#define RF_ASSERT( tcond ) {   if (!(tcond)) { throw std::runtime_error(__FILE__); }; }


#endif
