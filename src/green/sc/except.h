/*
 * Copyright (c) 2023 University of Michigan
 *
 */
#ifndef SC_EXCEPT_H
#define SC_EXCEPT_H

#include <stdexcept>

namespace green::sc {

  class sc_incorrect_damping_error : std::runtime_error {
  public:
    sc_incorrect_damping_error(const std::string& what) : std::runtime_error(what) {}
  };

}  // namespace green::sc

#endif  // SC_EXCEPT_H
