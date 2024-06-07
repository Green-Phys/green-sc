/*
 * Copyright (c) 2023-2024 University of Michigan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the “Software”), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#ifndef GREEN_SC_COMMON_DEFS_H
#define GREEN_SC_COMMON_DEFS_H

#include <green/ndarray/ndarray.h>

namespace green::sc {
  /**
   *
   */
  enum mixing_type { NO_MIXING, G_DAMPING, SIGMA_DAMPING, DIIS, CDIIS };
  // Tensor types
  template <typename prec, size_t Dim>
  using tensor = green::ndarray::ndarray<prec, Dim>;
  template <size_t Dim>
  using ztensor = green::ndarray::ndarray<std::complex<double>, Dim>;
  template <size_t Dim>
  using ztensor_view = green::ndarray::ndarray<std::complex<double>, Dim>;
  template <size_t Dim>
  using ztensor_base = green::ndarray::ndarray<std::complex<double>, Dim>;
  template <size_t Dim>
  using ctensor = green::ndarray::ndarray<std::complex<float>, Dim>;
  template <size_t Dim>
  using dtensor = green::ndarray::ndarray<double, Dim>;
  template <size_t Dim>
  using ltensor = green::ndarray::ndarray<long, Dim>;
  template <size_t Dim>
  using itensor = green::ndarray::ndarray<int, Dim>;

  template <typename T, size_t D>
  inline std::array<size_t, D + 1> operator+(const std::array<size_t, D>& a, T b) {
    std::array<size_t, D + 1> result;
    std::copy(a.begin(), a.end(), result.begin());
    result[D] = size_t(b);
    return result;
  }

  template <typename T, size_t D>
  inline std::array<size_t, D + 1> operator+(T b, const std::array<size_t, D>& a) {
    std::array<size_t, D + 1> result;
    std::copy(a.begin(), a.end(), result.begin() + 1);
    result[0] = size_t(b);
    return result;
  }

  inline void define_parameters(params::params& p) {
    p.define<mixing_type>("mixing_type", "Type of iteration convergence mixing. We use no mixing by default", SIGMA_DAMPING);
    p.define<double>("damping",
                     "Simple mixing paramters between current ad previous iteration. Should be between 0 and 1: 1 - no damping (direct Roothaan steps), "
                     "0 - full damping (and no update).",
                     0.7);
    p.define<int>("diis_start", "Iteration number when we start using DIIS", 2);
    p.define<int>("diis_size", "Size of DIIS subspace", 3);
    p.define<std::string>("diis_file", "File to store results", "diis.h5");
    p.define<bool>("diis_restart", "Restart DIIS from previous data", true);
    p.define<std::string>("results_file", "File to store results", "sim.h5");
    p.define<int>("verbose", "Print verbose output.", 0);
    p.define<bool>("restart", "Try to restart simulation from the previously obtained results", false);
    p.define<unsigned>("itermax", "Maximum number of iterations", 1);
    p.define<double>("E_thr,threshold", "Convergence threshold", 1e-9);
    p.define<double>("E_thr_sp", "Energy convergence threshold if run in single precision", 1e-5);
    p.define<std::string>("input_file,", "File with input data", "input.h5");
    p.define<bool>("const_density", "Maintain constant number of electrons through iterations", true);
  }
}  // namespace green::sc
#endif  // GREEN_SC_COMMON_DEFS_H
