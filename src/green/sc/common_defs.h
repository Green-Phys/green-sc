/*
 * Copyright (c) 2023 University of Michigan
 *
 */
#ifndef SC_COMMON_DEFS_H
#define SC_COMMON_DEFS_H

#include <green/ndarray/ndarray.h>

#include <Eigen/Dense>

namespace green::sc {
  /**
   *
   */
  enum mixing_type { NO_MIXING, G_DAMPING, SIGMA_DAMPING, DIIS };
  // Matrix types
  template <typename prec>
  using MatrixX   = Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixXcd = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixXcf = Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixXd  = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  // Matrix-Map types
  template <typename prec>
  using MMatrixX   = Eigen::Map<Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using MMatrixXcd = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using MMatrixXcf = Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using MMatrixXd  = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  // Const Matrix-Map types
  template <typename prec>
  using CMMatrixX   = Eigen::Map<const Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using CMMatrixXcd = Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using CMMatrixXcf = Eigen::Map<const Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using CMMatrixXd  = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  // Column type
  using column      = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1, Eigen::ColMajor>;
  using Mcolumn     = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1, Eigen::ColMajor>>;
  using CMcolumn    = Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1, Eigen::ColMajor>>;
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

  template <typename prec, typename = std::enable_if_t<std::is_same_v<prec, std::remove_const_t<prec>>>>
  auto matrix(green::ndarray::ndarray<prec, 2>& array) {
    return MMatrixX<prec>(array.data(), array.shape()[0], array.shape()[1]);
  }

  template <typename prec, typename = std::enable_if_t<std::is_same_v<prec, std::remove_const_t<prec>>>>
  auto matrix(green::ndarray::ndarray<prec, 2>&& array) {
    return MMatrixX<prec>(array.data(), array.shape()[0], array.shape()[1]);
  }

  template <typename prec>
  auto matrix(const green::ndarray::ndarray<const prec, 2>& array) {
    return CMMatrixX<prec>(const_cast<prec*>(array.data()), array.shape()[0], array.shape()[1]);
  }

  template <typename prec>
  auto matrix(green::ndarray::ndarray<const prec, 2>&& array) {
    return CMMatrixX<prec>(const_cast<prec*>(array.data()), array.shape()[0], array.shape()[1]);
  }

  template <typename prec>
  auto matrix(const green::ndarray::ndarray<prec, 2>& array) {
    return CMMatrixX<prec>(array.data(), array.shape()[0], array.shape()[1]);
  }

  template <size_t N>
  void make_hermitian(ndarray::ndarray<std::complex<double>, N>& X) {
    // check that two innermost dimensions form a matrix
    assert(X.shape()[N - 1] == X.shape()[N - 2]);
    // Dimension of the rest of arrays
    size_t dim1 = std::accumulate(X.shape().begin(), X.shape().end() - 2, 1ul, std::multiplies<size_t>());
    size_t nao  = X.shape()[N - 1];
    for (size_t i = 0; i < dim1; ++i) {
      MMatrixXcd Xm(X.data() + i * nao * nao, nao, nao);
      Xm = 0.5 * (Xm + Xm.conjugate().transpose().eval());
    }
  }

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
    p.define<mixing_type>("mixing_type", "Type of iteration convergence mixing. We use no mixing by default", NO_MIXING);
    p.define<double>("damping",
                     "Simple mixing paramters between current ad previous iteration. Should be between 0 and 1: 0 - no damping, "
                     "1 - full damping.",
                     0.5);
    p.define<std::string>("results_file,", "File to store results", "sim.h5");
    p.define<std::string>("input_file,", "File with input data", "input.h5");
    p.define<bool>("restart", "Try to restart simulation from the previously obtained results", false);
    p.define<unsigned>("itermax", "Maximum number of iterations", 1);
    p.define<bool>("const_density", "Maintain constant number of electrons through iterations", true);
    p.define<double>("E_thr", "Energy convergence threshold", 1e-9);
    p.define<double>("E_thr_sp", "Energy convergence threshold if run in single precision", 1e-5);
  }
}  // namespace green::sc
#endif  // SC_COMMON_DEFS_H
