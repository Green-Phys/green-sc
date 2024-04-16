/*
 * Copyright (c) 2024 University of Michigan
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

#ifndef GREEN_SC_FOCK_SIGMA_H
#define GREEN_SC_FOCK_SIGMA_H

#include <Eigen/Dense>

#include "common_defs.h"
#include "common_utils.h"

namespace green::opt {

  template <typename prec>
  using MMatrixX = Eigen::Map<Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  /**
   *
   * \brief fock_sigma
   *
   * The class provides a vector definition for combined Fock matrix and Self-energy.
   *
   */
  template <typename S1, typename St>
  class fock_sigma {
  private:
    S1 _m_Fock;
    St _m_Sigma;

  public:
    fock_sigma() = default;
    fock_sigma(const fock_sigma& rhs) :
        _m_Fock(sc::internal::init_data(rhs._m_Fock)), _m_Sigma(sc::internal::init_data(rhs._m_Sigma)) {
      sc::internal::assign(_m_Fock, rhs._m_Fock);
      sc::internal::assign(_m_Sigma, rhs._m_Sigma);
    }
    fock_sigma(const S1& fock, const St& sigma) :
        _m_Fock(sc::internal::init_data(fock)), _m_Sigma(sc::internal::init_data(sigma)) {
      sc::internal::assign(_m_Fock, fock);
      sc::internal::assign(_m_Sigma, sigma);
    }

    /**
     * Assign Fock-Sigma pair
     * @param rhs right-hand side
     * @return updated Fock-Sigma pair
     */
    fock_sigma& operator=(const fock_sigma& rhs) {
      sc::internal::assign(_m_Fock, rhs._m_Fock);
      sc::internal::assign(_m_Sigma, rhs._m_Sigma);
      return *this;
    }

    S1&  get_fock() { return _m_Fock; }
    St&  get_sigma() { return _m_Sigma; }
    void set_fock(S1& F_) { sc::internal::assign(_m_Fock, F_); }
    void set_sigma(St& S_) { sc::internal::assign(_m_Sigma, S_); }
    void set_fock_sigma(S1& F_, St& S_) {
      set_fock(F_);
      set_sigma(S_);
    }
    const S1&             get_fock() const { return _m_Fock; }
    const St&             get_sigma() const { return _m_Sigma; }

    std::complex<double>* get_fock_data() { return sc::internal::get_ref(_m_Fock); }
    std::complex<double>* get_sigma_data() { return sc::internal::get_ref(_m_Sigma); }

    void                  set_zero() {
      sc::internal::set_zero(_m_Fock);
      sc::internal::set_zero(_m_Sigma);
    }

    template <typename T>
    fock_sigma operator*=(T c) {
      using namespace sc::internal;
      _m_Fock *= c;
      _m_Sigma *= c;
      return *this;
    }

    fock_sigma operator+=(const fock_sigma& vec) {
      using namespace sc::internal;
      _m_Fock += vec.get_fock();
      _m_Sigma += vec.get_sigma();
      return *this;
    }

    fock_sigma operator+=(fock_sigma&& vec) {
      using namespace sc::internal;
      _m_Fock += vec.get_fock();
      _m_Sigma += vec.get_sigma();
      return *this;
    }
  };

  template <typename prec, typename = std::enable_if_t<std::is_same_v<prec, std::remove_const_t<prec>>>>
  auto matrix(ndarray::ndarray<prec, 2>&& array) {
    return MMatrixX<prec>(array.data(), array.shape()[0], array.shape()[1]);
  }

  template <typename T, typename C>
  void add(T& res, const T& b, C c) {
    res = b;
    res *= c;
  }

  template <typename T, typename C>
  void add(T& res, const T& a, const T& b, C c) {
    res = b;
    res *= c;
    res += a;
  }

}  // namespace green::opt

#endif  // GREEN_SC_FOCK_SIGMA_H
