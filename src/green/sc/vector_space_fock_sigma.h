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

#ifndef GREEN_OPT_VECTOR_SPACE_FOCK_SIGMA
#define GREEN_OPT_VECTOR_SPACE_FOCK_SIGMA

#include <green/grids/transformer_t.h>
#include <green/h5pp/archive.h>

#include <Eigen/Dense>
#include <complex>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "common_defs.h"
#include "common_utils.h"

namespace green::opt {

  template <size_t N>
  using ztensor = sc::ztensor<N>;

  template <typename prec>
  using MMatrixX    = Eigen::Map<Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using MMatrixXcd  = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using CMMatrixXcd = Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  /**
   *
   * \brief FockSigma
   *
   * The class provides a vector definition for combined Fock matrix and Self-energy.
   *
   */
  template <typename S1, typename St>
  class FockSigma {
  private:
    S1 _m_Fock;
    St _m_Sigma;

  public:
    FockSigma() = default;
    FockSigma(const FockSigma& rhs) :
        _m_Fock(sc::internal::init_data(rhs._m_Fock)), _m_Sigma(sc::internal::init_data(rhs._m_Sigma)) {
      sc::internal::assign(_m_Fock, rhs._m_Fock);
      sc::internal::assign(_m_Sigma, rhs._m_Sigma);
    }
    FockSigma(const S1& fock, const St& sigma) :
        _m_Fock(sc::internal::init_data(fock)), _m_Sigma(sc::internal::init_data(sigma)) {
      sc::internal::assign(_m_Fock, fock);
      sc::internal::assign(_m_Sigma, sigma);
    }

    /**
     * Assign Fock-Sigma pair
     * @param rhs right-hand side
     * @return updated Fock-Sigma pair
     */
    FockSigma& operator=(const FockSigma& rhs) {
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
    FockSigma operator*=(T c) {
      using namespace sc::internal;
      _m_Fock *= c;
      _m_Sigma *= c;
      return *this;
    }

    FockSigma operator+=(const FockSigma& vec) {
      using namespace sc::internal;
      _m_Fock += vec.get_fock();
      _m_Sigma += vec.get_sigma();
      return *this;
    }

    FockSigma operator+=(FockSigma&& vec) {
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

  /** \brief Vector space implementation
   *
   *  This template specialization considers cases when the Vector type is
   *  ztensor<4> (Fock matrix), ztensor<5> (Self-energy), or FockSigma (both).
   *
   *  The class stores the subspace on disk in HDF5 format.
   *  Publickly available functionality: access to vectors,
   *  addition to the vector space, removal from the vector space,
   *  evaluation of Euclidian overlaps (without any affine metrics/preconditioners),
   *  evaluation of a linear combination of vectors.
   *
   *  TODO: MPI parallelization; "move" operation can be done MUCH more efficiently with HDF5.
   */
  template <typename S1, typename St>
  class VSpaceFockSigma {
  private:
    size_t                           _m_size;
    std::string                      _m_dbase;  // Name of the file where the vectors will be saved
    std::string                      _vecname;  // Name of the vector to be saved

    std::weak_ptr<FockSigma<S1, St>> _vec_i;
    std::weak_ptr<FockSigma<S1, St>> _vec_j;

    /** \brief Read vector \f[i\f] from file
     *  \param i
     */
    const FockSigma<S1, St>& read_from_dbase(const size_t i) {
      FockSigma<S1, St>& obj = *_vec_i.lock();
      read_from_dbase(i, obj);
      return obj;
    }

    void read_from_dbase(const size_t i, FockSigma<S1, St>& res) {
      h5pp::archive vsp_ar(_m_dbase, "r");
      sc::internal::read(res.get_fock(), _vecname + "/vec" + std::to_string(i) + "/" + "Fock" + "/data", vsp_ar);
      sc::internal::read(res.get_sigma(), _vecname + "/vec" + std::to_string(i) + "/" + "Selfenergy" + "/data", vsp_ar);
      vsp_ar.close();
    }

    /** \brief Write vector to position \f[i\f] in the file
     *  \param i
     * **/
    void write_to_dbase(const size_t i, const FockSigma<S1, St>& Vec) {
      h5pp::archive vsp_ar(_m_dbase, "a");
      sc::internal::write(Vec.get_fock(), _vecname + "/vec" + std::to_string(i) + "/Fock/data", vsp_ar);
      sc::internal::write(Vec.get_sigma(), _vecname + "/vec" + std::to_string(i) + "/Selfenergy/data", vsp_ar);
      vsp_ar.close();
    }

  public:
    void init(std::shared_ptr<FockSigma<S1, St>>& vec_i, std::shared_ptr<FockSigma<S1, St>>& vec_j) {
      _vec_i = vec_i;
      _vec_j = vec_j;
    }

    void reset() {
      _vec_i.reset();
      _vec_j.reset();
    }

    /**
     * Create Fock-Sigma virtual space with specific database file name
     *
     * @param db database file name
     */
    explicit VSpaceFockSigma(std::string db, std::string vecname = "FockSelfenergy") :
        _m_dbase(std::move(db)), _m_size(0), _vecname(std::move(vecname)) {}

    /**
     * Read and return vector from database at a given index
     *
     * @param i vector index
     * @return vector at index i
     */
    const FockSigma<S1, St>& get(size_t i) {
      if (i >= _m_size) {
        throw std::runtime_error("Vector index of the VSpace container is out of bounds");
      }
      return read_from_dbase(i);
    }

    void get(size_t i, FockSigma<S1, St>& r) {
      if (i >= _m_size) {
        throw std::runtime_error("Vector index of the VSpace container is out of bounds");
      }
      return read_from_dbase(i, r);
    }

    void add(const FockSigma<S1, St>& Vec) {
      write_to_dbase(_m_size, Vec);
      _m_size++;
    }

    [[nodiscard]] std::complex<double> overlap(const size_t i, const FockSigma<S1, St>& vec_j) {
      FockSigma<S1, St>& vec_i = *_vec_i.lock();
      get(i, vec_i);
      return sc::internal::overlap(vec_i.get_fock(), vec_j.get_fock()) +
             sc::internal::overlap(vec_i.get_sigma(), vec_j.get_sigma());
    }

    [[nodiscard]] std::complex<double> overlap(const size_t i, const size_t j) {
      FockSigma<S1, St>& vec_i = *_vec_i.lock();
      FockSigma<S1, St>& vec_j = *_vec_j.lock();
      get(i, vec_i);
      get(j, vec_j);
      return sc::internal::overlap(vec_i.get_fock(), vec_j.get_fock()) +
             sc::internal::overlap(vec_i.get_sigma(), vec_j.get_sigma());
    }

    [[nodiscard]] std::complex<double> overlap(const FockSigma<S1, St>& Vec_v, const FockSigma<S1, St>& Vec_u) {
      return sc::internal::overlap(Vec_v.get_fock(), Vec_u.get_fock()) +
             sc::internal::overlap(Vec_v.get_sigma(), Vec_u.get_sigma());
    }

    [[nodiscard]] size_t size() const { return _m_size; };

    // TODO implement "move" operation in the ALPS
    void purge(const size_t i) {
      if (i >= _m_size) {
        throw std::runtime_error("Vector index of the VSpace container is out of bounds");
      }
      if (_m_size == 0) {
        throw std::runtime_error("VSpace container is of zero size, no vectors can be deleted");
      }
      h5pp::archive vsp_ar(_m_dbase, "a");
      for (size_t j = i + 1; j < size(); j++) {
        vsp_ar.move("vec" + std::to_string(j), "vec" + std::to_string(j - 1));
      }
      vsp_ar.close();

      _m_size--;
    }

    void make_linear_comb(const Eigen::VectorXcd& C, FockSigma<S1, St>& r) {
      // get(size() - 1, r);  // this is needed to initialize r
      r.set_zero();
      for (size_t i = 0; i < _m_size && i < C.size(); i++) {
        FockSigma<S1, St>& vec_i = *_vec_i.lock();
        get(size() - 1 - i, vec_i);
        double coeff = C(C.size() - 1 - i).real();
        vec_i *= coeff;
        r += vec_i;
      }
    }
  };

}  // namespace green::opt
#endif  // GREEN_OPT_VECTOR_SPACE_FOCK_SIGMA
