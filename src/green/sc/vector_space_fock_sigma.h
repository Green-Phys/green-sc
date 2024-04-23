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
#include "except.h"
#include "fock_sigma.h"

namespace green::opt {

  template <size_t N>
  using ztensor = sc::ztensor<N>;

  /** \brief Vector space implementation
   *
   *  The class stores the subspace on disk in HDF5 format.
   *  Publicly available functionality: access to vectors,
   *  addition to the vector space, removal from the vector space,
   *  evaluation of Euclidean overlaps (without any affine metrics/preconditioners),
   *  evaluation of a linear combination of vectors.
   */
  template <typename S1, typename St>
  class vector_space_fock_sigma {
  private:
    size_t                            _m_size;
    size_t                            _diis_size;
    std::string                       _m_dbase;  // Name of the file where the vectors will be saved
    std::string                       _vecname;  // Name of the vector to be saved

    std::weak_ptr<fock_sigma<S1, St>> _vec_i;
    std::weak_ptr<fock_sigma<S1, St>> _vec_j;

    /**
     * @brief Read and return vector from i-th position in the database file
     * @param i   - index of a vector to be read in the database
     */
    const fock_sigma<S1, St>& read_from_dbase(const size_t i) {
      fock_sigma<S1, St>& obj = *_vec_i.lock();
      read_from_dbase(i, obj);
      return obj;
    }

    /**
     * @brief Read vector from i-th position in the database file and write result into a given object
     * @param i   - index of a vector to be read in the database
     * @param res - object to read data into
     */
    void read_from_dbase(const size_t i, fock_sigma<S1, St>& res) {
      h5pp::archive vsp_ar(_m_dbase, "r");
      sc::internal::read(res.get_fock(), _vecname + "/vec" + std::to_string(i) + "/Fock/data", vsp_ar);
      sc::internal::read(res.get_sigma(), _vecname + "/vec" + std::to_string(i) + "/Selfenergy/data", vsp_ar);
      vsp_ar.close();
    }

    /**
     * Write a vector into a database. Here we (over)write garbage place and move it to the last vector position
     */
    void write_to_dbase(const fock_sigma<S1, St>& Vec) {
      h5pp::archive vsp_ar(_m_dbase, "a");
      // Overwrite garbage
      sc::internal::write(Vec.get_fock(), _vecname + "/garbage" + "/Fock/data", vsp_ar);
      sc::internal::write(Vec.get_sigma(), _vecname + "/garbage" + "/Selfenergy/data", vsp_ar);
      // Move the link to the last position
      std::string p   = "/" + _vecname;
      std::string dst = p + "/vec" + std::to_string(_m_size);
      std::string src = p + "/garbage";
      vsp_ar[p].move(src, dst);  // will throw if dst exists
      vsp_ar.close();
    }

  public:
    /**
     * assign data to weak pointers
     * @param vec_i
     * @param vec_j
     */
    void init(std::shared_ptr<fock_sigma<S1, St>>& vec_i, std::shared_ptr<fock_sigma<S1, St>>& vec_j) {
      _vec_i = vec_i;
      _vec_j = vec_j;
    }

    /**
     * release weak pointers
     */
    void reset() {
      _vec_i.reset();
      _vec_j.reset();
    }

    /**
     * Create Fock-Sigma virtual space with specific database file name
     *
     * @param db database file name
     * @param diis_size - size of virtual space
     * @param vecname - name of the HDF5 group in the database file
     */
    explicit vector_space_fock_sigma(std::string db, size_t diis_size, std::string vecname = "FockSelfenergy") :
        _m_size(0), _diis_size(diis_size), _m_dbase(std::move(db)), _vecname(std::move(vecname)) {}

    /**
     * Try to restore vector subspace state
     */
    bool restore() {
      if (!std::filesystem::exists(_m_dbase)) {
        return true;
      }
      h5pp::archive ar(_m_dbase, "r");
      // empty archive
      if (!ar.is_data(_vecname + "/m_size") || !ar.is_data(_vecname + "/diis_size")) {
        return true;
      }
      size_t m_size, diis_size;
      ar[_vecname + "/m_size"] >> m_size;
      ar[_vecname + "/diis_size"] >> diis_size;
      ar.close();
      if (diis_size == _diis_size) {
        _m_size = m_size;
        return true;
      }
      return false;
    }

    /**
     * update indices values
     */
    void update_indices() {
      if (!utils::context.global_rank) {
        h5pp::archive vsp_ar(_m_dbase, "a");
        vsp_ar[_vecname + "/m_size"] << _m_size;
        vsp_ar[_vecname + "/diis_size"] << _diis_size;
        vsp_ar.close();
      }
      MPI_Barrier(utils::context.global);
    }

    /**
     * Read and return vector from database at a given index
     *
     * @param i vector index
     * @return vector at index i
     */
    const fock_sigma<S1, St>& get(size_t i) {
      if (i >= _m_size) {
        throw sc::sc_diis_vsp_error("Vector index of the VSpace container is out of bounds");
      }
      return read_from_dbase(i);
    }

    /**
     * Read a vector from database at a given index and write it into a given object
     *
     * @param i vector index
     * @param r object to write data into
     */
    void get(size_t i, fock_sigma<S1, St>& r) {
      if (i >= _m_size) {
        throw sc::sc_diis_vsp_error("Vector index of the VSpace container is out of bounds");
      }
      return read_from_dbase(i, r);
    }

    /**
     * Add vector to the subspace overwriting garbage
     *
     * @param vec - vector to be added into a vector space
     */
    void add(const fock_sigma<S1, St>& vec) {
      // we don't have automatic purge therefore we have to make sure that vector space
      // does not grow above maximum capacity
      if (_m_size == _diis_size) throw sc::sc_diis_vsp_error("VSpace is at it's maximum capacity");
      if (!utils::context.global_rank) write_to_dbase(vec);
      MPI_Barrier(utils::context.global);
      _m_size++;
      update_indices();
    }

    /**
     * Compute Euclidean overlap between an i-th vector from vector subspace and a given right-side vector
     *
     * @param i - index of left-side vector in the subspace
     * @param vec_j - given right side vector
     * @return Euclidean overlap between i-th and a given right side vector
     */
    [[nodiscard]] std::complex<double> overlap(const size_t i, const fock_sigma<S1, St>& vec_j) {
      if (_vec_i.expired()) {
        throw sc::sc_diis_vsp_error("Uninitialized shared memory");
      }
      fock_sigma<S1, St>& vec_i = *_vec_i.lock();
      get(i, vec_i);
      return overlap(vec_i, vec_j);
    }

    /**
     * Compute Euclidean overlap between an i-th and j-th vectors from vector subspace
     *
     * @param i index of left vector
     * @param j index of right vector
     * @return Euclidean overlap between i-th and j-th vector
     */
    [[nodiscard]] std::complex<double> overlap(const size_t i, const size_t j) {
      if (_vec_i.expired() || _vec_j.expired()) {
        throw sc::sc_diis_vsp_error("Uninitialized shared memory");
      }
      fock_sigma<S1, St>& vec_i = *_vec_i.lock();
      fock_sigma<S1, St>& vec_j = *_vec_j.lock();
      get(i, vec_i);
      get(j, vec_j);
      return overlap(vec_i, vec_j);
    }

    /**
     * Compute Euclidean overlap between given two vectors
     *
     * @param vec_i left vector
     * @param vec_j right vector
     * @return Euclidean overlap
     */
    [[nodiscard]] std::complex<double> overlap(const fock_sigma<S1, St>& vec_i, const fock_sigma<S1, St>& vec_j) const {
      return sc::internal::overlap(vec_i.get_fock(), vec_j.get_fock()) +
             sc::internal::overlap(vec_i.get_sigma(), vec_j.get_sigma());
    }

    [[nodiscard]] size_t size() const { return _m_size; };

    /**
     * Move the link of vector i to garbage and shift all other links
     * No physical data is moved.
     *
     * @param i - index of a vector to be purged
     */
    void purge(const size_t i = 0) {
      if (_m_size == 0) {
        throw sc::sc_diis_vsp_error("VSpace container is of zero size, no vectors can be deleted");
      }
      if (i >= _m_size) {
        throw sc::sc_diis_vsp_error("Vector index of the VSpace container is out of bounds");
      }
      if (!utils::context.global_rank) {
        h5pp::archive vsp_ar(_m_dbase, "a");
        std::string   p   = "/" + _vecname;
        std::string   src = p + "/vec" + std::to_string(i);
        std::string   dst = p + "/garbage";
        vsp_ar[p].move(src, dst);
        for (size_t j = i + 1; j < _m_size; j++) {
          src = p + "/vec" + std::to_string(j);
          dst = p + "/vec" + std::to_string(j - 1);
          vsp_ar[p].move(src, dst);
        }
        vsp_ar.close();
      }
      MPI_Barrier(utils::context.global);
      _m_size--;
      update_indices();
    }

    void make_linear_comb(const Eigen::VectorXcd& C, fock_sigma<S1, St>& r) {
      // get(size() - 1, r);  // this is needed to initialize r
      r.set_zero();
      for (size_t i = 0; i < _m_size && i < C.size(); i++) {
        fock_sigma<S1, St>& vec_i = *_vec_i.lock();
        get(size() - 1 - i, vec_i);
        double coeff = C(C.size() - 1 - i).real();
        vec_i *= coeff;
        r += vec_i;
      }
    }
  };

}  // namespace green::opt
#endif  // GREEN_OPT_VECTOR_SPACE_FOCK_SIGMA
