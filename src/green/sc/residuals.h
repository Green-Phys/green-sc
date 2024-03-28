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

#ifndef GREEN_SC_RESIDUALS_H
#define GREEN_SC_RESIDUALS_H

#include <green/grids/transformer_t.h>
#include <green/utils/mpi_shared.h>

#include "vector_space_fock_sigma.h"

namespace green::opt {

  template <typename vec_t>
  class shared_optimization_problem {
  private:
    vec_t _vec;

  public:
    explicit shared_optimization_problem(vec_t& vec) : _vec(vec) {}

    vec_t&                     x() { return _vec; }
    [[nodiscard]] const vec_t& x() const { return _vec; }
  };

  template <>
  class shared_optimization_problem<FockSigma<ztensor<4>, utils::shared_object<ztensor<5>>>> {
  private:
    using vec_t = FockSigma<ztensor<4>, utils::shared_object<ztensor<5>>>;
    vec_t& _vec;

  public:
    explicit                   shared_optimization_problem(vec_t& vec) : _vec(vec) {}

    vec_t&                     x() { return _vec; }
    [[nodiscard]] const vec_t& x() const { return _vec; }
  };

  template <typename G, typename S1, typename St>
  void commutator_t(const grids::transformer_t& ft, St& C_t, G& G_t, FockSigma<S1, St>& FS_t, double mu, const S1& H,
                    const S1& S) { /*do nothing, used for the test purpose*/
  }

  /**
   * For a given Green's Function and Self-energy evaluate commutator [G, G^{-1}].
   * With G^{-1} evaluated as G^{-1} = (i w_n + mu) S - H - S1 - Sigma(w)
   *
   * @param ft - Fourier transform util
   * @param C_t - [OUT] ndarray to store the result
   * @param G_t - Green's function in time domain
   * @param S1 - static part of the self energy
   * @param S_t - dynamic part of the self energy
   * @param mu - chemical potential
   * @param H0 - non-interacting Hamiltonian
   * @param S - overlap matrix
   */
  inline void commutator_core(const grids::transformer_t& ft, ztensor<5>& C_t, ztensor<5>& G_t, ztensor<4> S1, ztensor<5>& S_t,
                              double mu, const ztensor<4>& H0, const ztensor<4>& S) {
    size_t     nts = G_t.shape()[0];
    size_t     ns  = G_t.shape()[1];
    size_t     nk  = G_t.shape()[2];
    size_t     nao = G_t.shape()[4];
    size_t     nw  = ft.wsample_fermi().size();

    ztensor<2> I(nao, nao);
    ztensor<3> Sigma_w(nw, nao, nao);
    ztensor<3> Sigma_k(nts, nao, nao);
    ztensor<3> G_w(nw, nao, nao);
    ztensor<3> C_w(nw, nao, nao);
    ztensor<3> C_t_slice(nts, nao, nao);
    ztensor<3> G_t_slice(nts, nao, nao);
    // PP: This one is needed if C_t is not allocated
    // (which I assume is the case since all the params,
    //  such as nts, ns, etc should not be known in the abstract classes)

    // k-points and spin are moved as an outer loop,
    // because in future it is possible to make it MPI-parallel
    for (size_t isk = utils::context.node_rank; isk < ns * nk; isk += utils::context.node_size) {
      size_t is = isk / nk;
      size_t ik = isk % nk;
      Sigma_k.set_zero();
      for (size_t it = 0; it < nts; ++it) matrix(Sigma_k(it)) = matrix(S_t(it, is, ik));
      ft.tau_to_omega(Sigma_k, Sigma_w, 1);
      for (size_t it = 0; it < nts; ++it) matrix(G_t_slice(it)) = matrix(G_t(it, is, ik));
      ft.tau_to_omega(G_t_slice, G_w, 1);
      for (size_t iw = 0; iw < nw; iw++) {
        // Take nao x nao matrices at certain omega, spin, and k-point
        std::complex<double> muomega = ft.omega(ft.sd().repn_fermi().wsample()(iw), 1) + mu;
        CMMatrixXcd          MO(S.data() + isk * nao * nao, nao, nao);
        CMMatrixXcd          MH(H0.data() + isk * nao * nao, nao, nao);
        MMatrixXcd           MI(I.data(), nao, nao);
        MMatrixXcd           MC(C_w.data() + iw * nao * nao, nao, nao);
        MMatrixXcd           MF(S1.data() + isk * nao * nao, nao, nao);
        MMatrixXcd           MS(Sigma_w.data() + iw * nao * nao, nao, nao);
        MMatrixXcd           MG(G_w.data() + iw * nao * nao, nao, nao);
        MI = muomega * MO - MH - MF - MS;
        MC = MG * MI - MI * MG;
      }
      ft.omega_to_tau(C_w, C_t_slice, 1);
      for (size_t it = 0; it < nts; ++it) matrix(C_t(it, is, ik)) = matrix(C_t_slice(it));
    }
  }

  /**
   * Evaluation of the commutator in the tau space between G and G_0^{-1} - Sigma
   *
   */
  template <>
  void commutator_t(const grids::transformer_t& ft, utils::shared_object<ztensor<5>>& C_t, utils::shared_object<ztensor<5>>& G_t,
                    FockSigma<ztensor<4>, utils::shared_object<ztensor<5>>>& FS_t, double mu, const ztensor<4>& H0,
                    const ztensor<4>& S) {
    auto& C_t_full = C_t.object();
    C_t.fence();
    commutator_core(ft, C_t_full, G_t.object(), FS_t.get_fock(), FS_t.get_sigma().object(), mu, H0, S);
    C_t.fence();
  }

  template <>
  void commutator_t(const grids::transformer_t& ft, ztensor<5>& C_t, ztensor<5>& G_t, FockSigma<ztensor<4>, ztensor<5>>& FS_t,
                    double mu, const ztensor<4>& H0, const ztensor<4>& S) {
    size_t       nao           = G_t.shape()[4];
    MPI_Datatype dt_matrix     = utils::create_matrix_datatype<std::complex<double>>(nao * nao);
    MPI_Op       matrix_sum_op = utils::create_matrix_operation<std::complex<double>>();
    commutator_core(ft, C_t, G_t, FS_t.get_fock(), FS_t.get_sigma(), mu, H0, S);
    utils::allreduce(MPI_IN_PLACE, C_t.data(), C_t.size() / (nao * nao), dt_matrix, matrix_sum_op, utils::context.internode_comm);
    MPI_Type_free(&dt_matrix);
    MPI_Op_free(&matrix_sum_op);
  }

}  // namespace green::opt

#endif  // GREEN_SC_RESIDUALS_H
