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
#ifndef GREEN_SC_MIXING_H
#define GREEN_SC_MIXING_H

#include <green/opt/diis_alg.h>
#include <green/params/params.h>
#include <green/utils/mpi_shared.h>

#include <memory>

#include "common_defs.h"
#include "common_utils.h"
#include "except.h"
#include "residuals.h"
#include "vector_space_fock_sigma.h"

using namespace std::string_literals;

namespace green::sc {
  /**
   * Base interface for iteration mixing
   *
   * @tparam G - type of the Green's function obejct
   * @tparam S1 - type of static self-energy
   * @tparam St - type of dynamic self-energy
   */
  template <typename G, typename S1, typename St>
  class base_mixing {
  public:
    /**
     * Method that mix result from current iteration with results from previous iterations.
     * Various strategies can be implemented, such simple damping on Green's function or self-energy,
     * DIIS and so on.
     *
     * @param iter - current iteration
     * @param mu - chemical potential
     * @param ovlp - overlap matrix
     * @param g - Green's function for the current iteration
     * @param s1 - static self-energy for the current iteration
     * @param s_t - dynamic self-energy for the current iteration
     */
    virtual void update(size_t iter, double mu, const S1& h0, const S1& ovlp, G& g, S1& s1, St& s_t) = 0;

    virtual void print_name()                                                                        = 0;

    virtual ~base_mixing() {}
  };

  template <typename G, typename S1, typename St>
  class no_mixing : public base_mixing<G, S1, St> {
  public:
    void update(size_t, double, const S1&, const S1&, G&, S1&, St&) override {};

    void print_name() override {
      if (utils::context.global_rank == 0) std::cout << "No mixing strategy will be applied" << std::endl;
    }
  };

  /**
   *
   * @tparam G
   * @tparam S1
   * @tparam St
   */
  template <typename G, typename S1, typename St>
  class g_mixing : public base_mixing<G, S1, St> {
  public:
    g_mixing(double m, const std::string& res) : _mixing_weight(m), _results_file(res) {
      if (m <= 0.0 || m >= 2.0) {
        throw sc_incorrect_mixing_error("Mixing should be in (0,2) interval");
      }
      if (m > 1) {
        if (utils::context.global_rank == 0)
          std::cout << "Mixing parameter is set to be bigger than one, in the over-relaxation regime convergence potentially can "
                       "be unstable."
                    << std::endl;
      }
    }
    void update(size_t iter, double, const S1&, const S1&, G& g, S1&, St&) override {
      if (iter == 1) {
        return;
      }
      G g_tmp(internal::init_data(g));
      internal::read_data(g_tmp, _results_file, "iter" + std::to_string(iter - 1) + "/G_tau/data");
      internal::update(g, g_tmp, _mixing_weight);
      internal::cleanup_data(g_tmp);
    };

    void print_name() override {
      if (utils::context.global_rank == 0)
        std::cout << "Green's function mixing strategy will be applied: " << _mixing_weight << "*G_new + " << (1 - _mixing_weight)
                  << "*G_old" << std::endl;
    }

  private:
    double      _mixing_weight;
    std::string _results_file;
  };

  template <typename G, typename S1, typename St>
  class sigma_mixing : public base_mixing<G, S1, St> {
  public:
    sigma_mixing(double m, const std::string& res) : _mixing_weight(m), _results_file(res) {
      if (m <= 0.0 || m >= 2.0) {
        throw sc_incorrect_mixing_error("Mixing should be in (0,2) interval");
      }
      if (m > 1) {
        if (utils::context.global_rank == 0)
          std::cout << "Mixing parameter is set to be bigger than one, in the over-relaxation regime convergence potentially can "
                       "be unstable."
                    << std::endl;
      }
    }
    void update(size_t iter, double, const S1&, const S1&, G&, S1& s1, St& s_t) override {
      if (iter == 1) {
        return;
      }
      St st_tmp(internal::init_data(s_t));
      internal::read_data(st_tmp, _results_file, "iter" + std::to_string(iter - 1) + "/Selfenergy/data");
      internal::update(s_t, st_tmp, _mixing_weight);
      internal::cleanup_data(st_tmp);
      S1 s1_tmp(internal::init_data(s1));
      internal::read_data(s1_tmp, _results_file, "iter" + std::to_string(iter - 1) + "/Sigma1");
      internal::update(s1, s1_tmp, _mixing_weight);
      internal::cleanup_data(s1_tmp);
    };

    void print_name() override {
      if (utils::context.global_rank == 0)
        std::cout << "Self-energy mixing strategy will be applied: " << _mixing_weight << "*Sigma_new + " << (1 - _mixing_weight)
                  << "*Sigma_old" << std::endl;
    }

  private:
    double      _mixing_weight;
    std::string _results_file;
  };

  /**
   *
   * @tparam G
   * @tparam S1
   * @tparam St
   */
  template <typename G, typename S1, typename St>
  class diis : public sigma_mixing<G, S1, St> {
    using vec_t       = opt::fock_sigma<S1, St>;
    using problem_t   = opt::shared_optimization_problem<vec_t>;
    using vec_space_t = opt::vector_space_fock_sigma<S1, St>;
    using residual_t  = std::function<void(vec_space_t&, problem_t&, vec_t&)>;

  public:
    diis(const params::params& p, bool commutator) :
        sigma_mixing<G, S1, St>(p["mixing_weight"], p["results_file"]), _mixing(p["mixing_weight"]), _results_file(p["results_file"]),
        _diis_file(p["diis_file"]), _diis_start(p["diis_start"]), _diis_size(p["diis_size"]),
        _diis(_diis_start, _diis_size, p["verbose"]), _x_vsp(_diis_file, _diis_size + 1),
        _res_vsp(_diis_file, _diis_size, "residuals"), _ft(p), _commutator(commutator) {
      if (!p["restart"].as<bool>() || !p["diis_restart"].as<bool>()) {
        MPI_Barrier(utils::context.global);
        if (!utils::context.global_rank) std::filesystem::remove(_diis_file);
        MPI_Barrier(utils::context.global);
        _need_reinit = false;
        return;
      }
      if (!_res_vsp.restore() || !_x_vsp.restore()) {
        MPI_Barrier(utils::context.global);
        if (!utils::context.global_rank) std::filesystem::remove(_diis_file);
        MPI_Barrier(utils::context.global);
        _need_reinit = false;
        return;
      }
      _need_reinit = true;
    }

    void update(size_t iter, double mu, const S1& h0, const S1& ovlp, G& g, S1& s1, St& s_t) override {
      vec_t vec(s1, s_t);
      vec_t res(s1, s_t);
      auto  vec_i = std::make_shared<vec_t>(s1, s_t);
      auto  vec_j = std::make_shared<vec_t>(s1, s_t);
      auto  res_i = vec_i;
      auto  res_j = vec_j;
      _x_vsp.init(vec_i, vec_j);
      _res_vsp.init(res_i, res_j);
      if(_need_reinit) {
        _diis.reinit(_res_vsp);
        _need_reinit = false;
      }
      problem_t  problem(vec);
      residual_t residual;
      if (_commutator) {
        residual = [this, &g, &mu, &h0, &ovlp, &s1, &s_t](vec_space_t&, problem_t& problem, vec_t& res) {
          vec_t& x_last = problem.x();
          G      C_t(internal::init_data(g));
          S1     Fz(internal::init_data(s1));
          opt::commutator_t(_ft, C_t, g, x_last, mu, h0, ovlp);
          internal::set_zero(Fz);
          res.set_fock_sigma(Fz, C_t);
        };
      } else {
        residual = [this, &s1, &s_t](vec_space_t& x_vsp, problem_t& problem, vec_t& res) {
          const vec_t& last = x_vsp.get(x_vsp.size() - 1);
          add(res, problem.x(), last, -1.0);
        };
      }
      if (iter == 1) {
        _diis.next_step(vec, res, _x_vsp, _res_vsp, residual, problem);
        _x_vsp.reset();
        _res_vsp.reset();
        return;
      }
      if (iter - 1 <= _diis_start) {
        sigma_mixing<G, S1, St>::update(iter, mu, h0, ovlp, g, s1, s_t);
        internal::assign(problem.x().get_fock(), s1);
        internal::assign(problem.x().get_sigma(), s_t);
      }
      _diis.next_step(vec, res, _x_vsp, _res_vsp, residual, problem);
      if (iter - 1 > _diis_start) {
        internal::assign(s1, problem.x().get_fock());
        internal::assign(s_t, problem.x().get_sigma());
      }
      // release shared pointer
      _x_vsp.reset();
      _res_vsp.reset();
    };

    void print_name() override {
      if (utils::context.global_rank == 0) std::cout << "DDIIS/CDIIS mixing strategy will be applied" << std::endl;
    };

  private:
    double               _mixing;
    std::string          _results_file;
    std::string          _diis_file;
    size_t               _diis_start;
    size_t               _diis_size;
    opt::diis_alg<vec_t> _diis;
    vec_space_t          _x_vsp;
    vec_space_t          _res_vsp;
    grids::transformer_t _ft;
    bool                 _commutator;
    bool                 _need_reinit;
    residual_t           _residual;
  };

  /**
   * Class that decides what type of iteration mixing should be performed based on selected parameters
   *
   * @tparam G - type of the Green's function obejct
   * @tparam S1 - type of static self-energy
   * @tparam St - type of dynamic self-energy
   */
  template <typename G, typename S1, typename St>
  class mixing_strategy {
  public:
    explicit mixing_strategy(const params::params& p) : _mixing(nullptr), _verbose(p["verbose"]) {
      if (p.is_set("damping")) {
        throw sc_incorrect_mixing_error(
            "Parameter `--damping` was provided, please use `--mixing_weight` parameter instead. "
            "We use the following definition: X_n = a X_n + (1-a) X_{n-1}, where `a` is set by `--mixing_weight`.");
      }
      switch (p["mixing_type"].as<mixing_type>()) {
        case NO_MIXING:
          _mixing = std::make_unique<no_mixing<G, S1, St>>();
          break;
        case G_MIXING:
          _mixing = std::make_unique<g_mixing<G, S1, St>>(p["mixing_weight"], p["results_file"]);
          break;
        case SIGMA_MIXING:
          _mixing = std::make_unique<sigma_mixing<G, S1, St>>(p["mixing_weight"], p["results_file"]);
          break;
        case DIIS:
          _mixing = std::make_unique<diis<G, S1, St>>(p, false);
          break;
        case CDIIS:
          _mixing = std::make_unique<diis<G, S1, St>>(p, true);
          break;
      }
    }

    /**
     * Based on the chosen strategy mix result from current iteration with results from previous iterations.
     * @param iter - current iteration
     * @param mu - chemical potential
     * @param h0 - non-interacting hamiltonian
     * @param ovlp - overlap matrix
     * @param g - Green's function for the current iteration
     * @param s1 - static self-energy for the current iteration
     * @param s_t - dynamic self-energy for the current iteration
     */
    void update(size_t iter, double mu, const S1& h0, const S1& ovlp, G& g, S1& s1, St& s_t) const {
      if (_verbose > 0) _mixing->print_name();
      _mixing->update(iter, mu, h0, ovlp, g, s1, s_t);
    }

   
    /**
     * Scaling of the dynamical part of self-energy
     * @param s_t - dynamic self-energy for the current iteration
     * @param scaling_parameter - magnitude of scaling
     */
    virtual void sigma_scale(St& s_t, double scaling_parameter)  {
        if(scaling_parameter != 1.0) {
            internal::scale(s_t, scaling_parameter);
        }
    }

  private:
    std::unique_ptr<base_mixing<G, S1, St>> _mixing;
    int                                     _verbose;
  };

}  // namespace green::sc

#endif  // GREEN_SC_MIXING_H
