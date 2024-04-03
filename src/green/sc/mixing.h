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
#ifndef SC_MIXING_H
#define SC_MIXING_H

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
     * Various stratagise can be implemented, such simple damping on Green's function or self-energy,
     * DIIS and so on.
     *
     * @param iter - current relative iteration
     * @param mu - chemical potential
     * @param ovlp - overlap matrix
     * @param g - Green's function for the current iteration
     * @param s1 - static self-energy for the current iteration
     * @param s_t - dynamic self-energy for the current iteration
     */
    virtual void update(size_t iter, double mu, const S1& h0, const S1& ovlp, G& g, S1& s1, St& s_t) = 0;

    virtual void print_name() = 0;

    virtual ~    base_mixing() {}
  };

  template <typename G, typename S1, typename St>
  class no_mixing : public base_mixing<G, S1, St> {
  public:
    void update(size_t, double, const S1&, const S1&, G&, S1&, St&) override{};

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
  class g_damping : public base_mixing<G, S1, St> {
  public:
    g_damping(double d, const std::string& res) : _damping(d), _results_file(res) {
      if (d < 0.0 || d >= 1.0) {
        throw sc_incorrect_damping_error("Damping should be in [0,1) interval");
      }
    }
    void update(size_t iter, double, const S1&, const S1&, G& g, S1&, St&) override {
      if (iter == 0) {
        return;
      }
      G g_tmp(internal::init_data(g));
      internal::read_data(g_tmp, _results_file, "iter" + std::to_string(iter - 1) + "/G_tau/data");
      internal::update(g, g_tmp, _damping);
      internal::cleanup_data(g_tmp);
    };

    void print_name() override {
        if (utils::context.global_rank == 0) std::cout << "Green's function mixing strategy will be applied: " 
                                                       << _damping << "*G_old + " 
                                                       << (1-_damping) << "*G_new" << std::endl;
    }

  private:
    double      _damping;
    std::string _results_file;
  };

  template <typename G, typename S1, typename St>
  class sigma_damping : public base_mixing<G, S1, St> {
  public:
    sigma_damping(double d, const std::string& res) : _damping(d), _results_file(res) {
      if (d < 0.0 || d >= 1.0) {
        throw sc_incorrect_damping_error("Damping should be in [0,1) interval");
      }
    }
    void update(size_t iter, double, const S1&, const S1&, G&, S1& s1, St& s_t) override {
      if (iter == 0) {
        return;
      }
      St st_tmp(internal::init_data(s_t));
      internal::read_data(st_tmp, _results_file, "iter" + std::to_string(iter - 1) + "/Selfenergy/data");
      internal::update(s_t, st_tmp, _damping);
      internal::cleanup_data(st_tmp);
      S1 s1_tmp(internal::init_data(s1));
      internal::read_data(s1_tmp, _results_file, "iter" + std::to_string(iter - 1) + "/Sigma1");
      internal::update(s1, s1_tmp, _damping);
      internal::cleanup_data(s1_tmp);
    };

    void print_name() override {
        if (utils::context.global_rank == 0) std::cout << "Self-energy mixing strategy will be applied: " 
                                                       << _damping << "*Sigma_old + " 
                                                       << (1-_damping) << "*Sigma_new" << std::endl;
    }

  private:
    double      _damping;
    std::string _results_file;
  };

  /**
   *
   * @tparam G
   * @tparam S1
   * @tparam St
   */
  template <typename G, typename S1, typename St>
  class diis : public sigma_damping<G, S1, St> {
    using vec_t       = opt::FockSigma<S1, St>;
    using problem_t   = opt::shared_optimization_problem<vec_t>;
    using vec_space_t = opt::VSpaceFockSigma<S1, St>;
    using residual_t  = std::function<void(vec_space_t&, problem_t&, vec_t&)>;

  public:
    diis(const params::params& p, bool commutator) :
        sigma_damping<G, S1, St>(p["damping"], p["results_file"]), _damping(p["damping"]), _results_file(p["results_file"]),
        _diis_file(p["diis_file"]), _diis_start(p["diis_start"]), _diis_size(p["diis_size"]),
        _diis(_diis_start, _diis_size, p["verbose"]), _x_vsp(_diis_file, _diis_size),
        _res_vsp(_diis_file, _diis_size - 1, "residuals"), _ft(p), _commutator(commutator) {}

    void update(size_t iter, double mu, const S1& h0, const S1& ovlp, G& g, S1& s1, St& s_t) override {
      vec_t vec(s1, s_t);
      vec_t res(s1, s_t);
      auto  vec_i = std::make_shared<vec_t>(s1, s_t);
      auto  vec_j = std::make_shared<vec_t>(s1, s_t);
      auto  res_i = std::make_shared<vec_t>(s1, s_t);
      auto  res_j = std::make_shared<vec_t>(s1, s_t);
      _x_vsp.init(vec_i, vec_j);
      _res_vsp.init(res_i, res_j);
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
      if (iter == 0) {
        _diis.next_step(vec, res, _x_vsp, _res_vsp, residual, problem);
        return;
      }
      if (iter <= _diis_start) {
        sigma_damping<G, S1, St>::update(iter, mu, h0, ovlp, g, s1, s_t);
        internal::assign(problem.x().get_fock(), s1);
        internal::assign(problem.x().get_sigma(), s_t);
      }
      _diis.next_step(vec, res, _x_vsp, _res_vsp, residual, problem);
      if (iter > _diis_start) {
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
    double               _damping;
    std::string          _results_file;
    std::string          _diis_file;
    size_t               _diis_start;
    size_t               _diis_size;
    opt::diis_alg<vec_t> _diis;
    vec_space_t          _x_vsp;
    vec_space_t          _res_vsp;
    grids::transformer_t _ft;
    bool                 _commutator;
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
    explicit mixing_strategy(const params::params& p) {
      switch (mixing_type E = p["mixing_type"]) {
        case NO_MIXING:
          _mixing = std::make_unique<no_mixing<G, S1, St>>();
          break;
        case G_DAMPING:
          _mixing = std::make_unique<g_damping<G, S1, St>>(p["damping"], p["results_file"]);
          break;
        case SIGMA_DAMPING:
          _mixing = std::make_unique<sigma_damping<G, S1, St>>(p["damping"], p["results_file"]);
          break;
        case DIIS:
          _mixing = std::make_unique<diis<G, S1, St>>(p, false);
          break;
        case CDIIS:
          _mixing = std::make_unique<diis<G, S1, St>>(p, true);
          break;
        default:
          _mixing = std::make_unique<no_mixing<G, S1, St>>();
          break;
      }
    }

    /**
     * Based on the chosen strategy mix result from current iteration with results from previous iterations.
     * @param iter
     * @param iter - current relative iteration
     * @param mu - chemical potential
     * @param h0 - non-interacting hamiltonian
     * @param ovlp - overlap matrix
     * @param g - Green's function for the current iteration
     * @param s1 - static self-energy for the current iteration
     * @param s_t - dynamic self-energy for the current iteration
     */
    void update(size_t iter, double mu, const S1& h0, const S1& ovlp, G& g, S1& s1, St& s_t) const {
      _mixing->update(iter, mu, h0, ovlp, g, s1, s_t);
    }

    void print_name() const {
      _mixing->print_name();
    }

  private:
    std::unique_ptr<base_mixing<G, S1, St>> _mixing;
  };

}  // namespace green::sc

#endif  // SC_MIXING_H
