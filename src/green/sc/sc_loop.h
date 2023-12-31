/*
 * Copyright (c) 2023 University of Michigan.
 *
 */

#ifndef GF2_SC_LOOP_H
#define GF2_SC_LOOP_H

#include <green/h5pp/archive.h>
#include <green/ndarray/ndarray.h>
#include <green/params/params.h>
#include <green/utils/mpi_utils.h>
#include <green/utils/timing.h>
#include <mpi.h>

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <vector>

#include "common_defs.h"
#include "mixing.h"
#include "solver.h"

namespace green::sc {
  /**
   * @brief sc_loop class perform the main self-consistency loop for self-energy
   * evaluation
   */
  template <typename DysonSolver>
  class sc_loop {
  private:
    using G  = typename DysonSolver::G;
    using S1 = typename DysonSolver::Sigma1;
    using St = typename DysonSolver::Sigma_tau;

    // maximum number of iterations
    size_t                     _itermax;
    // current iteration
    size_t                     _iter;
    // convergence criteria
    double                     _e_thr;
    // convergence criteria for single precision
    double                     _e_thr_sp;
    // path to inital solution in k-space
    std::string                _input_path;
    // path to the results file
    std::string                _results_file;
    // Restart calculation from existing Self-energy
    bool                       _restart;
    // Dyson Equation solver
    DysonSolver                _dyson_solver;
    // DIIS or simple damping object
    mixing_strategy<G, S1, St> _mix;
    // MPI
    green::utils::mpi_context  _context;

  public:
    sc_loop(MPI_Comm comm, green::params::params& p) :
        _itermax(p["itermax"]), _iter(0), _e_thr(p["threshold"]), _e_thr_sp(p["E_thr_sp"]), _input_path(p["input_file"]),
        _results_file(p["results_file"]), _restart(p["restart"]), _mix(p), _dyson_solver(p), _context(comm) {
      if (!_restart) {
        if (!_context.global_rank) {std::filesystem::remove(_results_file);}
        MPI_Barrier(comm);
      }
      if (!_context.global_rank) {
        internal::dump_parameters(p, _results_file);
        MPI_Barrier(comm);
      }
    }

    virtual ~sc_loop() {}

    /**
     * Solve iterative self-consistency equation
     *
     * @tparam Solver - type of diagrammatic solver
     * @tparam G - type of the Green's function
     * @tparam S1 - type of the static part of the Self-energy
     * @tparam St - type of the dynamical part of the Self-energy
     * @param solver -
     * @param g0_tau - [IN] starting guess for the Green's function, [OUT] resulting Green's function
     * @param sigma1 - [OUT] static part of the Self-energy
     * @param sigma_tau - [OUT] dynamic part of the Self-energy
     */
    template <typename Solver>
    void solve(Solver& solver, G& g0_tau, S1& sigma1, St& sigma_tau) {
      size_t         start_iter = 0, iter = 0;
      utils::timing& t = utils::timing::get_instance();
      t.start("Read results");
      if (_restart) {
        start_iter = read_results(g0_tau, sigma1, sigma_tau);
      }
      t.end();

      t.start("Self-consistency loop");
      t.start("Dyson");
      _dyson_solver.solve(g0_tau, sigma1, sigma_tau);
      t.end();
      for (_iter = start_iter, iter = 0; iter < _itermax; ++iter, ++_iter) {
        if (!_context.global_rank) {
          std::cout << "========== Starting iteration " << _iter << " out of " << _itermax << " ==========" << std::endl;
        }
        t.start("Diagrammatic solver");
        solver.solve(g0_tau, sigma1, sigma_tau);
        t.end();
        t.start("Iteration mixing");
        _mix.update(_iter, g0_tau, sigma1, sigma_tau);
        t.end();
        t.start("Check convergence");
        double diff = _dyson_solver.diff(g0_tau, sigma1, sigma_tau);
        t.end();
        // store results from current iteration
        t.start("Store results");
        if (!_context.global_rank) {
          dump_iteration(_iter, g0_tau, sigma1, sigma_tau);
          _dyson_solver.dump_iteration(_iter, _results_file);
        }
        t.end();
        if (std::abs(diff) < _e_thr) break;
        t.start("Dyson");
        _dyson_solver.solve(g0_tau, sigma1, sigma_tau);
        t.end();
      }
      t.end();
      t.print(_context.global);
    }

    /**
     * Read results of the previous unconverged simulation to proceed.
     *
     * @tparam G - type of the Green's function
     * @tparam S1 - type of the static part of the Self-energy
     * @tparam St - type of the dynamical part of the Self-energy
     * @param G_tau - [OUT]  Green's function
     * @param Sigma_1 - [OUT] static part of the Self-energy
     * @param Sigma_tau - [OUT] dynamic part of the Self-energy
     * @return last iteration number to start from
     */
    size_t read_results(G& g_tau, S1& sigma_1, St& sigma_tau) {
      if (!std::filesystem::exists(_results_file)) {
        return 0;
      }
      h5pp::archive ar(_results_file);
      if(!h5pp::dataset_exists(ar.current_id(), "iter")) {
        return 0;
      }
      ar["iter"] >> _iter;
      internal::read(sigma_1, "iter" + std::to_string(_iter) + "/Sigma1", ar);
      internal::read(sigma_tau, "iter" + std::to_string(_iter) + "/Selfenergy/data", ar);
      internal::read(g_tau, "iter" + std::to_string(_iter) + "/G_tau/data", ar);
      ar.close();
      return _iter + 1;
    }

    /**
     * Store results of the iteration `iter` into HDF5 file
     *
     * @tparam G - type of the Green's function
     * @tparam S1 - type of the static part of the Self-energy
     * @tparam St - type of the dynamical part of the Self-energy
     * @param iter - iteration number to be stored
     * @param g_tau - value of the Green's function at the iteration `iter`
     * @param sigma_1 - value of the static part of Self-energy at the iteration `iter`
     * @param sigma_tau - value of the dynamic part of Self-energy at the iteration `iter`
     */
    void dump_iteration(size_t iter, G& g_tau, S1& sigma_1, St& sigma_tau) const {
      h5pp::archive ar(_results_file, "a");
      ar["iter"] << iter;
      internal::write(sigma_1, "iter" + std::to_string(iter) + "/Sigma1", ar);
      internal::write(sigma_tau, "iter" + std::to_string(iter) + "/Selfenergy/data", ar);
      internal::write(g_tau, "iter" + std::to_string(iter) + "/G_tau/data", ar);
      ar.close();
    }
  };
}  // namespace green::sc

#endif  // GF2_SC_LOOP_H
