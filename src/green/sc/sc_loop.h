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

#ifndef GREEN_SC_LOOP_H
#define GREEN_SC_LOOP_H

#include <green/h5pp/archive.h>
#include <green/ndarray/ndarray.h>
#include <green/params/params.h>
#include <green/utils/mpi_utils.h>
#include <green/utils/timing.h>
#include <green/grids/common_defs.h>
#include <mpi.h>

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <vector>

#include "common_defs.h"
#include "mixing.h"
#include "solver.h"
#include "except.h"

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
    size_t _itermax;
    // current iteration
    size_t _iter;
    // convergence criteria
    double _e_thr;
    // convergence criteria for single precision
    double _e_thr_sp;
    // path to inital solution in k-space
    std::string _input_path;
    // path to the results file
    std::string _results_file;
    // Restart calculation from existing Self-energy
    bool _restart;
    // Dyson Equation solver
    DysonSolver _dyson_solver;
    // DIIS or simple mixing object
    mixing_strategy<G, S1, St> _mix;
    // MPI
    utils::mpi_context _context;

  public:
    sc_loop(MPI_Comm comm, params::params& p) :
        _itermax(p["itermax"]), _iter(0), _e_thr(p["threshold"]), _e_thr_sp(p["E_thr_sp"]), _input_path(p["input_file"]),
        _results_file(p["results_file"]), _restart(p["restart"]), _dyson_solver(p), _mix(p), _context(comm) {
      if (_restart) {
        check_grids_version_consistency();
      }
      
    }

    virtual ~sc_loop() = default;

    /**
     * Solve iterative self-consistency equation
     *
     * @tparam Solver - type of diagrammatic solver
     * @param solver - diagrammatic solver
     * @param ovlp  - [IN] overlap matrix for the problem
     * @param g0_tau - [IN] starting guess for the Green's function, [OUT] resulting Green's function
     * @param sigma1 - [OUT] static part of the Self-energy
     * @param sigma_tau - [OUT] dynamic part of the Self-energy
     */
    template <typename Solver>
    void solve(Solver& solver, const S1& h0, const S1& ovlp, G& g0_tau, S1& sigma1, St& sigma_tau) {
      size_t         start_iter = 1, iter = 0;
      utils::timing& t = utils::timing::get_instance();
      t.start("Read results");
      if (_restart) {
        start_iter = read_results(_dyson_solver.mu(), g0_tau, sigma1, sigma_tau, _results_file);
      } else {
        if (!_context.global_rank) {
          std::filesystem::remove(_results_file);
        }
        MPI_Barrier(_context.global);
      }
      MPI_Barrier(_context.global);
      t.end();

      t.start("Self-consistency loop");
      t.start("Dyson");
      _dyson_solver.solve(g0_tau, sigma1, sigma_tau);
      t.end();
      for (_iter = start_iter, iter = 0; iter < _itermax; ++iter, ++_iter) {
        if (!_context.global_rank) {
          std::cout << std::endl;
          std::cout << "========== Starting iteration " << _iter << " out of " << (_itermax + start_iter - 1)
                    << " ==========" << std::endl;
        }
        t.start("Diagrammatic solver");
        solver.solve(g0_tau, sigma1, sigma_tau);
        t.end();
        t.start("Iteration mixing");
        _mix.update(_iter, _dyson_solver.mu(), h0, ovlp, g0_tau, sigma1, sigma_tau);
        t.end();
        t.start("Check convergence");
        double diff = _dyson_solver.diff(g0_tau, sigma1, sigma_tau);
        t.end();
        // store results from current iteration
        t.start("Store results");
        if (!_context.global_rank) {
          dump_iteration(_iter, g0_tau, sigma1, sigma_tau);
          _dyson_solver.dump_iteration(_iter, g0_tau, sigma1, sigma_tau, _results_file);
        }
        t.end();
        if (!_context.global_rank) {
          std::stringstream ss;
          ss << std::scientific << std::setprecision(15);
          ss << std::setw(38) << std::right << "|ΔE_1b| + |ΔE_HF| + |ΔE_corr| = " << std::setw(22) << std::right << std::abs(diff)
             << std::endl;
          std::cout << ss.str();
        }
        if (std::abs(diff) < _e_thr) {
          if (!_context.global_rank) std::cout << "============== Simulation Converged ==============" << std::endl;
          break;
        }
        t.start("Dyson");
        _dyson_solver.solve(g0_tau, sigma1, sigma_tau);
        t.end();
      }
      if (!_context.global_rank && iter == _itermax)
        std::cout << "====== Reached Maximum number of iterations ======" << std::endl;
      t.end();
      t.print(_context.global);
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
      if (!ar.has_attribute("__grids_version__")) {
        // Grids attribute -- will only be performed once
        const std::string& grid_version = _dyson_solver.get_grids_version();
        ar.set_attribute<std::string>("__grids_version__", grid_version);
      }
      ar["iter"] << iter;
      internal::write(sigma_1, "iter" + std::to_string(iter) + "/Sigma1", ar);
      internal::write(sigma_tau, "iter" + std::to_string(iter) + "/Selfenergy/data", ar);
      internal::write(g_tau, "iter" + std::to_string(iter) + "/G_tau/data", ar);
      ar.close();
    }

    /**
     * @brief Checks consistency between grid-file version used in current run vs. the version
     * used to generate the results file that is being restarted from.
     * This is to prevent users from accidentally restarting from a results file that was generated with
     * an older version of green-grids, which can lead to silent errors in the results.
     * 
     * 1. If the results file does not have a green-grids version attribute, then check if current version is 0.2.4 (BASE VERSION)
     * 2. If the results file has a green-grids version attribute, check if it >= 0.2.4
     */
    void check_grids_version_consistency() {
      // If results file does not exist, nothing to check
      if (!std::filesystem::exists(_results_file)) return;

      // Read grid-file version
      const std::string& grid_file_version = _dyson_solver.get_grids_version();

      h5pp::archive ar(_results_file, "r");
      if (ar.has_attribute("__grids_version__")) {
        std::string grids_version_in_results;
        grids_version_in_results = ar.get_attribute<std::string>("__grids_version__");
        ar.close(); // safely close before throwing
        if (compare_version_strings(grid_file_version, grids_version_in_results) < 0) {
          throw green::grids::outdated_grids_file_error("The current green-grids version (" + grid_file_version +
                                            ") is older than the green-grids version used to create the original results file ("
                                            + grids_version_in_results +
                                            "). Please update green-grids to version " + grids_version_in_results);
        } else if (compare_version_strings(grid_file_version, grids_version_in_results) > 0) {
          throw outdated_results_file_error("The green-grids version used to create the results file (" + grids_version_in_results +
                                            ") is older than the current specified grid file (" + grid_file_version +
                                            "). Please download the appropriate version from: " +
                                            "https://github.com/Green-Phys/green-grids/releases/ or https://github.com/Green-Phys/green-grids/tags");
        }
      } else if (compare_version_strings(grid_file_version, green::grids::GRIDS_MIN_VERSION) > 0) {
        ar.close(); // safely close before throwing
        throw outdated_results_file_error("The results file was created using un-versioned grid file (equiv. to " + green::grids::GRIDS_MIN_VERSION +
                                          ") and the current green-grids version (" + grid_file_version + ") is newer.\n" + 
                                          "Please use old grid files from: https://github.com/Green-Phys/green-grids/releases/tag/v0.2.4.");
      }
    }

    DysonSolver& dyson_solver() { return _dyson_solver; }
  };
}  // namespace green::sc

#endif  // GREEN_SC_LOOP_H
