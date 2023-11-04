/*
 * Copyright (c) 2023 University of Michigan
 *
 */
#ifndef SC_SOLVER_H
#define SC_SOLVER_H

namespace green::sc {
  /**
   * Do nothing solver.
   * This solver doing nothing but he is doing it in stlye
   */
  class noop_solver {
  public:
    template <typename G, typename S1, typename St>
    void solve(G& g_tau, S1& sigma1, St& sigma_tau) {
      return;
    }
  };

  /**
   * Solver that is composed of other solvers and call them in predefined order
   *
   * @tparam Solvers - types of solvers
   */
  template <typename... Solvers>
  class composition_solver {
  public:
    composition_solver(Solvers&... solvers) : _solvers(std::ref(solvers)...) {}

    template <typename G, typename S1, typename St>
    void solve(G& g_tau, S1& sigma1, St& sigma_tau) {
      std::apply([&](auto&... solvers) { (solvers.solve(g_tau, sigma1, sigma_tau), ...); }, _solvers);
      return;
    }

  private:
    std::tuple<Solvers&...> _solvers;
  };
}  // namespace green::sc
#endif  // SC_SOLVER_H
