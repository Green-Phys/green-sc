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

#ifndef GREEN_SC_SOLVER_H
#define GREEN_SC_SOLVER_H

namespace green::sc {
  /**
   * Do nothing solver.
   * This solver doing nothing but he is doing it in stlye
   */
  class noop_solver {
  public:
    template <typename G, typename S1, typename St>
    void solve(G& g_tau, S1& sigma1, St& sigma_tau) {}
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
    }

  private:
    std::tuple<Solvers&...> _solvers;
  };
}  // namespace green::sc
#endif  // GREEN_SC_SOLVER_H
