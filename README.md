[![GitHub license](https://img.shields.io/github/license/Green-Phys/green-sc?cacheSeconds=3600&color=informational&label=License)](./LICENSE)
[![GitHub license](https://img.shields.io/badge/C%2B%2B-17-blue)](https://en.cppreference.com/w/cpp/compiler_support/17)

![sc](https://github.com/Green-Phys/h5pp/actions/workflows/h5pp-test.yaml/badge.svg)
[![codecov](https://codecov.io/gh/Green-Phys/green-sc/graph/badge.svg?token=EW687Z97D3)](https://codecov.io/gh/Green-Phys/green-sc)

# green-sc
Light-weight header-only implementation of generic iterative self-consistency loop.

## Basic usage
`green-sc` is a header-only C++ template library that implements generic iterative self-consistency loop.

To add this library into your project, first 

```CMake
Include(FetchContent)

FetchContent_Declare(
        green-sc
        GIT_REPOSITORY https://github.com/Green-Phys/green-sc.git
        GIT_TAG origin/main # or a later release
)
FetchContent_MakeAvailable(green-sc)
```
Add predefined alias `GREEN::SC` it to your target:
```CMake
target_link_libraries(<target> PUBLIC GREEN::SC)
```
And then simply include the following header:
```cpp
#include <green/sc/sc_loop.h>
```
It uses [`green-params`](https://github.com/Green-Phys/green-params.git) library to read parameters either from a command line or from parameter string.
Please check `green-params` documentation for more details.
The following parameters have to be specified:

- `itermax` - maximum number of iterations to run
- `mixing_type` - type of mixing between iterations
- `mixing_weight` - how much of a previous iteration to be mixed with the current iteration results (`1`: no contribution from previous iteration, `0`: no contribution from current iteration)
- `results_file` - file to store results at each iteration
- `restart` - checkpointing
- `threshold` - convergence threshold, iterations will be stopped if convergence creteria is smaller than `threshold`.

Class has to be parametrized with user defined `Dyson` type. `Dyson` publicly defines the following three types: `G` - type for Green's function, `Sigma1` - type for static part of self-energy, 
`Sigma_tau` - type for dynamic-part of self-energy, and implements the following methods:

- `solve(G&, Sigma1&, Sigma_tau&)` - for a given self-energies
- `diff(G&, Sigma1&, Sigma_tau&)` - computes convergence creteria
- `dump_iteration(int iter, const std::string&f)` - store additional data into a results file `f` for a given iteration `iter`.

The method `solve` needs four parameters:

- Solver - solver object that implements
  - `solve(A, B, C)`

***
Even though we use diagrammatic names such as Green's function and self-energy, this library can be used to solve arbitrary equation that can be written
in iterative form. Here is a small example of solving square equation $\beta X^2 - X + \alpha = 0$

```cpp
using namespace green::sc;

// define dyson solver
class square_equation_dyson {
public:
  using G         = double;
  using Sigma1    = double;
  using Sigma_tau = double;
  double _alpha;
  double _beta;
  double _diff;

  square_equation_dyson(const green::params::params &p) : _alpha(p["a"]), _beta(p["b"]) {}

  void solve(G& g, Sigma1& sigma1, Sigma_tau& sigma_tau) {
    double g_new = _alpha + _alpha * sigma1 * g;
    _diff        = std::abs(g - g_new);
    g            = g_new;
  }
  double diff(const G& g, const Sigma1& sigma1, const Sigma_tau& sigma_tau) { return _diff; }
  void   dump_iteration(size_t iter, const std::string& result_file){};
};

// define solver class
class square_equation_solver {
public:
  double _U;
  second_power_equation_solver(double alpha, double beta) : _U(beta / alpha) {}
  void solve(const G& g, Sigma1& sigma1, Sigma_tau& sigma_tau) { sigma1 = _U * g; }
};

p.define<double>("a", "a", 1.0);
p.define<double>("b", "b", 0.5);
// create with a given green-params parameter object:
sc_loop<square_equation_dyson> sc(MPI_COMM_WORLD, p);
double g = 0, s1=0, st = 0;
// create solver for square equation
square_equation_solver solver(p["a"], p["b"]);

// solve equation
sc.solve(solver, g, s1, st);
```

# Acknowledgements

This work is supported by National Science Foundation under the award OAC-2310582
