//
// Created by iskakoff on 9/20/23.
//
#include <green/sc/sc_loop.h>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>

inline std::string random_name() {
  std::string        str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
  std::random_device rd;
  std::mt19937       generator(rd());
  std::shuffle(str.begin(), str.end(), generator);
  return str.substr(0, 32) + ".h5";  // assumes 32 < number of characters in str
}

// Try to solve equation B*X^4 + C*X^2 - A*X + D = 0
// It leads to following itertative scheme:
// X = \alpha + \beta X^4 + \gamma X^2,
// where
// \alpha = D/A
// \beta = B/A
// \gamma = C/A
// Which can be written as Dyson-like equation as
// X = \alpha + \alpha ( sigma1 + sigma2 ) X,
// where
// sigma1 = \gamma/\alpha X - first order-like term
// sigma2 = \beta/\alpha X^3 - second order-like term
class fourth_power_equation_dyson {
public:
  using G         = double;
  using Sigma1    = double;
  using Sigma_tau = double;

       fourth_power_equation_dyson(green::params::params& p) : _alpha(p["alpha"]), _beta(p["beta"]) {}

  void solve(G& g, Sigma1& sigma1, Sigma_tau& sigma_tau) {
    double g_new = _alpha + _alpha * (sigma1 + sigma_tau) * g;
    _diff        = std::abs(g - g_new);
    g            = g_new;
  }
  double diff(const G& g, const Sigma1& sigma1, const Sigma_tau& sigma_tau) { return _diff; }
  void   dump_iteration(size_t iter, const std::string& result_file){};

private:
  double _alpha;
  double _beta;
  double _diff;
};

class fourth_power_equation_solver {
public:
  using G         = double;
  using Sigma1    = double;
  using Sigma_tau = double;

       fourth_power_equation_solver(double alpha, double beta) : _U(std::sqrt(beta / alpha)) {}

  void solve(const G& g, Sigma1& sigma1, Sigma_tau& sigma_tau) { sigma_tau = _U * g * g * _U * g; }

private:
  double _U;
};

class second_power_equation_solver {
public:
  using G         = double;
  using Sigma1    = double;
  using Sigma_tau = double;

       second_power_equation_solver(double alpha, double gamma) : _U(gamma / alpha) {}

  void solve(const G& g, Sigma1& sigma1, Sigma_tau& sigma_tau) { sigma1 = _U * g; }

private:
  double _U;
};

void solve_with_damping(const std::string& damping_type, const std::string& damping) {
  auto        p        = green::params::params("DESCR");
  std::string res_file = random_name();
  // std::string input_file = TEST_PATH + "/test.h5"s;
  std::string args     = "test --restart 0 --itermax 1000 --E_thr 1e-13 --mixing_type " + damping_type + " --damping " + damping +
                     " --results_file=" + res_file;
  green::sc::define_parameters(p);
  p.define<double>("alpha", "", 0.45);
  p.define<double>("beta", "", 0.5);
  p.define<double>("gamma", "", 0.25);
  p.define<double>("x0", "", 1.0);
  p.parse(args);
  double                                          alpha     = p["alpha"];
  double                                          beta      = p["beta"];
  double                                          gamma     = p["gamma"];
  double                                          g         = p["x0"];
  double                                          sigma1    = 0;
  double                                          sigma_tau = 0;
  green::sc::sc_loop<fourth_power_equation_dyson> sc(MPI_COMM_WORLD, p);

  fourth_power_equation_solver                    solver(p["alpha"], p["beta"]);
  sc.solve(solver, g, sigma1, sigma_tau);
  // REQUIRE(std::abs(beta * g * g * g * g - g + alpha) < 1e-12);
  REQUIRE(std::abs(g - 0.47557728309) < 1e-10);
  SECTION("Composite solver") {
    double                                                                              g         = p["x0"];
    double                                                                              sigma1    = 0;
    double                                                                              sigma_tau = 0;
    green::sc::noop_solver                                                              noop;
    green::sc::composition_solver<green::sc::noop_solver, fourth_power_equation_solver> comp_solver(noop, solver);
    sc.solve(comp_solver, g, sigma1, sigma_tau);
    REQUIRE(std::abs(beta * g * g * g * g - g + alpha) < 1e-12);
    REQUIRE(std::abs(g - 0.47557728309) < 1e-10);
  }
  SECTION("Composite solver sigma1 + sigma2") {
    double                                                                                    g         = p["x0"];
    double                                                                                    sigma1    = 0;
    double                                                                                    sigma_tau = 0;
    second_power_equation_solver                                                              another(alpha, gamma);
    green::sc::composition_solver<second_power_equation_solver, fourth_power_equation_solver> comp_solver(another, solver);
    sc.solve(comp_solver, g, sigma1, sigma_tau);
    REQUIRE(std::abs(beta * g * g * g * g + gamma * g * g - g + alpha) < 1e-12);
    REQUIRE(std::abs(g - 0.619914177733761) < 1e-10);
  }
  std::filesystem::remove(res_file);
}

TEST_CASE("Self-consistency") {
  SECTION("Solve simple") {
    auto        p        = green::params::params("DESCR");
    std::string res_file = random_name();
    // std::string input_file = TEST_PATH + "/test.h5"s;
    std::string args     = "test --restart 0 --itermax 100 --E_thr 1e-13 --results_file=" + res_file;
    green::sc::define_parameters(p);
    p.define<double>("alpha", "", 0.45);
    p.define<double>("beta", "", 0.5);
    p.define<double>("gamma", "", 0.25);
    p.define<double>("x0", "", 0.2);
    p.parse(args);
    double                                          alpha     = p["alpha"];
    double                                          beta      = p["beta"];
    double                                          gamma     = p["gamma"];
    double                                          g         = p["x0"];
    double                                          sigma1    = 0;
    double                                          sigma_tau = 0;
    green::sc::sc_loop<fourth_power_equation_dyson> sc(MPI_COMM_WORLD, p);

    fourth_power_equation_solver                    solver(p["alpha"], p["beta"]);
    sc.solve(solver, g, sigma1, sigma_tau);
    REQUIRE(std::abs(beta * g * g * g * g - g + alpha) < 1e-12);
    REQUIRE(std::abs(g - 0.47557728309) < 1e-10);
    SECTION("Composite solver") {
      double                                                                              g         = p["x0"];
      double                                                                              sigma1    = 0;
      double                                                                              sigma_tau = 0;
      green::sc::noop_solver                                                              noop;
      green::sc::composition_solver<green::sc::noop_solver, fourth_power_equation_solver> comp_solver(noop, solver);
      sc.solve(comp_solver, g, sigma1, sigma_tau);
      REQUIRE(std::abs(beta * g * g * g * g - g + alpha) < 1e-12);
      REQUIRE(std::abs(g - 0.47557728309) < 1e-10);
    }
    SECTION("Composite solver sigma1 + sigma2") {
      double                                                                                    g         = p["x0"];
      double                                                                                    sigma1    = 0;
      double                                                                                    sigma_tau = 0;
      second_power_equation_solver                                                              another(alpha, gamma);
      green::sc::composition_solver<second_power_equation_solver, fourth_power_equation_solver> comp_solver(another, solver);
      sc.solve(comp_solver, g, sigma1, sigma_tau);
      REQUIRE(std::abs(beta * g * g * g * g + gamma * g * g - g + alpha) < 1e-12);
      REQUIRE(std::abs(g - 0.619914177733761) < 1e-10);
    }
    std::filesystem::remove(res_file);
  }

  SECTION("Solve with damping") {
    solve_with_damping("G_DAMPING", "0.8");
    solve_with_damping("SIGMA_DAMPING", "0.8");
    REQUIRE_THROWS_AS(solve_with_damping("DIIS", "0.8"), green::sc::sc_unknown_mixing_error);
    REQUIRE_THROWS_AS(solve_with_damping("G_DAMPING", "1.8"), green::sc::sc_incorrect_damping_error);
    REQUIRE_THROWS_AS(solve_with_damping("SIGMA_DAMPING", "1.8"), green::sc::sc_incorrect_damping_error);
  }

  SECTION("Restart") {
    auto        p          = green::params::params("DESCR");
    auto        p2         = green::params::params("DESCR");
    std::string res_file_1 = random_name();
    std::string res_file_2 = random_name();
    std::string args_1     = "test --restart 0 --itermax 4 --E_thr 1e-13 --results_file=" + res_file_1;
    std::string args_2     = "test --restart 1 --itermax 2 --E_thr 1e-13 --results_file=" + res_file_2;
    green::sc::define_parameters(p);
    green::sc::define_parameters(p2);
    p.define<double>("alpha", "", 0.45);
    p.define<double>("beta", "", 0.5);
    p.define<double>("gamma", "", 0.25);
    p.define<double>("x0", "", 0.2);
    p2.define<double>("alpha", "", 0.45);
    p2.define<double>("beta", "", 0.5);
    p2.define<double>("gamma", "", 0.25);
    p2.define<double>("x0", "", 0.2);
    p.parse(args_1);
    p2.parse(args_2);
    double g           = p["x0"];
    double sigma1      = 0;
    double sigma_tau   = 0;
    double g2          = p2["x0"];
    double sigma1_2    = 0;
    double sigma_tau_2 = 0;
    {
      green::sc::sc_loop<fourth_power_equation_dyson> sc(MPI_COMM_WORLD, p);
      fourth_power_equation_solver                    solver(p["alpha"], p["beta"]);
      sc.solve(solver, g, sigma1, sigma_tau);
    }
    {
      green::sc::sc_loop<fourth_power_equation_dyson> sc(MPI_COMM_WORLD, p2);
      fourth_power_equation_solver                    solver(p2["alpha"], p2["beta"]);
      sc.solve(solver, g2, sigma1_2, sigma_tau_2);
    }
    {
      green::sc::sc_loop<fourth_power_equation_dyson> sc(MPI_COMM_WORLD, p2);
      fourth_power_equation_solver                    solver(p2["alpha"], p2["beta"]);
      sc.solve(solver, g2, sigma1_2, sigma_tau_2);
    }
    REQUIRE(std::abs(sigma_tau_2 - sigma_tau) < 1e-14);
    std::filesystem::remove(res_file_2);
    {
      // create empty file to check that
      green::h5pp::archive ar(res_file_2, "w");
      ar["test"] << 1;
      ar.close();
    }
    {
      p2["itermax"] = 4;
      g2            = p2["x0"];
      sigma1_2      = 0;
      sigma_tau_2   = 0;
      green::sc::sc_loop<fourth_power_equation_dyson> sc(MPI_COMM_WORLD, p2);
      fourth_power_equation_solver                    solver(p2["alpha"], p2["beta"]);
      sc.solve(solver, g2, sigma1_2, sigma_tau_2);
    }

    std::filesystem::remove(res_file_1);
    std::filesystem::remove(res_file_2);
  }
}

TEST_CASE("Shared object init") {
  green::utils::shared_object x(green::ndarray::ndarray<double, 2>(nullptr, 10, 10));
  green::utils::shared_object x_tmp(green::sc::internal::init_data(x));
  std::string                 res_file_1 = random_name();
  {
    x.fence();
    if (!green::utils::context.node_rank) x.object()(0, 0) = 100;
    x.fence();
    green::h5pp::archive ar(res_file_1, "w");
    ar["test"] << x.object();
    ar.close();
  }
  REQUIRE(x.size() == x_tmp.size());
  green::sc::internal::read_data(x_tmp, res_file_1, "test");
  REQUIRE(x.object()(0, 0) == x_tmp.object()(0, 0));
  std::filesystem::remove(res_file_1);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
