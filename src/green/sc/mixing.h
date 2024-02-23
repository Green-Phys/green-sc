/*
 * Copyright (c) 2023 University of Michigan
 *
 */
#ifndef SC_MIXING_H
#define SC_MIXING_H

#include <green/params/params.h>
#include <green/utils/mpi_shared.h>

#include "common_defs.h"
#include "common_utils.h"
#include "except.h"

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
     * @param g - Green's function for the current iteration
     * @param s1 - static self-energy for the current iteration
     * @param s_t - dynamic self-energy for the current iteration
     */
    virtual void update(size_t iter, G& g, S1& s1, St& s_t) const = 0;

    virtual ~base_mixing() {}
  };

  template <typename G, typename S1, typename St>
  class no_mixing : public base_mixing<G, S1, St> {
  public:
    void update(size_t iter, G& g, S1& s1, St& s_t) const override{};
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
    void update(size_t iter, G& g, S1& s1, St& s_t) const override {
      if (iter == 0) {
        return;
      }
      G g_tmp(internal::init_data(g));
      internal::read_data(g_tmp, _results_file, "iter" + std::to_string(iter - 1) + "/G_tau/data");
      internal::update(g, g_tmp, _damping);
      internal::cleanup_data(g_tmp);
    };

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
    void update(size_t iter, G& g, S1& s1, St& s_t) const override {
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

  private:
    double      _damping;
    std::string _results_file;
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
    mixing_strategy(const params::params& p) {
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
        default:
          throw sc_unknown_mixing_error("Mixing " + std::string(magic_enum::enum_name(E)) + " has not yet been implemented");
      }
    }

    /**
     * Based on the chosen strategy mix result from current iteration with results from previous iterations.
     * @param iter
     * @param iter - current relative iteration
     * @param g - Green's function for the current iteration
     * @param s1 - static self-energy for the current iteration
     * @param s_t - dynamic self-energy for the current iteration
     */
    void update(size_t iter, G& g, S1& s1, St& s_t) const { _mixing.get()->update(iter, g, s1, s_t); }

  private:
    std::unique_ptr<base_mixing<G, S1, St>> _mixing;
  };

}  // namespace green::sc

#endif  // SC_MIXING_H
