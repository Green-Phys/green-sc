/*
 * Copyright (c) 2023 University of Michigan
 *
 */
#ifndef SC_COMMON_UTILS_H
#define SC_COMMON_UTILS_H

#include <green/h5pp/archive.h>
#include <green/ndarray/ndarray.h>
#include <green/ndarray/ndarray_math.h>
#include <green/utils/mpi_shared.h>

namespace green::sc::internal {

  template <typename T>
  class is_default_constructible {
    typedef char yes;
    typedef struct {
      char arr[2];
    } no;

    template <typename U>
    static decltype(U(), yes()) test(int);

    template <typename>
    static no test(...);

  public:
    static constexpr bool value = sizeof(test<T>(0)) == sizeof(yes);
  };

  template <typename T>
  constexpr bool is_default_constructible_v = is_default_constructible<T>::value;

  template <typename T>
  void cleanup_data(T& old) {}

  template <typename T>
  std::enable_if_t<is_default_constructible_v<T>, T> init_data(const T&) {
    T tmp;
    return tmp;
  }

  template <typename T>
  void read_data(T& tmp, const std::string& fname, const std::string& data_path) {
    h5pp::archive ar(fname);
    ar[data_path] >> tmp;
    ar.close();
  }

  template <typename T>
  void update(T& old, T& tmp, double damping) {
    old *= damping;
    old += tmp * (1.0 - damping);
  }

  template <typename T, size_t N, typename C>
  utils::shared_object<tensor<T, N>>& operator*=(utils::shared_object<tensor<T, N>>& lhs, C rhs) {
    lhs.fence();
    if (!utils::context.node_rank) lhs.object() *= rhs;
    lhs.fence();
    return lhs;
  }

  template <typename T, size_t N>
  utils::shared_object<tensor<T, N>>& operator+=(utils::shared_object<tensor<T, N>>&       lhs,
                                                 const utils::shared_object<tensor<T, N>>& rhs) {
    lhs.fence();
    if (!utils::context.node_rank) lhs.object() += rhs.object();
    lhs.fence();
    return lhs;
  }

  template <typename T, size_t N>
  utils::shared_object<tensor<T, N>> init_data(const utils::shared_object<tensor<T, N>>& g) {
    return utils::shared_object<tensor<T, N>>(g.object().shape());
  }

  template <typename T, size_t N>
  tensor<T, N> init_data(const tensor<T, N>& g) {
    return tensor<T, N>(g.shape());
  }

  template <typename T>
  std::enable_if_t<h5pp::is_scalar<T>, T*> get_ref(T& tmp) {
    return &tmp;
  }

  template <size_t N>
  std::complex<double>* get_ref(ztensor<N>& tmp) {
    return tmp.data();
  }

  template <typename T>
  std::enable_if_t<h5pp::is_scalar<T>, void> assign(T& lhs, const T& rhs) {
    lhs = rhs;
  }

  template <size_t N>
  void assign(ztensor<N>& lhs, const ztensor<N>& rhs) {
    lhs << rhs;
  }

  template <typename T, size_t N>
  void assign(utils::shared_object<tensor<T, N>>& lhs, const utils::shared_object<tensor<T, N>>& rhs) {
    lhs.fence();
    if (!utils::context.node_rank) lhs.object() << rhs.object();
    lhs.fence();
  }

  template <typename T>
  std::enable_if_t<h5pp::is_scalar<T>, void> set_zero(T& lhs) {
    lhs = T(0);
  }

  template <size_t N>
  void set_zero(ztensor<N>& lhs) {
    lhs.set_zero();
  }
  template <typename T, size_t N>
  void set_zero(utils::shared_object<tensor<T, N>>& lhs) {
    lhs.fence();
    if (!utils::context.node_rank) lhs.object().set_zero();
    lhs.fence();
  }

  template <typename T>
  std::enable_if_t<h5pp::is_scalar<T>, std::complex<double>> overlap(const T& vec_v, const T& vec_u) {
    return vec_v * vec_u;
  }

  template <size_t N>
  std::complex<double> overlap(const ztensor<N>& vec_v, const ztensor<N>& vec_u) {
    using CMcolumn = Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1, Eigen::ColMajor>>;
    CMcolumn MFVec_v(vec_v.data(), vec_v.size());
    CMcolumn MFVec_u(vec_u.data(), vec_u.size());

    // TODO: think whether a rescaling of Sigma is needed...
    return MFVec_v.dot(MFVec_u);
  }
  template <typename T, size_t N>
  std::complex<double> overlap(const utils::shared_object<tensor<T, N>>& vec_v, const utils::shared_object<tensor<T, N>>& vec_u) {
    using CMcolumn = Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1, Eigen::ColMajor>>;
    CMcolumn MFVec_v(vec_v.object().data(), vec_v.size());
    CMcolumn MFVec_u(vec_u.object().data(), vec_u.size());

    // TODO: think whether a rescaling of Sigma is needed...
    return MFVec_v.dot(MFVec_u);
  }

  template <typename T>
  void read_data(utils::shared_object<T>& tmp, const std::string& fname, const std::string& data_path) {
    tmp.fence();
    if (!utils::context.node_rank) {
      h5pp::archive ar(fname);
      ar[data_path] >> tmp.object();
      ar.close();
    }
    tmp.fence();
  }

  inline void update(utils::shared_object<ztensor<5>>& old, utils::shared_object<ztensor<5>>& tmp, double damping) {
    old.fence();
    if (!utils::context.node_rank) {
      old.object() *= damping;
      old.object() += tmp.object() * (1.0 - damping);
    }
    old.fence();
  }

  inline void cleanup_data(utils::shared_object<ztensor<5>>&) {}

  template <typename T>
  void write(const T& v, const std::string& path, h5pp::archive& ar) {
    ar[path] << v;
  }

  inline void write(const utils::shared_object<ztensor<5>>& v, const std::string& path, h5pp::archive& ar) {
    ar[path] << v.object();
  }

  template <typename T>
  void read(T& v, const std::string& path, h5pp::archive& ar) {
    ar[path] >> v;
  }

  template <size_t N>
  void read(utils::shared_object<ztensor<N>>& v, const std::string& path, h5pp::archive& ar) {
    v.fence();
    if (!utils::context.node_rank) ar[path] >> v.object();
    v.fence();
  }

  template <size_t N>
  void read(ztensor<N>& v, const std::string& path, h5pp::archive& ar) {
    ar[path] >> v;
  }
}  // namespace green::sc::internal

namespace green::sc {
  /**
   * Read results of the previous unconverged simulation to proceed.
   *
   * @tparam G - type of the Green's function
   * @tparam S1 - type of the static part of the Self-energy
   * @tparam St - type of the dynamical part of the Self-energy
   * @param g_tau - [OUT]  Green's function
   * @param sigma_1 - [OUT] static part of the Self-energy
   * @param sigma_tau - [OUT] dynamic part of the Self-energy
   * @param results_file - [IN] name of a file to read data from
   * @return iteration number to start from
   */
  template <typename G, typename S1, typename St>
  size_t read_results(G& g_tau, S1& sigma_1, St& sigma_tau, const std::string& results_file) {
    if (!std::filesystem::exists(results_file)) {
      return 0;
    }
    h5pp::archive ar(results_file);
    if (!h5pp::dataset_exists(ar.current_id(), "iter")) {
      return 0;
    }
    size_t iter;
    ar["iter"] >> iter;
    internal::read(sigma_1, "iter" + std::to_string(iter) + "/Sigma1", ar);
    internal::read(sigma_tau, "iter" + std::to_string(iter) + "/Selfenergy/data", ar);
    internal::read(g_tau, "iter" + std::to_string(iter) + "/G_tau/data", ar);
    ar.close();
    return iter + 1;
  }

  inline void dump_parameters(const params::params& p, const std::string& results_file) {
    h5pp::archive ar(results_file, "a");
    for (const auto& param : p.params_set()) {
      const params::params_item& item = *param.get();
      ar["params/" + item.name() + "/value"] << item.entry()->print();
      if (!item.aka().empty()) {
        ar["params/" + item.name() + "/aliases"] << item.aka();
      }
    }
    ar.close();
  }
}  // namespace green::sc

#endif  // SC_COMMON_UTILS_H
