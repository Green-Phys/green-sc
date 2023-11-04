/*
 * Copyright (c) 2023 University of Michigan
 *
 */
#ifndef SC_COMMON_UTILS_H
#define SC_COMMON_UTILS_H

#include <green/h5pp/archive.h>
#include <green/ndarray/ndarray.h>
#include <green/utils/mpi_shared.h>

namespace green::sc::internal {
  template <typename T>
  void cleanup_data(T& old) {}

  template <typename T>
  T init_data(T& old) {
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
    old *= (1.0 - damping);
    old += tmp * damping;
  }

  utils::shared_object<ztensor<5>> init_data(utils::shared_object<ztensor<5>>& g) {
    ztensor<5> tmp(nullptr, g.object().shape());
    return utils::shared_object<ztensor<5>>(tmp);
  }

  void read_data(utils::shared_object<ztensor<5>>& tmp, const std::string& fname, const std::string& data_path) {
    tmp.fence();
    if (!utils::context.node_rank) {
      h5pp::archive ar(fname);
      ar[data_path] >> tmp.object();
      ar.close();
    }
    tmp.fence();
  }

  void update(utils::shared_object<ztensor<5>>& old, utils::shared_object<ztensor<5>>& tmp, double damping) {
    old.fence();
    if (!utils::context.node_rank) {
      old.object() *= damping;
      old.object() += tmp.object() * (1.0 - damping);
    }
    old.fence();
  }

  void cleanup_data(utils::shared_object<ztensor<5>>& g_tmp) {}

  template <typename T>
  void write(const T& v, const std::string& path, h5pp::archive& ar) {
    ar[path] << v;
  }

  void write(const utils::shared_object<ztensor<5>>& v, const std::string& path, h5pp::archive& ar) { ar[path] << v.object(); }

  template <typename T>
  void read(T& v, const std::string& path, h5pp::archive& ar) {
    ar[path] >> v;
  }

  void read(utils::shared_object<ztensor<5>>& v, const std::string& path, h5pp::archive& ar) { ar[path] >> v.object(); }
}  // namespace green::sc::internal

#endif  // SC_COMMON_UTILS_H
