// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <cstdlib>

namespace {
template <typename F> static auto catchAll(F &&func) noexcept {
  try {
    return func();
  } catch (const std::exception &e) {
    fprintf(stdout, "An exception was thrown: %s\n", e.what());
    fflush(stdout);
    abort();
  } catch (...) {
    fprintf(stdout, "An unknown exception was thrown\n");
    fflush(stdout);
    abort();
  }
}
} // namespace
