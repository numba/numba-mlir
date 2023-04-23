// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sycl/sycl.hpp>

struct GPUModule;
struct GPUKernel;

GPUModule* createGPUModule(sycl::queue& queue, const void *data, size_t dataSize);
void destoyGPUModule(GPUModule* mod);

GPUKernel* getGPUKernel(GPUModule* mod, const char *name);
void destroyGPUKernel(GPUKernel* kernel);

sycl::kernel getSYCLKernel(GPUKernel* kernel);

void suggestGPUBlockSize(GPUKernel* kernel, const uint32_t *gridSize, uint32_t *blockSize, size_t numDims);