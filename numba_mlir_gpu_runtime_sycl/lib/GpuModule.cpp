// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "GpuModule.hpp"

#include "LevelZeroWrapper.hpp"

#include <memory>

struct GPUModule {
    sycl::queue* queue = nullptr;
    ze::Module l0module;
};

struct GPUKernel {
    sycl::kernel syclKernel;
    ze::Kernel l0kernel;
};

GPUModule* createGPUModule(sycl::queue& queue, const void *data, size_t dataSize) {
    auto zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      queue.get_device());
    auto zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      queue.get_context());

    auto ret = std::make_unique<GPUModule>();
    ret->queue = &queue;

    ze_module_desc_t desc = {};
    desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    desc.pInputModule = static_cast<const uint8_t *>(data);
    desc.inputSize = dataSize;
    ret->l0module = ze::Module::create(zeContext, zeDevice, desc).first;

    return ret.release();
}

void destoyGPUModule(GPUModule* mod) {
    delete mod;
}

GPUKernel* getGPUKernel(GPUModule* mod, const char *name) {
    auto queue = mod->queue;
    assert(queue);

    auto zeModule = mod->l0module.get();

    ze_kernel_desc_t desc = {};
    desc.pKernelName = name;
    auto zeKernel = ze::Kernel::create(zeModule, desc);

    sycl::kernel_bundle<sycl::bundle_state::executable> kernelBundle =
      sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                               sycl::bundle_state::executable>(
          {zeModule}, queue->get_context());
    auto syclKernel = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {kernelBundle, zeKernel.get()}, queue->get_context());

    return new GPUKernel{syclKernel, std::move(zeKernel)};
}

void destroyGPUKernel(GPUKernel* kernel) {
    delete kernel;
}

sycl::kernel getSYCLKernel(GPUKernel* kernel) {
    return kernel->syclKernel;
}

void suggestGPUBlockSize(GPUKernel* kernel, const uint32_t *gridSize, uint32_t *blockSize, size_t numDims) {
    auto l0Kernel = kernel->l0kernel.get();
    assert(numDims > 0 && numDims <= 3);
    uint32_t gSize[3] = {};
    uint32_t *bSize[3] = {};
    for (size_t i = 0; i < numDims; ++i) {
      gSize[i] = gridSize[i];
      bSize[i] = &blockSize[i];
    }

    CHECK_ZE_RESULT(zeKernelSuggestGroupSize(
        l0Kernel, gSize[0], gSize[1], gSize[2], bSize[0], bSize[1], bSize[2]));
}