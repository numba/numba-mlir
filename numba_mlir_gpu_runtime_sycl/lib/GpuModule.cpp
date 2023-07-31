// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "GpuModule.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <type_traits>

#ifdef __linux__
#include <dlfcn.h>
#elif defined(_WIN32) || defined(_WIN64)
#define NOMINMAX
#include <windows.h>
#endif // __linux__

#include <level_zero/ze_api.h>

#include <CL/cl.h>
#if __has_include(<sycl/backend/opencl.hpp>)
#include <sycl/backend/opencl.hpp>
#else
#include <CL/sycl/backend/opencl.hpp>
#endif

[[noreturn]] static void reportError(std::string &&str) {
  throw std::runtime_error(std::move(str));
}

namespace {
class DynamicLibHelper final {
public:
  DynamicLibHelper &operator=(const DynamicLibHelper &) = delete;
  DynamicLibHelper() = delete;
  DynamicLibHelper(const DynamicLibHelper &) = delete;
  DynamicLibHelper(const char *libName) {
    assert(libName);

#ifdef __linux__
    _handle = dlopen(libName, RTLD_NOLOAD | RTLD_NOW | RTLD_LOCAL);
    if (!_handle) {
      char *error = dlerror();
      reportError(
          "Could not load library " + std::string(libName) +
          ". Error encountered: " + std::string(error ? error : "<null>"));
    }
#elif defined(_WIN32) || defined(_WIN64)
    _handle = LoadLibraryExA(libName, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!_handle)
      reportError("Could not load library " + std::string(libName));
#endif
  }

  ~DynamicLibHelper() {
#ifdef __linux__
    dlclose(_handle);
#elif defined(_WIN32) || defined(_WIN64)
    FreeLibrary((HMODULE)_handle);
#endif
  };

  template <typename T> T getSymbol(const char *symName) {
#ifdef __linux__
    void *sym = dlsym(_handle, symName);

    if (!sym) {
      char *error = dlerror();
      reportError("Could not retrieve symbol " + std::string(symName) +
                  ". Error encountered: " + std::string(error));
    }

#elif defined(_WIN32) || defined(_WIN64)
    void *sym = (void *)GetProcAddress((HMODULE)_handle, symName);

    if (!sym)
      reportError("Could not retrieve symbol " + std::string(symName));
#endif

    return (T)sym;
  }

private:
  void *_handle = nullptr;
};

#define DECL_TYPE(func) using func##_T = decltype(&func)
#define INIT_FUNC(func)                                                        \
  do {                                                                         \
    this->func = this->dynLib.getSymbol<func##_T>(#func);                      \
  } while (false)
#define DECL_FUNC(func) func##_T func

static constexpr const char *zeLibName =
#ifdef __linux__
    "libze_loader.so.1";
#elif defined(_WIN32) || defined(_WIN64)
    "ze_loader.dll";
#endif

struct ZELoader {
  DECL_TYPE(zeModuleCreate);
  DECL_TYPE(zeModuleBuildLogGetString);
  DECL_TYPE(zeModuleDestroy);
  DECL_TYPE(zeModuleBuildLogDestroy);
  DECL_TYPE(zeKernelCreate);
  DECL_TYPE(zeKernelDestroy);
  DECL_TYPE(zeKernelSuggestGroupSize);

  ZELoader() : dynLib(zeLibName) {
    INIT_FUNC(zeModuleCreate);
    INIT_FUNC(zeModuleBuildLogGetString);
    INIT_FUNC(zeModuleDestroy);
    INIT_FUNC(zeModuleBuildLogDestroy);
    INIT_FUNC(zeKernelCreate);
    INIT_FUNC(zeKernelDestroy);
    INIT_FUNC(zeKernelSuggestGroupSize);
  }

  DECL_FUNC(zeModuleCreate);
  DECL_FUNC(zeModuleBuildLogGetString);
  DECL_FUNC(zeModuleDestroy);
  DECL_FUNC(zeModuleBuildLogDestroy);
  DECL_FUNC(zeKernelCreate);
  DECL_FUNC(zeKernelDestroy);
  DECL_FUNC(zeKernelSuggestGroupSize);

private:
  DynamicLibHelper dynLib;
};

static ZELoader &getZeLoader() {
  static ZELoader loader;
  return loader;
}

static std::string getZeErrorString(ze_result_t val) {
#define ZE_ENUM_VAL(arg)                                                       \
  case arg:                                                                    \
    return #arg
  switch (val) {
    ZE_ENUM_VAL(ZE_RESULT_SUCCESS);
    ZE_ENUM_VAL(ZE_RESULT_NOT_READY);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_DEVICE_LOST);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_MODULE_BUILD_FAILURE);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_MODULE_LINK_FAILURE);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_NOT_AVAILABLE);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_UNINITIALIZED);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_UNSUPPORTED_VERSION);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_INVALID_ARGUMENT);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_INVALID_NULL_HANDLE);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_INVALID_SIZE);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_UNSUPPORTED_SIZE);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_INVALID_ENUMERATION);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_INVALID_NATIVE_BINARY);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_INVALID_GLOBAL_NAME);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_INVALID_KERNEL_NAME);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_INVALID_FUNCTION_NAME);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_OVERLAPPING_REGIONS);
    ZE_ENUM_VAL(ZE_RESULT_ERROR_UNKNOWN);

  default:
    return "Unknown error: " + std::to_string(val);
  }
#undef ZE_ENUM_VAL
}

static void checkZeResult(ze_result_t res, const char *func) {
  if (res != ZE_RESULT_SUCCESS)
    reportError(std::string(func) + " failed: " + getZeErrorString(res));
}

#define CHECK_ZE_RESULT(expr) checkZeResult((expr), #expr)

struct ZeModuleDeleter {
  void operator()(ze_module_handle_t module) const {
    CHECK_ZE_RESULT(getZeLoader().zeModuleDestroy(module));
  }
};
using ZeModule =
    std::unique_ptr<std::remove_pointer_t<ze_module_handle_t>, ZeModuleDeleter>;

struct ZeBuildLogDeleter {
  void operator()(ze_module_build_log_handle_t log) const {
    CHECK_ZE_RESULT(getZeLoader().zeModuleBuildLogDestroy(log));
  }
};
using ZeBuildLog =
    std::unique_ptr<std::remove_pointer_t<ze_module_build_log_handle_t>,
                    ZeBuildLogDeleter>;

struct ZeKernelDeleter {
  void operator()(ze_kernel_handle_t module) const {
    CHECK_ZE_RESULT(getZeLoader().zeKernelDestroy(module));
  }
};
using ZeKernel =
    std::unique_ptr<std::remove_pointer_t<ze_kernel_handle_t>, ZeKernelDeleter>;

static constexpr const char *clLibName =
#ifdef __linux__
    "libOpenCL.so.1";
#elif defined(_WIN32) || defined(_WIN64)
    "OpenCL.dll";
#endif

struct OCLLoader {
  DECL_TYPE(clCreateProgramWithIL);
  DECL_TYPE(clReleaseProgram);
  DECL_TYPE(clBuildProgram);
  DECL_TYPE(clGetProgramBuildInfo);
  DECL_TYPE(clCreateKernel);
  DECL_TYPE(clReleaseKernel);

  OCLLoader() : dynLib(clLibName) {
    INIT_FUNC(clCreateProgramWithIL);
    INIT_FUNC(clReleaseProgram);
    INIT_FUNC(clBuildProgram);
    INIT_FUNC(clGetProgramBuildInfo);
    INIT_FUNC(clCreateKernel);
    INIT_FUNC(clReleaseKernel);
  }

  DECL_FUNC(clCreateProgramWithIL);
  DECL_FUNC(clReleaseProgram);
  DECL_FUNC(clBuildProgram);
  DECL_FUNC(clGetProgramBuildInfo);
  DECL_FUNC(clCreateKernel);
  DECL_FUNC(clReleaseKernel);

private:
  DynamicLibHelper dynLib;
};

static OCLLoader &getClLoader() {
  static OCLLoader loader;
  return loader;
}

static std::string getClErrorString(cl_int val) {
#define CL_ENUM_VAL(arg)                                                       \
  case arg:                                                                    \
    return #arg
  switch (val) {
    CL_ENUM_VAL(CL_SUCCESS);
    CL_ENUM_VAL(CL_BUILD_PROGRAM_FAILURE);
    CL_ENUM_VAL(CL_INVALID_CONTEXT);
    CL_ENUM_VAL(CL_INVALID_DEVICE);
    CL_ENUM_VAL(CL_INVALID_VALUE);
    CL_ENUM_VAL(CL_OUT_OF_RESOURCES);
    CL_ENUM_VAL(CL_OUT_OF_HOST_MEMORY);
    CL_ENUM_VAL(CL_INVALID_OPERATION);
    CL_ENUM_VAL(CL_INVALID_BINARY);
  default:
    return "Unknown error: " + std::to_string(val);
  }
#undef CL_ENUM_VAL
}

static void checkClResult(const char *func, cl_int res) {
  if (res != CL_SUCCESS)
    reportError(std::string(func) + " failed: " + getClErrorString(res));
}

#define CHECK_CL_RESULT(arg) checkClResult(#arg, arg)

struct ClProgramDeleter {
  void operator()(cl_program program) const {
    CHECK_CL_RESULT(getClLoader().clReleaseProgram(program));
  }
};
using ClProgram =
    std::unique_ptr<std::remove_pointer_t<cl_program>, ClProgramDeleter>;

struct ClKernelDeleter {
  void operator()(cl_kernel kernel) const {
    CHECK_CL_RESULT(getClLoader().clReleaseKernel(kernel));
  }
};
using ClKernel =
    std::unique_ptr<std::remove_pointer_t<cl_kernel>, ClKernelDeleter>;

static constexpr const auto ze_be = sycl::backend::ext_oneapi_level_zero;
static constexpr const auto cl_be = sycl::backend::opencl;
} // namespace

struct GPUModule {
  sycl::queue *queue = nullptr;
  sycl::kernel_bundle<sycl::bundle_state::executable> kernelBundle;
  ZeModule zeModule;
  ClProgram clProgram;
};

struct GPUKernel {
  sycl::kernel syclKernel;
  ZeKernel zeKernel;
  ClKernel clKernel;
  uint32_t maxWgSize = 0;
};

GPUModule *createGPUModule(sycl::queue &queue, const void *data,
                           size_t dataSize) {
  auto ctx = queue.get_context();
  auto backend = ctx.get_platform().get_backend();

  if (backend == ze_be) {
    auto &loader = getZeLoader();
    auto zeDevice = sycl::get_native<ze_be>(queue.get_device());
    auto zeContext = sycl::get_native<ze_be>(queue.get_context());

    ze_module_desc_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    desc.pInputModule = static_cast<const uint8_t *>(data);
    desc.inputSize = dataSize;

    ze_module_handle_t moduleHandle = nullptr;
    ze_module_build_log_handle_t logHandle = nullptr;
    try {
      CHECK_CL_RESULT(loader.zeModuleCreate(zeContext, zeDevice, &desc,
                                            &moduleHandle, &logHandle));
    } catch (std::exception &e) {
      if (!logHandle)
        throw;

      ZeBuildLog log(logHandle);

      size_t len = 0;
      CHECK_ZE_RESULT(
          loader.zeModuleBuildLogGetString(log.get(), &len, nullptr));

      std::string str;
      str.resize(len);
      CHECK_ZE_RESULT(
          loader.zeModuleBuildLogGetString(log.get(), &len, str.data()));
      throw std::runtime_error(e.what() + std::string("\n") + str);
    }
    ZeBuildLog log(logHandle);
    ZeModule zeModule(moduleHandle);

    auto kernelBundle =
        sycl::make_kernel_bundle<ze_be, sycl::bundle_state::executable>(
            {zeModule.get()}, ctx);
    return new GPUModule{&queue, std::move(kernelBundle), std::move(zeModule),
                         nullptr};
  }
  if (backend == cl_be) {
    auto &loader = getClLoader();

    auto clContext = sycl::get_native<cl_be>(ctx);
    auto clDevice = sycl::get_native<cl_be>(queue.get_device());

    cl_int errCode = CL_SUCCESS;
    ClProgram program(
        loader.clCreateProgramWithIL(clContext, data, dataSize, &errCode));
    checkClResult("clCreateProgramWithILF", errCode);

    try {
      CHECK_CL_RESULT(loader.clBuildProgram(program.get(), 1, &clDevice, "",
                                            nullptr, nullptr));
    } catch (std::exception &e) {
      size_t len = 0;
      auto ret = loader.clGetProgramBuildInfo(
          program.get(), clDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
      if (ret != CL_SUCCESS)
        throw;

      std::string str;
      str.resize(len);
      ret = loader.clGetProgramBuildInfo(program.get(), clDevice,
                                         CL_PROGRAM_BUILD_LOG, len, str.data(),
                                         nullptr);
      if (ret != CL_SUCCESS)
        throw;

      throw std::runtime_error(e.what() + std::string("\n") + str);
    }

    auto kernelBundle =
        sycl::make_kernel_bundle<cl_be, sycl::bundle_state::executable>(
            program.get(), ctx);
    return new GPUModule{&queue, std::move(kernelBundle), nullptr,
                         std::move(program)};
  }

  reportError("Backend is not supported: " +
              std::to_string(static_cast<int>(backend)));
}

void destoyGPUModule(GPUModule *mod) { delete mod; }

GPUKernel *getGPUKernel(GPUModule *mod, const char *name) {
  auto queue = mod->queue;
  assert(queue);
  auto ctx = queue->get_context();

  auto maxWgSize = static_cast<uint32_t>(
      queue->get_device().get_info<sycl::info::device::max_work_group_size>());
  if (mod->zeModule) {
    auto &loader = getZeLoader();
    auto zeModule = mod->zeModule.get();

    ze_kernel_desc_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    desc.pKernelName = name;

    ze_kernel_handle_t kernelHandle = nullptr;
    CHECK_CL_RESULT(loader.zeKernelCreate(zeModule, &desc, &kernelHandle));
    ZeKernel zeKernel(kernelHandle);

    auto syclKernel =
        sycl::make_kernel<ze_be>({mod->kernelBundle, zeKernel.get()}, ctx);
    return new GPUKernel{syclKernel, std::move(zeKernel), nullptr, maxWgSize};
  }
  if (mod->clProgram) {
    auto &loader = getClLoader();

    cl_int errCode = CL_SUCCESS;
    ClKernel clKernel(
        loader.clCreateKernel(mod->clProgram.get(), name, &errCode));
    checkClResult("clCreateKernel", errCode);

    auto syclKernel = sycl::make_kernel<cl_be>(clKernel.get(), ctx);
    return new GPUKernel{syclKernel, nullptr, std::move(clKernel), maxWgSize};
  }
  reportError("Invalid module");
}

void destroyGPUKernel(GPUKernel *kernel) { delete kernel; }

sycl::kernel getSYCLKernel(GPUKernel *kernel) { return kernel->syclKernel; }

static uint32_t downPow2(uint32_t x) {
  assert(x > 0);
  x |= (x >> 1);
  x |= (x >> 2);
  x |= (x >> 4);
  x |= (x >> 8);
  x |= (x >> 16);
  return x - (x >> 1);
}

static uint32_t upPow2(uint32_t x) {
  assert(x > 0);
  x--;
  x |= (x >> 1);
  x |= (x >> 2);
  x |= (x >> 4);
  x |= (x >> 8);
  x |= (x >> 16);
  x++;
  return x;
}

void suggestGPUBlockSize(GPUKernel *kernel, const uint32_t *gridSize,
                         uint32_t *blockSize, size_t numDims) {
  assert(numDims > 0 && numDims <= 3);
  if (kernel->zeKernel) {
    auto zeKernel = kernel->zeKernel.get();
    uint32_t gSize[3] = {};
    uint32_t *bSize[3] = {};
    for (size_t i = 0; i < numDims; ++i) {
      gSize[i] = gridSize[i];
      bSize[i] = &blockSize[i];
    }

    CHECK_ZE_RESULT(getZeLoader().zeKernelSuggestGroupSize(
        zeKernel, gSize[0], gSize[1], gSize[2], bSize[0], bSize[1], bSize[2]));
    return;
  }

  auto maxWgSize = downPow2(kernel->maxWgSize);
  for (size_t i = 0; i < numDims; ++i) {
    auto gsize = gridSize[i];
    if (gsize > 1) {
      auto lsize = std::min(gsize, maxWgSize);
      blockSize[i] = lsize;
      maxWgSize /= upPow2(lsize);
    } else {
      blockSize[i] = 1;
    }
  }
}
