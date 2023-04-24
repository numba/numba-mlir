// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "GpuModule.hpp"

#include <algorithm>
#include <stdexcept>
#include <memory>
#include <type_traits>

#ifdef __linux__
#include <dlfcn.h>
#elif defined(_WIN32) || defined(_WIN64)
#define NOMINMAX
#include <windows.h>
#endif // __linux__

#include "LevelZeroWrapper.hpp"

#include <CL/cl.h>
#if __has_include(<sycl/backend/opencl.hpp>)
#include <sycl/backend/opencl.hpp>
#else
#include <CL/sycl/backend/opencl.hpp>
#endif


[[noreturn]] static void reportError(std::string&& str) {
    throw std::runtime_error(std::move(str));
}

namespace {
class DynamicLibHelper final
{
public:
    DynamicLibHelper &operator=(const DynamicLibHelper &) = delete;
    DynamicLibHelper() = delete;
    DynamicLibHelper(const DynamicLibHelper &) = delete;
    DynamicLibHelper(const char *libName)
    {
        assert(libName);

#ifdef __linux__
        _handle = dlopen(libName, RTLD_NOLOAD | RTLD_NOW | RTLD_LOCAL);
        if (!_handle) {
            char *error = dlerror();
            reportError("Could not load library " + std::string(libName) +
                        ". Error encountered: " + std::string(error));
        }
#elif defined(_WIN32) || defined(_WIN64)
        _handle =
            LoadLibraryExA(libName, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if(!_handle)
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

static constexpr const char *clLibName =
#ifdef __linux__
"libOpenCL.so";
#elif defined(_WIN32) || defined(_WIN64)
"OpenCL.dll";
#endif

struct OCLLoader {
    using clCreateProgramWithIL_T = cl_program(*)(cl_context,const void *,size_t,cl_int *);
    using clReleaseProgram_T = cl_int (*)(cl_program);
    using clBuildProgram_T = cl_int(*)(cl_program,cl_uint,const cl_device_id *,const char *,void (*)(cl_program, void *),void *);
    using clGetProgramBuildInfo_T = cl_int (*)(cl_program,cl_device_id,cl_program_build_info,size_t,void *,size_t *);
    using clCreateKernel_T = cl_kernel (*)(cl_program, const char *, cl_int *);
    using clReleaseKernel_T = cl_int (*) (cl_kernel);

    OCLLoader(): dynLib(clLibName) {
        clCreateProgramWithIL = dynLib.getSymbol<clCreateProgramWithIL_T>("clCreateProgramWithIL");
        clReleaseProgram = dynLib.getSymbol<clReleaseProgram_T>("clReleaseProgram");
        clBuildProgram = dynLib.getSymbol<clBuildProgram_T>("clBuildProgram");
        clGetProgramBuildInfo = dynLib.getSymbol<clGetProgramBuildInfo_T>("clGetProgramBuildInfo");
        clCreateKernel = dynLib.getSymbol<clCreateKernel_T>("clCreateKernel");
        clReleaseKernel = dynLib.getSymbol<clReleaseKernel_T>("clReleaseKernel");
    }

    clCreateProgramWithIL_T clCreateProgramWithIL;
    clReleaseProgram_T clReleaseProgram;
    clBuildProgram_T clBuildProgram;
    clGetProgramBuildInfo_T clGetProgramBuildInfo;
    clCreateKernel_T clCreateKernel;
    clReleaseKernel_T clReleaseKernel;
private:
    DynamicLibHelper dynLib;
};


static OCLLoader& getClLoader() {
    static OCLLoader loader;
    return loader;
}

static std::string getClErrorString(cl_int val) {
#define CL_ENUM_VAL(arg) case arg: return #arg
    switch(val) {
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

static void checkClResult(const char* func, cl_int res) {
    if (res != CL_SUCCESS)
        reportError(std::string(func) + " failed: " + getClErrorString(res));
}

#define CHECK_CL_RESULT(arg) checkClResult(#arg, arg)

struct ClProgramDeleter {
    void operator()(cl_program program) const {
        CHECK_CL_RESULT(getClLoader().clReleaseProgram(program));
    }
};

struct ClKernelDeleter {
    void operator()(cl_kernel kernel) const {
        CHECK_CL_RESULT(getClLoader().clReleaseKernel(kernel));
    }
};

using ClProgram = std::unique_ptr<std::remove_pointer_t<cl_program>, ClProgramDeleter>;
using ClKernel = std::unique_ptr<std::remove_pointer_t<cl_kernel>, ClKernelDeleter>;

static constexpr const auto ze_be = sycl::backend::ext_oneapi_level_zero;
static constexpr const auto cl_be = sycl::backend::opencl;
}

struct GPUModule {
    sycl::queue* queue = nullptr;
    sycl::kernel_bundle<sycl::bundle_state::executable> kernelBundle;
    ze::Module l0module;
    ClProgram clProgram;
};

struct GPUKernel {
    sycl::kernel syclKernel;
    ze::Kernel l0kernel;
    ClKernel clKernel;
    uint32_t maxWgSize = 0;
};

GPUModule* createGPUModule(sycl::queue& queue, const void *data, size_t dataSize) {
    auto ctx = queue.get_context();
    auto backend = ctx.get_platform().get_backend();

    if (backend == ze_be) {
        auto zeDevice = sycl::get_native<ze_be>(
          queue.get_device());
        auto zeContext = sycl::get_native<ze_be>(
          queue.get_context());


        ze_module_desc_t desc = {};
        desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
        desc.pInputModule = static_cast<const uint8_t *>(data);
        desc.inputSize = dataSize;

        auto l0module = ze::Module::create(zeContext, zeDevice, desc).first;
        auto kernelBundle =
          sycl::make_kernel_bundle<ze_be, sycl::bundle_state::executable>(
              {l0module.get()}, ctx);
        return new GPUModule{&queue, std::move(kernelBundle), std::move(l0module), nullptr};
    }
    if (backend == cl_be) {
        auto& loader = getClLoader();

        auto clContext = sycl::get_native<cl_be>(ctx);
        auto clDevice = sycl::get_native<cl_be>(queue.get_device());

        cl_int errCode = CL_SUCCESS;
        ClProgram program(loader.clCreateProgramWithIL(clContext, data, dataSize, &errCode));
        checkClResult("clCreateProgramWithILF", errCode);

        try {
            CHECK_CL_RESULT(loader.clBuildProgram(program.get(), 1, &clDevice, "", nullptr, nullptr));
        } catch (std::exception& e) {
            size_t len = 0;
            auto ret = loader.clGetProgramBuildInfo(program.get(), clDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
            if (ret != CL_SUCCESS)
                throw;

            std::string str;
            str.resize(len);
            ret = loader.clGetProgramBuildInfo(program.get(), clDevice, CL_PROGRAM_BUILD_LOG, len, str.data(), nullptr);
            if (ret != CL_SUCCESS)
                throw;

            throw std::runtime_error(e.what() + std::string("\n") + str);
        }

        auto kernelBundle = sycl::make_kernel_bundle<cl_be, sycl::bundle_state::executable>(program.get(), ctx);
        return new GPUModule{&queue, std::move(kernelBundle), nullptr, std::move(program)};
    }

    reportError("Backend is not supported: " + std::to_string(static_cast<int>(backend)));
}

void destoyGPUModule(GPUModule* mod) {
    delete mod;
}

GPUKernel* getGPUKernel(GPUModule* mod, const char *name) {
    auto queue = mod->queue;
    assert(queue);
    auto ctx = queue->get_context();

    auto maxWgSize = static_cast<uint32_t>(queue->get_device().get_info<sycl::info::device::max_work_group_size>());
    if (mod->l0module) {
        auto zeModule = mod->l0module.get();

        ze_kernel_desc_t desc = {};
        desc.pKernelName = name;
        auto zeKernel = ze::Kernel::create(zeModule, desc);

        auto syclKernel = sycl::make_kernel<ze_be>({mod->kernelBundle, zeKernel.get()}, ctx);
        return new GPUKernel{syclKernel, std::move(zeKernel), nullptr, maxWgSize};
    }
    if (mod->clProgram) {
        auto& loader = getClLoader();

        cl_int errCode = CL_SUCCESS;
        ClKernel clKernel(loader.clCreateKernel(mod->clProgram.get(), name, &errCode));
        checkClResult("clCreateKernel", errCode);

        auto syclKernel = sycl::make_kernel<cl_be>(clKernel.get(), ctx);
        return new GPUKernel{syclKernel, nullptr, std::move(clKernel), maxWgSize};
    }
    reportError("Invalid module");
}

void destroyGPUKernel(GPUKernel* kernel) {
    delete kernel;
}

sycl::kernel getSYCLKernel(GPUKernel* kernel) {
    return kernel->syclKernel;
}

void suggestGPUBlockSize(GPUKernel* kernel, const uint32_t *gridSize, uint32_t *blockSize, size_t numDims) {
    assert(numDims > 0 && numDims <= 3);
    if (kernel->l0kernel) {
        auto l0Kernel = kernel->l0kernel.get();
        uint32_t gSize[3] = {};
        uint32_t *bSize[3] = {};
        for (size_t i = 0; i < numDims; ++i) {
          gSize[i] = gridSize[i];
          bSize[i] = &blockSize[i];
        }

        CHECK_ZE_RESULT(zeKernelSuggestGroupSize(
            l0Kernel, gSize[0], gSize[1], gSize[2], bSize[0], bSize[1], bSize[2]));
        return;
    }

    for (size_t i=0 ; i< numDims; ++i)
        blockSize[i] = 1;

    for (size_t i=0 ; i< numDims; ++i) {
        auto gsize = gridSize[i];
        if (gsize > 1) {
            blockSize[i] = std::min(gsize, kernel->maxWgSize);
            break;
        }
    }
}