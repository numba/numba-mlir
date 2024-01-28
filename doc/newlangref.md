# RFC: High level GPU kernel API

Alexander Kalistratov, Ivan Butygin

## Summary

We propose new high-level kernel API (TBD)

## Motivation

Current low-level Kernel API is too verbose not very convenient for fast
prototyping.
Current high-level APIs (array API and prange), on the other hand, provide too
little low level control over GPU execution.

## Proposal

We propose a new Workgroup-level API, with direct access to Numpy array
operations and ability to acess workitem level API directly.

### Kernel definition
Simple example of pairwise distance kernel:
```
# Current OpenCL/SYCL style kernel
@kernel
def pairwise_distance_kernel(X1, X2, D):
    i, j = nb.get_global_id()

    if i < X1.shape[0] and j < X2.shape[0]:
        d = 0.0
        # calculating distance with loop by dimensions
        for k in range(X1.shape[1]):
            tmp = X1[i, k] - X2[j, k]
            d += tmp * tmp
        D[i, j] = np.sqrt(d)

# New api, immediately swithing to workitem level.
@new_kernel
def pairwise_distance_kernel(group, X1, X2, D):
    # switch to workitem level
    # parallel loop over work items
    for ind in group.wi_range():
        i, j = ind.global_id()

        if i < X1.shape[0] and j < X2.shape[0]:
            # using high-level array api to calculate distance
            d = ((X1[i] - X2[j])**2).sum()
            D[i, j] = np.sqrt(d)

# Using WG level api
@new_kernel
def pairwise_distance_kernel(group, X1, X2, D):
    gid = group.work_offset() # global offset to current WG (i.e. group_size * group_id)

    # Create tensor of specified shape, but with boundary checks of X1 and X2
    x1 = group.load(X1[gid[0]:], shape=(group.shape[0], X1.shape[1]))
    x2 = group.load(X2[gid[0]:], shape=(group.shape[0], X2.shape[1]))

    # calculating pairwise distance with numpy-style broadcasting
    diff = ((x1[None, :, :] - x2[:, None, :])**2).sum(axis=2)

    # store result to D, but with boundary checks
    group.store(D[gid[0]:, gid[1]:], np.sqrt(diff))
```
### Launching the kernel:
```
# Current kernel API
pairwise_distance_kernel[global_size, local_size](X1, X2, D)

# New API, `group` kernel arg directly corresponds to host API param, using
# to setup iteration dimensions.
group = Group(work_shape=global_size, group_shape_hint=local_size, subgroup_size_hint=sg_size)
pairwise_distance_kernel(group, X1, X2, D)

# Local and subgroup sizes are only hints and are subject to heuristica and
# autotuning.
```

### Tensors (Name TBD) and arrays

Numpy arrays passed as arguments to the kernel can be accessed directly inside
but we also provide `tensor` object as a convenient way to access data inside
the kernel.

Tensors can be of arbitrary, possibly dynamic, shape and support masking access.

Creating tensor from array
```
x1 = group.load(X1[gid[0]:], shape=(group.shape[0], X1.shape[1]))
```

Resulting tensor is always of requested shape, but if source slice was of
smaller shape, some elements will be masked.

Copying data back into array
```
group.store(D[gid[0]:, gid[1]:], tensor)
```

If tensor is masked, only active elements will be written.

Tensor data created from `group.load` can either be direct view into source
array or local copy. Any changes made to tensor may or amy not be visible to
source array. If user wants to make make changes visible, it must call
`group.store` explicitly.

Allocating new tensor:
```
arr = group.empty(shape=(...), dtype=dtyp)
arr = group.zeros(shape=(...), dtype=dtyp)
arr = group.ones(shape=(...), dtype=dtyp)
arr = group.full(shape=(...), dtype=dtyp, fill_value=...)
```
Tensors can be allocated either in Shared Local Memory or in Global memory,
actual allocation placement is left to the compler.
(Placement hints TBD)

Tensors support usual numpy operations, including fancy indexing and
broadcasting:
```
diff = ((x1[None, :, :] - x2[:, None, :])**2).sum(axis=2)
```
Numpy ops follows usual Numpy semantics by returning newly allocated tensor as
result, but compiler is heavily rely on ops fusion to remove intermedialte
allocations. If some of the intermediate allocation wasn't removed it will
result in compiler warning.

User also can pass out buffer explcitly:
```
arr = group.zeros(shape=(...), dtype=dtyp)
res = np.subtract(x1, x2, out=arr)
```
Explicit allocations won't generate such warnings, but still can be removed by
compiler due ops fusion and/or DCE.

### Switching to SubGroup or WorkItem scope

While the main execution model is WorkGroup scope execution, it's possible to
swhich to subgroup or workitem scope for convenience.

SG Level:
```
@new_kernel
def foo(group, X1, X2, D):
    for sg in group.subgroups():
        id = sg.id
        size = sg.size
```

Workitem scope:
```
@new_kernel
def foo(group, X1, X2, D):
    for wi in group.workitems():
        i, j, k = wi.id
```

Programming on workitem scope is close to usual OpenCL programming.

### Extending

Free functions:
```
@kernel.func
def add(a, b):
    return a + b

@newkernel
def foo(group, ...):
    c = add(a, b)
```
`@kernel.func` functions are callable from any (WG/SG/WI) scope.


Functions overloads for specific scope:
```
def foo(a, b):
    pass

@kernel.func(foo, scope=WorkGroup)
def foo_wg(g, a, b):
    i,j,k = g.id()
    ...

@kernel.func(foo, scope=SubGroup)
def foo_wg(sg, a, b):
    i = group.id()
    ...

@kernel.func(foo, scope=WorkItem)
def foo_wg(wi, a, b):
    i,j,k = wi.id()
    ...

@newkernel
def bar(group, ...):
    # Index objects are passed implicitly
    c1 = foo(a1, b1)
    for sg in group.subgroups():
        c2 = foo(a2, b2)

    for wi in group.workitems():
        c3 = foo(a3, b3)
```

Defining low level intrinsics/codegen:
```
def my_intrinsic(a, b):
    pass

@kernel.intrinsic(my_intrinsic, scope=WorkGroup):
def my_intrinsic_impl(a, b):
    # Can query 'a' and 'b' types here
    if is_fp16(a) and is_int8(b):
        def func(builder, a, b):
            # Use low level MLIR python builder API here, LLVM, SPIR-V or similar
            c = builder.create(spirv.call)("__my_intrinsic", a, b)
            return c

        return func

    # Can return None if intrinsic doesn't supported for specific data types.
    return None

@newkernel
def foo(group, ...):
    ...
    c = my_intrinsic(a, b)
```

Putting everything together:
```
# module device_lib.py

def my_hw_gemm(a, b, acc):
    pass

def my_tile_load(arr):
    pass

def my_tile_store(arr, data):
    pass

@kernel.intrinsic(my_hw_gemm, scope=SubGroup):
def my_hw_gemm_impl(a, b, acc):
    def func(builder, a, b, acc):
        return builder.create(spirv.call)("__my_hw_gemm", a, b, acc)

    return func

@kernel.intrinsic(my_tile_load, scope=SubGroup):
def my_tile_load_impl(arr):
    ...

@kernel.intrinsic(my_tile_store, scope=SubGroup):
def my_tile_store_impl(arr, data):
    ...

@kernel.func(foo, scope=WorkGroup)
def my_gemm(group, a, b, acc):
    i, j = group.id()
    for sg in group.subgroups():
        M, N = 8, 16 # Or from autotuning
        tile_acc = my_tile_load(...)
        for k in range(0, K, TK):
            tile_a = my_tile_load(a[i * group.shape[0]:, k:])
            tile_b = my_tile_load(b[k:, j * group.shape[1]:])
            my_hw_gemm(tile_a, tile_b, tile_acc)

        my_tile_store(acc[...], tile_acc)


# module main.py
import device_lib

@newkernel
def my_kernel(group, a, b, res, device_lib)
    acc = group.zeros(a.shape[0], b.shape[1])
    device_lib.my_gemm(a, b, acc)
    group.store(res, acc)
```

### Autotuning

(TBD)
