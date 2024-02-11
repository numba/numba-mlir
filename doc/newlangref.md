# RFC: High level GPU kernel API

Alexander Kalistratov, Ivan Butygin

## Summary

We propose new high-level kernel API (TBD)

## Motivation

Current low-level Kernel API is too verbose and not very convenient for fast
prototyping.
Current high-level APIs (array API and prange), on the other hand, provide too
little low level control over GPU execution.

## Proposal

We propose a new Workgroup-level API, with direct access to Numpy array
operations and ability to acess workitem level API directly.


### Kernel definition
Simple example of pairwise distance kernel:
```python
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
@kernel
def pairwise_distance_kernel(group, X1, X2, D):
    # switch to workitem level
    # parallel loop over work items
    @group.workitems
    def inner(ind):
        i, j = ind.global_id()

        if i < X1.shape[0] and j < X2.shape[0]:
            # using high-level array api to calculate distance
            d = ((X1[i] - X2[j])**2).sum()
            D[i, j] = np.sqrt(d)

    inner()

# Using WG level api
@kernel
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
```python
# Current kernel API
pairwise_distance_kernel[global_size, local_size](X1, X2, D)

# New API, `group` kernel arg directly corresponds to host API param, using
# to setup iteration dimensions.
group = Group(work_shape=global_size, group_shape_hint=local_size, subgroup_size_hint=sg_size)
pairwise_distance_kernel(group, X1, X2, D)

# Local and subgroup sizes are only hints and are subject to heuristica and
# autotuning.
```


### Tensors and arrays

Numpy arrays passed as arguments to the kernel can be accessed directly inside
but we also provide `tensor` object as a convenient way to access data inside
the kernel.

Tensors can be of arbitrary, possibly dynamic, shape and support masking access.

Creating tensor from array
```python
x1 = group.load(X1[gid[0]:], shape=(group.shape[0], X1.shape[1]))
```

Resulting tensor is always of requested shape, but if source slice was of
smaller shape, some elements will be masked.

Copying data back into array:
```python
group.store(D[gid[0]:, gid[1]:], tensor)
```

If tensor is masked, only active elements will be written.

Tensor data created from `group.load` can either be direct view into source
array or local copy. Any changes made to tensor may or may not be visible to
source array. If user wants to make make changes visible, it must call
`group.store` explicitly.

Allocating new tensor:
```python
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
```python
diff = ((x1[None, :, :] - x2[:, None, :])**2).sum(axis=2)
```
Numpy ops follows usual Numpy semantics by returning newly allocated tensor as
result, but compiler is heavily rely on ops fusion to remove intermedialte
allocations. If some of the intermediate allocation wasn't removed it will
result in compiler warning.

Supported Numpy ops on tensors: (TBD)

User also can pass out buffer explcitly:
```python
arr = group.zeros(shape=(...), dtype=dtyp)
res = np.subtract(x1, x2, out=arr)
```
Explicit allocations won't generate such warnings, but still can be removed by
compiler due ops fusion and/or DCE.

Tensor allocation is only allowed on workgroup level, they are not allowed on
subgroup or workitem level.


### Vectors

In addition to `tensor` objects compiler supports operations over `vector` types.
Vectors are immutable and statically-sized.


New vector allocation:
```python
arr = group.vzeros(shape=(...), dtype=dtyp)
arr = group.vones(shape=(...), dtype=dtyp)
arr = group.vfull(shape=(...), dtype=dtyp, fill_value=...)
```

Creating vector from array:
```python
x1 = group.vload(X1[gid[0]:], shape=(W, H))
```
Vector allocation shape must be deteminable at compile time.

Creating vector from tensor:
```python
x1 = group.load(X1[gid[0]:], shape=(W, H)).vec();
x2 = group.load(X1[gid[0]:], shape=(group.shape[0], X1.shape[1]))[x:x+W,y:y+W].vec()
x3 = group.load(X1[gid[0]:], shape=(group.shape[0], X1.shape[1])).vec(shape=(W,H))
```
`vec()` shape must be deteminable at compile time. If shape is omitted, source
tensor shape will be used and must be static.


Storing vector back into array/tensor:
```python
group.store(D[gid[0]:, gid[1]:], vector)
```
Note: we don't distinguish between tensor/vector store and they have the same
semantics.

Vectors can be masked. If source tensor was masked, resulting vector will be
masked as well.

Vector operations are permitted on workgroup, subgroup and workitem level.

Vectors generally support same set of Numpy ops as Tensors. Numpy-style
broadcasting is supported. `out=` argument is not supported.

Supported Numpy ops on vectors: (TBD)


### Masking

Tensors and vectors are masked, i.e. individual elements can be marked active or
inactive.

`group.load/vload` will mark elements which are outside of requested shape as
inactive.

`group.store` will only update destination elements which have source mask
active.

Assigning tensor elements via `[]` will mark them as active.

There is intentionally no way to mark element as inactive other than creating
new tensor/vector.

For numpy functions operating on tensors/vectors, specific element will be
marked as active only if all source elements it accesses are marked as active.

Reduction functions will only consider active elements.

Allocation functions `group.(v)zeros`,`group.(v)ones`,`group.(v)full` will mark
all elements as active, `group.empty` will ask all elements as inactive.

Mask is allocated in the same storage type as tensor/vector data. In some cases
(allocation function) compiler can elide mask allocation and return pseudo-mask,
always active.

Mask can be accessed directly via `tensor.mask` property. Returned mask will
have same shape as the original array and has dtype of `bool`. Returned mask is
read-only. Active elements are marked as `False` as the returned mask (following
Numpy masked arrays convention).


### Switching to SubGroup or WorkItem scope

While the main execution model is WorkGroup scope execution, it's possible to
swihch to subgroup or workitem scope for convenience.

SG Level:
```python
@kernel
def foo(group, X1, X2, D):
    @group.subgroups
    def inner(sg):
        id = sg.subgroup_id()
        size = sg.size()

    inner()
```

Workitem scope:
```python
@kernel
def foo(group, X1, X2, D):
    @group.workitems
    def inner(wi):
        i, j, k = wi.global_id()

    inner()
```

Programming on workitem scope is close to usual OpenCL programming.


### Extending

Free functions:
```python
@kernel.func
def add(a, b):
    return a + b

@kernel
def foo(group, ...):
    c = add(a, b)
```
`@kernel.func` functions are callable from any (WG/SG/WI) scope.


Functions overloads for specific scope:
```python
def foo(a, b):
    pass

@kernel.func(foo, scope=WorkGroup)
def foo_wg(g, a, b):
    i,j,k = g.group_id()
    ...

@kernel.func(foo, scope=SubGroup)
def foo_wg(sg, a, b):
    i = sg.subgroup_id()
    ...

@kernel.func(foo, scope=WorkItem)
def foo_wg(wi, a, b):
    i,j,k = wi.global_id()
    ...

@kernel
def bar(group, ...):
    # Index objects are passed implicitly
    c1 = foo(a1, b1)
    @group.subgroups
    def inner1(sg):
        c2 = foo(a2, b2)

    inner1()

    @group.workitems
    def inner2(wi):
        c3 = foo(a3, b3)

    inner2()
```

Defining low level intrinsics/codegen:
```python
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

@kernel
def foo(group, ...):
    ...
    c = my_intrinsic(a, b)
```

Putting everything together:
```python
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
    @group.subgroups
    def inner(sg):
        M, N = 8, 16 # Or from autotuning
        tile_acc = my_tile_load(...)
        for k in range(0, K, TK):
            tile_a = my_tile_load(a[i * group.shape[0]:, k:])
            tile_b = my_tile_load(b[k:, j * group.shape[1]:])
            my_hw_gemm(tile_a, tile_b, tile_acc)

        my_tile_store(acc[...], tile_acc)

    inner()


# module main.py
import device_lib

@kernel
def my_kernel(group, a, b, res, device_lib)
    acc = group.zeros(a.shape[0], b.shape[1])
    device_lib.my_gemm(a, b, acc)
    group.store(res, acc)
```


### Autotuning

(TBD)
