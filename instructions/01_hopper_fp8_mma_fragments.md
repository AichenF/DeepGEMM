##### 9.7.14.5.10. [Matrix Fragments for `mma.m16n8k32`](#warp-level-matrix-fragment-mma-16832)

A warp executing `mma.m16n8k32` will compute an MMA operation of shape `.m16n8k32`.

Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds
a fragment of the matrix.

* Multiplicand A:

  + `.s8` or `.u8` or `.e4m3` or `.e5m2` or `.e3m2` or `.e2m3` or `.e2m1`:

    > | .atype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.s8` / `.u8` | A vector expression containing four `.b32` registers, with each register containing four `.s8` / `.u8` elements from the matrix A. | a0, a1, .., a14, a15 |
    > | `.e4m3` / `.e5m2` / `.e3m2` / `.e2m3` / `.e2m1` | A vector expression containing four `.b32` registers, with each register containing four `.e4m3` / `.e5m2` / `.e3m2` / `.e2m3` / `.e2m1` elements from the matrix A. | a0, a1, …, a14, a15 |
    >
    > The layout of the fragments held by different threads is shown in [Figure 88](#mma-16832-a-i8).
    >
    > The row and column of a matrix fragment can be computed as:
    >
    > ```ptx
    > groupID = %laneid >> 2
    > threadID_in_group = %laneid % 4
    >
    > row =   groupID                                        for ai where 0 <= i < 4 || 8 <= i < 12
    >        groupID + 8                                     otherwise
    >
    > col =    (threadID_in_group * 4) + (i & 0x3)           for ai where i < 8
    >          (threadID_in_group * 4) + (i & 0x3) + 16      for ai where i >= 8
    > ```

* Multiplicand B:

  + `.s8` or `.u8` or `.e4m3` or `.e5m2` or `.e3m2` or `.e2m3` or `.e2m1`:

    > | .btype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.s8` / `.u8` | A vector expression containing two `.b32` registers, with each register containing four `.s8` / `.u8` elements from the matrix B. | b0, b1, b2, b3, b4, b5, b6, b7 |
    > | `.e4m3` / `.e5m2` / `.e3m2` / `.e2m3` / `.e2m1` | A vector expression containing two `.b32` registers, with each register containing four `.e4m3` / `.e5m2` / `.e3m2` / `.e2m3` / `.e2m1` elements from the matrix B. | b0, b1, b2, b3, b4, b5, b6, b7 |
    >
    > The row and column of a matrix fragment can be computed as:
    >
    > ```ptx
    > groupID           = %laneid >> 2
    > threadID_in_group = %laneid % 4
    >
    > row =      (threadID_in_group * 4) + (i & 0x3)           for bi where i < 4
    >            (threadID_in_group * 4) + (i & 0x3) + 16      for bi where i >= 4
    >
    > col =   groupID
    > ```

* Accumulators (C or D):

  | .ctype / .dtype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.s32` | A vector expression containing four `.s32` registers, containing four `.s32` elements from the matrix C (or D). | c0, c1, c2, c3 |
  | `.f32` | A vector expression containing four `.f32` registers, containing four `.f32` elements from the matrix C (or D). | c0, c1, c2, c3 |
  | `.f16` | A vector expression containing two `.f16x2` registers, with each register containing two `.f16` elements from the matrix C (or D). | c0, c1, c2, c3 |

  The row and column of a matrix fragment can be computed as:

  ```ptx
  groupID           = %laneid >> 2
  threadID_in_group = %laneid % 4

  row =      groupID                           for ci where i <  2
           groupID + 8                         for ci where i >= 2

  col =  (threadID_in_group * 2) + (i & 0x1)    for ci where i = {0,..,3}
  ```

Alternate floating point type:

```ptx
mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32        d, a, b, c;
mma.sync.aligned.m16n8k8.row.col.f32.atype.btype.f32      d, a, b, c;
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32       d, a, b, c;
mma.sync.aligned.shape.row.col.dtype.f8type.f8type.ctype  d, a, b, c;

.f8type     = {.e4m3, .e5m2};
.ctype      = {.f16, .f32};
.dtype      = {.f16, .f32};
.shape      = {.m16n8k16, .m16n8k32};
```

The mandatory `.sync` qualifier indicates that `mma` instruction causes the executing thread to
wait until all threads in the warp execute the same `mma` instruction before resuming execution.

The mandatory `.aligned` qualifier indicates that all threads in the warp must execute the same
`mma` instruction. In conditionally executed code, a `mma` instruction should only be used if it
is known that all threads in the warp evaluate the condition identically, otherwise behavior is
undefined.

`.e4m3` and `.e5m2` alternate floating point type `mma` operation requires `sm_89` or higher.

```ptx
.reg .b32 %Ra<4>, %Rb<4>;
.reg .f32 %Rc<4>, %Rd<4>;
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32
  {%Rd0, %Rd1, %Rd2, %Rd3},
  {%Ra0, %Ra1, %Ra2, %Ra3},
  {%Rb0, %Rb1},
  {%Rc0, %Rc1, %Rc2, %Rc3};
```
