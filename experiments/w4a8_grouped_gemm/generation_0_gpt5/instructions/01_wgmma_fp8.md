##### WGMMA Operation
- **Instruction**: `wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16`
- **Tile Size**: Computes a 64x64 output tile using 64x16 (A) and 16x64 (B) input tiles
- **Data Flow**: D[64x64] = A[64x16] x B[16x64] + D[64x64]
- **K-dimension**: Must iterate K/16 times to accumulate full result

##### Instruction Syntax

###### Description

Instruction wgmma.mma_async issues a MxNxK matrix multiply and accumulate operation, D = A*B+D, where the A matrix is MxK, the B matrix is KxN, and the D matrix is MxN.

The operation of the form D = A*B is issued when the input predicate argument scale-d is false.

wgmma.fence instruction must be used to fence the register accesses of wgmma.mma_async instruction from their prior accesses. Otherwise, the behavior is undefined.

wgmma.commit_group and wgmma.wait_group operations must be used to wait for the completion of the asynchronous matrix multiply and accumulate operations before the results are accessed.

Register operand d represents the accumulator matrix as well as the destination matrix, distributed across the participating threads. Register operand a represents the multiplicand matrix A in register distributed across the participating threads. The 64-bit register operands a-desc and b-desc are the matrix descriptors which represent the multiplicand matrices A and B in shared memory respectively. The contents of a matrix descriptor must be same across all the warps in the warpgroup. The format of the matrix descriptor is described in Matrix Descriptor Format.

Matrices A and B are stored in row-major and column-major format respectively. For certain floating point variants, the input matrices A and B can be transposed by specifying the value 1 for the immediate integer arguments imm-trans-a and imm-trans-b respectively. A value of 0 can be used to avoid the transpose operation. The valid values of imm-trans-a and imm-trans-b are 0 and 1. The transpose operation is only supported for the wgmma.mma_async variants with .f16/ .bf16 types on matrices accessed from shared memory using matrix descriptors.

###### FP8 floating point type

wgmma.mma_async.sync.aligned.shape.dtype.atype.btype  d, a-desc, b-desc, scale-d, imm-scale-a, imm-scale-b;

wgmma.mma_async.sync.aligned.shape.dtype.atype.btype  d, a, b-desc, scale-d, imm-scale-a, imm-scale-b;

.shape   = {.m64n8k32, .m64n16k32, .m64n24k32, .m64n32k32,
            .m64n40k32, .m64n48k32, .m64n56k32, .m64n64k32,
            .m64n72k32, .m64n80k32, .m64n88k32, .m64n96k32,
            .m64n104k32, .m64n112k32, .m64n120k32, .m64n128k32,
            .m64n136k32, .m64n144k32, .m64n152k32, .m64n160k32,
            .m64n168k32, .m64n176k32, .m64n184k32, .m64n192k32,
            .m64n200k32, .m64n208k32, .m64n216k32, .m64n224k32,
            .m64n232k32, .m64n240k32, .m64n248k32, .m64n256k32};
.atype  = {.e4m3, .e5m2};
.btype  = {.e4m3, .e5m2};
.dtype  = {.f16, .f32};
