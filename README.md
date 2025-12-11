# blrmat: BLR Matrix Solver

A lightweight **Block Low-Rank (BLR) Matrix** library implemented in CUDA.  This project leverages **cuBLAS** and **cuSOLVER** to accelerate dense matrix operations by compressing off-diagonal blocks into low-rank approximations ($U \times V^T$) via SVD.

## Features

* **Adaptive Compression**: automatically identifies and compresses low-rank blocks using SVD based on a specified tolerance.
* **High-Performance Kernels**:
    * **GEMM**: Matrix-Matrix multiplication exploiting low-rank structure ($O(N \cdot k)$ vs $O(N^2)$).
    * **Cholesky Decomposition**: In-place factorization for Symmetric Positive Definite (SPD) matrices using Left-Looking TRSM updates.
    * **LU Decomposition**: Standard block LU factorization (simplified proof-of-concept).
* **Accuracy Verification**: Built-in validation routines to compute Frobenius norm errors against dense reference solutions.


##  Project Structure

```text
.
├── CMakeLists.txt       # Build configuration
├── main_test.cu         # Benchmark and Test Suite
├── include/
│   └── blr_matrix.h     # Header definitions
└── src/
    └── blr_matrix.cu    # CUDA implementation
```


## Simulations


All tests were performed on a Dell PowerEdge R750xa server with 2 TB of memory. The system is equipped with four NVIDIA A100 80GB PCIe GPUs and two Intel Xeon Gold 6330 processors (totaling 56 cores, 2.00 GHz). Benchmarks were conducted on a single GPU instance. Check details on [front.convergence.lip6](https://front.convergence.lip6.fr/).


The GEMM with BLR format reaches close to 2 times speedup compared with cuBLAS GEMM in our example:

```
==================================================
  BLR Matrix Benchmark & Accuracy Test            
==================================================
Matrix: 10000x10000, Block: 2000, Tol: 0.0001

[Test 1] GEMM Performance
   -> Running Dense cuBLAS GEMM (4 runs):
      Run 0: 43.435 ms (Warmup)
      Run 1: 3.485 ms
      Run 2: 3.473 ms
      Run 3: 3.471 ms
      >> Avg Time (Runs 1-3): 3.476 ms

   -> Running BLR GEMM (4 runs):
      Run 0: 81.950 ms (Warmup)
      Run 1: 2.147 ms
      Run 2: 1.937 ms
      Run 3: 1.736 ms
      >> Avg Time (Runs 1-3): 1.940 ms

   [Speedup Report]
      Speedup (Dense/BLR): 1.792x
      GEMM Relative Error: 5.342e-08

[Test 2] LU Decomposition
   LU Time        : 171.044 ms
   Verifying Accuracy (Computing A - L*U)...
   LU Error       : 7.426e-02

[Test 3] Cholesky Decomposition
   Cholesky Time  : 255.214 ms
   Verifying Accuracy (Computing A - L*L^T)...
   Cholesky Error : 1.197e-09

```
