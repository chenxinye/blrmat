/**
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 * 
 * Implementation of high-performance BLR operations.
 */



#include "blr_matrix.h"
#include <algorithm>
#include <cmath>
#include <cstdio>


void check_cublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) { // Helper wrapper for cuBLAS/cuSOLVER errors
        fprintf(stderr, "cuBLAS Error code: %d\n", status); exit(1);
    }
}
void check_cusolver(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "cuSOLVER Error code: %d\n", status); exit(1);
    }
}

BLRMatrix::BLRMatrix(int rows, int cols, int blockSize, double tolerance)
    : M_(rows), N_(cols), nb_(blockSize), tol_(tolerance) {
    
    check_cublas(cublasCreate(&cublasH_));
    check_cusolver(cusolverDnCreate(&cusolverH_));

    grid_rows_ = (rows + blockSize - 1) / blockSize;
    grid_cols_ = (cols + blockSize - 1) / blockSize;

    tiles_.resize(grid_rows_);
    for (int i = 0; i < grid_rows_; ++i) {
        tiles_[i].resize(grid_cols_, nullptr);
        for (int j = 0; j < grid_cols_; ++j) {
            tiles_[i][j] = new Tile();
            tiles_[i][j]->row_idx = i;
            tiles_[i][j]->col_idx = j;
            
            // Handle edge cases for last block
            tiles_[i][j]->rows = std::min(blockSize, rows - i * blockSize);
            tiles_[i][j]->cols = std::min(blockSize, cols - j * blockSize);
        }
    }
}

BLRMatrix::~BLRMatrix() {
    for (auto& row : tiles_) {
        for (auto& tile : row) {
            delete tile;
        }
    }
    cublasDestroy(cublasH_);
    cusolverDnDestroy(cusolverH_);
}



void BLRMatrix::compressTile(Tile* tile, const double* d_src, int lda) {
    int m = tile->rows;
    int n = tile->cols;

    // Define constants for cuBLAS pointer arguments
    double one = 1.0;
    double zero = 0.0;

    // Allocate temporary memory for SVD
    double *d_U_full, *d_S, *d_VT_full;
    double *d_work;
    int *d_info, lwork;
    int min_dim = std::min(m, n);

    CUDA_CHECK(cudaMalloc(&d_U_full, m * m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_S, min_dim * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_VT_full, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // Query workspace for SVD (gesvd)
    check_cusolver(cusolverDnDgesvd_bufferSize(
        cusolverH_, m, n, &lwork));
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));

    // We need a copy of src because gesvd destroys input
    double* d_A_copy;
    CUDA_CHECK(cudaMalloc(&d_A_copy, m * n * sizeof(double)));
    

    // Copy sub-matrix from global matrix to contiguous block
    check_cublas(cublasDgeam(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N, 
                             m, n, 
                             &one, d_src, lda, 
                             &zero, nullptr, lda, // B=nullptr
                             d_A_copy, m));

    // Perform SVD: A = U * S * VT
    check_cusolver(cusolverDnDgesvd(
        cusolverH_, 'S', 'S', m, n, d_A_copy, m,
        d_S, d_U_full, m, d_VT_full, n,
        d_work, lwork, nullptr, d_info));

    // --- Truncation Logic ---
    std::vector<double> h_S(min_dim);
    CUDA_CHECK(cudaMemcpy(h_S.data(), d_S, min_dim * sizeof(double), cudaMemcpyDeviceToHost));

    int rank = 0;
    for (int k = 0; k < min_dim; ++k) {
        if (h_S[k] > tol_) rank++;
    }
    if (rank == 0) rank = 1; 
    tile->rank = rank;

    // Allocate Compressed Tiles
    CUDA_CHECK(cudaMalloc(&tile->d_U, m * rank * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&tile->d_V, n * rank * sizeof(double))); 

    // Copy U (subset)
    check_cublas(cublasDgeam(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
                             m, rank, 
                             &one, d_U_full, m, 
                             &zero, nullptr, m, // B=nullptr
                             tile->d_U, m));

    // Prepare V: U <- U * diag(S)
    check_cublas(cublasDdgmm(cublasH_, CUBLAS_SIDE_RIGHT, m, rank, 
                             tile->d_U, m, d_S, 1, tile->d_U, m));

    // V <- VT_full (first rank rows), then Transpose
    double* d_VT_trunc; // rank x n
    CUDA_CHECK(cudaMalloc(&d_VT_trunc, rank * n * sizeof(double)));
    

    // Extract top rank rows of VT
    check_cublas(cublasDgeam(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
                             rank, n, 
                             &one, d_VT_full, n, 
                             &zero, nullptr, n, // B=nullptr
                             d_VT_trunc, rank));
    
    // Transpose VT to get V (n x rank)
    check_cublas(cublasDgeam(cublasH_, CUBLAS_OP_T, CUBLAS_OP_N,
                             n, rank, 
                             &one, d_VT_trunc, rank, 
                             &zero, nullptr, n, // B=nullptr, ldb doesn't matter much but 'n' is safe
                             tile->d_V, n));

    // Cleanup
    cudaFree(d_U_full); cudaFree(d_S); cudaFree(d_VT_full); 
    cudaFree(d_work); cudaFree(d_info); cudaFree(d_A_copy); cudaFree(d_VT_trunc);
}

void BLRMatrix::buildFromDense(const double* d_A_flat, int lda) {
    double one = 1.0;
    double zero = 0.0;

    for (int i = 0; i < grid_rows_; ++i) {
        for (int j = 0; j < grid_cols_; ++j) {
            Tile* tile = tiles_[i][j];
            const double* d_block_start = d_A_flat + (i * nb_) + (j * nb_) * lda;

            if (i == j) {
                // Diagonal: Keep Dense
                tile->type = TILE_DENSE;
                CUDA_CHECK(cudaMalloc(&tile->d_Dense, tile->rows * tile->cols * sizeof(double)));
                
                check_cublas(cublasDgeam(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
                                         tile->rows, tile->cols, 
                                         &one, d_block_start, lda,
                                         &zero, nullptr, lda, // B=nullptr
                                         tile->d_Dense, tile->rows));
            } else {
                // Off-Diagonal: Compress
                tile->type = TILE_LOW_RANK;
                compressTile(tile, d_block_start, lda);
            }
        }
    }
}


//---------------------------------------------------
// Matrix-Matrix Multiplication (GEMM)
//---------------------------------------------------
// Y = alpha * A * X + beta * Y
void BLRMatrix::matmul(const double* d_X, double* d_Y, int width_X, double alpha, double beta) {
    double one = 1.0;
    double zero = 0.0;

    // Scale Y by beta first
    int total_rows = M_;
    check_cublas(cublasDscal(cublasH_, total_rows * width_X, &beta, d_Y, 1));

    // Loop through blocks
    for (int i = 0; i < grid_rows_; ++i) {
        for (int j = 0; j < grid_cols_; ++j) {
            Tile* tile = tiles_[i][j];
            
            // Input X chunk for this block column
            const double* d_X_ptr = d_X + (j * nb_); 
            
            // Output Y chunk for this block row
            double* d_Y_ptr = d_Y + (i * nb_);

            if (tile->type == TILE_DENSE) {
                // Standard Dense GEMM: Y_i += A_ij * X_j
                check_cublas(cublasDgemm(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
                                         tile->rows, width_X, tile->cols,
                                         &alpha,
                                         tile->d_Dense, tile->rows,
                                         d_X_ptr, M_, 
                                         &one,        
                                         d_Y_ptr, M_));
            } else {
                // Low-Rank GEMM: Y_i += (U * V^T) * X_j
                // 1. Tmp = V^T * X_j (rank x width_X)
                // 2. Y_i += U * Tmp  (rows x width_X)
                
                double* d_tmp;
                CUDA_CHECK(cudaMalloc(&d_tmp, tile->rank * width_X * sizeof(double)));

                // 1. Tmp = V^T * X
                check_cublas(cublasDgemm(cublasH_, CUBLAS_OP_T, CUBLAS_OP_N,
                                         tile->rank, width_X, tile->cols,
                                         &one, 
                                         tile->d_V, tile->cols,
                                         d_X_ptr, M_,
                                         &zero, 
                                         d_tmp, tile->rank));

                // 2. Y += U * Tmp
                check_cublas(cublasDgemm(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
                                         tile->rows, width_X, tile->rank,
                                         &alpha,
                                         tile->d_U, tile->rows,
                                         d_tmp, tile->rank,
                                         &one,
                                         d_Y_ptr, M_));
                
                cudaFree(d_tmp);
            }
        }
    }
}

//---------------------------------------------------
// Cholesky Decomposition
//---------------------------------------------------
void BLRMatrix::factorizeCholesky() {
    double one = 1.0;
    double minus_one = -1.0;
    
    int* d_info;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    
    // 1. Pre-calculate max workspace size needed to avoid malloc in loop
    //    We assume the diagonal blocks (Dense) are the largest operations.
    int max_lwork = 0;
    int cur_lwork = 0;
    
    // Check buffer size for the first block (representative of max size nb x nb)
    if (grid_cols_ > 0) {
        Tile* A_00 = tiles_[0][0];
        check_cusolver(cusolverDnDpotrf_bufferSize(
            cusolverH_, CUBLAS_FILL_MODE_LOWER, 
            A_00->rows, A_00->d_Dense, A_00->rows, &cur_lwork));
        max_lwork = cur_lwork;
    }

    double* d_work = nullptr;
    if (max_lwork > 0) {
        CUDA_CHECK(cudaMalloc(&d_work, max_lwork * sizeof(double)));
    }

    for (int k = 0; k < grid_cols_; ++k) {
        Tile* A_kk = tiles_[k][k];

        // 1. Factorize Diagonal (POTRF) -> Dense
        // We use the pre-allocated d_work
        check_cusolver(cusolverDnDpotrf(cusolverH_, CUBLAS_FILL_MODE_LOWER, 
                                        A_kk->rows, A_kk->d_Dense, A_kk->rows, 
                                        d_work, max_lwork, d_info));

        // Optional: Synchronize/Check d_info here if you suspect numerical issues
        // int h_info = 0;
        // cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
        // if (h_info != 0) printf("Warning: Cholesky pivot %d failed\n", k);

        // 2. Panel Update (TRSM)
        // Solve L_ik * L_kk^T = A_ik  =>  L_ik = A_ik * L_kk^{-T}
        for (int i = k + 1; i < grid_rows_; ++i) {
            Tile* A_ik = tiles_[i][k];
            
            if (A_ik->type == TILE_DENSE) {
                // Dense Case: Standard TRSM
                // X * L^T = B  (Side Right, Transpose L)
                check_cublas(cublasDtrsm(cublasH_, 
                                         CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, 
                                         CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                                         A_ik->rows, A_ik->cols,
                                         &one,
                                         A_kk->d_Dense, A_kk->rows,
                                         A_ik->d_Dense, A_ik->rows));
            } else {
                // Low Rank Case: A_ik = U * V^T
                // We want: (U * V^T) * L_kk^{-T} = U * (V^T * L_kk^{-T})
                // Let V_new^T = V^T * L_kk^{-T}
                // Transpose both sides: V_new = L_kk^{-1} * V
                // This is a LEFT solve on V.
                
                // V is stored as (cols x rank) in memory?
                // Wait, in buildFromDense/compressTile: 
                // We perform Transpose at the end. 
                // So tile->d_V stores the matrix V (size: n x rank).
                // Correct.
                
                // Solve L_kk * X = V  =>  X = L_kk^{-1} * V
                // Side: Left. 
                // Uplo: Lower (L_kk is Lower).
                // Op: NoTrans (we use L directly).
                // M = rows of V (which is tile->cols / original N dim), N = rank
                check_cublas(cublasDtrsm(cublasH_,
                                         CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                         CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                         A_ik->cols, A_ik->rank, // m, n
                                         &one,
                                         A_kk->d_Dense, A_kk->rows,
                                         A_ik->d_V, A_ik->cols)); // ldb = rows of V
            }
        }

        // 3. Trailing Submatrix Update (Schur Complement)
        // A_ij = A_ij - A_ik * A_jk^T
        // Only implementing Dense-Dense update for simplicity in this demo
        for (int i = k + 1; i < grid_rows_; ++i) {
            for (int j = k + 1; j <= i; ++j) {
                Tile* A_ij = tiles_[i][j];
                Tile* A_ik = tiles_[i][k]; 
                Tile* A_jk = tiles_[j][k]; 

                if (A_ij->type == TILE_DENSE && A_ik->type == TILE_DENSE && A_jk->type == TILE_DENSE) {
                     check_cublas(cublasDgemm(cublasH_, CUBLAS_OP_N, CUBLAS_OP_T,
                                              A_ij->rows, A_ij->cols, A_ik->cols,
                                              &minus_one,
                                              A_ik->d_Dense, A_ik->rows,
                                              A_jk->d_Dense, A_jk->rows,
                                              &one,
                                              A_ij->d_Dense, A_ij->rows));
                }
            }
        }
    }
    
    CUDA_CHECK(cudaFree(d_info));
    if(d_work) CUDA_CHECK(cudaFree(d_work));
}



//---------------------------------------------------
//  LU Decomposition
//---------------------------------------------------
void BLRMatrix::factorizeLU() {
    int* d_info; cudaMalloc(&d_info, sizeof(int));
    int lwork = 0;
    double* d_work = nullptr;

    for (int k = 0; k < std::min(grid_rows_, grid_cols_); ++k) {
        Tile* A_kk = tiles_[k][k];

        check_cusolver(cusolverDnDgetrf_bufferSize(cusolverH_, A_kk->rows, A_kk->cols, A_kk->d_Dense, A_kk->rows, &lwork));
        if (d_work) cudaFree(d_work);
        cudaMalloc(&d_work, lwork * sizeof(double));
        
        int* d_ipiv; cudaMalloc(&d_ipiv, A_kk->rows * sizeof(int));
        
        check_cusolver(cusolverDnDgetrf(cusolverH_, A_kk->rows, A_kk->cols, 
                                        A_kk->d_Dense, A_kk->rows, 
                                        d_work, d_ipiv, d_info));

        cudaFree(d_ipiv);
    }
    cudaFree(d_info);
    if (d_work) cudaFree(d_work);
}





//---------------------------------------------------
//  Back to Dense
//---------------------------------------------------
void BLRMatrix::expandToDense(double* d_A_dense, int lda) {
    double one = 1.0;
    double zero = 0.0;
    
    // Fill with zero first just in case
    // (Optional, dependent on if we fill all blocks)
    // cudaMemset(d_A_dense, 0, M_ * N_ * sizeof(double)); 

    for (int i = 0; i < grid_rows_; ++i) {
        for (int j = 0; j < grid_cols_; ++j) {
            Tile* tile = tiles_[i][j];
            
            // Calculate pointer to the block in the global dense matrix
            double* d_block_dst = d_A_dense + (i * nb_) + (j * nb_) * lda;
            
            if (tile->type == TILE_DENSE) {
                // Copy Dense tile directly
                check_cublas(cublasDgeam(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
                                         tile->rows, tile->cols,
                                         &one, tile->d_Dense, tile->rows,
                                         &zero, nullptr, tile->rows,
                                         d_block_dst, lda));
            } else {
                // Expand Low-Rank tile: Block = U * V^T
                // tile->d_U is (rows x rank)
                // tile->d_V is (cols x rank) -- stored as V, need V^T
                
                check_cublas(cublasDgemm(cublasH_, CUBLAS_OP_N, CUBLAS_OP_T,
                                         tile->rows, tile->cols, tile->rank,
                                         &one,
                                         tile->d_U, tile->rows,
                                         tile->d_V, tile->cols,
                                         &zero,
                                         d_block_dst, lda));
            }
        }
    }
}