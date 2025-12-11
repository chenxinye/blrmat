/**
 * Author: Xinye Chen
 * Benchmark and accuracy test for BLR Matrix vs Dense Matrix operations.
 *
 * - Detailed timing per run (Warmup + 3 runs)
 * - Speedup
 * - Accuracy verify
 */

#include "blr_matrix.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>



#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
#endif



// Generate Physics-like Matrix (Diagonally Dominant for stable LU)
void generatePhysicsLikeMatrix(double* h_A, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double val = 1.0 / (std::abs(i - j) + 1.0);
            if (i == j) val += 5.0; // Boost diagonal
            h_A[i + j * rows] = val; // Column-major
        }
    }
}

// Generate SPD Matrix for Cholesky
void generateSPDMatrix(double* h_A, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double val = 1.0 / (std::abs(i - j) + 1.0);
            if (i == j) val += (double)N; // Huge diagonal to guarantee SPD
            h_A[i + j * N] = val;
        }
    }
}

void fillRandomMatrix(double* h_A, int size) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < size; ++i) {
        h_A[i] = dis(gen);
    }
}

// Relative Frobenius Norm Error
double computeRelativeError(const double* ref, const double* res, int size) {
    double diff_sq_sum = 0.0;
    double ref_sq_sum = 0.0;
    for (int i = 0; i < size; ++i) {
        double diff = ref[i] - res[i];
        diff_sq_sum += diff * diff;
        ref_sq_sum += ref[i] * ref[i];
    }
    return std::sqrt(diff_sq_sum) / std::sqrt(ref_sq_sum);
}



double verifyLUAccuracy(BLRMatrix& blr, const double* d_A_ref, int N, cublasHandle_t handle) {
    double* d_LU_packed; // Verifies || A - L * U || / || A ||
    double *d_L, *d_U, *d_Recon;
    size_t size = N * N * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_LU_packed, size));
    CUDA_CHECK(cudaMalloc(&d_L, size));
    CUDA_CHECK(cudaMalloc(&d_U, size));
    CUDA_CHECK(cudaMalloc(&d_Recon, size));

    // Expand BLR result to dense
    blr.expandToDense(d_LU_packed, N);

    // Split L and U on CPU
    std::vector<double> h_LU(N * N);
    std::vector<double> h_L(N * N, 0.0);
    std::vector<double> h_U(N * N, 0.0);

    CUDA_CHECK(cudaMemcpy(h_LU.data(), d_LU_packed, size, cudaMemcpyDeviceToHost));

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            double val = h_LU[i + j * N];
            if (i > j) {
                h_L[i + j * N] = val;       // Strictly Lower
            } else if (i == j) {
                h_L[i + j * N] = 1.0;       // Diagonal L = 1
                h_U[i + j * N] = val;       // Diagonal U
            } else {
                h_U[i + j * N] = val;       // Strictly Upper
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(d_L, h_L.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), size, cudaMemcpyHostToDevice));

    // Recon = L * U
    double one = 1.0; double zero = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, 
                &one, d_L, N, d_U, N, &zero, d_Recon, N);

    std::vector<double> h_Recon(N * N);
    std::vector<double> h_Ref(N * N);
    CUDA_CHECK(cudaMemcpy(h_Recon.data(), d_Recon, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Ref.data(), d_A_ref, size, cudaMemcpyDeviceToHost));

    double err = computeRelativeError(h_Ref.data(), h_Recon.data(), N * N);

    cudaFree(d_LU_packed); cudaFree(d_L); cudaFree(d_U); cudaFree(d_Recon);
    return err;
}



double verifyCholeskyAccuracy(BLRMatrix& blr, const double* d_A_ref, int N, cublasHandle_t handle) {
    double* d_L_full; // Verifies || A - L * L^T || / || A ||
    double* d_Recon;
    size_t size = N * N * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_L_full, size));
    CUDA_CHECK(cudaMalloc(&d_Recon, size));

    blr.expandToDense(d_L_full, N);

    
    std::vector<double> h_L(N * N); // Zero out upper triangle manually on CPU for safety
    CUDA_CHECK(cudaMemcpy(h_L.data(), d_L_full, size, cudaMemcpyDeviceToHost));
    for(int j=0; j<N; ++j) {
        for(int i=0; i<j; ++i) h_L[i + j*N] = 0.0; 
    }
    CUDA_CHECK(cudaMemcpy(d_L_full, h_L.data(), size, cudaMemcpyHostToDevice));

    
    double one = 1.0; double zero = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, 
                &one, d_L_full, N, d_L_full, N, &zero, d_Recon, N); // Recon = L * L^T

    std::vector<double> h_Recon(N * N);
    std::vector<double> h_Ref(N * N);
    CUDA_CHECK(cudaMemcpy(h_Recon.data(), d_Recon, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Ref.data(), d_A_ref, size, cudaMemcpyDeviceToHost));

    double err = computeRelativeError(h_Ref.data(), h_Recon.data(), N * N);
    cudaFree(d_L_full); cudaFree(d_Recon);
    return err;
}

int main() {
    const int N = 10000;        
    const int B = 2000;         
    const int RHS_K = 128;     
    const double TOL = 1e-4;   

    std::cout << "==================================================" << std::endl;
    std::cout << "  BLR Matrix Benchmark & Accuracy Test            " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Matrix: " << N << "x" << N << ", Block: " << B << ", Tol: " << TOL << std::endl;
    std::cout << std::fixed << std::setprecision(3);

    cublasHandle_t handle; cublasCreate(&handle);
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

    double *d_A, *d_X, *d_Y_dense, *d_Y_blr;
    CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_X, N * RHS_K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Y_dense, N * RHS_K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Y_blr, N * RHS_K * sizeof(double)));

    std::vector<double> h_A(N * N);
    std::vector<double> h_X(N * RHS_K);
    generatePhysicsLikeMatrix(h_A.data(), N, N);
    fillRandomMatrix(h_X.data(), N * RHS_K);
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), N * RHS_K * sizeof(double), cudaMemcpyHostToDevice));

    //---------------------------------------------------
    // TEST 1: GEMM
    //---------------------------------------------------
    std::cout << "\n[Test 1] GEMM Performance" << std::endl;
    double alpha = 1.0, beta = 0.0;
    
    // 1.1 Dense GEMM
    std::cout << "   -> Running Dense cuBLAS GEMM (4 runs):" << std::endl;
    float total_dense = 0;
    for(int i=0; i<4; ++i) {
        cudaEventRecord(start);
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, RHS_K, N, 
                    &alpha, d_A, N, d_X, N, &beta, d_Y_dense, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        
        std::cout << "      Run " << i << ": " << ms << " ms";
        if (i == 0) std::cout << " (Warmup)";
        else total_dense += ms;
        std::cout << std::endl;
    }
    float avg_dense = total_dense / 3.0f;
    std::cout << "      >> Avg Time (Runs 1-3): " << avg_dense << " ms" << std::endl;

    // 1.2 BLR GEMM
    BLRMatrix blr_gemm(N, N, B, TOL);
    blr_gemm.buildFromDense(d_A, N);
    
    std::cout << "\n   -> Running BLR GEMM (4 runs):" << std::endl;
    float total_blr = 0;
    for(int i=0; i<4; ++i) {
        cudaEventRecord(start);
        blr_gemm.matmul(d_X, d_Y_blr, RHS_K, alpha, beta);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        
        std::cout << "      Run " << i << ": " << ms << " ms";
        if (i == 0) std::cout << " (Warmup)";
        else total_blr += ms;
        std::cout << std::endl;
    }
    float avg_blr = total_blr / 3.0f;
    std::cout << "      >> Avg Time (Runs 1-3): " << avg_blr << " ms" << std::endl;

    // 1.3 Speedup & Error
    double speedup = avg_dense / avg_blr;
    std::cout << "\n   [Speedup Report]" << std::endl;
    std::cout << "      Speedup (Dense/BLR): " << speedup << "x" << std::endl;

    std::vector<double> h_Y1(N*RHS_K), h_Y2(N*RHS_K);
    cudaMemcpy(h_Y1.data(), d_Y_dense, N*RHS_K*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Y2.data(), d_Y_blr, N*RHS_K*sizeof(double), cudaMemcpyDeviceToHost);
    double gemm_err = computeRelativeError(h_Y1.data(), h_Y2.data(), N*RHS_K);
    std::cout << "      GEMM Relative Error: " << std::scientific << gemm_err << std::endl;


    //---------------------------------------------------
    // TEST 2: LU Decomposition
    //---------------------------------------------------
    std::cout << "\n[Test 2] LU Decomposition" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N*N*sizeof(double), cudaMemcpyHostToDevice)); // Reset Data
    BLRMatrix blr_lu(N, N, B, TOL);
    blr_lu.buildFromDense(d_A, N);

    cudaEventRecord(start);
    blr_lu.factorizeLU();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float lu_ms; cudaEventElapsedTime(&lu_ms, start, stop);
    std::cout << "   LU Time        : " << lu_ms << " ms" << std::endl;

    std::cout << "   Verifying Accuracy (Computing A - L*U)..." << std::endl;
    double lu_err = verifyLUAccuracy(blr_lu, d_A, N, handle);
    std::cout << "   LU Error       : " << std::scientific << lu_err << std::endl;


    //---------------------------------------------------
    // TEST 3: Cholesky Decomposition
    //---------------------------------------------------
    std::cout << "\n[Test 3] Cholesky Decomposition" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    generateSPDMatrix(h_A.data(), N);
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N*N*sizeof(double), cudaMemcpyHostToDevice));

    BLRMatrix blr_chol(N, N, B, TOL);
    blr_chol.buildFromDense(d_A, N);

    cudaEventRecord(start);
    blr_chol.factorizeCholesky();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float chol_ms; cudaEventElapsedTime(&chol_ms, start, stop);
    std::cout << "   Cholesky Time  : " << chol_ms << " ms" << std::endl;

    std::cout << "   Verifying Accuracy (Computing A - L*L^T)..." << std::endl;
    double chol_err = verifyCholeskyAccuracy(blr_chol, d_A, N, handle);
    std::cout << "   Cholesky Error : " << std::scientific << chol_err << std::endl;

    // Cleanup
    cudaFree(d_A); cudaFree(d_X); cudaFree(d_Y_dense); cudaFree(d_Y_blr);
    cublasDestroy(handle);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}