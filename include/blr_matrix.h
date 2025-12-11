/**
 *  Header for Block Low-Rank Matrix implementation using cuBLAS/cuSOLVER.
 */

#ifndef BLR_MATRIX_H
#define BLR_MATRIX_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <iostream>


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Types of tiles in the matrix
enum TileType {
    TILE_DENSE,
    TILE_LOW_RANK
};

/**
 * @brief Represents a single block in the BLR matrix.
 * * If type == TILE_DENSE: Only d_Dense is used.
 * If type == TILE_LOW_RANK: d_U and d_V are used to represent A ~ U * V^T.
 */
struct Tile {
    int row_idx;
    int col_idx;
    int rows;
    int cols;
    int rank;        // Rank for Low-Rank tile
    TileType type;

    double* d_Dense = nullptr; // Size: rows * cols
    double* d_U = nullptr;     // Size: rows * rank
    double* d_V = nullptr;     // Size: cols * rank (stored as V^T usually, but here storing V)

    ~Tile() {
        if (d_Dense) cudaFree(d_Dense);
        if (d_U) cudaFree(d_U);
        if (d_V) cudaFree(d_V);
    }
};

class BLRMatrix {
public:
    /**
     * @brief Constructor: Initializes the grid and handles.
     * @param rows Total rows of the matrix.
     * @param cols Total cols of the matrix.
     * @param blockSize The size of each square block.
     * @param tolerance SVD truncation tolerance for compression.
     */
    BLRMatrix(int rows, int cols, int blockSize, double tolerance);
    

    ~BLRMatrix();

    void expandToDense(double* d_A_dense, int lda);

    /**
     * @brief Construct BLR matrix from a flat dense GPU array.
     * Performs SVD compression on off-diagonal blocks.
     */
    void buildFromDense(const double* d_A_flat, int lda);

    /**
     * @brief Matrix-Matrix Multiplication: Y = alpha * A * X + beta * Y
     * where A is this BLR matrix, X and Y are dense matrices.
     */
    void matmul(const double* d_X, double* d_Y, int width_X, double alpha, double beta);

    /**
     * @brief Block Cholesky Decomposition (In-place).
     * Assumes the matrix is Symmetric Positive Definite (SPD).
     * Simplified logic: Standard Block Cholesky adapting to tile types.
     */
    void factorizeCholesky();

    /**
     * @brief Block LU Decomposition (In-place).
     * Simplified logic without pivoting for clarity.
     */
    void factorizeLU();

private:
    int M_, N_;           // Matrix dimensions
    int nb_;              // Block size
    int grid_rows_;       // Number of block rows
    int grid_cols_;       // Number of block cols
    double tol_;          // Compression tolerance

    cublasHandle_t cublasH_;
    cusolverDnHandle_t cusolverH_;

    // 2D grid of tiles
    std::vector<std::vector<Tile*>> tiles_;

    /**
     * @brief Helper to compress a dense block into Low-Rank (U, V) using SVD.
     */
    void compressTile(Tile* tile, const double* d_src, int lda);
};

#endif // BLR_MATRIX_H