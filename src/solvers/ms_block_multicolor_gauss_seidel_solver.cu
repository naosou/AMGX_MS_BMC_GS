// ms_block_multicolor_gauss_seidel_solver.cu
// MS-BMC-Jacobi CUDA kernel (warp-per-block) with shared memory and full parameterization

#include <vector>
#include <set>
#include <algorithm>
#include <omp.h>
#include "matrix.h"
#include "solver.h"
#include "solver_factory.h"
#include "metis.h"

#include <cuda_runtime.h>

namespace amgx {

// CUDA kernel: warp-per-block, weighted Jacobi smoothing
__global__ void ms_bmc_jacobi_kernel(const int *row_ptr, const int *col_ind, const float *values,
                                     const float *b, const float *x_old, float *x_new,
                                     const int *row_ids, int block_size, int alpha, float weight)
{
    int warp_id = blockIdx.x;
    int lane_id = threadIdx.x % 32;

    extern __shared__ float x_shared[]; // shared memory buffer, size = block_size

    for (int rep = 0; rep < alpha; ++rep)
    {
        for (int i = lane_id; i < block_size; i += 32)
        {
            int row = row_ids[warp_id * block_size + i];
            float diag = 0.0f;
            float sum = 0.0f;

            for (int jj = row_ptr[row]; jj < row_ptr[row + 1]; ++jj)
            {
                int col = col_ind[jj];
                float val = __ldg(&values[jj]);
                if (col == row)
                    diag = val;
                else
                    sum += val * __ldg(&x_old[col]);
            }

            float gs_update = (b[row] - sum) / diag;
            x_shared[i] = fmaf(weight, gs_update - x_old[row], x_old[row]);
        }
        __syncthreads();

        for (int i = lane_id; i < block_size; i += 32)
        {
            int row = row_ids[warp_id * block_size + i];
            x_new[row] = x_shared[i];
        }
        __syncthreads();

        float *tmp = (float *)x_old;
        x_old = x_new;
        x_new = tmp;
    }
}

template <typename ValueType>
class MSBlockMultiColorwJACOBISolver : public Solver<ValueType>
{
public:
    void initialize();
    void solve(Matrix<ValueType> &A, std::vector<ValueType> &x_host, const std::vector<ValueType> &b_host);
    static void registerSolver(SolverFactory<ValueType> &factory);

private:
    int alpha;
    int block_size;
    float weight;
    int warp_size;

    std::vector<std::vector<int>> block_to_rows;
    void partition_blocks(const Matrix<ValueType> &A);
};

template <typename ValueType>
void MSBlockMultiColorwJACOBISolver<ValueType>::initialize()
{
    this->alpha = this->getParameter<int>("alpha", 1);
    this->block_size = this->getParameter<int>("block_size", 64);
    this->weight = this->getParameter<ValueType>("weight", 0.8f);
    this->warp_size = this->getParameter<int>("warp_size", 32); // now customizable
}

template <typename ValueType>
void MSBlockMultiColorwJACOBISolver<ValueType>::partition_blocks(const Matrix<ValueType> &A)
{
    int num_rows = A.num_rows;
    int num_blocks = (num_rows + block_size - 1) / block_size;
    block_to_rows.resize(num_blocks);
    for (int i = 0; i < num_rows; ++i)
        block_to_rows[i / block_size].push_back(i);
}

template <typename ValueType>
void MSBlockMultiColorwJACOBISolver<ValueType>::solve(Matrix<ValueType> &A, std::vector<ValueType> &x_host, const std::vector<ValueType> &b_host)
{
    if (block_to_rows.empty())
        partition_blocks(A);

    int num_blocks = block_to_rows.size();
    std::vector<int> row_ids;
    for (const auto &blk : block_to_rows)
    {
        for (int i : blk)
            row_ids.push_back(i);
        int pad = block_size - blk.size();
        for (int i = 0; i < pad; ++i)
            row_ids.push_back(0);
    }

    int *d_row_ptr, *d_col_ind, *d_row_ids;
    float *d_vals, *d_x0, *d_x1, *d_b;

    cudaMalloc(&d_row_ptr, sizeof(int) * (A.num_rows + 1));
    cudaMalloc(&d_col_ind, sizeof(int) * A.nnz());
    cudaMalloc(&d_vals, sizeof(float) * A.nnz());
    cudaMalloc(&d_x0, sizeof(float) * A.num_rows);
    cudaMalloc(&d_x1, sizeof(float) * A.num_rows);
    cudaMalloc(&d_b, sizeof(float) * A.num_rows);
    cudaMalloc(&d_row_ids, sizeof(int) * row_ids.size());

    cudaMemcpy(d_row_ptr, A.row_offsets, sizeof(int) * (A.num_rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, A.col_indices, sizeof(int) * A.nnz(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, A.values, sizeof(float) * A.nnz(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x0, x_host.data(), sizeof(float) * A.num_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_host.data(), sizeof(float) * A.num_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ids, row_ids.data(), sizeof(int) * row_ids.size(), cudaMemcpyHostToDevice);

    dim3 grid(num_blocks);
    dim3 block(warp_size); // warp based assignment
    size_t shmem = sizeof(float) * block_size;

    ms_bmc_jacobi_kernel<<<grid, block, shmem>>>(d_row_ptr, d_col_ind, d_vals, d_b, d_x0, d_x1, d_row_ids, block_size, alpha, weight);

    cudaMemcpy(x_host.data(), d_x0, sizeof(float) * A.num_rows, cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_vals);
    cudaFree(d_x0);
    cudaFree(d_x1);
    cudaFree(d_b);
    cudaFree(d_row_ids);
}

template <typename ValueType>
void MSBlockMultiColorwJACOBISolver<ValueType>::registerSolver(SolverFactory<ValueType> &factory)
{
    factory.registerSolver("MS_BLOCK_MULTICOLOR_wJACOBI", []() -> Solver<ValueType> *
                           { return new MSBlockMultiColorwJACOBISolver<ValueType>(); });
}

template class MSBlockMultiColorwJACOBISolver<float>;
template class MSBlockMultiColorwJACOBISolver<double>;

template void MSBlockMultiColorwJACOBISolver<float>::registerSolver(SolverFactory<float> &);
template void MSBlockMultiColorwJACOBISolver<double>::registerSolver(SolverFactory<double> &);

} // namespace amgx
