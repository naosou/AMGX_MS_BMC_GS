
#include <string.h>
#include <cutil.h>
#include <miscmath.h>
#include <amgx_cusparse.h>
#include <thrust/copy.h>
#include <solvers/ms_block_multicolor_gauss_seidel_solver.h>
#include <solvers/block_common_solver.h>
#include <gaussian_elimination.h>
#include <basic_types.h>
#include <util.h>
#include <texture.h>
#include <device_properties.h>
#include <stream.h>

#include "sm_utils.inl"

namespace amgx {

template <class T_Config>
MSBlockMultiColorGSSolver<T_Config>::MSBlockMultiColorGSSolver(AMG_Config &cfg, const std::string &cfg_scope)
{
    this->alpha = cfg.template getParameter<int>("alpha", cfg_scope);
    this->block_size = cfg.template getParameter<int>("block_size", cfg_scope);
    this->warp_size = cfg.template getParameter<int>("warp_size", cfg_scope);
    this->omega = cfg.template getParameter<ValueType>("omega", cfg_scope);
}

__global__ void ms_bmc_jacobi_kernel(const int *row_ptr, const int *col_ind, const float *values,
                                     const float *b, const float *x_old, float *x_new,
                                     const int *row_ids, int block_size, int alpha, float omega)
{
    int warp_id = blockIdx.x;
    int lane_id = threadIdx.x % 32;

    extern __shared__ float x_shared[];

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
            x_shared[i] = fmaf(omega, gs_update - x_old[row], x_old[row]);
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

template <class T_Config>
bool MSBlockMultiColorGSSolver<T_Config>::solve_iteration(ValueType &b, ValueType &x, bool xIsZero)
{
    using IndexType = typename T_Config::IndPrec;
    using Value = typename T_Config::VecPrec;

    const Matrix<T_Config> &A = *(this->matrix);
    const int num_rows = A.get_num_rows();

    std::vector<int> row_ids;
    int num_blocks = (num_rows + block_size - 1) / block_size;

    for (int b = 0; b < num_blocks; ++b)
    {
        for (int i = 0; i < block_size; ++i)
        {
            int row = b * block_size + i;
            row_ids.push_back(row < num_rows ? row : 0);
        }
    }

    int *d_row_ptr, *d_col_ind, *d_row_ids;
    float *d_vals, *d_x0, *d_x1, *d_b;

    cudaMalloc(&d_row_ptr, sizeof(int) * (num_rows + 1));
    cudaMalloc(&d_col_ind, sizeof(int) * A.get_num_nz());
    cudaMalloc(&d_vals, sizeof(float) * A.get_num_nz());
    cudaMalloc(&d_x0, sizeof(float) * num_rows);
    cudaMalloc(&d_x1, sizeof(float) * num_rows);
    cudaMalloc(&d_b, sizeof(float) * num_rows);
    cudaMalloc(&d_row_ids, sizeof(int) * row_ids.size());

    cudaMemcpy(d_row_ptr, A.row_offsets.raw(), sizeof(int) * (num_rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, A.col_indices.raw(), sizeof(int) * A.get_num_nz(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, A.values.raw(), sizeof(float) * A.get_num_nz(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x0, x.raw(), sizeof(float) * num_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.raw(), sizeof(float) * num_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ids, row_ids.data(), sizeof(int) * row_ids.size(), cudaMemcpyHostToDevice);

    dim3 grid(num_blocks);
    dim3 block(warp_size);
    size_t shmem = sizeof(float) * block_size;

    ms_bmc_jacobi_kernel<<<grid, block, shmem>>>(d_row_ptr, d_col_ind, d_vals, d_b, d_x0, d_x1, d_row_ids, block_size, alpha, omega);

    cudaMemcpy(x.raw(), d_x0, sizeof(float) * num_rows, cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_vals);
    cudaFree(d_x0);
    cudaFree(d_x1);
    cudaFree(d_b);
    cudaFree(d_row_ids);

    return true;
}

template <class T_Config>
void MSBlockMultiColorGSSolver<T_Config>::print_solver_parameters() const
{
    std::cout << "MS-BMC-GS solver parameters:" << std::endl;
    std::cout << "  alpha: " << alpha << std::endl;
    std::cout << "  block_size: " << block_size << std::endl;
    std::cout << "  warp_size: " << warp_size << std::endl;
    std::cout << "  omega: " << omega << std::endl;
}

#define AMGX_CASE_LINE(CASE) template class MSBlockMultiColorGSSolver<CASE>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_DECLARE_FACTORY(CASE) \ 
    template void MSBlockMultiColorGSSolver<CASE>::registerFactory();

} // namespace amgx
