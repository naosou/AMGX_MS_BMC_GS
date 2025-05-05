#pragma once

#include <string>
#include <solvers/solver.h>
#include <basic_types.h>
#include <matrix.h>

namespace amgx {

template <AMGX_MemorySpace t_memSpace, AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class MSBlockMultiColorGSSolver;

DECLARE_SOLVER_FACTORY(MSBlockMultiColorGSSolver);

template <AMGX_MemorySpace t_memSpace, AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class MSBlockMultiColorGSSolver : public Solver<TemplateConfig<t_memSpace, t_vecPrec, t_matPrec, t_indPrec>>
{
    public:
        typedef TemplateConfig<t_memSpace, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef typename TConfig::VecPrec ValueType;
        typedef typename TConfig::MemSpace MemorySpace;
        typedef typename TConfig::IndPrec IndexType;

        MSBlockMultiColorGSSolver(AMG_Config &cfg, const std::string &cfg_scope);

        void solver_setup(bool reuse_matrix_structure = true) override {}
        bool solve_iteration(ValueType &b, ValueType &x, bool xIsZero = false) override;
        void print_solver_parameters() const override;

    private:
        int alpha;
        int block_size;
        int warp_size;
        ValueType omega;
};

} // namespace amgx
