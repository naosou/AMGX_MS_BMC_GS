#pragma once

#include "solver.h"

namespace amgx {

template <class T_Config>
class MSBlockMultiColorGSSolver;

DECLARE_SOLVER_FACTORY(MSBlockMultiColorGSSolver);

template <class T_Config>
class MSBlockMultiColorGSSolver : public Solver<T_Config>
{
    public:
        typedef T_Config TConfig;
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
        float omega;
};

} // namespace amgx
