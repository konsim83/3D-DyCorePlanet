#pragma once

// Deal.ii
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

// Aquaplanet3D
#include <base/config.h>

DYCOREPLANET_OPEN_NAMESPACE

namespace ExtersiorCalculus
{
  namespace Assembly
  {
    namespace Scratch
    {
      ////////////////////////////////////
      /// NSE system
      ////////////////////////////////////

      template <int dim>
      struct NSESystem
      {
        NSESystem(const double              time_step,
                  const double              time_index,
                  const FiniteElement<dim> &nse_fe,
                  const Mapping<dim> &      mapping,
                  const Quadrature<dim> &   nse_quadrature,
                  const UpdateFlags         nse_update_flags,
                  const FiniteElement<dim> &temperature_fe,
                  const UpdateFlags         temperature_update_flags);

        NSESystem(const NSESystem<dim> &data);

        FEValues<dim> temperature_fe_values;
        FEValues<dim> nse_fe_values;

        std::vector<Tensor<1, dim>> phi_w;
        std::vector<Tensor<1, dim>> curl_phi_w;
        std::vector<Tensor<1, dim>> phi_u;
        std::vector<double>         div_phi_u;
        std::vector<double>         phi_p;

        std::vector<double>         old_temperature_values;
        std::vector<Tensor<1, dim>> old_velocity_values;
        std::vector<Tensor<1, dim>> old_vorticity_values;

        const double time_step;
        const double time_index;
      };


      ////////////////////////////////////
      /// Temperature matrix
      ////////////////////////////////////

      template <int dim>
      struct TemperatureMatrix
      {
        TemperatureMatrix(const double              time_step,
                          const double              time_index,
                          const FiniteElement<dim> &temperature_fe,
                          const Mapping<dim> &      mapping,
                          const Quadrature<dim> &   temperature_quadrature);

        TemperatureMatrix(const TemperatureMatrix<dim> &data);

        FEValues<dim> temperature_fe_values;

        std::vector<double>         phi_T;
        std::vector<Tensor<1, dim>> grad_phi_T;

        const double time_step;
        const double time_index;
      };


      ////////////////////////////////////
      /// Temperture RHS
      ////////////////////////////////////

      template <int dim>
      struct TemperatureRHS
      {
        TemperatureRHS(const double              time_step,
                       const double              time_index,
                       const FiniteElement<dim> &temperature_fe,
                       const FiniteElement<dim> &nse_fe,
                       const Mapping<dim> &      mapping,
                       const Quadrature<dim> &   quadrature);

        TemperatureRHS(const TemperatureRHS<dim> &data);

        FEValues<dim> temperature_fe_values;
        FEValues<dim> nse_fe_values;

        std::vector<double>         phi_T;
        std::vector<Tensor<1, dim>> grad_phi_T;

        std::vector<Tensor<1, dim>> old_velocity_values;
        std::vector<double>         old_temperature_values;
        std::vector<Tensor<1, dim>> old_temperature_grads;

        const double time_step;
        const double time_index;
      };

    } // namespace Scratch



    namespace CopyData
    {
      ////////////////////////////////////
      /// NSE system copy
      ////////////////////////////////////

      template <int dim>
      struct NSESystem
      {
        NSESystem(const FiniteElement<dim> &nse_fe);

        NSESystem(const NSESystem<dim> &data);

        NSESystem<dim> &
        operator=(const NSESystem<dim> &) = default;

        Vector<double>                       local_rhs;
        FullMatrix<double>                   local_matrix;
        std::vector<types::global_dof_index> local_dof_indices;
      };


      ////////////////////////////////////
      /// Temperature system copy
      ////////////////////////////////////

      template <int dim>
      struct TemperatureMatrix
      {
        TemperatureMatrix(const FiniteElement<dim> &temperature_fe);

        TemperatureMatrix(const TemperatureMatrix<dim> &data);

        FullMatrix<double> local_mass_matrix;
        FullMatrix<double> local_advection_matrix;
        FullMatrix<double> local_stiffness_matrix;

        std::vector<types::global_dof_index> local_dof_indices;
      };


      ////////////////////////////////////
      /// Temperature RHS copy
      ////////////////////////////////////

      template <int dim>
      struct TemperatureRHS
      {
        TemperatureRHS(const FiniteElement<dim> &temperature_fe);
        TemperatureRHS(const TemperatureRHS<dim> &data);
        Vector<double>                       local_rhs;
        std::vector<types::global_dof_index> local_dof_indices;
        FullMatrix<double>                   matrix_for_bc;
      };

    } // namespace CopyData
  }   // namespace Assembly
} // namespace ExtersiorCalculus

// Extern template instantiations
extern template class ExtersiorCalculus::Assembly::Scratch::NSEPreconditioner<
  2>;
extern template class ExtersiorCalculus::Assembly::Scratch::NSESystem<2>;
extern template class ExtersiorCalculus::Assembly::Scratch::TemperatureMatrix<
  2>;
extern template class ExtersiorCalculus::Assembly::Scratch::TemperatureRHS<2>;

extern template class ExtersiorCalculus::Assembly::CopyData::NSEPreconditioner<
  2>;
extern template class ExtersiorCalculus::Assembly::CopyData::NSESystem<2>;
extern template class ExtersiorCalculus::Assembly::CopyData::TemperatureMatrix<
  2>;
extern template class ExtersiorCalculus::Assembly::CopyData::TemperatureRHS<2>;

extern template class ExtersiorCalculus::Assembly::Scratch::NSEPreconditioner<
  3>;
extern template class ExtersiorCalculus::Assembly::Scratch::NSESystem<3>;
extern template class ExtersiorCalculus::Assembly::Scratch::TemperatureMatrix<
  3>;
extern template class ExtersiorCalculus::Assembly::Scratch::TemperatureRHS<3>;

extern template class ExtersiorCalculus::Assembly::CopyData::NSEPreconditioner<
  3>;
extern template class ExtersiorCalculus::Assembly::CopyData::NSESystem<3>;
extern template class ExtersiorCalculus::Assembly::CopyData::TemperatureMatrix<
  3>;
extern template class ExtersiorCalculus::Assembly::CopyData::TemperatureRHS<3>;

DYCOREPLANET_CLOSE_NAMESPACE
