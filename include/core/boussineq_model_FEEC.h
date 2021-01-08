#pragma once

// Deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
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
#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

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

// STL
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <string>
#include <vector>


// AquaPlanet
#include <base/config.h>
#include <base/utilities.h>
#include <core/boussineq_model_assembly_FEEC.h>
#include <core/planet_geometry.h>
#include <linear_algebra/approximate_inverse.hpp>
#include <linear_algebra/approximate_schur_complement.hpp>
#include <linear_algebra/block_schur_preconditioner.hpp>
#include <linear_algebra/inverse_matrix.hpp>
#include <linear_algebra/nested_schur_complement.hpp>
#include <linear_algebra/preconditioner.h>
#include <linear_algebra/preconditioner_block_identity.hpp>
#include <linear_algebra/schur_complement.hpp>
#include <linear_algebra/shifted_schur_complement.hpp>
#include <model_data/boussinesq_model_data.h>
#include <model_data/boussinesq_model_parameters.h>
#include <model_data/core_model_data.h>


DYCOREPLANET_OPEN_NAMESPACE

namespace ExteriorCalculus
{
  /*!
   * @class BoussinesqModel
   *
   * @brief Class to implement a 3D bouyancy Boussinesq model on a rotating spherical shell.
   *
   * Implements a bouyancy Boussinesq model in non dimensional form on a
   * rotating spherical shell, i.e., a Navier-Stokes-Model with temperature
   * forcing:
   * @f{eqnarray*}
   * \partial_t u +(u\cdot\nabla)u + 2\Omega\times u + \nabla p & =
   * \frac{1}{\mathrm{Re}}\Delta u - \rho(T) \hat g e_r\\
   * \nabla \cdot u & = 0 \\
   * \partial_t + u\cdot\nabla T & = \nabla\cdot\left(\frac{1}{\hat A}\nabla
   * T\right) + \gamma \\
   * @f}
   * where \f$\mathrm{Re}=L_{\mathrm{ref}}u_{\mathrm{ref}}/\nu\f$ is the
   * Reynolds number and \f$\hat A=L_{\mathrm{ref}}u_{\mathrm{ref}}/A\f$ is
   * something like a varying the P\'eclet number. This is necessray if the
   * thermal diffusivity is spatially varying and has high contrast which makes
   * it difficult to identify a single constant number like the P\'eclet number
   * to determine if the a regime under consideration is either dominated by
   * convective effects or by diffusive effects globally.
   * \f$\rho(T)=1-\beta(T-T_{\mathrm{ref}})\f$ is a dimensionless scaling factor
   * and \f$\hat g\f$ is the scaled gravity acceleartion given by
   * @f{eqnarray*}{
   * \hat g = \frac{L_{\mathrm{ref}}}{u_{\mathrm{ref}}^2}g
   * @f}
   *
   * Quantities marked with \f$(\cdot)_{\mathrm{ref}}\f$ are reference
   * quantities.
   */
  template <int dim>
  class BoussinesqModel : protected PlanetGeometry<dim>
  {
  public:
    BoussinesqModel(CoreModelData::Parameters &parameters);
    ~BoussinesqModel();

    void
    run();

  private:
    class Postprocessor : public DataPostprocessor<dim>
    {
    public:
      Postprocessor(const unsigned int partition);

      virtual void
      evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim> &inputs,
        std::vector<Vector<double>> &computed_quantities) const override;

      virtual std::vector<std::string>
      get_names() const override;

      virtual std::vector<
        DataComponentInterpretation::DataComponentInterpretation>
      get_data_component_interpretation() const override;

      virtual UpdateFlags
      get_needed_update_flags() const override;

    private:
      const unsigned int partition;
    };

    void
    setup_dofs();

    void
    setup_nse_matrices(const std::vector<IndexSet> &nse_partitioning,
                       const std::vector<IndexSet> &nse_relevant_partitioning);

    void
    setup_nse_preconditioner(
      const std::vector<IndexSet> &nse_partitioning,
      const std::vector<IndexSet> &nse_relevant_partitioning);

    void
    setup_temperature_matrices(
      const IndexSet &temperature_partitioning,
      const IndexSet &temperature_relevant_partitioning);

    void
    assemble_nse_preconditioner(const double time_index);

    void
    build_nse_preconditioner(const double time_index);

    void
    assemble_nse_system(const double time_index);

    void
    assemble_temperature_matrix(const double time_index);

    void
    assemble_temperature_rhs(const double time_index);

    double
    get_maximal_velocity() const;

    double
    get_cfl_number() const;

    void
    recompute_time_step();

    void
    solve_NSE_block_preconditioned();

    void
    solve_NSE_Schur_complement();

    void
    solve_temperature();

    void
    refine_and_coarsen(const unsigned int max_level);

    void
    output_results();

    void
    print_paramter_info() const;

    /*!
     * Parameter class.
     */
    CoreModelData::Parameters &parameters;

    const MappingQ<dim> temperature_mapping;

    const FESystem<dim>       nse_fe;
    DoFHandler<dim>           nse_dof_handler;
    AffineConstraints<double> nse_constraints;

    IndexSet              nse_index_set, nse_relevant_set;
    std::vector<IndexSet> nse_partitioning, nse_relevant_partitioning;

    LA::BlockSparseMatrix nse_matrix;
    LA::BlockSparseMatrix nse_preconditioner_matrix;
    LA::MPI::BlockVector  nse_rhs;

    LA::MPI::BlockVector nse_solution;
    LA::MPI::BlockVector old_nse_solution;

    FE_Q<dim>                 temperature_fe;
    DoFHandler<dim>           temperature_dof_handler;
    AffineConstraints<double> temperature_constraints;

    LA::SparseMatrix temperature_matrix;
    LA::SparseMatrix temperature_mass_matrix;
    LA::SparseMatrix temperature_advection_matrix;
    LA::SparseMatrix temperature_stiffness_matrix;
    LA::MPI::Vector  temperature_rhs;

    LA::MPI::Vector temperature_solution;
    LA::MPI::Vector old_temperature_solution;

    unsigned int timestep_number;

    std::shared_ptr<LA::PreconditionJacobi> T_preconditioner;

    bool rebuild_nse_matrix;
    bool rebuild_nse_preconditioner;
    bool rebuild_temperature_matrices;
    bool rebuild_temperature_preconditioner;

    void
    local_assemble_nse_preconditioner(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      Assembly::Scratch::NSEPreconditioner<dim> &           scratch,
      Assembly::CopyData::NSEPreconditioner<dim> &          data);

    void
    copy_local_to_global_nse_preconditioner(
      const Assembly::CopyData::NSEPreconditioner<dim> &data);

    void
    local_assemble_nse_system(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      Assembly::Scratch::NSESystem<dim> &                   scratch,
      Assembly::CopyData::NSESystem<dim> &                  data);

    void
    copy_local_to_global_nse_system(
      const Assembly::CopyData::NSESystem<dim> &data);

    void
    local_assemble_temperature_matrix(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      Assembly::Scratch::TemperatureMatrix<dim> &           scratch,
      Assembly::CopyData::TemperatureMatrix<dim> &          data);

    void
    copy_local_to_global_temperature_matrix(
      const Assembly::CopyData::TemperatureMatrix<dim> &data);

    void
    local_assemble_temperature_rhs(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      Assembly::Scratch::TemperatureRHS<dim> &              scratch,
      Assembly::CopyData::TemperatureRHS<dim> &             data);

    void
    copy_local_to_global_temperature_rhs(
      const Assembly::CopyData::TemperatureRHS<dim> &data);

    class Postprocessor;

    using VorticitySystemPreconType = LA::PreconditionAMG;
    std::shared_ptr<VorticitySystemPreconType> vorticity_system_preconditioner;

    using MassPerconditionerType = LA::PreconditionJacobi;
    using MassInverseType =
      LinearAlgebra::InverseMatrix<LA::SparseMatrix, LA::PreconditionJacobi>;
    std::shared_ptr<MassPerconditionerType> Mw_inverse_preconditioner;
    std::shared_ptr<MassPerconditionerType> Mu_inverse_preconditioner;
    std::shared_ptr<MassInverseType>        Mw_inverse;
    std::shared_ptr<MassInverseType>        Mu_inverse;

    using ApproxShiftedSchurComplementInverseType =
      LinearAlgebra::ApproxShiftedSchurComplementInverse<
        LA::BlockSparseMatrix,
        LA::MPI::Vector,
        //        VorticitySystemPreconType,
        MassPerconditionerType,
        MassPerconditionerType>;
    std::shared_ptr<ApproxShiftedSchurComplementInverseType>
      approx_Mu_minus_Sw_inverse;

    using SchurComplementLowerBlockType =
      LinearAlgebra::SchurComplementLowerBlock<
        LA::BlockSparseMatrix,
        LA::MPI::Vector,
        ApproxShiftedSchurComplementInverseType,
        MassPerconditionerType>;
    // MassInverseType>;

    std::shared_ptr<SchurComplementLowerBlockType>
      pressure_system_approx_schur_compl_matrix;

    using ApproxNesteSchurComplementInverseType =
      LinearAlgebra::ApproxNestedSchurComplementInverse<
        SchurComplementLowerBlockType,
        LA::MPI::Vector,
        DoFHandler<dim>>;
    std::shared_ptr<ApproxNesteSchurComplementInverseType>
      approx_nested_schur_complement_inverse;

    using BlockSchurPreconditionerFEECType =
      LinearAlgebra::BlockSchurPreconditionerFEEC<
        // MassInverseType,
        MassPerconditionerType,
        ApproxShiftedSchurComplementInverseType,
        ApproxNesteSchurComplementInverseType>;

    std::shared_ptr<const BlockSchurPreconditionerFEECType> preconditioner_feec;

    std::shared_ptr<const typename LinearAlgebra::PreconditionerBlockIdentity<
      DoFHandler<dim>>>
      preconditioner_identity;
  };

} // namespace ExteriorCalculus

/*
 * Extern template instantiations. Do not instantiate for dim=2 since the
 * meaning of curls is different there. This needs template specialization that
 * is not implemented yet..
 */
// extern template class ExteriorCalculus::BoussinesqModel<2>;
extern template class ExteriorCalculus::BoussinesqModel<3>;

DYCOREPLANET_CLOSE_NAMESPACE
