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

#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
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


// STL
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <string>


// AquaPlanet
#include <base/config.h>
#include <core/aqua_planet.h>
#include <model_data/boussinesq_model_data.h>


AQUAPLANET_OPEN_NAMESPACE

class BoussinesqModel : protected AquaPlanet
{
public:
  struct Parameters;

  BoussinesqModel(Parameters &parameters);
  ~BoussinesqModel();

  void
  run();

private:
  void
  setup_dofs();
  void
  assemble_nse_preconditioner();
  void
  build_nse_preconditioner();
  void
  assemble_nse_system();
  void
  assemble_temperature_matrix();
  void
  assemble_temperature_system();
  double
  get_maximal_velocity() const;
  double
  get_cfl_number() const;
  void
  solve();
  void
  output_results();

public:
  struct Parameters
  {
    Parameters(const std::string &parameter_filename);

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);

    double       final_time;
    unsigned int initial_global_refinement;

    double       nse_theta;
    unsigned int nse_velocity_degree;
    bool         use_locally_conservative_discretization;

    double       temperature_theta;
    unsigned int temperature_degree;
  };

private:
  Parameters &parameters;

  const MappingQ<3> mapping;

  const FESystem<3>         nse_fe;
  DoFHandler<3>             nse_dof_handler;
  AffineConstraints<double> nse_constraints;

  LA::BlockSparseMatrix nse_matrix;
  LA::BlockSparseMatrix nse_mass_matrix;
  LA::BlockSparseMatrix nse_advection_matrix;
  LA::BlockSparseMatrix nse_diffusion_matrix;
  LA::BlockSparseMatrix nse_coriolis_matrix;
  LA::MPI::BlockVector  nse_rhs;
  LA::MPI::BlockVector  nse_solution;
  LA::MPI::BlockVector  old_nse_solution;

  FE_Q<3>                   temperature_fe;
  DoFHandler<3>             temperature_dof_handler;
  AffineConstraints<double> temperature_constraints;

  LA::SparseMatrix temperature_matrix;
  LA::SparseMatrix temperature_mass_matrix;
  LA::SparseMatrix temperature_advection_matrix;
  LA::SparseMatrix temperature_stiffness_matrix;
  LA::MPI::Vector  temperature_solution;
  LA::MPI::Vector  old_temperature_solution;
  LA::MPI::Vector  temperature_rhs;

  double       time_step;
  unsigned int timestep_number;
};

AQUAPLANET_CLOSE_NAMESPACE
