#include <core/boussinesq_model.h>
#include <core/planet_geometry.h>

DYCOREPLANET_OPEN_NAMESPACE

namespace Standard
{
  //////////////////////////////////////////////////////
  /// Standard Boussinesq model in H1-L2
  //////////////////////////////////////////////////////

  template <int dim>
  BoussinesqModel<dim>::BoussinesqModel(CoreModelData::Parameters &parameters_)
    : PlanetGeometry<dim>(parameters_.physical_constants.R0,
                          parameters_.physical_constants.R1)
    , parameters(parameters_)
    , mapping(3)
    , nse_fe(FE_Q<dim>(parameters.nse_velocity_degree),
             dim,
             (parameters.use_locally_conservative_discretization ?
                static_cast<const FiniteElement<dim> &>(
                  FE_DGP<dim>(parameters.nse_velocity_degree - 1)) :
                static_cast<const FiniteElement<dim> &>(
                  FE_Q<dim>(parameters.nse_velocity_degree - 1))),
             1)
    , nse_dof_handler(this->triangulation)
    , temperature_fe(parameters.temperature_degree)
    , temperature_dof_handler(this->triangulation)
    , timestep_number(0)
  {
    TimerOutput::Scope timing_section(
      this->computing_timer,
      "BoussinesqModel - constructor and grid rescaling");

    /*
     * Rescale the original this->triangulation to the one scaled by the
     * reference length.
     */
    //    GridTools::scale(1 / parameters.reference_quantities.length,
    //                     this->triangulation);
  }



  template <int dim>
  BoussinesqModel<dim>::~BoussinesqModel()
  {}



  /////////////////////////////////////////////////////////////
  // System and dof setup
  /////////////////////////////////////////////////////////////

  template <int dim>
  void
  BoussinesqModel<dim>::setup_nse_matrices(
    const std::vector<IndexSet> &nse_partitioning,
    const std::vector<IndexSet> &nse_relevant_partitioning)
  {
    nse_matrix.clear();
    TrilinosWrappers::BlockSparsityPattern sp(nse_partitioning,
                                              nse_partitioning,
                                              nse_relevant_partitioning,
                                              this->mpi_communicator);
    Table<2, DoFTools::Coupling>           coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        if (!((c == dim) && (d == dim)))
          coupling[c][d] = DoFTools::always;
        else
          coupling[c][d] = DoFTools::none;

    DoFTools::make_sparsity_pattern(nse_dof_handler,
                                    coupling,
                                    sp,
                                    nse_constraints,
                                    false,
                                    Utilities::MPI::this_mpi_process(
                                      this->mpi_communicator));
    sp.compress();

    nse_matrix.reinit(sp);
    nse_mass_matrix.reinit(sp);
    nse_advection_matrix.reinit(sp);
    nse_diffusion_matrix.reinit(sp);
    nse_coriolis_matrix.reinit(sp);
  }



  template <int dim>
  void
  BoussinesqModel<dim>::setup_nse_preconditioner(
    const std::vector<IndexSet> &nse_partitioning,
    const std::vector<IndexSet> &nse_relevant_partitioning)
  {
    Amg_preconditioner.reset();
    Mp_preconditioner.reset();

    nse_preconditioner_matrix.clear();
    TrilinosWrappers::BlockSparsityPattern sp(nse_partitioning,
                                              nse_partitioning,
                                              nse_relevant_partitioning,
                                              this->mpi_communicator);

    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        if (c == d)
          coupling[c][d] = DoFTools::always;
        else
          coupling[c][d] = DoFTools::none;

    DoFTools::make_sparsity_pattern(nse_dof_handler,
                                    coupling,
                                    sp,
                                    nse_constraints,
                                    false,
                                    Utilities::MPI::this_mpi_process(
                                      this->mpi_communicator));
    sp.compress();

    nse_preconditioner_matrix.reinit(sp);
  }



  template <int dim>
  void
  BoussinesqModel<dim>::setup_temperature_matrices(
    const IndexSet &temperature_partitioner,
    const IndexSet &temperature_relevant_partitioner)
  {
    T_preconditioner.reset();
    temperature_mass_matrix.clear();
    temperature_stiffness_matrix.clear();
    temperature_matrix.clear();

    TrilinosWrappers::SparsityPattern sp(temperature_partitioner,
                                         temperature_partitioner,
                                         temperature_relevant_partitioner,
                                         this->mpi_communicator);
    DoFTools::make_sparsity_pattern(temperature_dof_handler,
                                    sp,
                                    temperature_constraints,
                                    false,
                                    Utilities::MPI::this_mpi_process(
                                      this->mpi_communicator));
    sp.compress();

    temperature_matrix.reinit(sp);
    temperature_mass_matrix.reinit(sp);
    temperature_advection_matrix.reinit(sp);
    temperature_stiffness_matrix.reinit(sp);
  }



  template <int dim>
  void
  BoussinesqModel<dim>::setup_dofs()
  {
    TimerOutput::Scope timing_section(
      this->computing_timer, "BoussinesqModel - setup dofs of systems");

    /*
     * Setup dof handlers for nse and temperature
     */
    std::vector<unsigned int> nse_sub_blocks(dim + 1, 0);
    nse_sub_blocks[dim] = 1;

    nse_dof_handler.distribute_dofs(nse_fe);
    if (parameters.use_schur_complement_solver)
      {
        DoFRenumbering::Cuthill_McKee(nse_dof_handler);
        //  DoFRenumbering::boost::king_ordering(nse_dof_handler);
      }

    DoFRenumbering::component_wise(nse_dof_handler, nse_sub_blocks);

    temperature_dof_handler.distribute_dofs(temperature_fe);

    /*
     * Count dofs
     */
    std::vector<types::global_dof_index> nse_dofs_per_block(2);
    DoFTools::count_dofs_per_block(nse_dof_handler,
                                   nse_dofs_per_block,
                                   nse_sub_blocks);
    const unsigned int n_u = nse_dofs_per_block[0], n_p = nse_dofs_per_block[1],
                       n_T = temperature_dof_handler.n_dofs();

    /*
     * Comma separated large numbers
     */
    std::locale s = this->pcout.get_stream().getloc();
    this->pcout.get_stream().imbue(std::locale(""));

    /*
     * Print some mesh and dof info
     */
    this->pcout << "Number of active cells: "
                << this->triangulation.n_global_active_cells() << " (on "
                << this->triangulation.n_levels() << " levels)" << std::endl
                << "Number of degrees of freedom: " << n_u + n_p + n_T << " ("
                << n_u << '+' << n_p << '+' << n_T << ')' << std::endl
                << std::endl;
    this->pcout.get_stream().imbue(s);

    /*
     * Setup partitioners to store what dofs and matrix entries are stored on
     * the local processor
     */
    IndexSet temperature_partitioning(n_T),
      temperature_relevant_partitioning(n_T);

    {
      nse_index_set = nse_dof_handler.locally_owned_dofs();
      nse_partitioning.push_back(nse_index_set.get_view(0, n_u));
      nse_partitioning.push_back(nse_index_set.get_view(n_u, n_u + n_p));
      DoFTools::extract_locally_relevant_dofs(nse_dof_handler,
                                              nse_relevant_set);
      nse_relevant_partitioning.push_back(nse_relevant_set.get_view(0, n_u));
      nse_relevant_partitioning.push_back(
        nse_relevant_set.get_view(n_u, n_u + n_p));
      temperature_partitioning = temperature_dof_handler.locally_owned_dofs();
      DoFTools::extract_locally_relevant_dofs(
        temperature_dof_handler, temperature_relevant_partitioning);
    }


    /*
     * Setup constraints and boundary values for NSE. Make sure this is
     * consistent with the initial data.
     */
    {
      nse_constraints.clear();
      nse_constraints.reinit(nse_relevant_set);
      DoFTools::make_hanging_node_constraints(nse_dof_handler, nse_constraints);

      FEValuesExtractors::Vector velocity_components(0);

      // No-slip on boundary 0 (lower)
      VectorTools::interpolate_boundary_values(
        nse_dof_handler,
        0,
        Functions::ZeroFunction<dim>(dim + 1),
        nse_constraints,
        nse_fe.component_mask(velocity_components));

      // No-flux on upper boundary
      std::set<types::boundary_id> no_normal_flux_boundaries;
      no_normal_flux_boundaries.insert(1);

      VectorTools::compute_no_normal_flux_constraints(nse_dof_handler,
                                                      0,
                                                      no_normal_flux_boundaries,
                                                      nse_constraints,
                                                      mapping);

      nse_constraints.close();
    }

    /*
     * Setup temperature constraints and boundary values
     */
    {
      temperature_constraints.clear();
      temperature_constraints.reinit(temperature_relevant_partitioning);
      DoFTools::make_hanging_node_constraints(temperature_dof_handler,
                                              temperature_constraints);
      // Lower boundary
      VectorTools::interpolate_boundary_values(
        temperature_dof_handler,
        0,
        CoreModelData::Boussinesq::TemperatureInitialValues<dim>(
          parameters.physical_constants.R0, parameters.physical_constants.R1),
        temperature_constraints);

      temperature_constraints.close();
    }

    /*
     * Setup the matrix and vector objects.
     */
    setup_nse_matrices(nse_partitioning, nse_relevant_partitioning);
    setup_nse_preconditioner(nse_partitioning, nse_relevant_partitioning);

    setup_temperature_matrices(temperature_partitioning,
                               temperature_relevant_partitioning);

    nse_rhs.reinit(nse_partitioning,
                   nse_relevant_partitioning,
                   this->mpi_communicator,
                   true);
    nse_solution.reinit(nse_relevant_partitioning, this->mpi_communicator);
    old_nse_solution.reinit(nse_solution);

    temperature_rhs.reinit(temperature_partitioning,
                           temperature_relevant_partitioning,
                           this->mpi_communicator,
                           true);
    temperature_solution.reinit(temperature_relevant_partitioning,
                                this->mpi_communicator);
    old_temperature_solution.reinit(temperature_solution);
  }



  /////////////////////////////////////////////////////////////
  // Assembly NSE preconditioner
  /////////////////////////////////////////////////////////////


  template <int dim>
  void
  BoussinesqModel<dim>::local_assemble_nse_preconditioner(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::NSEPreconditioner<dim> &           scratch,
    Assembly::CopyData::NSEPreconditioner<dim> &          data)
  {
    const unsigned int dofs_per_cell = nse_fe.dofs_per_cell;
    const unsigned int n_q_points = scratch.nse_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    const double one_over_reynolds_number =
      (1. / CoreModelData::get_reynolds_number(
              parameters.reference_quantities.velocity,
              parameters.reference_quantities.length,
              parameters.physical_constants.kinematic_viscosity));

    scratch.nse_fe_values.reinit(cell);

    cell->get_dof_indices(data.local_dof_indices);
    data.local_matrix = 0;

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            scratch.phi_u[k] = scratch.nse_fe_values[velocities].value(k, q);
            scratch.grad_phi_u[k] =
              scratch.nse_fe_values[velocities].gradient(k, q);
            scratch.phi_p[k] = scratch.nse_fe_values[pressure].value(k, q);
          }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            data.local_matrix(i, j) +=
              (scratch.phi_u[i] * scratch.phi_u[j] +
               parameters.time_step * one_over_reynolds_number *
                 scalar_product(scratch.grad_phi_u[i], scratch.grad_phi_u[j]) +
               scratch.phi_p[i] * scratch.phi_p[j]) *
              scratch.nse_fe_values.JxW(q);
      }
  }



  template <int dim>
  void
  BoussinesqModel<dim>::copy_local_to_global_nse_preconditioner(
    const Assembly::CopyData::NSEPreconditioner<dim> &data)
  {
    nse_constraints.distribute_local_to_global(data.local_matrix,
                                               data.local_dof_indices,
                                               nse_preconditioner_matrix);
  }


  template <int dim>
  void
  BoussinesqModel<dim>::assemble_nse_preconditioner(const double time_index)
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Assembly NSE preconditioner");

    nse_preconditioner_matrix = 0;
    const QGauss<dim> quadrature_formula(parameters.nse_velocity_degree + 1);
    using CellFilter =
      FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

    WorkStream::run(
      CellFilter(IteratorFilters::LocallyOwnedCell(),
                 nse_dof_handler.begin_active()),
      CellFilter(IteratorFilters::LocallyOwnedCell(), nse_dof_handler.end()),
      std::bind(&BoussinesqModel<dim>::local_assemble_nse_preconditioner,
                this,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3),
      std::bind(&BoussinesqModel<dim>::copy_local_to_global_nse_preconditioner,
                this,
                std::placeholders::_1),
      Assembly::Scratch::NSEPreconditioner<dim>(parameters.time_step,
                                                time_index,
                                                nse_fe,
                                                quadrature_formula,
                                                mapping,
                                                update_JxW_values |
                                                  update_values |
                                                  update_gradients),
      Assembly::CopyData::NSEPreconditioner<dim>(nse_fe));

    nse_preconditioner_matrix.compress(VectorOperation::add);
  }



  template <int dim>
  void
  BoussinesqModel<dim>::build_nse_preconditioner(const double time_index)
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Build NSE preconditioner");

    this->pcout
      << "   Assembling and building Navier-Stokes block preconditioner..."
      << std::flush;

    assemble_nse_preconditioner(time_index);

    std::vector<std::vector<bool>> constant_modes;
    FEValuesExtractors::Vector     velocity_components(0);
    DoFTools::extract_constant_modes(nse_dof_handler,
                                     nse_fe.component_mask(velocity_components),
                                     constant_modes);
    Mp_preconditioner =
      std::make_shared<TrilinosWrappers::PreconditionJacobi>();
    Amg_preconditioner = std::make_shared<TrilinosWrappers::PreconditionAMG>();

    TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
    Amg_data.constant_modes        = constant_modes;
    Amg_data.elliptic              = true;
    Amg_data.higher_order_elements = true;
    Amg_data.smoother_sweeps       = 1;
    Amg_data.aggregation_threshold = 0.02;

    Mp_preconditioner->initialize(nse_preconditioner_matrix.block(1, 1));
    Amg_preconditioner->initialize(nse_preconditioner_matrix.block(0, 0),
                                   Amg_data);

    this->pcout << std::endl;
  }



  /////////////////////////////////////////////////////////////
  // Assembly NSE system
  /////////////////////////////////////////////////////////////

  template <int dim>
  void
  BoussinesqModel<dim>::local_assemble_nse_system(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::NSESystem<dim> &                   scratch,
    Assembly::CopyData::NSESystem<dim> &                  data)
  {
    const unsigned int dofs_per_cell =
      scratch.nse_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points = scratch.nse_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    const double one_over_reynolds_number =
      (1. / CoreModelData::get_reynolds_number(
              parameters.reference_quantities.velocity,
              parameters.reference_quantities.length,
              parameters.physical_constants.kinematic_viscosity));

    scratch.nse_fe_values.reinit(cell);

    typename DoFHandler<dim>::active_cell_iterator temperature_cell(
      &this->triangulation,
      cell->level(),
      cell->index(),
      &temperature_dof_handler);

    scratch.temperature_fe_values.reinit(temperature_cell);

    data.local_matrix = 0;
    data.local_rhs    = 0;

    scratch.temperature_fe_values.get_function_values(
      old_temperature_solution, scratch.old_temperature_values);

    scratch.nse_fe_values[velocities].get_function_values(
      old_nse_solution, scratch.old_velocity_values);
    scratch.nse_fe_values[velocities].get_function_gradients(
      old_nse_solution, scratch.old_velocity_grads);

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double old_temperature = scratch.old_temperature_values[q];
        const double density_scaling = CoreModelData::density_scaling(
          parameters.physical_constants.density,
          old_temperature,
          parameters.reference_quantities.temperature_ref);
        const Tensor<1, dim> old_velocity = scratch.old_velocity_values[q];
        const Tensor<2, dim> old_velocity_grads =
          transpose(scratch.old_velocity_grads[q]);

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            scratch.phi_u[k] = scratch.nse_fe_values[velocities].value(k, q);

            scratch.grads_phi_u[k] = transpose(
              scratch.nse_fe_values[velocities].symmetric_gradient(k, q));

            scratch.div_phi_u[k] =
              scratch.nse_fe_values[velocities].divergence(k, q);

            scratch.phi_p[k] = scratch.nse_fe_values[pressure].value(k, q);
          }

        const Tensor<1, dim> coriolis =
          parameters.reference_quantities.length *
          CoreModelData::coriolis_vector(scratch.nse_fe_values.quadrature_point(
                                           q),
                                         parameters.physical_constants.omega) /
          parameters.reference_quantities.velocity;

        /*
         * Move everything to the LHS here.
         */
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            data.local_matrix(i, j) +=
              (scratch.phi_u[i] * scratch.phi_u[j] // mass term
               + parameters.time_step *
                   (one_over_reynolds_number * 2 * scratch.grads_phi_u[i] *
                    scratch.grads_phi_u[j]) // eps(v):sigma(eps(u))
               -
               (scratch.div_phi_u[i] *
                scratch
                  .phi_p[j]) // div(v)*p ---> we solve for scaled pressure dt*p
               - (scratch.phi_p[i] * scratch.div_phi_u[j]) // q*div(u)
               ) *
              scratch.nse_fe_values.JxW(q);

        const Tensor<1, dim> gravity =
          (parameters.reference_quantities.length /
           (parameters.reference_quantities.velocity *
            parameters.reference_quantities.velocity)) *
          CoreModelData::gravity_vector(
            scratch.nse_fe_values.quadrature_point(q),
            parameters.physical_constants.gravity_constant);

        /*
         * This is only the RHS
         */
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          data.local_rhs(i) +=
            (scratch.phi_u[i] * old_velocity +
             parameters.time_step * density_scaling * gravity *
               scratch.phi_u[i] -
             parameters.time_step * scratch.phi_u[i] *
               (old_velocity * old_velocity_grads) // advection at previous time
             - (dim == 2 ? -parameters.time_step * 2 *
                             parameters.physical_constants.omega *
                             scratch.phi_u[i] * cross_product_2d(old_velocity) :
                           parameters.time_step * 2 * scratch.phi_u[i] *
                             cross_product_3d(coriolis,
                                              old_velocity)) // coriolis force)
             ) *
            scratch.nse_fe_values.JxW(q);
      }

    cell->get_dof_indices(data.local_dof_indices);
  }



  template <int dim>
  void
  BoussinesqModel<dim>::copy_local_to_global_nse_system(
    const Assembly::CopyData::NSESystem<dim> &data)
  {
    nse_constraints.distribute_local_to_global(data.local_matrix,
                                               data.local_rhs,
                                               data.local_dof_indices,
                                               nse_matrix,
                                               nse_rhs);
  }



  template <int dim>
  void
  BoussinesqModel<dim>::assemble_nse_system(const double time_index)
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Assemble NSE system");

    this->pcout << "   Assembling Navier-Stokes system..." << std::flush;

    nse_matrix           = 0;
    nse_mass_matrix      = 0;
    nse_advection_matrix = 0;
    nse_coriolis_matrix  = 0;
    nse_diffusion_matrix = 0;

    nse_rhs = 0;

    const QGauss<dim> quadrature_formula(parameters.nse_velocity_degree + 1);
    using CellFilter =
      FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

    WorkStream::run(
      CellFilter(IteratorFilters::LocallyOwnedCell(),
                 nse_dof_handler.begin_active()),
      CellFilter(IteratorFilters::LocallyOwnedCell(), nse_dof_handler.end()),
      std::bind(&BoussinesqModel<dim>::local_assemble_nse_system,
                this,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3),
      std::bind(&BoussinesqModel<dim>::copy_local_to_global_nse_system,
                this,
                std::placeholders::_1),
      Assembly::Scratch::NSESystem<dim>(parameters.time_step,
                                        time_index,
                                        nse_fe,
                                        mapping,
                                        quadrature_formula,
                                        (update_values |
                                         update_quadrature_points |
                                         update_JxW_values | update_gradients),
                                        temperature_fe,
                                        update_values),
      Assembly::CopyData::NSESystem<dim>(nse_fe));

    nse_matrix.compress(VectorOperation::add);
    nse_rhs.compress(VectorOperation::add);

    this->pcout << std::endl;
  }



  /////////////////////////////////////////////////////////////
  // Assembly temperature
  /////////////////////////////////////////////////////////////

  template <int dim>
  void
  BoussinesqModel<dim>::local_assemble_temperature_matrix(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::TemperatureMatrix<dim> &           scratch,
    Assembly::CopyData::TemperatureMatrix<dim> &          data)
  {
    const unsigned int dofs_per_cell =
      scratch.temperature_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points =
      scratch.temperature_fe_values.n_quadrature_points;

    const double one_over_peclet_number =
      (1. / CoreModelData::get_peclet_number(
              parameters.reference_quantities.velocity,
              parameters.reference_quantities.length,
              parameters.physical_constants.thermal_diffusivity));

    scratch.temperature_fe_values.reinit(cell);

    cell->get_dof_indices(data.local_dof_indices);

    data.local_mass_matrix      = 0;
    data.local_advection_matrix = 0;
    data.local_stiffness_matrix = 0;

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            scratch.grad_phi_T[k] =
              scratch.temperature_fe_values.shape_grad(k, q);
            scratch.phi_T[k] = scratch.temperature_fe_values.shape_value(k, q);
          }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              data.local_mass_matrix(i, j) +=
                (scratch.phi_T[i] * scratch.phi_T[j] *
                 scratch.temperature_fe_values.JxW(q));

              /*
               * TODO!!!
               */
              data.local_advection_matrix(i, j) += 0;

              data.local_stiffness_matrix(i, j) +=
                (one_over_peclet_number * scratch.grad_phi_T[i] *
                 scratch.grad_phi_T[j] * scratch.temperature_fe_values.JxW(q));
            }
      }
  }



  template <int dim>
  void
  BoussinesqModel<dim>::copy_local_to_global_temperature_matrix(
    const Assembly::CopyData::TemperatureMatrix<dim> &data)
  {
    temperature_constraints.distribute_local_to_global(data.local_mass_matrix,
                                                       data.local_dof_indices,
                                                       temperature_mass_matrix);

    temperature_constraints.distribute_local_to_global(
      data.local_stiffness_matrix,
      data.local_dof_indices,
      temperature_stiffness_matrix);
  }



  template <int dim>
  void
  BoussinesqModel<dim>::assemble_temperature_matrix(const double time_index)
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Assemble temperature matrices");

    this->pcout << "   Assembling temperature matrix..." << std::flush;

    temperature_mass_matrix      = 0;
    temperature_advection_matrix = 0;
    temperature_stiffness_matrix = 0;

    const QGauss<dim> quadrature_formula(parameters.temperature_degree + 2);

    using CellFilter =
      FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

    WorkStream::run(
      CellFilter(IteratorFilters::LocallyOwnedCell(),
                 temperature_dof_handler.begin_active()),
      CellFilter(IteratorFilters::LocallyOwnedCell(),
                 temperature_dof_handler.end()),
      std::bind(&BoussinesqModel<dim>::local_assemble_temperature_matrix,
                this,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3),
      std::bind(&BoussinesqModel<dim>::copy_local_to_global_temperature_matrix,
                this,
                std::placeholders::_1),
      Assembly::Scratch::TemperatureMatrix<dim>(parameters.time_step,
                                                time_index,
                                                temperature_fe,
                                                mapping,
                                                quadrature_formula),
      Assembly::CopyData::TemperatureMatrix<dim>(temperature_fe));

    temperature_mass_matrix.compress(VectorOperation::add);
    temperature_advection_matrix.compress(VectorOperation::add);
    temperature_stiffness_matrix.compress(VectorOperation::add);

    this->pcout << std::endl;
  }



  /////////////////////////////////////////////////////////////
  // Assembly temperature RHS
  /////////////////////////////////////////////////////////////


  template <int dim>
  void
  BoussinesqModel<dim>::local_assemble_temperature_rhs(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::TemperatureRHS<dim> &              scratch,
    Assembly::CopyData::TemperatureRHS<dim> &             data)
  {
    const unsigned int dofs_per_cell =
      scratch.temperature_fe_values.get_fe().dofs_per_cell;

    const unsigned int n_q_points =
      scratch.temperature_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector velocities(0);

    data.local_rhs     = 0;
    data.matrix_for_bc = 0;

    cell->get_dof_indices(data.local_dof_indices);

    scratch.temperature_fe_values.reinit(cell);

    typename DoFHandler<dim>::active_cell_iterator nse_cell(
      &this->triangulation, cell->level(), cell->index(), &nse_dof_handler);
    scratch.nse_fe_values.reinit(nse_cell);

    scratch.temperature_fe_values.get_function_values(
      old_temperature_solution, scratch.old_temperature_values);
    scratch.temperature_fe_values.get_function_gradients(
      old_temperature_solution, scratch.old_temperature_grads);

    scratch.nse_fe_values[velocities].get_function_values(
      nse_solution, scratch.old_velocity_values);

    const double one_over_peclet_number =
      (1. / CoreModelData::get_peclet_number(
              parameters.reference_quantities.velocity,
              parameters.reference_quantities.length,
              parameters.physical_constants.thermal_diffusivity));

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            scratch.phi_T[k] = scratch.temperature_fe_values.shape_value(k, q);
            scratch.grad_phi_T[k] =
              scratch.temperature_fe_values.shape_grad(k, q);
          }

        const double gamma =
          (parameters.reference_quantities.length /
           (parameters.reference_quantities.velocity *
            parameters.reference_quantities.temperature_ref)) *
          0; // CoreModelData::Boussinesq::TemperatureRHS value at quad point

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            data.local_rhs(i) +=
              (scratch.phi_T[i] * scratch.old_temperature_values[q] -
               parameters.time_step / (parameters.NSE_solver_interval) *
                 scratch.phi_T[i] * scratch.old_velocity_values[q] *
                 scratch.old_temperature_grads[q] -
               parameters.time_step / (parameters.NSE_solver_interval) * gamma *
                 scratch.phi_T[i]) *
              scratch.temperature_fe_values.JxW(q);

            if (temperature_constraints.is_inhomogeneously_constrained(
                  data.local_dof_indices[i]))
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  data.matrix_for_bc(j, i) +=
                    (scratch.phi_T[i] * scratch.phi_T[j] +
                     parameters.time_step / (parameters.NSE_solver_interval) *
                       one_over_peclet_number * scratch.grad_phi_T[i] *
                       scratch.grad_phi_T[j]) *
                    scratch.temperature_fe_values.JxW(q);
              }
          }
      }
  }


  template <int dim>
  void
  BoussinesqModel<dim>::copy_local_to_global_temperature_rhs(
    const Assembly::CopyData::TemperatureRHS<dim> &data)
  {
    temperature_constraints.distribute_local_to_global(data.local_rhs,
                                                       data.local_dof_indices,
                                                       temperature_rhs,
                                                       data.matrix_for_bc);
  }

  template <int dim>
  void
  BoussinesqModel<dim>::assemble_temperature_rhs(const double time_index)
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Assemble temperature RHS");

    this->pcout << "   Assembling temperature right-hand side..." << std::flush;

    temperature_matrix.copy_from(temperature_mass_matrix);
    temperature_matrix.add(parameters.time_step /
                             (parameters.NSE_solver_interval),
                           temperature_stiffness_matrix);

    if (rebuild_temperature_preconditioner == true)
      {
        T_preconditioner =
          std::make_shared<TrilinosWrappers::PreconditionJacobi>();
        T_preconditioner->initialize(temperature_matrix);

        //      rebuild_temperature_preconditioner = false;
      }

    temperature_rhs = 0;

    const QGauss<dim> quadrature_formula(parameters.temperature_degree + 2);


    using CellFilter =
      FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

    WorkStream::run(
      CellFilter(IteratorFilters::LocallyOwnedCell(),
                 temperature_dof_handler.begin_active()),
      CellFilter(IteratorFilters::LocallyOwnedCell(),
                 temperature_dof_handler.end()),
      std::bind(&BoussinesqModel<dim>::local_assemble_temperature_rhs,
                this,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3),
      std::bind(&BoussinesqModel<dim>::copy_local_to_global_temperature_rhs,
                this,
                std::placeholders::_1),
      Assembly::Scratch::TemperatureRHS<dim>(parameters.time_step,
                                             time_index,
                                             temperature_fe,
                                             nse_fe,
                                             mapping,
                                             quadrature_formula),
      Assembly::CopyData::TemperatureRHS<dim>(temperature_fe));

    temperature_rhs.compress(VectorOperation::add);

    this->pcout << std::endl;
  }


  template <int dim>
  double
  BoussinesqModel<dim>::get_maximal_velocity() const
  {
    const QIterated<dim> quadrature_formula(QTrapez<1>(),
                                            parameters.nse_velocity_degree);
    const unsigned int   n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values(mapping, nse_fe, quadrature_formula, update_values);
    std::vector<Tensor<1, dim>> velocity_values(n_q_points);

    const FEValuesExtractors::Vector velocities(0);
    double                           max_local_velocity = 0;

    for (const auto &cell : nse_dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          fe_values[velocities].get_function_values(nse_solution,
                                                    velocity_values);
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              max_local_velocity =
                std::max(max_local_velocity, velocity_values[q].norm());
            }
        }

    return Utilities::MPI::max(max_local_velocity, this->mpi_communicator);
  }


  template <int dim>
  double
  BoussinesqModel<dim>::get_cfl_number() const
  {
    const QIterated<dim> quadrature_formula(QTrapez<1>(),
                                            parameters.nse_velocity_degree);
    const unsigned int   n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values(mapping, nse_fe, quadrature_formula, update_values);
    std::vector<Tensor<1, dim>> velocity_values(n_q_points);

    const FEValuesExtractors::Vector velocities(0);
    double                           max_local_cfl = 0;

    for (const auto &cell : nse_dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          fe_values[velocities].get_function_values(nse_solution,
                                                    velocity_values);
          double max_local_velocity = 1e-10;
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              max_local_velocity =
                std::max(max_local_velocity, velocity_values[q].norm());
            }
          max_local_cfl =
            std::max(max_local_cfl, max_local_velocity / cell->diameter());
        }

    return Utilities::MPI::max(max_local_cfl, this->mpi_communicator);
  }


  template <int dim>
  void
  BoussinesqModel<dim>::recompute_time_step()
  {
    /*
     * Since we have the same geometry as in Deal.ii's mantle convection code
     * (step-32) we can determine the new step similarly.
     */
    const double scaling = (dim == 3 ? 0.25 : 1.0);
    parameters.time_step = (scaling / (2.1 * dim * std::sqrt(1. * dim)) /
                            (parameters.temperature_degree * get_cfl_number()));

    const double maximal_velocity = get_maximal_velocity();

    this->pcout << "   Max velocity (dimensionsless): " << maximal_velocity
                << std::endl;
    this->pcout << "   Max velocity (with dimensions): "
                << maximal_velocity * parameters.reference_quantities.velocity
                << " m/s" << std::endl;

    this->pcout << "   New Time step (dimensionsless): " << parameters.time_step
                << std::endl;
    this->pcout << "   New Time step (with dimensions): "
                << parameters.time_step * parameters.reference_quantities.time
                << " s" << std::endl;
  }

  /////////////////////////////////////////////////////////////
  // solve
  /////////////////////////////////////////////////////////////

  template <int dim>
  void
  BoussinesqModel<dim>::solve_NSE_block_preconditioned()
  {
    if ((timestep_number == 0) ||
        ((timestep_number > 0) &&
         (timestep_number % parameters.NSE_solver_interval == 0)))
      {
        TimerOutput::Scope timer_section(this->computing_timer,
                                         "   Solve Stokes system");
        this->pcout
          << "   Solving Navier-Stokes system for one time step with (block preconditioned solver)... "
          << std::flush;

        TrilinosWrappers::MPI::BlockVector distributed_nse_solution(nse_rhs);
        distributed_nse_solution = nse_solution;

        const unsigned int
          start = (distributed_nse_solution.block(0).size() +
                   distributed_nse_solution.block(1).local_range().first),
          end   = (distributed_nse_solution.block(0).size() +
                 distributed_nse_solution.block(1).local_range().second);

        for (unsigned int i = start; i < end; ++i)
          if (nse_constraints.is_constrained(i))
            distributed_nse_solution(i) = 0;

        PrimitiveVectorMemory<TrilinosWrappers::MPI::BlockVector> mem;
        unsigned int  n_iterations     = 0;
        const double  solver_tolerance = 1e-8 * nse_rhs.l2_norm();
        SolverControl solver_control(30, solver_tolerance);

        /*
         * We have only the actual pressure but need
         * to solve for a scaled pressure to keep the
         * system symmetric. Hence for the initial guess
         * we need to transform to the rescaled version.
         */
        distributed_nse_solution.block(1) *= parameters.time_step;

        try
          {
            const LinearAlgebra::BlockSchurPreconditioner<
              TrilinosWrappers::PreconditionAMG,
              TrilinosWrappers::PreconditionJacobi>
              preconditioner(nse_matrix,
                             nse_preconditioner_matrix,
                             *Mp_preconditioner,
                             *Amg_preconditioner,
                             false);

            SolverFGMRES<TrilinosWrappers::MPI::BlockVector> solver(
              solver_control,
              mem,
              SolverFGMRES<TrilinosWrappers::MPI::BlockVector>::AdditionalData(
                30));

            solver.solve(nse_matrix,
                         distributed_nse_solution,
                         nse_rhs,
                         preconditioner);

            n_iterations = solver_control.last_step();
          }
        catch (SolverControl::NoConvergence &)
          {
            const LinearAlgebra::BlockSchurPreconditioner<
              TrilinosWrappers::PreconditionAMG,
              TrilinosWrappers::PreconditionJacobi>
              preconditioner(nse_matrix,
                             nse_preconditioner_matrix,
                             *Mp_preconditioner,
                             *Amg_preconditioner,
                             true);

            SolverControl solver_control_refined(nse_matrix.m(),
                                                 solver_tolerance);

            SolverFGMRES<TrilinosWrappers::MPI::BlockVector> solver(
              solver_control_refined,
              mem,
              SolverFGMRES<TrilinosWrappers::MPI::BlockVector>::AdditionalData(
                50));

            solver.solve(nse_matrix,
                         distributed_nse_solution,
                         nse_rhs,
                         preconditioner);

            n_iterations =
              (solver_control.last_step() + solver_control_refined.last_step());
          }
        nse_constraints.distribute(distributed_nse_solution);

        /*
         * We solved only for a scaled pressure to
         * keep the system symmetric. So retransform.
         */
        distributed_nse_solution.block(1) /= parameters.time_step;

        nse_solution = distributed_nse_solution;

        this->pcout << n_iterations << " iterations." << std::endl;
      } // solver time intervall constraint
  }


  template <int dim>
  void
  BoussinesqModel<dim>::solve_NSE_Schur_complement()
  {
    if ((timestep_number == 0) ||
        ((timestep_number > 0) &&
         (timestep_number % parameters.NSE_solver_interval == 0)))
      {
        TimerOutput::Scope timer_section(this->computing_timer,
                                         "   Solve NSE system");
        this->pcout
          << "   Solving Navier-Stokes system for one time step with (preconditioned Schur complement solver)... "
          << std::endl;

        /*
         * Initialize the inner preconditioner.
         */
        inner_schur_preconditioner =
          std::make_shared<InnerPreconditionerType>();

        // Fill preconditioner with life
        inner_schur_preconditioner->initialize(nse_matrix.block(0, 0), data);

        using BlockInverseType =
          LinearAlgebra::InverseMatrix<LA::SparseMatrix,
                                       InnerPreconditionerType>;
        const BlockInverseType block_inverse(nse_matrix.block(0, 0),
                                             *inner_schur_preconditioner);

        TrilinosWrappers::MPI::BlockVector distributed_nse_solution(nse_rhs);
        distributed_nse_solution = nse_solution;

        const unsigned int
          start = (distributed_nse_solution.block(0).size() +
                   distributed_nse_solution.block(1).local_range().first),
          end   = (distributed_nse_solution.block(0).size() +
                 distributed_nse_solution.block(1).local_range().second);

        for (unsigned int i = start; i < end; ++i)
          if (nse_constraints.is_constrained(i))
            distributed_nse_solution(i) = 0;

        // tmp of size block(0)
        LA::MPI::Vector tmp(nse_partitioning[0], this->mpi_communicator);

        // Set up Schur complement
        LinearAlgebra::SchurComplement<LA::BlockSparseMatrix,
                                       LA::MPI::Vector,
                                       BlockInverseType>
          schur_complement(nse_matrix,
                           block_inverse,
                           nse_partitioning,
                           this->mpi_communicator);

        // Compute schur_rhs = -g + C*A^{-1}*f
        LA::MPI::Vector schur_rhs(nse_partitioning[1], this->mpi_communicator);

        this->pcout
          << std::endl
          << "      Apply inverse of block (0,0) for Schur complement solver RHS..."
          << std::endl;

        block_inverse.vmult(tmp, nse_rhs.block(0));
        nse_matrix.block(1, 0).vmult(schur_rhs, tmp);
        schur_rhs -= nse_rhs.block(1);

        this->pcout << "      Schur complement solver RHS computation done..."
                    << std::endl
                    << std::endl;

        {
          TimerOutput::Scope t(
            this->computing_timer,
            "      Solve NSE system - Schur complement solver (for pressure)");

          this->pcout << "      Apply Schur complement solver..." << std::endl;

          // Set Solver parameters for solving for u
          SolverControl                solver_control(nse_matrix.m(),
                                       1e-6 * schur_rhs.l2_norm());
          SolverGMRES<LA::MPI::Vector> schur_solver(solver_control);

          /*
           * Precondition the Schur complement with
           * the approximate inverse of an approximate
           * Schur complement.
           */
          using ApproxSchurComplementType =
            LinearAlgebra::ApproximateSchurComplement<LA::BlockSparseMatrix,
                                                      LA::MPI::Vector,
                                                      LA::PreconditionILU>;
          ApproxSchurComplementType approx_schur(nse_matrix,
                                                 nse_partitioning,
                                                 this->mpi_communicator);

          using ApproxSchurComplementPreconditionerType =
            LA::PreconditionIdentity;
          ApproxSchurComplementPreconditionerType precondition_identity;
#ifdef DEBUG
          LinearAlgebra::ApproximateInverseMatrix<
            ApproxSchurComplementType,
            ApproxSchurComplementPreconditionerType>
            preconditioner_for_schur_solver(approx_schur,
                                            precondition_identity,
                                            /* n_iter */ 2500);
#else
          //          LA::PreconditionIdentity preconditioner_for_schur_solver;
          LinearAlgebra::ApproximateInverseMatrix<
            ApproxSchurComplementType,
            ApproxSchurComplementPreconditionerType>
            preconditioner_for_schur_solver(approx_schur,
                                            precondition_identity,
                                            /* n_iter */ 2500);
#endif

          schur_solver.solve(schur_complement,
                             distributed_nse_solution.block(1),
                             schur_rhs,
                             preconditioner_for_schur_solver);

          this->pcout << "      Iterative Schur complement solver converged in "
                      << solver_control.last_step() << " iterations."
                      << std::endl
                      << std::endl;

          nse_constraints.distribute(distributed_nse_solution);
        } // solve for pressure

        {
          TimerOutput::Scope t(
            this->computing_timer,
            "      Solve NSE system - outer CG solver (for u)");

          this->pcout << "      Apply outer solver..." << std::endl;

          // use computed u to solve for sigma
          nse_matrix.block(0, 1).vmult(tmp, distributed_nse_solution.block(1));
          tmp *= -1;
          tmp += nse_rhs.block(0);

          // Solve for velocity
          block_inverse.vmult(distributed_nse_solution.block(0), tmp);

          this->pcout << "      Outer solver completed." << std::endl
                      << std::endl;

          nse_constraints.distribute(distributed_nse_solution);

          /*
           * We solved only for a scaled pressure to
           * keep the system symmetric. So retransform.
           */
          distributed_nse_solution.block(1) /= parameters.time_step;

          nse_solution = distributed_nse_solution;

        } // solve for velocity
      }   // solver time intervall constraint
  }


  template <int dim>
  void
  BoussinesqModel<dim>::solve_temperature()
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Solve temperature system");

    this->pcout << "      Apply temperature solver..." << std::endl;

    SolverControl solver_control(temperature_matrix.m(),
                                 1e-12 * temperature_rhs.l2_norm());

    SolverCG<TrilinosWrappers::MPI::Vector> cg(solver_control);

    TrilinosWrappers::MPI::Vector distributed_temperature_solution(
      temperature_rhs);

    distributed_temperature_solution = temperature_solution;

    cg.solve(temperature_matrix,
             distributed_temperature_solution,
             temperature_rhs,
             *T_preconditioner);

    temperature_constraints.distribute(distributed_temperature_solution);

    temperature_solution = distributed_temperature_solution;

    this->pcout << "      " << solver_control.last_step()
                << " CG iterations for temperature" << std::endl;

    /*
     * Compute global max and min temerature. Needs MPI communication.
     */
    double temperature[2] = {std::numeric_limits<double>::max(),
                             -std::numeric_limits<double>::max()};
    double global_temperature[2];

    for (unsigned int i = distributed_temperature_solution.local_range().first;
         i < distributed_temperature_solution.local_range().second;
         ++i)
      {
        temperature[0] =
          std::min<double>(temperature[0], distributed_temperature_solution(i));
        temperature[1] =
          std::max<double>(temperature[1], distributed_temperature_solution(i));
      }
    temperature[0] *= -1.0;

    Utilities::MPI::max(temperature,
                        this->mpi_communicator,
                        global_temperature);

    global_temperature[0] *= -1.0;

    this->pcout << "      Temperature range: " << global_temperature[0] << ' '
                << global_temperature[1] << std::endl
                << std::endl;
  }

  /////////////////////////////////////////////////////////////
  // Postprocessor
  /////////////////////////////////////////////////////////////


  template <int dim>
  BoussinesqModel<dim>::Postprocessor::Postprocessor(
    const unsigned int partition)
    : partition(partition)
  {}



  template <int dim>
  std::vector<std::string>
  BoussinesqModel<dim>::Postprocessor::get_names() const
  {
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("p");
    solution_names.emplace_back("T");
    solution_names.emplace_back("partition");
    return solution_names;
  }



  template <int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  BoussinesqModel<dim>::Postprocessor::get_data_component_interpretation() const
  {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim,
                     DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    return interpretation;
  }



  template <int dim>
  UpdateFlags
  BoussinesqModel<dim>::Postprocessor::get_needed_update_flags() const
  {
    return update_values | update_gradients | update_quadrature_points;
  }



  template <int dim>
  void
  BoussinesqModel<dim>::Postprocessor::evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>> &               computed_quantities) const
  {
    const unsigned int n_quadrature_points = inputs.solution_values.size();

    Assert(inputs.solution_gradients.size() == n_quadrature_points,
           ExcInternalError());
    Assert(computed_quantities.size() == n_quadrature_points,
           ExcInternalError());
    Assert(inputs.solution_values[0].size() == dim + 2, ExcInternalError());

    /*
     * TODO: Rescale to physical quantities here.
     */
    for (unsigned int q = 0; q < n_quadrature_points; ++q)
      {
        for (unsigned int d = 0; d < dim; ++d)
          computed_quantities[q](d) = inputs.solution_values[q](d);

        const double pressure       = (inputs.solution_values[q](dim));
        computed_quantities[q](dim) = pressure;

        const double temperature        = inputs.solution_values[q](dim + 1);
        computed_quantities[q](dim + 1) = temperature;

        computed_quantities[q](dim + 2) = partition;
      }
  }



  /////////////////////////////////////////////////////////////
  // Output results
  /////////////////////////////////////////////////////////////


  template <int dim>
  void
  BoussinesqModel<dim>::output_results()
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "Postprocessing and output");

    this->pcout << "   Writing Boussinesq solution for one timestep... "
                << std::flush;

    const FESystem<dim> joint_fe(nse_fe, 1, temperature_fe, 1);

    DoFHandler<dim> joint_dof_handler(this->triangulation);
    joint_dof_handler.distribute_dofs(joint_fe);

    Assert(joint_dof_handler.n_dofs() ==
             nse_dof_handler.n_dofs() + temperature_dof_handler.n_dofs(),
           ExcInternalError());

    TrilinosWrappers::MPI::Vector joint_solution;

    joint_solution.reinit(joint_dof_handler.locally_owned_dofs(),
                          this->mpi_communicator);

    {
      std::vector<types::global_dof_index> local_joint_dof_indices(
        joint_fe.dofs_per_cell);
      std::vector<types::global_dof_index> local_nse_dof_indices(
        nse_fe.dofs_per_cell);
      std::vector<types::global_dof_index> local_temperature_dof_indices(
        temperature_fe.dofs_per_cell);

      typename DoFHandler<dim>::active_cell_iterator
        joint_cell       = joint_dof_handler.begin_active(),
        joint_endc       = joint_dof_handler.end(),
        nse_cell         = nse_dof_handler.begin_active(),
        temperature_cell = temperature_dof_handler.begin_active();
      for (; joint_cell != joint_endc;
           ++joint_cell, ++nse_cell, ++temperature_cell)
        if (joint_cell->is_locally_owned())
          {
            joint_cell->get_dof_indices(local_joint_dof_indices);
            nse_cell->get_dof_indices(local_nse_dof_indices);
            temperature_cell->get_dof_indices(local_temperature_dof_indices);

            for (unsigned int i = 0; i < joint_fe.dofs_per_cell; ++i)
              if (joint_fe.system_to_base_index(i).first.first == 0)
                {
                  Assert(joint_fe.system_to_base_index(i).second <
                           local_nse_dof_indices.size(),
                         ExcInternalError());

                  joint_solution(local_joint_dof_indices[i]) = nse_solution(
                    local_nse_dof_indices[joint_fe.system_to_base_index(i)
                                            .second]);
                }
              else
                {
                  Assert(joint_fe.system_to_base_index(i).first.first == 1,
                         ExcInternalError());
                  Assert(joint_fe.system_to_base_index(i).second <
                           local_temperature_dof_indices.size(),
                         ExcInternalError());

                  joint_solution(local_joint_dof_indices[i]) =
                    temperature_solution(
                      local_temperature_dof_indices
                        [joint_fe.system_to_base_index(i).second]);
                }
          } // end if is_locally_owned()
    }       // end for ++joint_cell

    joint_solution.compress(VectorOperation::insert);

    IndexSet locally_relevant_joint_dofs(joint_dof_handler.n_dofs());
    DoFTools::extract_locally_relevant_dofs(joint_dof_handler,
                                            locally_relevant_joint_dofs);

    TrilinosWrappers::MPI::Vector locally_relevant_joint_solution;
    locally_relevant_joint_solution.reinit(locally_relevant_joint_dofs,
                                           this->mpi_communicator);
    locally_relevant_joint_solution = joint_solution;

    Postprocessor postprocessor(
      Utilities::MPI::this_mpi_process(this->mpi_communicator));

    DataOut<dim> data_out;
    data_out.attach_dof_handler(joint_dof_handler);

    data_out.add_data_vector(locally_relevant_joint_solution, postprocessor);

    data_out.build_patches();

    static int        out_index = 0;
    const std::string filename =
      (parameters.filename_output + "-" +
       Utilities::int_to_string(out_index, 5) + "." +
       Utilities::int_to_string(this->triangulation.locally_owned_subdomain(),
                                4) +
       ".vtu");
    std::ofstream output(parameters.dirname_output + "/" + filename);
    data_out.write_vtu(output);

    /*
     * Write pvtu record
     */
    if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i = 0;
             i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
             ++i)
          filenames.push_back(std::string(parameters.filename_output + "-") +
                              Utilities::int_to_string(out_index, 5) + "." +
                              Utilities::int_to_string(i, 4) + ".vtu");

        const std::string pvtu_master_filename =
          (parameters.filename_output + "-" +
           Utilities::int_to_string(out_index, 5) + ".pvtu");
        std::ofstream pvtu_master(parameters.dirname_output + "/" +
                                  pvtu_master_filename);
        data_out.write_pvtu_record(pvtu_master, filenames);
      }
    out_index++;

    this->pcout << std::endl;
  }


  /////////////////////////////////////////////////////////////
  // Print parameters
  /////////////////////////////////////////////////////////////

  template <int dim>
  void
  BoussinesqModel<dim>::print_paramter_info() const
  {
    this->pcout << "-------------------- Paramter info --------------------"
                << std::endl
                << "Earth radius                         :   "
                << parameters.physical_constants.R0 << std::endl
                << "Atmosphere height                    :   "
                << parameters.physical_constants.atm_height << std::endl
                << std::endl
                << "Reference pressure                   :   "
                << parameters.physical_constants.pressure << std::endl
                << "Reference length                     :   "
                << parameters.reference_quantities.length << std::endl
                << "Reference velocity                   :   "
                << parameters.reference_quantities.velocity << std::endl
                << "Reference time                       :   "
                << parameters.reference_quantities.time << std::endl
                << "Reference atmosphere temperature     :   "
                << parameters.reference_quantities.temperature_ref << std::endl
                << "Atmosphere temperature change        :   "
                << parameters.reference_quantities.temperature_change
                << std::endl
                << std::endl
                << "Reynolds number                      :   "
                << CoreModelData::get_reynolds_number(
                     parameters.reference_quantities.velocity,
                     parameters.reference_quantities.length,
                     parameters.physical_constants.kinematic_viscosity)
                << std::endl
                << "Peclet number                        :   "
                << CoreModelData::get_peclet_number(
                     parameters.reference_quantities.velocity,
                     parameters.reference_quantities.length,
                     parameters.physical_constants.thermal_diffusivity)
                << std::endl
                << "Rossby number                        :   "
                << CoreModelData::get_rossby_number(
                     parameters.reference_quantities.length,
                     parameters.physical_constants.omega,
                     parameters.reference_quantities.velocity)
                << std::endl
                << "Reference accelertion                :   "
                << CoreModelData::get_reference_accelleration(
                     parameters.reference_quantities.length,
                     parameters.reference_quantities.velocity)
                << std::endl
                << "Grashoff number                      :   "
                << CoreModelData::get_grashoff_number(
                     dim,
                     parameters.physical_constants.gravity_constant,
                     parameters.physical_constants.expansion_coefficient,
                     parameters.reference_quantities.temperature_change,
                     parameters.reference_quantities.length,
                     parameters.physical_constants.kinematic_viscosity)
                << std::endl
                << "Prandtl number                       :   "
                << CoreModelData::get_prandtl_number(
                     parameters.physical_constants.kinematic_viscosity,
                     parameters.physical_constants.thermal_diffusivity)
                << std::endl
                << "Rayleigh number                      :   "
                << CoreModelData::get_rayleigh_number(
                     dim,
                     parameters.physical_constants.gravity_constant,
                     parameters.physical_constants.expansion_coefficient,
                     parameters.reference_quantities.temperature_change,
                     parameters.reference_quantities.length,
                     parameters.physical_constants.kinematic_viscosity,
                     parameters.physical_constants.thermal_diffusivity)
                << std::endl
                << "-------------------------------------------------------"
                << std::endl
                << std::endl;
  }



  /////////////////////////////////////////////////////////////
  // Run function
  /////////////////////////////////////////////////////////////


  template <int dim>
  void
  BoussinesqModel<dim>::run()
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "BoussinesqModel - global run function");

    // call refinement routine in base class
    this->refine_global(parameters.initial_global_refinement);

    setup_dofs();

    print_paramter_info();

    /*
     * Initial values.
     */
    nse_solution = 0;

    TrilinosWrappers::MPI::Vector solution_tmp(
      temperature_dof_handler.locally_owned_dofs());

    VectorTools::project(
      temperature_dof_handler,
      temperature_constraints,
      QGauss<dim>(parameters.temperature_degree + 2),
      CoreModelData::Boussinesq::TemperatureInitialValues<dim>(
        parameters.physical_constants.R0, parameters.physical_constants.R1),
      solution_tmp);

    old_nse_solution         = nse_solution;
    temperature_solution     = solution_tmp;
    old_temperature_solution = solution_tmp;

    try
      {
        Tools::create_data_directory(parameters.dirname_output);
      }
    catch (std::runtime_error &e)
      {
        // No exception handling here.
      }
    output_results();

    double time_index = 0;
    do
      {
        this->pcout << "----------------------------------------" << std::endl
                    << "Time step " << timestep_number << ":  t=" << time_index
                    << std::endl;

        if (timestep_number == 0)
          {
            assemble_nse_system(time_index);

            if (!parameters.use_schur_complement_solver)
              build_nse_preconditioner(time_index);
          }
        else if ((timestep_number > 0) &&
                 (timestep_number % parameters.NSE_solver_interval == 0))
          {
            recompute_time_step();

            assemble_nse_system(time_index);

            if (!parameters.use_schur_complement_solver)
              build_nse_preconditioner(time_index);
          }

        assemble_temperature_matrix(time_index);
        assemble_temperature_rhs(time_index);

        if (parameters.use_direct_solver)
          {
            TimerOutput::Scope t(this->computing_timer,
                                 " direct solver (MUMPS)");

            throw std::runtime_error(
              "Solver not implemented: MUMPS does not work on "
              "TrilinosWrappers::MPI::BlockSparseMatrix classes.");
          }

        if (parameters.use_schur_complement_solver)
          {
            solve_NSE_Schur_complement();
          }
        else
          {
            solve_NSE_block_preconditioned();
          }

        solve_temperature();

        output_results();

        /*
         * Print summary after a NSE system has been solved.
         */
        if ((timestep_number > 0) &&
            (timestep_number % parameters.NSE_solver_interval == 0))
          {
            this->computing_timer.print_summary();
          }

        time_index += parameters.time_step;
        ++timestep_number;

        old_nse_solution         = nse_solution;
        old_temperature_solution = temperature_solution;

        this->pcout << "----------------------------------------" << std::endl;
      }
    while (time_index <= parameters.final_time);
  }

} // namespace Standard

DYCOREPLANET_CLOSE_NAMESPACE
