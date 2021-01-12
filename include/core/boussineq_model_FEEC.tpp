#pragma once

#include <core/boussineq_model_FEEC.h>
#include <core/planet_geometry.h>

DYCOREPLANET_OPEN_NAMESPACE

namespace ExteriorCalculus
{
  //////////////////////////////////////////////////////
  /// Standard Boussinesq model in H1-L2
  //////////////////////////////////////////////////////

  template <int dim>
  BoussinesqModel<dim>::BoussinesqModel(CoreModelData::Parameters &parameters_)
    : PlanetGeometry<dim>(parameters_.physical_constants.R0,
                          parameters_.physical_constants.R1,
                          parameters_.cuboid_geometry)
    , parameters(parameters_)
    , temperature_mapping(1)
    , nse_fe(
        static_cast<const FiniteElement<dim> &>(FE_NedelecSZ<dim>(
          std::max(static_cast<int>(parameters.nse_velocity_degree) - 1, 0))),
        1,
        static_cast<const FiniteElement<dim> &>(FE_RaviartThomas<dim>(
          std::max(static_cast<int>(parameters.nse_velocity_degree) - 1, 0))),
        1,
        static_cast<const FiniteElement<dim> &>(FE_DGQ<dim>(
          std::max(static_cast<int>(parameters.nse_velocity_degree) - 1, 0))),
        1)
    , nse_dof_handler(this->triangulation)
    , temperature_fe(parameters.temperature_degree)
    , temperature_dof_handler(this->triangulation)
    , timestep_number(0)
    , rebuild_nse_matrix(true)
    , rebuild_nse_preconditioner(true)
    , rebuild_temperature_matrices(true)
    , rebuild_temperature_preconditioner(true)
  {
    TimerOutput::Scope timing_section(
      this->computing_timer,
      "BoussinesqModel - constructor and grid rescaling");

    /*
     * Rescale the original this->triangulation to the one scaled by the
     * reference length.
     */
    GridTools::scale(1 / parameters.reference_quantities.length,
                     this->triangulation);

    {
      /*
       * We must also rescale the domain parameters since this enters the data
       * of other objects (initial conditions etc)
       */
      if (parameters.cuboid_geometry)
        {
          /*
           * Note that this assumes that the lower left corner is the origin.
           */
          this->center /= parameters.reference_quantities.length;
        }

      this->inner_radius /= parameters.reference_quantities.length;
      this->outer_radius /= parameters.reference_quantities.length;
      this->global_Omega_diameter /= parameters.reference_quantities.length;
      parameters.physical_constants.R0 /=
        parameters.reference_quantities.length;
      parameters.physical_constants.R1 /=
        parameters.reference_quantities.length;
    }
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
    LA::BlockSparsityPattern sp(nse_partitioning,
                                nse_partitioning,
                                nse_relevant_partitioning,
                                this->mpi_communicator);

    Table<2, DoFTools::Coupling> coupling(2 * dim + 1, 2 * dim + 1);
    for (unsigned int c = 0; c < 2 * dim + 1; ++c)
      {
        for (unsigned int d = 0; d < 2 * dim + 1; ++d)
          {
            if (c < dim)
              {
                if (d < 2 * dim - 1)
                  coupling[c][d] = DoFTools::always;
                else
                  coupling[c][d] = DoFTools::none;
              }
            else if ((c >= dim) && (c < 2 * dim))
              {
                coupling[c][d] = DoFTools::always;
              }
            else if (c == 2 * dim)
              {
                if ((d >= dim) && (d < 2 * dim))
                  coupling[c][d] = DoFTools::always;
                else
                  coupling[c][d] = DoFTools::none;
              }
          }
      }

    DoFTools::make_sparsity_pattern(nse_dof_handler,
                                    coupling,
                                    sp,
                                    nse_constraints,
                                    false,
                                    Utilities::MPI::this_mpi_process(
                                      this->mpi_communicator));
    sp.compress();

    nse_matrix.reinit(sp);
  }


  template <int dim>
  void
  BoussinesqModel<dim>::setup_nse_preconditioner(
    const std::vector<IndexSet> &nse_partitioning,
    const std::vector<IndexSet> &nse_relevant_partitioning)
  {
    // pressure_system_preconditioner.reset();

    nse_preconditioner_matrix.clear();
    LA::BlockSparsityPattern sp(nse_partitioning,
                                nse_partitioning,
                                nse_relevant_partitioning,
                                this->mpi_communicator);

    Table<2, DoFTools::Coupling> coupling(2 * dim + 1, 2 * dim + 1);
    //    for (unsigned int c = 0; c < 2 * dim + 1; ++c)
    //      for (unsigned int d = 0; d < 2 * dim + 1; ++d)
    //        if (c == d)
    //          coupling[c][d] = DoFTools::always;
    //        else
    //          coupling[c][d] = DoFTools::none;
    for (unsigned int c = 0; c < 2 * dim + 1; ++c)
      {
        for (unsigned int d = 0; d < 2 * dim + 1; ++d)
          {
            if (c < dim)
              {
                if (d < 2 * dim - 1)
                  coupling[c][d] = DoFTools::always;
                else
                  coupling[c][d] = DoFTools::none;
              }
            else if ((c >= dim) && (c < 2 * dim))
              {
                if (d == 2 * dim)
                  coupling[c][d] = DoFTools::none;
                else
                  coupling[c][d] = DoFTools::always;
              }
            else if (c == 2 * dim)
              {
                if ((d >= dim) && (d < 2 * dim))
                  coupling[c][d] = DoFTools::none;
                else
                  coupling[c][d] = DoFTools::always;
              }
          }
      }

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

    LA::SparsityPattern sp(temperature_partitioner,
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
    nse_dof_handler.distribute_dofs(nse_fe);

    DoFRenumbering::Cuthill_McKee(nse_dof_handler);
    //  DoFRenumbering::boost::king_ordering(nse_dof_handler);
    DoFRenumbering::block_wise(nse_dof_handler);

    temperature_dof_handler.distribute_dofs(temperature_fe);

    /*
     * Count dofs
     */

    std::vector<types::global_dof_index> nse_dofs_per_block =
      DoFTools::count_dofs_per_fe_block(nse_dof_handler);
    const unsigned int n_w = nse_dofs_per_block[0], n_u = nse_dofs_per_block[1],
                       n_p = nse_dofs_per_block[2],
                       n_T = temperature_dof_handler.n_dofs();

    /*
     * Comma separated large numbers
     */
    std::locale s = this->pcout.get_stream().getloc();
    this->pcout.get_stream().imbue(std::locale(""));

    /*
     * Print some mesh and dof info
     */
    this->pcout << "   Number of active cells: "
                << this->triangulation.n_global_active_cells() << " (on "
                << this->triangulation.n_levels() << " levels)" << std::endl
                << "   Number of degrees of freedom: " << n_w + n_u + n_p + n_T
                << " (" << n_w << " + " << n_u << " + " << n_p << " + " << n_T
                << ")" << std::endl
                << std::endl;
    this->pcout.get_stream().imbue(s);

    /*
     * Setup partitioners to store what dofs and matrix entries are stored on
     * the local processor
     */
    IndexSet temperature_partitioning(n_T),
      temperature_relevant_partitioning(n_T);

    {
      nse_index_set.clear();
      nse_relevant_set.clear();
      nse_partitioning.clear();
      nse_relevant_partitioning.clear();

      nse_index_set = nse_dof_handler.locally_owned_dofs();
      nse_partitioning.push_back(nse_index_set.get_view(0, n_w));
      nse_partitioning.push_back(nse_index_set.get_view(n_w, n_w + n_u));
      nse_partitioning.push_back(
        nse_index_set.get_view(n_w + n_u, n_w + n_u + n_p));

      DoFTools::extract_locally_relevant_dofs(nse_dof_handler,
                                              nse_relevant_set);
      nse_relevant_partitioning.push_back(nse_relevant_set.get_view(0, n_w));
      nse_relevant_partitioning.push_back(
        nse_relevant_set.get_view(n_w, n_w + n_u));
      nse_relevant_partitioning.push_back(
        nse_relevant_set.get_view(n_w + n_u, n_w + n_u + n_p));

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

      if (parameters.cuboid_geometry)
        {
          std::vector<GridTools::PeriodicFacePair<
            typename DoFHandler<dim>::cell_iterator>>
            periodicity_vector;

          /*
           * All dimensions up to the last are periodic (z-direction is always
           * bounded from below and form above)
           */
          for (unsigned int d = 0; d < dim - 1; ++d)
            {
              GridTools::collect_periodic_faces(nse_dof_handler,
                                                /*b_id1*/ 2 * (d + 1) - 2,
                                                /*b_id2*/ 2 * (d + 1) - 1,
                                                /*direction*/ d,
                                                periodicity_vector);
            }

          DoFTools::make_periodicity_constraints<DoFHandler<dim>>(
            periodicity_vector, nse_constraints);

          /*
           * Lower boundary (id=4)
           */
          VectorTools::project_boundary_values_curl_conforming_l2(
            nse_dof_handler,
            /*first vector component */ 0,
            Functions::ZeroFunction<dim>(2 * dim + 1),
            /*boundary id*/ 4,
            nse_constraints);

          VectorTools::project_boundary_values_div_conforming(
            nse_dof_handler,
            /*first vector component */
            3,
            Functions::ZeroFunction<dim>(dim),
            /*boundary id*/ 4,
            nse_constraints);

          /*
           * Upper boundary (id=5)
           */
          VectorTools::project_boundary_values_curl_conforming_l2(
            nse_dof_handler,
            /*first vector component */ 0,
            Functions::ZeroFunction<dim>(2 * dim + 1),
            /*boundary id*/ 5,
            nse_constraints);

          VectorTools::project_boundary_values_div_conforming(
            nse_dof_handler,
            /*first vector component */
            3,
            Functions::ZeroFunction<dim>(dim),
            //        ConstantFunction<dim>(1500, dim),
            /*boundary id*/ 5,
            nse_constraints);
        }
      else // shell geometry
        {
          /*
           * Lower boundary (id=0), upper boundary (id=1)
           */
          for (const auto &id : this->triangulation.get_boundary_ids())
            {
              if (id == 0)
                {
                  VectorTools::project_boundary_values_curl_conforming_l2(
                    nse_dof_handler,
                    /*first vector component */ 0,
                    VectorFunctionFromTensorFunction<dim>(
                      CoreModelData::TangentialFunction<dim>(0.0),
                      /* selected_component = */ 0,
                      /* n_components = */ 2 * dim + 1),
                    /*boundary id*/ id,
                    nse_constraints);
                  VectorTools::project_boundary_values_div_conforming(
                    nse_dof_handler,
                    /*first vector component */
                    3,
                    VectorFunctionFromTensorFunction<dim>(
                      CoreModelData::RadialFunction<dim>(0.0)),
                    /*boundary id*/ id,
                    nse_constraints);
                }
              else
                {
                  VectorTools::project_boundary_values_curl_conforming_l2(
                    nse_dof_handler,
                    /*first vector component */ 0,
                    VectorFunctionFromTensorFunction<dim>(
                      CoreModelData::TangentialFunction<dim>(0.0),
                      /* selected_component = */ 0,
                      /* n_components = */ 2 * dim + 1),
                    /*boundary id*/ id,
                    nse_constraints);
                  VectorTools::project_boundary_values_div_conforming(
                    nse_dof_handler,
                    /*first vector component */
                    3,
                    VectorFunctionFromTensorFunction<dim>(
                      CoreModelData::RadialFunction<dim>(0.0)),
                    /*boundary id*/ id,
                    nse_constraints);
                }
            }
        }

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

      if (parameters.cuboid_geometry)
        {
          std::vector<GridTools::PeriodicFacePair<
            typename DoFHandler<dim>::cell_iterator>>
            periodicity_vector;

          /*
           * All dimensions up to the last are periodic (z-direction is always
           * bounded from below and form above)
           */
          for (unsigned int d = 0; d < dim - 1; ++d)
            {
              GridTools::collect_periodic_faces(temperature_dof_handler,
                                                /*b_id1*/ 2 * (d + 1) - 2,
                                                /*b_id2*/ 2 * (d + 1) - 1,
                                                /*direction*/ d,
                                                periodicity_vector);
            }

          DoFTools::make_periodicity_constraints<DoFHandler<dim>>(
            periodicity_vector, temperature_constraints);

          // Dirchlet on boundary id 2/4 (lower in 2d/3d)
          VectorTools::interpolate_boundary_values(
            temperature_dof_handler,
            (dim == 2 ? 2 : 4),
            CoreModelData::Boussinesq::TemperatureInitialValuesCuboid<dim>(
              this->center, this->global_Omega_diameter),
            temperature_constraints);
        }
      else
        {
          // Lower boundary is Dirichlet, upper is no-flux and natural
          VectorTools::interpolate_boundary_values(
            temperature_dof_handler,
            /*boundary id*/ 0,
            CoreModelData::Boussinesq::TemperatureInitialValues<dim>(
              parameters.physical_constants.R0,
              parameters.physical_constants.R1),
            temperature_constraints);
        }

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

    rebuild_nse_matrix                 = true;
    rebuild_nse_preconditioner         = true;
    rebuild_temperature_matrices       = true;
    rebuild_temperature_preconditioner = true;
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

    const FEValuesExtractors::Vector vorticity(0);
    const FEValuesExtractors::Vector velocities(dim);
    const FEValuesExtractors::Scalar pressure(2 * dim);

    scratch.nse_fe_values.reinit(cell);

    cell->get_dof_indices(data.local_dof_indices);
    data.local_matrix = 0;

    const double one_over_reynolds_number =
      (1. / CoreModelData::get_reynolds_number(
              parameters.reference_quantities.velocity,
              parameters.reference_quantities.length,
              parameters.physical_constants.kinematic_viscosity));

    std::vector<double> sign_change(dofs_per_cell, 1.0);
    Tools::get_face_sign_change_raviart_thomas(cell, nse_fe, sign_change);

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            scratch.phi_w[k] = scratch.nse_fe_values[vorticity].value(k, q);

            scratch.curl_phi_w[k] = scratch.nse_fe_values[vorticity].curl(k, q);

            scratch.phi_u[k] =
              sign_change[k] * scratch.nse_fe_values[velocities].value(k, q);

            scratch.phi_p[k] = scratch.nse_fe_values[pressure].value(k, q);
          }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              const double phi_u_i_times_phi_w_j =
                scratch.phi_u[i] * scratch.phi_w[j];
              const double phi_w_i_times_phi_u_j =
                scratch.phi_w[i] * scratch.phi_u[j];
              data.local_matrix(i, j) +=
                //              scratch.phi_w[i] * scratch.phi_w[j] +
                parameters.time_step * one_over_reynolds_number *
                  scratch.curl_phi_w[i] * scratch.curl_phi_w[j] +
                +(std::fabs(phi_u_i_times_phi_w_j) > 1.0e-9 ?
                    -2 * (std::signbit(phi_u_i_times_phi_w_j) - 0.5) :
                    0.0) +
                (std::fabs(phi_w_i_times_phi_u_j) > 1.0e-9 ?
                   -2 * (std::signbit(phi_w_i_times_phi_u_j) - 0.5) :
                   0.0) +
                scratch.phi_p[i] * scratch.phi_p[j] *
                  scratch.nse_fe_values.JxW(q);
            }
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
    if (rebuild_nse_preconditioner == false)
      return;

    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Build NSE FEEC preconditioner");

    this->pcout
      << "   Assembling and building Navier-Stokes FEEC block preconditioner..."
      << std::flush;

    assemble_nse_preconditioner(time_index);

    // std::vector<std::vector<bool>> constant_modes_vorticity;
    // FEValuesExtractors::Vector     vorticity_component(0);
    // DoFTools::extract_constant_modes(nse_dof_handler,
    //                                  nse_fe.component_mask(vorticity_component),
    //                                  constant_modes_vorticity);
    // vorticity_system_preconditioner =
    //   std::make_shared<VorticitySystemPreconType>();
    // typename VorticitySystemPreconType::AdditionalData
    //   vorticity_system_preconditioner_data;
    // /*
    //  * This is relevant to AMG preconditioners
    //  */
    // vorticity_system_preconditioner_data.constant_modes =
    //   constant_modes_vorticity;
    // vorticity_system_preconditioner_data.elliptic              = true;
    // vorticity_system_preconditioner_data.higher_order_elements = false;
    // vorticity_system_preconditioner_data.smoother_sweeps       = 1;
    // vorticity_system_preconditioner_data.aggregation_threshold = 0.02;

    // vorticity_system_preconditioner->initialize(
    //   nse_preconditioner_matrix.block(0, 0),
    //   vorticity_system_preconditioner_data);

    rebuild_nse_preconditioner = false;

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

    const FEValuesExtractors::Vector vorticity(0);
    const FEValuesExtractors::Vector velocities(dim);
    const FEValuesExtractors::Scalar pressure(2 * dim);

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

    // Only rebuild matrix upon refinement
    if (rebuild_nse_matrix)
      {
        data.local_matrix = 0;
      }
    // This needs always to be rebuilt
    data.local_rhs = 0;

    /*
     * Get some values at the previous time step
     */
    scratch.temperature_fe_values.get_function_values(
      old_temperature_solution, scratch.old_temperature_values);
    scratch.nse_fe_values[vorticity].get_function_values(
      old_nse_solution, scratch.old_vorticity_values);
    scratch.nse_fe_values[velocities].get_function_values(
      old_nse_solution, scratch.old_velocity_values);

    std::vector<double> sign_change(dofs_per_cell, 1.0);
    Tools::get_face_sign_change_raviart_thomas(cell, nse_fe, sign_change);

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double old_temperature = scratch.old_temperature_values[q];
        const double density_scaling = CoreModelData::density_scaling(
          parameters.physical_constants.expansion_coefficient,
          old_temperature,
          parameters.reference_quantities.temperature_ref);
        const Tensor<1, dim> old_vorticity = scratch.old_vorticity_values[q];
        const Tensor<1, dim> old_velocity  = scratch.old_velocity_values[q];

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            scratch.phi_w[k] = scratch.nse_fe_values[vorticity].value(k, q);

            scratch.curl_phi_w[k] = scratch.nse_fe_values[vorticity].curl(k, q);

            scratch.phi_u[k] =
              sign_change[k] * scratch.nse_fe_values[velocities].value(k, q);

            scratch.div_phi_u[k] =
              sign_change[k] *
              scratch.nse_fe_values[velocities].divergence(k, q);

            scratch.phi_p[k] = scratch.nse_fe_values[pressure].value(k, q);
          }

        Tensor<1, dim> coriolis;
        if (parameters.cuboid_geometry)
          coriolis = parameters.reference_quantities.length *
                     CoreModelData::coriolis_vector(
                       scratch.nse_fe_values.quadrature_point(q),
                       parameters.physical_constants.omega) /
                     parameters.reference_quantities.velocity;

        /*
         * Move everything to the LHS here.
         */
        if (rebuild_nse_matrix)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    data.local_matrix(i, j) +=
                      (scratch.phi_w[i] *
                         scratch.phi_w[j] // mass_w term block(0,0)
                       - scratch.curl_phi_w[i] *
                           scratch.phi_u[j] // rot_w term block(0,1)
                       + scratch.phi_u[i] *
                           scratch.phi_u[j] // mass_u term block(1,1)
                       + parameters.time_step * one_over_reynolds_number *
                           (scratch.phi_u[i] *
                            scratch
                              .curl_phi_w[j]) // 1/Re * v * curl(w) block(1,0)
                       - (scratch.div_phi_u[i] *
                          scratch.phi_p[j]) // div(v)*p ---> scaled pressure
                                            // dt*p block(1,2)
                       - (scratch.phi_p[i] *
                          scratch.div_phi_u[j]) // q * div(u) block(2, 1)
                       ) *
                      scratch.nse_fe_values.JxW(q);
                  }
              }
          }
        const Tensor<1, dim> gravity =
          (parameters.reference_quantities.length /
           (parameters.reference_quantities.velocity *
            parameters.reference_quantities.velocity)) *
          (parameters.cuboid_geometry ?
             CoreModelData::vertical_gravity_vector(
               scratch.nse_fe_values.quadrature_point(q),
               parameters.physical_constants.gravity_constant) :
             CoreModelData::gravity_vector(
               scratch.nse_fe_values.quadrature_point(q),
               parameters.physical_constants.gravity_constant));

        /*
         * This is only the RHS
         */
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            data.local_rhs(i) +=
              (scratch.phi_u[i] * old_velocity +
               parameters.time_step * density_scaling * gravity *
                 scratch.phi_u[i] -
               parameters.time_step *
                 (scratch.div_phi_u[i] * 0.5 *
                    scalar_product(old_velocity, old_velocity) +
                  scratch.phi_u[i] *
                    cross_product_3d(
                      old_vorticity,
                      old_velocity)) // advection at previous time
               - parameters.time_step * 2 * scratch.phi_u[i] *
                   cross_product_3d(coriolis,
                                    old_velocity) // coriolis force
               ) *
              scratch.nse_fe_values.JxW(q);
          }
      }

    cell->get_dof_indices(data.local_dof_indices);
  }



  template <int dim>
  void
  BoussinesqModel<dim>::copy_local_to_global_nse_system(
    const Assembly::CopyData::NSESystem<dim> &data)
  {
    if (rebuild_nse_matrix)
      {
        nse_constraints.distribute_local_to_global(data.local_matrix,
                                                   data.local_rhs,
                                                   data.local_dof_indices,
                                                   nse_matrix,
                                                   nse_rhs);
      }
    else
      {
        nse_constraints.distribute_local_to_global(data.local_rhs,
                                                   data.local_dof_indices,
                                                   nse_rhs);
      }
  }



  template <int dim>
  void
  BoussinesqModel<dim>::assemble_nse_system(const double time_index)
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Assemble NSE system");

    this->pcout << "   Assembling Navier-Stokes system..." << std::flush;

    if (rebuild_nse_matrix)
      {
        nse_matrix = 0;
      }
    nse_rhs = 0;

    const QGauss<dim> quadrature_formula(parameters.nse_velocity_degree + 2);
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
                                        temperature_mapping,
                                        quadrature_formula,
                                        (update_values |
                                         update_quadrature_points |
                                         update_JxW_values | update_gradients),
                                        temperature_fe,
                                        update_values),
      Assembly::CopyData::NSESystem<dim>(nse_fe));

    if (rebuild_nse_matrix)
      {
        nse_matrix.compress(VectorOperation::add);
      }
    nse_rhs.compress(VectorOperation::add);

    rebuild_nse_matrix = false;

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
    if (rebuild_temperature_matrices == false)
      return;

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
                                                temperature_mapping,
                                                quadrature_formula),
      Assembly::CopyData::TemperatureMatrix<dim>(temperature_fe));

    temperature_mass_matrix.compress(VectorOperation::add);
    temperature_advection_matrix.compress(VectorOperation::add);
    temperature_stiffness_matrix.compress(VectorOperation::add);

    rebuild_temperature_matrices       = false;
    rebuild_temperature_preconditioner = true;

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

    const FEValuesExtractors::Vector velocities(dim);

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
        T_preconditioner = std::make_shared<LA::PreconditionJacobi>();
        T_preconditioner->initialize(temperature_matrix);

        rebuild_temperature_preconditioner = false;
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
                                             temperature_mapping,
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
                                            parameters.nse_velocity_degree + 1);
    const unsigned int   n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values(nse_fe, quadrature_formula, update_values);
    std::vector<Tensor<1, dim>> velocity_values(n_q_points);

    const FEValuesExtractors::Vector velocities(dim);
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

    double max_global_velocity =
      Utilities::MPI::max(max_local_velocity, this->mpi_communicator);

    this->pcout << "   Max velocity (dimensionsless): " << max_global_velocity
                << std::endl;
    this->pcout << "   Max velocity (with dimensions): "
                << max_global_velocity *
                     parameters.reference_quantities.velocity
                << " m/s" << std::endl;

    return max_global_velocity;
  }


  template <int dim>
  double
  BoussinesqModel<dim>::get_cfl_number() const
  {
    const QIterated<dim> quadrature_formula(QTrapez<1>(),
                                            parameters.nse_velocity_degree + 1);
    const unsigned int   n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values(nse_fe, quadrature_formula, update_values);
    std::vector<Tensor<1, dim>> velocity_values(n_q_points);

    const FEValuesExtractors::Vector velocities(dim);
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

    double max_global_cfl =
      Utilities::MPI::max(max_local_cfl, this->mpi_communicator);

    this->pcout << "   Max of local CFL numbers: " << max_local_cfl
                << std::endl;

    return max_global_cfl;
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
                            (std::max(parameters.temperature_degree,
                                      parameters.nse_velocity_degree) *
                             get_cfl_number()));

    get_maximal_velocity();

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
                                         "   Solve NSE system");

        this->pcout
          << "   Solving Navier-Stokes system for one time step with (block preconditioned solver)... "
          << std::endl;

        if (parameters.use_block_preconditioner_feec)
          {
            /*
             * Setup mass matrix inverses
             */
            Mw_inverse_preconditioner =
              std::make_shared<MassPerconditionerType>();
            Mu_inverse_preconditioner =
              std::make_shared<MassPerconditionerType>();

            Mw_inverse_preconditioner->initialize(nse_matrix.block(0, 0));
            Mw_inverse =
              std::make_shared<MassInverseType>(nse_matrix.block(0, 0),
                                                *Mw_inverse_preconditioner,
                                                /* use_simple_cg */ true);

            Mu_inverse_preconditioner->initialize(nse_matrix.block(1, 1));
            Mu_inverse =
              std::make_shared<MassInverseType>(nse_matrix.block(1, 1),
                                                *Mu_inverse_preconditioner,
                                                /* use_simple_cg */ true);
          }

        if (parameters.use_block_preconditioner_feec)
          {
            /*
             * Setup shifted and nested Schur complements
             */
            approx_Mu_minus_Sw_inverse =
              std::make_shared<ApproxShiftedSchurComplementInverseType>(
                //                nse_preconditioner_matrix,
                //                *vorticity_system_preconditioner,
                //                *Mu_inverse_preconditioner,
                nse_matrix,
                *Mw_inverse_preconditioner,
                *Mu_inverse_preconditioner,
                nse_partitioning,
                this->mpi_communicator);

            pressure_system_approx_schur_compl_matrix =
              std::make_shared<SchurComplementLowerBlockType>(
                nse_matrix,
                *approx_Mu_minus_Sw_inverse, // This is stronger but slower
                                             // (only applied with
                                             // _do_full_solve)
                *Mu_inverse_preconditioner,  // This is weaker but faster
                // *Mu_inverse, // This is weaker but faster
                nse_partitioning,
                /* do_full_solve */ false,
                this->mpi_communicator);

            approx_nested_schur_complement_inverse =
              std::make_shared<ApproxNesteSchurComplementInverseType>(
                nse_preconditioner_matrix,
                *pressure_system_approx_schur_compl_matrix);
          }


        LA::MPI::BlockVector distributed_nse_solution(nse_rhs);
        distributed_nse_solution = nse_solution;
        /*
         * We solved only for a scaled pressure to
         * keep the system symmetric. So transform now and rescale later.
         */
        distributed_nse_solution.block(2) *= parameters.time_step;

        /*
         * Set values at constrained local pressure dofs to zero in order not
         * to bother the Schur complement solver with irrelevant values.
         */
        const unsigned int
          start = (distributed_nse_solution.block(0).size() +
                   distributed_nse_solution.block(1).size() +
                   distributed_nse_solution.block(2).local_range().first),
          end   = (distributed_nse_solution.block(0).size() +
                 distributed_nse_solution.block(1).size() +
                 distributed_nse_solution.block(2).local_range().second);

        for (unsigned int i = start; i < end; ++i)
          if (nse_constraints.is_constrained(i))
            distributed_nse_solution(i) = 0;

        PrimitiveVectorMemory<LA::MPI::BlockVector> mem;
        unsigned int                                n_iterations = 0;
        const double  solver_tolerance = 1e-8 * nse_rhs.l2_norm();
        SolverControl solver_control(
          /* n_max_iter */ (parameters.use_block_preconditioner_feec ? 500 :
                                                                       15000),
          solver_tolerance,
          /* log_history */ true,
          /* log_result */ true);

        if (parameters.correct_pressure_to_zero_mean)
          {
            const double mean_pressure =
              // Tools::compute_pressure_mean_value(nse_dof_handler,
              //                                    QGauss<dim>(2),
              //                                    distributed_nse_solution);
              VectorTools::compute_mean_value(nse_dof_handler,
                                              QGauss<dim>(2),
                                              nse_solution,
                                              2 * dim);
            distributed_nse_solution.block(2).add(-mean_pressure);

            this->pcout
              << "      Blocksolver RHS pre-correction: The mean value "
                 "was adjusted by "
              << -mean_pressure << "    -> new mean:   "
              << Tools::compute_pressure_mean_value(nse_dof_handler,
                                                    QGauss<dim>(2),
                                                    distributed_nse_solution)
              << std::endl;
          }

        {
          SolverGMRES<LA::MPI::BlockVector> solver(
            solver_control,
            mem,
            SolverGMRES<LA::MPI::BlockVector>::AdditionalData(100));

          /*
           * Now build the actual block preconditioner
           */
          if (parameters.use_block_preconditioner_feec)
            {
              preconditioner_feec =
                std::make_shared<BlockSchurPreconditionerFEECType>(
                  nse_matrix,
                  *Mw_inverse_preconditioner,
                  *approx_Mu_minus_Sw_inverse,
                  *approx_nested_schur_complement_inverse,
                  nse_dof_handler,
                  parameters.correct_pressure_to_zero_mean);

              solver.solve(nse_matrix,
                           distributed_nse_solution,
                           nse_rhs,
                           *preconditioner_feec);
            }
          else
            {
              preconditioner_identity = std::make_shared<
                const typename LinearAlgebra::PreconditionerBlockIdentity<
                  DoFHandler<dim>>>(nse_dof_handler,
                                    parameters.correct_pressure_to_zero_mean);

              solver.solve(nse_matrix,
                           distributed_nse_solution,
                           nse_rhs,
                           *preconditioner_identity);
            }

          n_iterations = solver_control.last_step();
        }

        nse_constraints.distribute(distributed_nse_solution);

        /*
         * We solved only for a scaled pressure to
         * keep the system symmetric. So retransform.
         */
        distributed_nse_solution.block(2) /= parameters.time_step;

        nse_solution = distributed_nse_solution;

        this->pcout << "      " << n_iterations
                    << "   GMRES iterations for NSE system." << std::endl;
      } // solver time intervall constraint
  }


  template <int dim>
  void
  BoussinesqModel<dim>::solve_temperature()
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Solve temperature system");

    this->pcout << "   Apply temperature solver..." << std::endl;

    SolverControl solver_control(temperature_matrix.m(),
                                 1e-12 * temperature_rhs.l2_norm(),
                                 /* log_history */ false,
                                 /* log_result */ false);

    SolverCG<LA::MPI::Vector> cg(solver_control);

    LA::MPI::Vector distributed_temperature_solution(temperature_rhs);

    distributed_temperature_solution = temperature_solution;

    cg.solve(temperature_matrix,
             distributed_temperature_solution,
             temperature_rhs,
             *T_preconditioner);

    temperature_constraints.distribute(distributed_temperature_solution);

    temperature_solution = distributed_temperature_solution;

    this->pcout << "      " << solver_control.last_step()
                << "   CG iterations for temperature system" << std::endl;

    /*
     * Compute global max and min temperature. Needs MPI communication.
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
    std::vector<std::string> solution_names(dim, "vorticity");
    solution_names.emplace_back("velocity");
    solution_names.emplace_back("velocity");
    solution_names.emplace_back("velocity");
    solution_names.emplace_back("p");
    solution_names.emplace_back("T");
    solution_names.emplace_back("partition");

    return solution_names;
  }



  template <int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  BoussinesqModel<dim>::Postprocessor::get_data_component_interpretation() const
  {
    // vorticity
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim,
                     DataComponentInterpretation::component_is_part_of_vector);

    // velocity
    interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);

    // pressure
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    // temperature
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    // partition
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
    Assert(inputs.solution_values[0].size() == 2 * dim + 2, ExcInternalError());

    /*
     * TODO: Rescale to physical quantities here.
     */
    for (unsigned int q = 0; q < n_quadrature_points; ++q)
      {
        for (unsigned int d = 0; d < dim; ++d)
          computed_quantities[q](d) = inputs.solution_values[q](d);

        for (unsigned int d = dim; d < 2 * dim; ++d)
          computed_quantities[q](d) = inputs.solution_values[q](d);

        const double pressure           = (inputs.solution_values[q](2 * dim));
        computed_quantities[q](2 * dim) = pressure;

        const double temperature = inputs.solution_values[q](2 * dim + 1);
        computed_quantities[q](2 * dim + 1) = temperature;

        computed_quantities[q](2 * dim + 2) = partition;
      }
  }


  /////////////////////////////////////////////////////////////
  // Refinement and coarsening
  /////////////////////////////////////////////////////////////


  template <int dim>
  void
  BoussinesqModel<dim>::refine_and_coarsen(const unsigned int max_level)
  {
    AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
                ExcMessage("Refinement only works on 1 MPI process."));

    parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector>
      temperature_transfer(temperature_dof_handler);
    parallel::distributed::SolutionTransfer<dim, LA::MPI::BlockVector>
      nse_transfer(nse_dof_handler);

    {
      TimerOutput::Scope timer_section(this->computing_timer,
                                       "Refinement and coarsening, part 1");

      Vector<float> estimated_error_per_cell(
        this->triangulation.n_active_cells());
      KellyErrorEstimator<dim>::estimate(
        temperature_dof_handler,
        QGauss<dim - 1>(parameters.temperature_degree + 1),
        std::map<types::boundary_id, const Function<dim> *>(),
        temperature_solution,
        estimated_error_per_cell,
        ComponentMask(),
        nullptr,
        0,
        this->triangulation.locally_owned_subdomain());

      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
        this->triangulation, estimated_error_per_cell, 0.3, 0.1);

      /*
       * Clear flag if refinement is too deep
       */
      if (this->triangulation.n_levels() > max_level)
        {
          for (typename Triangulation<dim>::active_cell_iterator cell =
                 this->triangulation.begin_active(max_level);
               cell != this->triangulation.end();
               ++cell)
            {
              cell->clear_refine_flag();
            }
        }

      std::vector<const LA::MPI::Vector *> x_temperature(2);
      x_temperature[0] = &temperature_solution;
      x_temperature[1] = &old_temperature_solution;
      std::vector<const LA::MPI::BlockVector *> x_nse(2);
      x_nse[0] = &nse_solution;
      x_nse[1] = &old_nse_solution;

      this->triangulation.prepare_coarsening_and_refinement();
      temperature_transfer.prepare_for_coarsening_and_refinement(x_temperature);
      nse_transfer.prepare_for_coarsening_and_refinement(x_nse);
      this->triangulation.execute_coarsening_and_refinement();
    }

    /*
     * Need to do another setup of dofs
     */
    setup_dofs();

    {
      TimerOutput::Scope timer_section(this->computing_timer,
                                       "Refinement and coarsening, part 2");
      {
        LA::MPI::Vector                distributed_temp1(temperature_rhs);
        LA::MPI::Vector                distributed_temp2(temperature_rhs);
        std::vector<LA::MPI::Vector *> tmp(2);
        tmp[0] = &(distributed_temp1);
        tmp[1] = &(distributed_temp2);
        temperature_transfer.interpolate(tmp);
        temperature_constraints.distribute(distributed_temp1);
        temperature_constraints.distribute(distributed_temp2);
        temperature_solution     = distributed_temp1;
        old_temperature_solution = distributed_temp2;
      }

      {
        LA::MPI::BlockVector                distributed_stokes(nse_rhs);
        LA::MPI::BlockVector                old_distributed_stokes(nse_rhs);
        std::vector<LA::MPI::BlockVector *> nse_tmp(2);
        nse_tmp[0] = &(distributed_stokes);
        nse_tmp[1] = &(old_distributed_stokes);
        nse_transfer.interpolate(nse_tmp);
        nse_constraints.distribute(distributed_stokes);
        nse_constraints.distribute(old_distributed_stokes);
        nse_solution     = distributed_stokes;
        old_nse_solution = old_distributed_stokes;
      }
    }

    // GridTools::remove_hanging_nodes(this->triangulation);
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

    DataOut<dim> data_out;

    if (true)
      {
        const FESystem<dim> joint_fe(nse_fe, 1, temperature_fe, 1);

        DoFHandler<dim> joint_dof_handler(this->triangulation);
        joint_dof_handler.distribute_dofs(joint_fe);

        Assert(joint_dof_handler.n_dofs() ==
                 nse_dof_handler.n_dofs() + temperature_dof_handler.n_dofs(),
               ExcInternalError());

        LA::MPI::Vector joint_solution;

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
                temperature_cell->get_dof_indices(
                  local_temperature_dof_indices);

                for (unsigned int i = 0; i < joint_fe.dofs_per_cell; ++i)
                  {
                    /*
                     * A vorticity/velocity/pressure dof
                     */
                    if (joint_fe.system_to_base_index(i).first.first == 0)
                      {
                        Assert(joint_fe.system_to_base_index(i).second <
                                 local_nse_dof_indices.size(),
                               ExcInternalError());

                        joint_solution(local_joint_dof_indices[i]) =
                          nse_solution(
                            local_nse_dof_indices
                              [joint_fe.system_to_base_index(i).second]);
                      }
                    /*
                     * A temperature dof
                     */
                    else
                      {
                        Assert(joint_fe.system_to_base_index(i).first.first ==
                                 1,
                               ExcInternalError());
                        Assert(joint_fe.system_to_base_index(i).second <
                                 local_temperature_dof_indices.size(),
                               ExcInternalError());

                        joint_solution(local_joint_dof_indices[i]) =
                          temperature_solution(
                            local_temperature_dof_indices
                              [joint_fe.system_to_base_index(i).second]);
                      }
                  }
              } // namespace ExteriorCalculus
        }       // end for ++joint_cell

        joint_solution.compress(VectorOperation::insert);

        IndexSet locally_relevant_joint_dofs(joint_dof_handler.n_dofs());
        DoFTools::extract_locally_relevant_dofs(joint_dof_handler,
                                                locally_relevant_joint_dofs);

        LA::MPI::Vector locally_relevant_joint_solution;
        locally_relevant_joint_solution.reinit(locally_relevant_joint_dofs,
                                               this->mpi_communicator);
        locally_relevant_joint_solution = joint_solution;

        Postprocessor postprocessor(
          Utilities::MPI::this_mpi_process(this->mpi_communicator));

        data_out.attach_dof_handler(joint_dof_handler);

        data_out.add_data_vector(locally_relevant_joint_solution,
                                 postprocessor);

        data_out.build_patches(std::min(parameters.nse_velocity_degree,
                                        parameters.temperature_degree));
      }
    else
      {
        std::vector<std::string> nse_names(dim, "vorticity");
        nse_names.emplace_back("velocity");
        nse_names.emplace_back("velocity");
        nse_names.emplace_back("velocity");
        nse_names.emplace_back("p");

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          nse_component_interpretation(
            2 * dim + 1,
            DataComponentInterpretation::component_is_part_of_vector);
        nse_component_interpretation[2 * dim] =
          DataComponentInterpretation::component_is_scalar;

        data_out.add_data_vector(nse_dof_handler,
                                 nse_solution,
                                 nse_names,
                                 nse_component_interpretation);

        data_out.add_data_vector(temperature_dof_handler,
                                 temperature_solution,
                                 "T");

        data_out.build_patches(std::min(parameters.nse_velocity_degree,
                                        parameters.temperature_degree));
      }

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

    unsigned int pre_refinement_step = 0;

  start_time_iteration:

    /*
     * Initial values.
     */
    nse_solution = 0;

    LA::MPI::Vector solution_tmp(temperature_dof_handler.locally_owned_dofs());

    if (parameters.cuboid_geometry)
      {
        VectorTools::project(
          temperature_dof_handler,
          temperature_constraints,
          QGauss<dim>(parameters.temperature_degree + 2),
          CoreModelData::Boussinesq::TemperatureInitialValuesCuboid<dim>(
            this->center, this->global_Omega_diameter),
          solution_tmp);
      }
    else
      {
        VectorTools::project(
          temperature_dof_handler,
          temperature_constraints,
          QGauss<dim>(parameters.temperature_degree + 2),
          CoreModelData::Boussinesq::TemperatureInitialValues<dim>(
            parameters.physical_constants.R0, parameters.physical_constants.R1),
          solution_tmp);
      }

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
        if ((timestep_number > 0) &&
            (timestep_number % parameters.NSE_solver_interval == 0) &&
            parameters.adapt_time_step)
          {
            recompute_time_step();
          }
        else
          {
            /*
             * This is just informative output
             */
            get_cfl_number();
            get_maximal_velocity();
          }

        this->pcout << "----------------------------------------" << std::endl
                    << "Time step " << timestep_number << ":  t=" << time_index
                    << " -> t=" << time_index + parameters.time_step
                    << "  (dt=" << parameters.time_step
                    << " | final time=" << parameters.final_time << ")"
                    << std::endl;

        if (timestep_number == 0)
          {
            assemble_nse_system(time_index);

            if ((!parameters.use_schur_complement_solver) &&
                (parameters.use_block_preconditioner_feec))
              build_nse_preconditioner(time_index);
          }
        else if ((timestep_number > 0) &&
                 (timestep_number % parameters.NSE_solver_interval == 0))
          {
            assemble_nse_system(time_index);

            if ((!parameters.use_schur_complement_solver) &&
                (parameters.use_block_preconditioner_feec))
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
              "LA::MPI::BlockSparseMatrix classes.");
          }

        if (parameters.use_schur_complement_solver)
          {
            throw std::runtime_error(
              "Schur complement solver not implemented for FEEC method.");
          }
        else
          {
            solve_NSE_block_preconditioned();
          }

        solve_temperature();

        if (parameters.adaptive_refinement)
          {
            if ((timestep_number == 0) &&
                (pre_refinement_step < parameters.initial_adaptive_refinement))
              {
                this->pcout << "   Performing pre-refinement of mesh   #"
                            << pre_refinement_step + 1 << "   out of   "
                            << parameters.initial_adaptive_refinement << "..."
                            << std::endl;

                refine_and_coarsen(parameters.initial_global_refinement +
                                   parameters.initial_adaptive_refinement);
                ++pre_refinement_step;
                goto start_time_iteration;
              }
            else if ((timestep_number > 0) &&
                     (timestep_number %
                        parameters.adaptive_refinement_interval ==
                      0))
              {
                this->pcout
                  << "   Refining and coarsening of mesh in time step   #"
                  << timestep_number << "..." << std::endl;

                refine_and_coarsen(parameters.initial_global_refinement +
                                   parameters.initial_adaptive_refinement);
              }
          }

        output_results();

        // Stop solving if time step is already past final time
        if (time_index > parameters.final_time)
          break;

        /*
         * Print summary after a NSE system has been solved.
         */
        if ((timestep_number > 0) &&
            (timestep_number % parameters.NSE_solver_interval == 0))
          {
            this->computing_timer.print_summary();
          }

        time_index += parameters.time_step / parameters.NSE_solver_interval;
        ++timestep_number;

        old_nse_solution         = nse_solution;
        old_temperature_solution = temperature_solution;

        this->pcout << "----------------------------------------" << std::endl;
      }
    while (true);
  }

} // namespace ExteriorCalculus

DYCOREPLANET_CLOSE_NAMESPACE
