#include <core/boussinesq_model.h>

AQUAPLANET_OPEN_NAMESPACE


//////////////////////////////////////////////////////
/// Boussinesq model parameters
//////////////////////////////////////////////////////

BoussinesqModel::Parameters::Parameters(const std::string &parameter_filename)
  : final_time(1.0)
  , initial_global_refinement(2)
  , nse_theta(0.5)
  , nse_velocity_degree(2)
  , use_locally_conservative_discretization(true)
  , temperature_theta(0.5)
  , temperature_degree(2)
{
  ParameterHandler prm;
  BoussinesqModel::Parameters::declare_parameters(prm);

  std::ifstream parameter_file(parameter_filename);
  if (!parameter_file)
    {
      parameter_file.close();
      std::ofstream parameter_out(parameter_filename);
      prm.print_parameters(parameter_out, ParameterHandler::Text);
      AssertThrow(false,
                  ExcMessage(
                    "Input parameter file <" + parameter_filename +
                    "> not found. Creating a template file of the same name."));
    }
  prm.parse_input(parameter_file,
                  /* filename = */ "generated_parameter.in",
                  /* last_line = */ "",
                  /* skip_undefined = */ true);
  parse_parameters(prm);
}


void
BoussinesqModel::Parameters::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Boussinesq Model");
  {
    prm.enter_subsection("Mesh parameters");
    {
      prm.declare_entry("initial global refinement",
                        "3",
                        Patterns::Integer(0),
                        "The number of global refinement steps performed on "
                        "the initial coarse mesh, before the problem is first "
                        "solved there.");
    }
    prm.leave_subsection();

    prm.declare_entry("final time",
                      "1.0",
                      Patterns::Double(0),
                      "The end time of the simulation in seconds.");

    prm.declare_entry("nse theta",
                      "0.5",
                      Patterns::Double(0.0, 1.0),
                      "Theta value for theta method.");

    prm.declare_entry("nse velocity degree",
                      "2",
                      Patterns::Integer(1),
                      "The polynomial degree to use for the velocity variables "
                      "in the Stokes system.");

    prm.declare_entry(
      "use locally conservative discretization",
      "true",
      Patterns::Bool(),
      "Whether to use a Navier-Stokes discretization that is locally "
      "conservative at the expense of a larger number of degrees "
      "of freedom, or to go with a cheaper discretization "
      "that does not locally conserve mass (although it is "
      "globally conservative.");

    prm.declare_entry("temperature theta",
                      "0.5",
                      Patterns::Double(0.0, 1.0),
                      "Theta value for theta method.");

    prm.declare_entry(
      "temperature degree",
      "2",
      Patterns::Integer(1),
      "The polynomial degree to use for the temperature variable.");
  }
  prm.leave_subsection();
}


void
BoussinesqModel::Parameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Boussinesq Model");
  {
    prm.enter_subsection("Mesh parameters");
    {
      initial_global_refinement = prm.get_integer("initial global refinement");
    }
    prm.leave_subsection();

    final_time = prm.get_double("final time");

    nse_theta           = prm.get_double("nse theta");
    nse_velocity_degree = prm.get_integer("nse velocity degree");
    use_locally_conservative_discretization =
      prm.get_bool("use locally conservative discretization");

    temperature_theta  = prm.get_double("temperature theta");
    temperature_degree = prm.get_integer("temperature degree");
  }
  prm.leave_subsection();
}



//////////////////////////////////////////////////////
/// Boussinesq model
//////////////////////////////////////////////////////



BoussinesqModel::BoussinesqModel(Parameters &parameters_)
  : AquaPlanet(CoreModelData::Boussinesq::R0, CoreModelData::Boussinesq::R1)
  , parameters(parameters_)
  , mapping(4)
  , nse_fe(FE_Q<3>(parameters.nse_velocity_degree),
           3,
           (parameters.use_locally_conservative_discretization ?
              static_cast<const FiniteElement<3> &>(
                FE_DGP<3>(parameters.nse_velocity_degree - 1)) :
              static_cast<const FiniteElement<3> &>(
                FE_Q<3>(parameters.nse_velocity_degree - 1))),
           1)
  , nse_dof_handler(triangulation)
  , temperature_fe(parameters.temperature_degree)
  , temperature_dof_handler(triangulation)
  , time_step(0)
  , timestep_number(0)
{}



BoussinesqModel::~BoussinesqModel()
{}



void
BoussinesqModel::run()
{
  // call refinement routine in base class
  refine_global(parameters.initial_global_refinement);

  write_mesh_vtu();

  //  setup_dofs();
  //
  // start_time_iteration:
  //  VectorTools::project(temperature_dof_handler,
  //                       temperature_constraints,
  //                       QGauss<3>(parameters.temperature_degree + 2),
  //                       EquationData::TemperatureInitialValues(),
  //                       old_temperature_solution);
  //
  //  double time_index = 0;
  //  do
  //    {
  //      std::cout << "Time step " << timestep_number << ":  t=" << time_index
  //                << std::endl;
  //      assemble_nse_system();
  //
  //      build_nse_preconditioner();
  //
  //      assemble_temperature_matrix();
  //
  //      solve();
  //
  //      output_results();
  //
  //      //      std::cout << std::endl;
  //      //      if ((timestep_number == 0) &&
  //      //          (pre_refinement_step < n_pre_refinement_steps))
  //      //        {
  //      //          refine_mesh(initial_refinement + n_pre_refinement_steps);
  //      //          ++pre_refinement_step;
  //      //          goto start_time_iteration;
  //      //        }
  //      //      else if ((timestep_number > 0) && (timestep_number % 5 == 0))
  //      //        refine_mesh(initial_refinement + n_pre_refinement_steps);
  //
  //      time_index += time_step;
  //      ++timestep_number;
  //
  //      old_nse_solution         = nse_solution;
  //      old_temperature_solution = temperature_solution;
  //    }
  //  while (time_index <= parameters.end_time);
}

AQUAPLANET_CLOSE_NAMESPACE
