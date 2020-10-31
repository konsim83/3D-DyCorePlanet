#include <model_data/boussinesq_model_parameters.h>

DYCOREPLANET_OPEN_NAMESPACE


CoreModelData::Parameters::Parameters(const std::string &parameter_filename)
  : space_dimension(2)
  , reference_quantities(parameter_filename)
  , physical_constants(parameter_filename, reference_quantities)
  , final_time(1.0)
  , time_step(0.1)
  , initial_global_refinement(2)
  , nse_theta(0.5)
  , nse_velocity_degree(2)
  , use_locally_conservative_discretization(true)
  , use_schur_complement_solver(true)
  , use_direct_solver(false)
  , NSE_solver_interval(1)
  , temperature_theta(0.5)
  , temperature_degree(2)
{
  ParameterHandler prm;
  CoreModelData::Parameters::declare_parameters(prm);

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
CoreModelData::Parameters::declare_parameters(ParameterHandler &prm)
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

    prm.declare_entry("space dimension",
                      "2",
                      Patterns::Integer(2, 3),
                      "Spatial dimension of the problem.");

    prm.declare_entry("final time",
                      "1.0",
                      Patterns::Double(0),
                      "The end time of the simulation in seconds.");

    prm.declare_entry("time step",
                      "0.1",
                      Patterns::Double(0),
                      "Time step size.");

    prm.declare_entry("nse theta",
                      "0.5",
                      Patterns::Double(0.0, 1.0),
                      "Theta value for theta method.");

    prm.declare_entry("nse velocity degree",
                      "2",
                      Patterns::Integer(1),
                      "The polynomial degree to use for the velocity variables "
                      "in the NSE system.");

    prm.declare_entry(
      "use locally conservative discretization",
      "true",
      Patterns::Bool(),
      "Whether to use a Navier-Stokes discretization that is locally "
      "conservative at the expense of a larger number of degrees "
      "of freedom, or to go with a cheaper discretization "
      "that does not locally conserve mass (although it is "
      "globally conservative.");

    prm.declare_entry(
      "use schur complement solver",
      "false",
      Patterns::Bool(),
      "Choose whether to use a preconditioned Schur complement solver or an iterative block system solver with block preconditioner.");

    prm.declare_entry(
      "use direct solver",
      "false",
      Patterns::Bool(),
      "Choose whether to use a direct solver for the Navier-Stokes part.");

    prm.declare_entry("NSE solver interval",
                      "1",
                      Patterns::Integer(1),
                      "Apply the NSE solver only every n-th time step.");

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
CoreModelData::Parameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Boussinesq Model");
  {
    prm.enter_subsection("Mesh parameters");
    {
      initial_global_refinement = prm.get_integer("initial global refinement");
    }
    prm.leave_subsection();

    space_dimension = prm.get_integer("space dimension");

    final_time = prm.get_double("final time");
    time_step  = prm.get_double("time step");

    nse_theta           = prm.get_double("nse theta");
    nse_velocity_degree = prm.get_integer("nse velocity degree");
    use_locally_conservative_discretization =
      prm.get_bool("use locally conservative discretization");
    use_schur_complement_solver = prm.get_bool("use schur complement solver");
    use_direct_solver           = prm.get_bool("use direct solver");

    NSE_solver_interval = prm.get_integer("NSE solver interval");

    temperature_theta  = prm.get_double("temperature theta");
    temperature_degree = prm.get_integer("temperature degree");
  }
  prm.leave_subsection();
}


DYCOREPLANET_CLOSE_NAMESPACE
