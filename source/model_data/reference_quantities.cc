#include <model_data/physical_constants.h>

DYCOREPLANET_OPEN_NAMESPACE


CoreModelData::ReferenceQuantities::ReferenceQuantities(
  const std::string &parameter_filename)
  : pressure(1.01325e+5)
  , omega(7.272205e-5)
  , density(1.29)
  , time(3.6e+3)
  , velocity(10)
  , length(1e+4)
  , temperature_bottom(273.15)
  , temperature_top(253.15)
  , temperature_change(5)
{
  ParameterHandler prm;
  ReferenceQuantities::declare_parameters(prm);

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
CoreModelData::ReferenceQuantities::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Boussinesq Model");
  {
    prm.enter_subsection("Reference quantities");
    {
      prm.declare_entry("omega",
                        "7.272205e-5",
                        Patterns::Double(0),
                        "Reference angular velocity [rad/s].");

      prm.declare_entry("pressure",
                        "1.01325e+5",
                        Patterns::Double(0),
                        "Reference pressure.");

      prm.declare_entry("density",
                        "1.29",
                        Patterns::Double(0),
                        "Reference density.");

      prm.declare_entry("time",
                        "3.6e+3",
                        Patterns::Double(0),
                        "Reference time.");

      prm.declare_entry("velocity",
                        "10",
                        Patterns::Double(0),
                        "Reference velocity");

      prm.declare_entry("length",
                        "1e+4",
                        Patterns::Double(0),
                        "Reference length.");

      prm.declare_entry("temperature_bottom",
                        "273.15",
                        Patterns::Double(0),
                        "Reference temperature at bottom.");

      prm.declare_entry("temperature_top",
                        "253.15",
                        Patterns::Double(0),
                        "Reference temperature at top.");

      prm.declare_entry("temperature_change",
                        "5",
                        Patterns::Double(0),
                        "Reference temperature change.");
    }
    prm.leave_subsection();
  }
  prm.leave_subsection();
}



void
CoreModelData::ReferenceQuantities::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Boussinesq Model");
  {
    prm.enter_subsection("Reference quantities");
    {
      pressure           = prm.get_double("pressure");           /* Pa */
      omega              = prm.get_double("omega");              /* 1/s */
      density            = prm.get_double("density");            /* kg / m^3 */
      time               = prm.get_double("time");               /* s */
      velocity           = prm.get_double("velocity");           /* m/s */
      length             = prm.get_double("length");             /* m */
      temperature_bottom = prm.get_double("temperature_bottom"); /* K */
      temperature_top    = prm.get_double("temperature_top");    /* K */
      temperature_change = prm.get_double("temperature_change"); /* K */
    }
    prm.leave_subsection();
  }
  prm.leave_subsection();
}

DYCOREPLANET_CLOSE_NAMESPACE
