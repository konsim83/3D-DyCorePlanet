#include <model_data/physical_constants.h>

DYCOREPLANET_OPEN_NAMESPACE


CoreModelData::PhysicalConstants::PhysicalConstants(
  const std::string & parameter_filename,
  ReferenceQuantities reference_quantities)
  : universal_gas_constant(8.31446261815324)
  , specific_gas_constant_dry(2)
  , expansion_coefficient(1 / 273.15)
  , dynamic_viscosity(1.82e-5)
  , kinematic_viscosity(dynamic_viscosity)
  , specific_heat_p(1.005)
  , specific_heat_v(0.718)
  , thermal_conductivity(2.62e-2)
  , thermal_diffusivity(thermal_conductivity / (specific_heat_p * 1.01325e+5))
  , radiogenic_heating(7.4e-12)
  , gravity_constant(9.81)
  , speed_of_sound(331.5)
  , atm_height(4)
  , R0(6.371000e+6)
  , R1(R0 + 1.0e+5)
{
  ParameterHandler prm;
  PhysicalConstants::declare_parameters(prm);

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
  parse_parameters(prm, reference_quantities);
}



void
CoreModelData::PhysicalConstants::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Physical Constants");
  {
    prm.declare_entry("universal_gas_constant",
                      "8.31446261815324",
                      Patterns::Double(0),
                      "Universal gas constant.");

    prm.declare_entry("specific_gas_constant_dry",
                      "287.0",
                      Patterns::Double(0),
                      "Specific gas constant of dry air.");

    prm.declare_entry("expansion coefficient",
                      "0.003661",
                      Patterns::Double(0),
                      "Thermal expansion coefficient std: ideal gas.");

    prm.declare_entry("dynamic_viscosity",
                      "1.82e-5",
                      Patterns::Double(0),
                      "Dynamic viscosity.");

    prm.declare_entry("specific_heat_p",
                      "1.005",
                      Patterns::Double(0),
                      "Specific heat constant (isobaric)");

    prm.declare_entry("specific_heat_v",
                      "0.718",
                      Patterns::Double(0),
                      "Specific heat constant (isochoric).");

    prm.declare_entry("thermal_conductivity",
                      "2.62e-2",
                      Patterns::Double(0),
                      "Thermal conductivity.");

    prm.declare_entry("radiogenic_heating",
                      "7.4e-12",
                      Patterns::Double(0),
                      "Radiogenic heating.");

    prm.declare_entry("gravity_constant",
                      "9.81",
                      Patterns::Double(0),
                      "Gravity constant");

    prm.declare_entry("speed_of_sound",
                      "331.5",
                      Patterns::Double(0),
                      "Speed of sound.");

    prm.declare_entry("atm_height",
                      "1.0e+5",
                      Patterns::Double(0),
                      "Height of atmosphere.");

    prm.declare_entry("R0",
                      "6.371000e+6",
                      Patterns::Double(0),
                      "Inner radius of atmosphere.");
  }
  prm.leave_subsection();
}



void
CoreModelData::PhysicalConstants::parse_parameters(
  ParameterHandler &  prm,
  ReferenceQuantities reference_quantities)
{
  prm.enter_subsection("Physical Constants");
  {
    universal_gas_constant    = prm.get_double("universal_gas_constant");
    specific_gas_constant_dry = prm.get_double("specific_gas_constant_dry");
    expansion_coefficient     = prm.get_double("expansion coefficient");

    dynamic_viscosity = prm.get_double("dynamic_viscosity");

    kinematic_viscosity = dynamic_viscosity / reference_quantities.density;

    specific_heat_p      = prm.get_double("specific_heat_p");
    specific_heat_v      = prm.get_double("specific_heat_v");
    thermal_conductivity = prm.get_double("thermal_conductivity");

    thermal_diffusivity =
      thermal_conductivity / (specific_heat_p * reference_quantities.pressure);

    radiogenic_heating = prm.get_double("radiogenic_heating");
    gravity_constant   = prm.get_double("gravity_constant");
    speed_of_sound     = prm.get_double("speed_of_sound");
    atm_height         = prm.get_double("atm_height");
    R0                 = prm.get_double("R0");

    R1 = R0 + atm_height;
  }
  prm.leave_subsection();
}

DYCOREPLANET_CLOSE_NAMESPACE
