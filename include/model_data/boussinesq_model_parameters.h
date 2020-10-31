#pragma once

// Deal.ii
#include <deal.II/base/parameter_handler.h>

// AquaPlanet
#include <base/config.h>
#include <model_data/physical_constants.h>
#include <model_data/reference_quantities.h>

DYCOREPLANET_OPEN_NAMESPACE

namespace CoreModelData
{
  /*!
   * @struct Paramters
   *
   * Struct holding parameters for a bouyancy Boussinesq model.
   */
  struct Parameters
  {
    Parameters(const std::string &parameter_filename);

    void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);

    unsigned int space_dimension;

    CoreModelData::ReferenceQuantities reference_quantities;
    CoreModelData::PhysicalConstants   physical_constants;

    double       final_time;
    double       time_step;
    unsigned int initial_global_refinement;

    double       nse_theta;
    unsigned int nse_velocity_degree;
    bool         use_locally_conservative_discretization;

    bool use_schur_complement_solver;
    bool use_direct_solver;

    unsigned int NSE_solver_interval;

    double       temperature_theta;
    unsigned int temperature_degree;
  };

} // namespace CoreModelData

DYCOREPLANET_CLOSE_NAMESPACE
