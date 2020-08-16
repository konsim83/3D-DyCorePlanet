#pragma once

// AquaPlanet
#include <base/config.h>
#include <model_data/physical_constants.h>
#include <model_data/reference_quantities.h>


DYCOREPLANET_OPEN_NAMESPACE

/*!
 * @namespace CoreModelData
 *
 * Namespace containing namespaces for different models such as Boussinesq
 * appromimation or the full primitive equations
 */
namespace CoreModelData
{
  /*!
   * Return the Reynolds number of the flow.
   */
  double
  get_reynolds_number(const double velocity,
                      const double length,
                      const double kinematic_viscosity);

  /*!
   * Return the Peclet number of the flow.
   */
  double
  get_peclet_number(const double velocity,
                    const double length,
                    const double thermal_diffusivity);

  /*!
   * Return the Grashoff number of the flow.
   */
  double
  get_grashoff_number(const int    dim,
                      const double gravity_constant,
                      const double expansion_coefficient,
                      const double temperature_change,
                      const double length,
                      const double kinematic_viscosity);

  /*!
   * Return the Prandtl number of the flow.
   */
  double
  get_prandtl_number(const double kinematic_viscosity,
                     const double thermal_diffusivity);

  /*!
   * Return the Rayleigh number of the flow.
   */
  double
  get_rayleigh_number(const int    dim,
                      const double gravity_constant,
                      const double expansion_coefficient,
                      const double temperature_change,
                      const double length,
                      const double kinematic_viscosity,
                      const double thermal_diffusivity);

  /*!
   * Density as function of temperature.
   *
   * @return desity
   */
  double
  density(const double density,
          const double expansion_coefficient,
          const double temperature,
          const double temperature_bottom);

  /*!
   * Density scaling as function of temperature. This is the density devided
   * by the reference density.
   *
   * @return desity
   */
  double
  density_scaling(const double expansion_coefficient,
                  const double temperature,
                  const double temperature_bottom);

  /*!
   * Compute gravity vector at a given point.
   *
   * @return gravity vector
   */
  template <int dim>
  Tensor<1, dim>
  gravity_vector(const Point<dim> &p, const double gravity_constant);

  /*!
   * Compute coriolis vector at a given point.
   *
   * @param p
   * @return coriolis vector
   */
  template <int dim>
  Tensor<1, dim>
  coriolis_vector(const Point<dim> &p, const double omega);

} // namespace CoreModelData

extern template Tensor<1, 2>
CoreModelData::gravity_vector<2>(const Point<2> &p,
                                 const double    gravity_constant);
extern template Tensor<1, 3>
CoreModelData::gravity_vector<3>(const Point<3> &p,
                                 const double    gravity_constant);

extern template Tensor<1, 2>
CoreModelData::coriolis_vector(const Point<2> & /*p*/, const double omega);

extern template Tensor<1, 3>
CoreModelData::coriolis_vector(const Point<3> & /*p*/, const double omega);

DYCOREPLANET_CLOSE_NAMESPACE
