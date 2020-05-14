#pragma once

// C++ STL
#include <string>

// Deal.ii
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/vector.h>

// AquaPlanet
#include <base/config.h>
#include <model_data/core_model_data.h>


AQUAPLANET_OPEN_NAMESPACE

namespace CoreModelData
{
  /*!
   * @namespace Boussinesq
   *
   * Namespace containing constants and helper functions relevant to a transient
   * Boussinesq model.
   */
  namespace Boussinesq
  {
    //////////////////////////////////////////////////
    /// Reference quantities..
    //////////////////////////////////////////////////


    /*!
     * Earth reference pressure.
     */
    constexpr double reference_pressure = 1.01325e+5; /* Pa */

    /*!
     * Earth angular velocity.
     */
    constexpr double reference_omega =
      2 * numbers::PI / (24 * 60 * 60); /* 1/s */

    /*!
     * Refence density of air at bottom reference
     * temperature.
     */
    constexpr double reference_density = 1.29; /* kg / m^3 */

    /*!
     * Refence temperature 273.15 K (0 degree Celsius) at bottom.
     */
    constexpr double reference_temperature_bottom = 273.15; /* K */

    /*!
     * Refence temperature at top of atmospere.
     */
    constexpr double reference_temperature_top = 253.15; /* K */

    /*!
     * Reference time is one hour.
     */
    constexpr double reference_time = 3.6e+3; /* s */

    /*!
     * Reference velocity.
     */
    constexpr double reference_velocity = 10; /* m/s */

    /*!
     * Reference length.
     */
    constexpr double reference_length = 1e+4; /* m */

    /*!
     * Reference temperature change.
     */
    constexpr double reference_temperature_change = 20; /* K */


    //////////////////////////////////////////////////
    /// Some physical constants.
    //////////////////////////////////////////////////


    /*!
     * Thermal expansion coefficient (beta) of air at bottom reference
     * temperature.
     */
    constexpr double expansion_coefficient =
      1 / reference_temperature_bottom; /* 1/K */

    /*!
     * Thermal diffusion coefficient (eta) of air at bottom reference
     * temperature.
     */
    constexpr double thermal_diffusivity = 1.76e-5; /* m^2/s */

    /*!
     * Thermal conductivity (kappa) of air at bottom reference
     * temperature.
     */
    constexpr double thermal_condictivity = 2.62e-2; /* W/(mK) */

    // constexpr double specific_heat         = 1250;    	/* J / K / kg */
    // constexpr double radiogenic_heating    = 7.4e-12; 	/* W / kg     */

    /*!
     * Gravity constant.
     */
    constexpr double gravity_constant = 9.81; /* m/s^2 */


    /*!
     * Height of atmosphere (here up to mesosphere)
     */
    constexpr double atm_height = 2.0e+6; /* m */

    /*!
     * Earth radius.
     */
    constexpr double R0 = 6.371000e+6; /* m */

    /*!
     * Earth radius plus height of mesosphere.
     */
    constexpr double R1 = R0 + atm_height; /* m */

    /*!
     * A year in seconds.
     */
    constexpr double year_in_seconds = 60 * 60 * 24 * 365.2425; /* s */


    //////////////////////////////////////////////////
    /// Some functions.
    //////////////////////////////////////////////////


    /*!
     * Density as function of temperature.
     *
     * @param temperature
     * @return desity
     */
    double
    density(const double temperature);

    /*!
     * Density scaling as function of temperature. This is the density devided
     * by the reference density.
     *
     * @param temperature
     * @return desity
     */
    double
    density_scaling(const double temperature);

    /*!
     * Compute gravity vector at a given point.
     *
     * @param p
     * @return garvity vector
     */
    Tensor<1, 3>
    gravity_vector(const Point<3> &p);


    /*!
     * Temerature initial values for rising warm bubble test.
     */
    class TemperatureInitialValues : public Function<3>
    {
    public:
      /*!
       * Constructor.
       */
      TemperatureInitialValues()
        : Function<3>(1)
      {}

      /*!
       * Return temperature value at a single point.
       *
       * @param p
       * @param component
       * @return
       */
      virtual double
      value(const Point<3> &p, const unsigned int component = 0) const override;

      /*!
       * Return temperature value as a vector at a single point.
       *
       * @param points
       * @param values
       */
      virtual void
      value_list(const std::vector<Point<3>> &points,
                 std::vector<double> &        values,
                 const unsigned int           component = 0) const override;
    };


    /*!
     * Temerature right-hand side for rising warm bubble test. This term
     * represents external heat sources.
     */
    class TemperatureRHS : public Function<3>
    {
    public:
      /*!
       * Constructor.
       */
      TemperatureRHS()
        : Function<3>(1)
      {}

      /*!
       * Return temperature value at a single point.
       *
       * @param p
       * @param component
       * @return
       */
      virtual double
      value(const Point<3> &p, const unsigned int component = 0) const override;

      /*!
       * Return temperature value as a vector at a single point.
       *
       * @param points
       * @param values
       */
      virtual void
      value_list(const std::vector<Point<3>> &points,
                 std::vector<double> &        value,
                 const unsigned int           component = 0) const override;
    };


    /*!
     * Velocity initial values for rising warm bubble test.
     */
    class VelocityInitialValues : public TensorFunction<1, 3>
    {
    public:
      /*!
       * Constructor.
       */
      VelocityInitialValues()
        : TensorFunction<1, 3>()
      {}

      /*!
       * Return velocity value at a single point.
       *
       * @param p
       * @return
       */
      virtual Tensor<1, 3>
      value(const Point<3> &p) const override;

      /*!
       * Return temperature value as a vector at a single point.
       *
       * @param points
       * @param values
       */
      virtual void
      value_list(const std::vector<Point<3>> &points,
                 std::vector<Tensor<1, 3>> &  values) const override;
    };

  } // namespace Boussinesq

} // namespace CoreModelData

AQUAPLANET_CLOSE_NAMESPACE
