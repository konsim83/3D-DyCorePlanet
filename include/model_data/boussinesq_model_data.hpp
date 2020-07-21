#pragma once

// C++ STL
#include <cmath>
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


DYCOREPLANET_OPEN_NAMESPACE

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



#if defined(NO_PHYSICAL_CONST)
    constexpr double reference_pressure = 1;
#else
    /*!
     * Earth reference pressure.
     */
    constexpr double reference_pressure = 1.01325e+5; /* Pa */
#endif


#if defined(NO_PHYSICAL_CONST)
    constexpr double reference_omega = 1;
#else
    /*!
     * Earth angular velocity.
     */
    constexpr double reference_omega =
      2 * numbers::PI / (24 * 60 * 60);                         /* 1/s */
#endif


#if defined(NO_PHYSICAL_CONST)
    constexpr double reference_density = 1;
#else
                                                                /*!
                                                                 * Refence density of air at bottom reference
                                                                 * temperature.
                                                                 */
    constexpr double reference_density = 1.29;                  /* kg / m^3 */
#endif


#if defined(NO_PHYSICAL_CONST)
    constexpr double reference_time = 1;
#else
                                                                /*!
                                                                 * Reference time is one hour.
                                                                 */
    constexpr double reference_time = 3.6e+3;                   /* s */
#endif


#if defined(NO_PHYSICAL_CONST)
    constexpr double reference_velocity = 1;
#else
                                                                /*!
                                                                 * Reference velocity.
                                                                 */
    constexpr double reference_velocity = 10;                   /* m/s */
#endif


#if defined(NO_PHYSICAL_CONST)
    constexpr double reference_length = 1;
#else
                                                                /*!
                                                                 * Reference length.
                                                                 */
    constexpr double reference_length = 1e+4;                   /* m */
#endif


#if defined(NO_PHYSICAL_CONST)
    constexpr double reference_temperature_bottom = 100;
#else
                                                                /*!
                                                                 * Refence temperature 273.15 K (0 degree Celsius) at bottom.
                                                                 */
    constexpr double reference_temperature_bottom = 273.15;     /* K */
#endif


#if defined(NO_PHYSICAL_CONST)
    constexpr double reference_temperature_top = 10;
#else
                                                                /*!
                                                                 * Refence temperature at top of atmospere.
                                                                 */
    constexpr double reference_temperature_top = 253.15;        /* K */
#endif


#if defined(NO_PHYSICAL_CONST)
    constexpr double reference_temperature_change = 45;
#else
                                                                /*!
                                                                 * Reference temperature change.
                                                                 */
    constexpr double reference_temperature_change = 20;         /* K */
#endif


    //////////////////////////////////////////////////
    /// Some physical constants.
    //////////////////////////////////////////////////


#if defined(NO_PHYSICAL_CONST)
    constexpr double universal_gas_constant = 1;
#else
                                                                /*!
                                                                 * Universal gas constant.
                                                                 */
    constexpr double universal_gas_constant = 8.31446261815324; /* J/(mol*K) */
#endif

    /*!
     * Specific gas constant of dry air.
     */
    //    constexpr double specific_gas_constant_dry = 287; /* J/(kg*K) */
#if defined(NO_PHYSICAL_CONST)
    constexpr double specific_gas_constant_dry = 1;
#else

#endif


#if defined(NO_PHYSICAL_CONST)
    constexpr double expansion_coefficient = 0.5;
#else
                                                                /*!
                                                                 * Thermal expansion coefficient (beta) of air at bottom reference
                                                                 * temperature.
                                                                 */
    constexpr double expansion_coefficient =
      1 / reference_temperature_bottom;           /* 1/K */
#endif


#if defined(NO_PHYSICAL_CONST)
    constexpr double dynamic_viscosity = 20;
#else
                                                  /*!
                                                   * Dynamic viscosity (eta or mu) of air at bottom reference
                                                   * temperature.
                                                   */
    constexpr double dynamic_viscosity = 1.82e-5; /* kg/(m*s) */
#endif


#if defined(NO_PHYSICAL_CONST)
    constexpr double kinematic_viscosity = 50;
#else
                                                  /*!
                                                   * Dynamics viscosity (nu) of air at bottom reference
                                                   * temperature.
                                                   */
    constexpr double kinematic_viscosity =
      dynamic_viscosity / reference_density;         /* m^2/s */
#endif


#if defined(NO_PHYSICAL_CONST)
    constexpr double specific_heat_p = 0.1;
#else
                                                     /*!
                                                      * Specific heat capacity of air under constant pressure.
                                                      */
    constexpr double specific_heat_p = 1.005;        /* J / (K*kg) */
#endif


#if defined(NO_PHYSICAL_CONST)
    constexpr double specific_heat_v = 0.1;
#else
                                                     /*!
                                                      * Specific heat capacity of air under isochoric changes of state.
                                                      */
    constexpr double specific_heat_v = 0.718;        /* J / (K*kg) */
#endif


#if defined(NO_PHYSICAL_CONST)
    constexpr double thermal_conductivity = 0.5;
#else
                                                     /*!
                                                      * Thermal conductivity (kappa, k or lambda) of air at bottom reference
                                                      * temperature.
                                                      */
    constexpr double thermal_conductivity = 2.62e-2; /* W/(m*K) */
#endif


#if defined(NO_PHYSICAL_CONST)
    constexpr double thermal_diffusivity = 0.025;
#else
                                                     /*!
                                                      * Thermal diffusivity (alpha or a) of air at bottom reference
                                                      * temperature.
                                                      */
    constexpr double thermal_diffusivity =
      thermal_conductivity / (specific_heat_p * reference_pressure); /*m^2/s */
#endif


#if defined(NO_PHYSICAL_CONST)
    constexpr double radiogenic_heating = 0.01;
#else
                                                                     /*!
                                                                      * A good part of the earth's heat loss through the surface is due
                                                                      * to the decay of radioactive elements (uranium, thorium,
                                                                      * potassium).
                                                                      */
    constexpr double radiogenic_heating = 7.4e-12; /* W / kg */
#endif


#if defined(NO_PHYSICAL_CONST)
    constexpr double gravity_constant = 100;
#else
                                                   /*!
                                                    * Gravity constant.
                                                    */
    constexpr double gravity_constant = 9.81;      /* m/s^2 */
#endif


#if defined(NO_PHYSICAL_CONST)
    constexpr double speed_of_sound = 0.01;
#else
                                                   /*!
                                                    * Speed of sound.
                                                    */
    constexpr double speed_of_sound = 331.5;       /* m/s */
#endif


#if defined(NO_PHYSICAL_CONST)
    constexpr double atm_height = 4;
#else
                                                   /*!
                                                    * Height of atmosphere
                                                    */
    constexpr double atm_height = 2.0e+6;          /* m */
#endif

    /*!
     * Earth radius.
     */
#if defined(NO_PHYSICAL_CONST)
    constexpr double R0 = 1;
#else
    constexpr double R0         = 6.371000e+6;     /* m */
#endif

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
     * Return the Reynolds number of the flow.
     */
    inline double
    get_reynolds_number();



    /*!
     * Return the Peclet number of the flow.
     */
    inline double
    get_peclet_number();



    /*!
     * Return the Grashoff number of the flow.
     */
    inline double
    get_grashoff_number(const int dim);



    /*!
     * Return the Prandtl number of the flow.
     */
    inline double
    get_prandtl_number();



    /*!
     * Return the Rayleigh number of the flow.
     */
    inline double
    get_rayleigh_number(const int dim);



    /*!
     * Density as function of temperature.
     *
     * @param temperature
     * @return desity
     */
    inline double
    density(const double temperature);

    /*!
     * Density scaling as function of temperature. This is the density devided
     * by the reference density.
     *
     * @param temperature
     * @return desity
     */
    inline double
    density_scaling(const double temperature);

    /*!
     * Compute gravity vector at a given point.
     *
     * @param p
     * @return gravity vector
     */
    template <int dim>
    inline Tensor<1, dim>
    gravity_vector(const Point<dim> &p);


    /*!
     * Compute coriolis vector at a given point.
     *
     * @param p
     * @return coriolis vector
     */
    template <int dim>
    inline Tensor<1, dim>
    coriolis_vector(const Point<dim> &p);



    /*!
     * Temerature initial values for rising warm bubble test.
     */
    template <int dim>
    class TemperatureInitialValues : public Function<dim>
    {
    public:
      /*!
       * Constructor.
       */
      TemperatureInitialValues()
        : Function<dim>(1)
      {}

      /*!
       * Return temperature value at a single point.
       *
       * @param p
       * @param component
       * @return
       */
      virtual double
      value(const Point<dim> & p,
            const unsigned int component = 0) const override;

      /*!
       * Return temperature value as a vector at a single point.
       *
       * @param points
       * @param values
       */
      virtual void
      value_list(const std::vector<Point<dim>> &points,
                 std::vector<double> &          values,
                 const unsigned int             component = 0) const override;
    };


    /*!
     * Temerature right-hand side for rising warm bubble test. This term
     * represents external heat sources.
     */
    template <int dim>
    class TemperatureRHS : public Function<dim>
    {
    public:
      /*!
       * Constructor.
       */
      TemperatureRHS()
        : Function<dim>(1)
      {}

      /*!
       * Return temperature value at a single point.
       *
       * @param p
       * @param component
       * @return
       */
      virtual double
      value(const Point<dim> & p,
            const unsigned int component = 0) const override;

      /*!
       * Return temperature value as a vector at a single point.
       *
       * @param points
       * @param values
       */
      virtual void
      value_list(const std::vector<Point<dim>> &points,
                 std::vector<double> &          value,
                 const unsigned int             component = 0) const override;
    };


    /*!
     * Velocity initial values for rising warm bubble test.
     */
    template <int dim>
    class VelocityInitialValues : public TensorFunction<1, dim>
    {
    public:
      /*!
       * Constructor.
       */
      VelocityInitialValues()
        : TensorFunction<1, dim>()
      {}

      /*!
       * Return velocity value at a single point.
       *
       * @param p
       * @return
       */
      virtual Tensor<1, dim>
      value(const Point<dim> &p) const override;

      /*!
       * Return temperature value as a vector at a single point.
       *
       * @param points
       * @param values
       */
      virtual void
      value_list(const std::vector<Point<dim>> &points,
                 std::vector<Tensor<1, dim>> &  values) const override;
    };

  } // namespace Boussinesq

} // namespace CoreModelData



//////////////////////////////////////////////////////////////
// Implementation
//////////////////////////////////////////////////////////////



inline double
CoreModelData::Boussinesq::get_reynolds_number()
{
  return (reference_velocity * reference_length) / kinematic_viscosity;
}



inline double
CoreModelData::Boussinesq::get_peclet_number()
{
  return (reference_velocity * reference_length) / thermal_diffusivity;
}



inline double
CoreModelData::Boussinesq::get_grashoff_number(const int dim)
{
  return (gravity_constant * expansion_coefficient *
          reference_temperature_change * std::pow(reference_length, dim) /
          kinematic_viscosity);
}



inline double
CoreModelData::Boussinesq::get_prandtl_number()
{
  return kinematic_viscosity / thermal_diffusivity;
}



inline double
CoreModelData::Boussinesq::get_rayleigh_number(const int dim)
{
  return gravity_constant * expansion_coefficient *
         reference_temperature_change * std::pow(reference_length, dim) *
         get_prandtl_number();
}



inline double
CoreModelData::Boussinesq::density(const double temperature)
{
  return reference_density *
         (1 -
          expansion_coefficient * (temperature - reference_temperature_bottom));
}



inline double
CoreModelData::Boussinesq::density_scaling(const double temperature)
{
  return (1 -
          expansion_coefficient * (temperature - reference_temperature_bottom));
}



template <int dim>
inline Tensor<1, dim>
CoreModelData::Boussinesq::gravity_vector(const Point<dim> &p)
{
  const double r = p.norm();
  return -gravity_constant * p / r;
}



template <int dim>
inline Tensor<1, dim>
CoreModelData::Boussinesq::coriolis_vector(const Point<dim> &p)
{
  Tensor<1, dim> z;

  z[0]   = 0;
  z[1]   = 0;
  ze[-1] = reference_omega;

  return z;
}



template <int dim>
double
CoreModelData::Boussinesq::TemperatureInitialValues<dim>::value(
  const Point<dim> &p,
  const unsigned int) const
{
  //  const double r = p.norm();
  //
  //  /*
  //   * Linear interpolation of temperature
  //   * between bottom and top of atmosphere.
  //   */
  //  double temperature =
  //    (reference_temperature_top - reference_temperature_bottom) * (r - R0) /
  //      (R1 - R0) +
  //    reference_temperature_bottom;
  //
  //	return temperature;

  const double r = p.norm();
  const double h = R1 - R0;
  const double s = (r - R0) / h;
  const double q =
    (dim == 2) ? 1.0 : std::max(0.0, cos(numbers::PI * abs(p(2) / R1)));
  const double phi = std::atan2(p(0), p(1));
  const double tau = s + s * (1 - s) * sin(6 * phi) * q;

  return reference_temperature_bottom * (1.0 - tau) +
         reference_temperature_top * tau;
}



template <int dim>
void
CoreModelData::Boussinesq::TemperatureInitialValues<dim>::value_list(
  const std::vector<Point<dim>> &points,
  std::vector<double> &          values,
  const unsigned int) const
{
  Assert(points.size() == values.size(),
         ExcDimensionMismatch(points.size(), values.size()));

  for (unsigned int p = 0; p < points.size(); ++p)
    {
      values[p] = value(points[p]);
    }
}



template <int dim>
double
CoreModelData::Boussinesq::TemperatureRHS<dim>::value(const Point<dim> &p,
                                                      const unsigned int) const
{
  return 0;
}



template <int dim>
void
CoreModelData::Boussinesq::TemperatureRHS<dim>::value_list(
  const std::vector<Point<dim>> &points,
  std::vector<double> &          values,
  const unsigned int) const
{
  Assert(points.size() == values.size(),
         ExcDimensionMismatch(points.size(), values.size()));

  for (unsigned int p = 0; p < points.size(); ++p)
    {
      values[p] = value(points[p]);
    }
}



template <int dim>
Tensor<1, dim>
CoreModelData::Boussinesq::VelocityInitialValues<dim>::value(
  const Point<dim> &) const
{
  // This initializes to zero.
  Tensor<1, dim> value;

  return value;
}



template <int dim>
void
CoreModelData::Boussinesq::VelocityInitialValues<dim>::value_list(
  const std::vector<Point<dim>> &points,
  std::vector<Tensor<1, dim>> &  values) const
{
  Assert(points.size() == values.size(),
         ExcDimensionMismatch(points.size(), values.size()));

  for (unsigned int p = 0; p < points.size(); ++p)
    {
      values[p].clear();
      values[p] = value(points[p]);
    }
}

DYCOREPLANET_CLOSE_NAMESPACE
