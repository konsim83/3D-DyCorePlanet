#include <model_data/boussinesq_model_data.h>

AQUAPLANET_OPEN_NAMESPACE


double
CoreModelData::Boussinesq::density(const double temperature)
{
  return reference_density *
         (1 -
          expansion_coefficient * (temperature - reference_temperature_bottom));
}



double
CoreModelData::Boussinesq::density_scaling(const double temperature)
{
  return (1 -
          expansion_coefficient * (temperature - reference_temperature_bottom));
}



Tensor<1, 3>
CoreModelData::Boussinesq::gravity_vector(const Point<3> &p)
{
  const double r = p.norm();
  return gravity_constant * p / r;
}



double
CoreModelData::Boussinesq::TemperatureInitialValues::value(
  const Point<3> &p,
  const unsigned int) const
{
  /*
   * Linear interpolation of temperature
   * between bottom and top of atmosphere.
   */
  const double r = p.norm();

  double temperature =
    (reference_temperature_top - reference_temperature_bottom) * (r - R0) /
      (R1 - R0) +
    reference_temperature_bottom;

  return temperature;
}



void
CoreModelData::Boussinesq::TemperatureInitialValues::value_list(
  const std::vector<Point<3>> &points,
  std::vector<double> &        values,
  const unsigned int) const
{
  Assert(points.size() == values.size(),
         ExcDimensionMismatch(points.size(), values.size()));

  for (unsigned int p = 0; p < points.size(); ++p)
    {
      values[p] = value(points[p]);
    }
}



double
CoreModelData::Boussinesq::TemperatureRHS::value(const Point<3> &p,
                                                 const unsigned int) const
{
  return 0;
}



void
CoreModelData::Boussinesq::TemperatureRHS::value_list(
  const std::vector<Point<3>> &points,
  std::vector<double> &        values,
  const unsigned int) const
{
  Assert(points.size() == values.size(),
         ExcDimensionMismatch(points.size(), values.size()));

  for (unsigned int p = 0; p < points.size(); ++p)
    {
      values[p] = value(points[p]);
    }
}



Tensor<1, 3>
CoreModelData::Boussinesq::VelocityInitialValues::value(const Point<3> &) const
{
  // This initializes to zero.
  Tensor<1, 3> value;

  return value;
}



void
CoreModelData::Boussinesq::VelocityInitialValues::value_list(
  const std::vector<Point<3>> &points,
  std::vector<Tensor<1, 3>> &  values) const
{
  Assert(points.size() == values.size(),
         ExcDimensionMismatch(points.size(), values.size()));

  for (unsigned int p = 0; p < points.size(); ++p)
    {
      values[p].clear();
      values[p] = value(points[p]);
    }
}

AQUAPLANET_CLOSE_NAMESPACE
