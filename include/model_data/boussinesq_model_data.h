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
    /// Initial velocity
    //////////////////////////////////////////////////

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

    //////////////////////////////////////////////////
    /// Initial temperature
    //////////////////////////////////////////////////

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
      TemperatureInitialValues(const double R0, const double R1)
        : Function<dim>(1)
      {
        covariance_matrix = 0;

        for (unsigned int d = 0; d < dim; ++d)
          {
            covariance_matrix[d][d] = 20 / ((R1 - R0) / 2);
          }

        center1(0) = R0 + (R1 - R0) * 0.35;
        center2(1) = R0 + (R1 - R0) * 0.65;
      }

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

    private:
      Point<dim>     center1, center2;
      Tensor<2, dim> covariance_matrix;
    };

    //////////////////////////////////////////////////
    /// Temperature RHS
    //////////////////////////////////////////////////

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

  } // namespace Boussinesq

} // namespace CoreModelData


template <int dim>
double
CoreModelData::Boussinesq::TemperatureInitialValues<dim>::value(
  const Point<dim> &p,
  const unsigned int) const
{
  //   const double r = p.norm();
  //   const double h = R1 - R0;
  //   const double s = (r - R0) / h;
  //   const double q =
  //    (dim == 2) ? 1.0 : std::max(0.0, cos(numbers::PI * abs(p(2) / R1)));
  //   const double phi = std::atan2(p(0), p(1));
  //   const double tau = s + s * (1 - s) * sin(6 * phi) * q;
  //
  //   return reference_temperature_bottom * (1.0 - tau) +
  //         reference_temperature_top * tau;

  double temperature =
    sqrt(determinant(covariance_matrix)) *
      exp(-0.5 *
          scalar_product(p - center1, covariance_matrix * (p - center1))) /
      sqrt(std::pow(2 * numbers::PI, dim)) +
    sqrt(determinant(covariance_matrix)) *
      exp(-0.5 *
          scalar_product(p - center2, covariance_matrix * (p - center2))) /
      sqrt(std::pow(2 * numbers::PI, dim));

  return temperature;
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
