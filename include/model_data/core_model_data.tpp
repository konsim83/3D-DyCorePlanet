#include <model_data/core_model_data.h>

DYCOREPLANET_OPEN_NAMESPACE

namespace CoreModelData
{
  template <int dim>
  Tensor<1, dim>
  CoreModelData::gravity_vector(const Point<dim> &p,
                                const double      gravity_constant)
  {
    const double r = p.norm();
    return -gravity_constant * p / r;
  }


  template <int dim>
  Tensor<1, dim>
  CoreModelData::coriolis_vector(const Point<dim> & /*p*/, const double omega)
  {
    Tensor<1, dim> z;

    z[dim - 1] = omega;

    return z;
  }

} // namespace CoreModelData

DYCOREPLANET_CLOSE_NAMESPACE
