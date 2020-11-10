#include <model_data/core_model_data.h>

DYCOREPLANET_OPEN_NAMESPACE

namespace CoreModelData
{
  template <int dim>
  Tensor<1, dim>
  vertical_gravity_vector(const Point<dim> & /* p */,
                          const double gravity_constant)
  {
    Tensor<1, dim> e_z;
    e_z[dim - 1] = 1;

    return -gravity_constant * e_z;
  }

  template <int dim>
  Tensor<1, dim>
  gravity_vector(const Point<dim> &p, const double gravity_constant)
  {
    const double r = p.norm();
    return -gravity_constant * p / r;
  }


  template <int dim>
  Tensor<1, dim>
  coriolis_vector(const Point<dim> & /*p*/, const double omega)
  {
    Tensor<1, dim> z;

    z[dim - 1] = omega;

    return z;
  }

} // namespace CoreModelData

DYCOREPLANET_CLOSE_NAMESPACE
