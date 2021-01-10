#pragma once

#include <deal.II/base/quadrature.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/trilinos_parallel_block_vector.h>

#include <base/config.h>
#include <sys/stat.h>

#include <stdexcept>
#include <string>

DYCOREPLANET_OPEN_NAMESPACE

/*!
 * @namespace Tools
 *
 * @brief Namespace containing all tools that do not fit other more specific namespaces.
 */
namespace Tools
{
  /*!
   * @brief Creates a directory with a name from a given string
   *
   * @param dir_name
   */
  void
  create_data_directory(std::string dir_name);


  template <int dim>
  double
  compute_pressure_mean_value(
    const DoFHandler<dim> &     dof,
    const Quadrature<dim> &     quadrature,
    const LA::MPI::BlockVector &distributed_pressure_vector);

  void
  get_face_sign_change_raviart_thomas(
    const DoFHandler<3>::active_cell_iterator &cell,
    const FiniteElement<3> &                   fe,
    std::vector<double> &                      face_sign);

  /*
   * Extern template instantiations
   */
  extern template double
  compute_pressure_mean_value<2>(
    const DoFHandler<2> &       dof,
    const Quadrature<2> &       quadrature,
    const LA::MPI::BlockVector &distributed_pressure_vector);

  extern template double
  compute_pressure_mean_value<3>(
    const DoFHandler<3> &       dof,
    const Quadrature<3> &       quadrature,
    const LA::MPI::BlockVector &distributed_pressure_vector);

} // namespace Tools

DYCOREPLANET_CLOSE_NAMESPACE
