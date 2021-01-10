#pragma once

#include <base/utilities.h>

DYCOREPLANET_OPEN_NAMESPACE

template <int dim>
double
Tools::compute_pressure_mean_value(
  const DoFHandler<dim> &     dof,
  const Quadrature<dim> &     quadrature,
  const LA::MPI::BlockVector &distributed_pressure_vector)
{
  const FESystem<dim> &fe_system =
    dynamic_cast<const FESystem<dim> &>(dof.get_fe());
  FEValues<dim, dim> fe_values(fe_system,
                               quadrature,
                               UpdateFlags(update_JxW_values | update_values));

  const FEValuesExtractors::Scalar pressure(2 * dim);

  std::vector<double> values(quadrature.size(), 0.0);
  double              mean = 0.;
  double              area = 0.;

  // Compute mean value
  for (const auto &cell : dof.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        fe_values[pressure].get_function_values(distributed_pressure_vector,
                                                values);
        for (unsigned int k = 0; k < quadrature.size(); ++k)
          {
            mean += fe_values.JxW(k) * values[k];
            area += fe_values.JxW(k);
          }
      }

#ifdef DEAL_II_WITH_MPI
  // if this was a distributed DoFHandler, we need to do the reduction
  // over the entire domain
  if (const parallel::TriangulationBase<dim, /* spacedim */ dim>
        *ptr_triangulation = dynamic_cast<
          const parallel::TriangulationBase<dim, /* spacedim */ dim> *>(
          &dof.get_triangulation()))
    {
      double mean_and_area[2] = {mean, area};
      double global_mean_and_area[2];

      const int ierr = MPI_Allreduce(mean_and_area,
                                     global_mean_and_area,
                                     2,
                                     MPI_DOUBLE,
                                     MPI_SUM,
                                     ptr_triangulation->get_communicator());
      AssertThrowMPI(ierr);

      mean = global_mean_and_area[0];
      area = global_mean_and_area[1];
    }
#endif

  return (mean / area);
}

DYCOREPLANET_CLOSE_NAMESPACE
