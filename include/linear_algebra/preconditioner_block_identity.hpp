#pragma once

// Deal.ii
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/numerics/vector_tools.h>


// AquaPlanet
#include <base/config.h>

DYCOREPLANET_OPEN_NAMESPACE

namespace LinearAlgebra
{
  /*!
   * @class PreconditionerBlockIdentity
   *
   * This is just the identity as a preconditioner.
   */
  template <typename DoFHandlerType>
  class PreconditionerBlockIdentity
  {
  public:
    PreconditionerBlockIdentity(DoFHandlerType &dof_handler,
                                const bool      correct_pressure_mean_value)
      : dof_handler(dof_handler)
      , correct_pressure_mean_value(correct_pressure_mean_value)
    {}

    void
    vmult(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const
    {
      dst = src;

      if (correct_pressure_mean_value)
        {
          const double mean_pressure = VectorTools::compute_mean_value(
            dof_handler, QGauss<3>(2), dst, /* 2*dim */ 6);
          dst.block(2).add(-mean_pressure);

          //          std::cout << "      Blocksolver internal correction: The
          //          mean value "
          //                       "was adjusted by "
          //                    << -mean_pressure << "    -> new mean:   "
          //                    << VectorTools::compute_mean_value(dof_handler,
          //                                                       QGauss<3>(2),
          //                                                       dst,
          //                                                       /* 2*dim */
          //                                                       6)
          //                    << std::endl;
        }
    }

  private:
    DoFHandlerType &dof_handler;
    const bool      correct_pressure_mean_value;
  };

} // namespace LinearAlgebra


DYCOREPLANET_CLOSE_NAMESPACE
