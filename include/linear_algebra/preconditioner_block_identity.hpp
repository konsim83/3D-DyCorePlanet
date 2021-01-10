#pragma once

// Deal.ii
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/numerics/vector_tools.h>


// AquaPlanet
#include <base/config.h>
#include <base/utilities.h>

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
      : dof_handler(&dof_handler)
      , correct_pressure_mean_value(correct_pressure_mean_value)
    {}

    void
    vmult(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const
    {
      dst = src;

      if (correct_pressure_mean_value)
        {
          const double mean_pressure =
            Tools::compute_pressure_mean_value(*dof_handler, QGauss<3>(1), dst);
          dst.block(2).add(-mean_pressure);
        }
    }

  private:
    const SmartPointer<const DoFHandlerType> dof_handler;
    const bool                               correct_pressure_mean_value;
  };


  ///////////////////////////////////////
  /// Implementation
  ///////////////////////////////////////


} // namespace LinearAlgebra


DYCOREPLANET_CLOSE_NAMESPACE
