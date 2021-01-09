#pragma once

// Deal.ii
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

// AquaPlanet
#include <base/config.h>
#include <linear_algebra/schur_complement.hpp>

DYCOREPLANET_OPEN_NAMESPACE

namespace LinearAlgebra
{
  /*!
   * @class BlockSchurPreconditionerFEEC
   *
   * This is a left block preconditioner meant for a GMRES solver for a
   * time-dependent Stokes problem with mass matrix.
   */
  template <class InverseType,
            class ApproxShiftedSchurComplementInverseType,
            class ApproxNestedSchurComplementInverseType>
  class BlockSchurPreconditionerFEEC : public Subscriptor
  {
  public:
    BlockSchurPreconditionerFEEC(const LA::BlockSparseMatrix &S,
                                 const InverseType &          Mw_inverse,
                                 const ApproxShiftedSchurComplementInverseType
                                   &_approx_Mu_minus_Sw_inverse,
                                 const ApproxNestedSchurComplementInverseType
                                   &_approx_nested_schur_complement_inverse,
                                 DoFHandler<3> &_dof_handler,
                                 const bool     correct_to_zero_mean)
      : nse_matrix(&S)
      , Mw_inverse(&Mw_inverse)
      , approx_Mu_minus_Sw_inverse(&_approx_Mu_minus_Sw_inverse)
      , approx_nested_schur_complement_inverse(
          &_approx_nested_schur_complement_inverse)
      , dof_handler(&_dof_handler)
      , correct_to_zero_mean(correct_to_zero_mean)
    {}

    void
    vmult(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const
    {
      {
        /*
         * First block is a fully converged CG solve. It is just a mass matrix
         * in H(curl).
         */
        Mw_inverse->vmult(dst.block(0), src.block(0));
      }

      LA::MPI::Vector utmp1(src.block(1));
      LA::MPI::Vector utmp2(src.block(1));
      {
        /*
         * Second component
         */
        nse_matrix->block(1, 0).vmult(utmp1, dst.block(0));
        utmp1 *= -1.0;
        utmp1.add(src.block(1));
        approx_Mu_minus_Sw_inverse->vmult(dst.block(1), utmp1);
      }

      LA::MPI::Vector ptmp(src.block(2));
      {
        /*
         * Third component
         */
        ptmp.add(src.block(2));
        ptmp *= -1.0;
        nse_matrix->block(2, 1).vmult_add(ptmp, dst.block(1));
        approx_nested_schur_complement_inverse->vmult(dst.block(2), ptmp);
      }

      if (correct_to_zero_mean)
        {
          const double mean_pressure = VectorTools::compute_mean_value(
            *dof_handler, QGauss<3>(2), dst, /* 2*dim */ 6);
          dst.block(2).add(-mean_pressure);
        }
    }

  private:
    const SmartPointer<const LA::BlockSparseMatrix> nse_matrix;
    const SmartPointer<const InverseType>           Mw_inverse;

    const SmartPointer<const ApproxShiftedSchurComplementInverseType>
      approx_Mu_minus_Sw_inverse;
    const SmartPointer<const ApproxNestedSchurComplementInverseType>
      approx_nested_schur_complement_inverse;

    const SmartPointer<const DoFHandler<3>> dof_handler;
    const bool                              correct_to_zero_mean;
  };


  ///////////////////////////////////////
  /// Implementation
  ///////////////////////////////////////


} // namespace LinearAlgebra


DYCOREPLANET_CLOSE_NAMESPACE
