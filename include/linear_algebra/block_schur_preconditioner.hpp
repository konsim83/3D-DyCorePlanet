#pragma once

// AquaPlanet
#include <base/config.h>
#include <linear_algebra/schur_complement.hpp>

DYCOREPLANET_OPEN_NAMESPACE

namespace LinearAlgebra
{
  /*!
   * @class BlockSchurPreconditioner
   *
   * This is a right block preconditioner meant for a FGMRES solver for a Stokes
   * problem.
   */
  template <class PreconditionerTypeA, class PreconditionerTypeMp>
  class BlockSchurPreconditioner : public Subscriptor
  {
  public:
    BlockSchurPreconditioner(const LA::BlockSparseMatrix &S,
                             const LA::BlockSparseMatrix &Spre,
                             const PreconditionerTypeMp & Mppreconditioner,
                             const PreconditionerTypeA &  Apreconditioner,
                             const bool                   do_solve_A,
                             const std::vector<IndexSet> &owned_partitioning,
                             MPI_Comm                     mpi_communicator)
      : nse_matrix(&S)
      , nse_preconditioner_matrix(&Spre)
      , mp_preconditioner(Mppreconditioner)
      , a_preconditioner(Apreconditioner)
      , schur_complement_Mp(S,
                            Apreconditioner,
                            owned_partitioning,
                            mpi_communicator)
      , do_solve_A(do_solve_A)
      , owned_partitioning(owned_partitioning)
      , mpi_communicator(mpi_communicator)
    {}

    void
    vmult(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const
    {
      LA::MPI::Vector utmp(src.block(0));
      {
        SolverControl solver_control(5000, 1e-6 * src.block(1).l2_norm());
        SolverGMRES<LA::MPI::Vector> solver(solver_control);
        solver.solve(schur_complement_Mp,
                     dst.block(1),
                     src.block(1),
                     LA::PreconditionIdentity());
        dst.block(1) *= -1.0;
      }
      {
        nse_matrix->block(0, 1).vmult(utmp, dst.block(1));
        utmp *= -1.0;
        utmp.add(src.block(0));
      }
      if (do_solve_A == true)
        {
          SolverControl   solver_control(5000, utmp.l2_norm() * 1e-2);
          LA::SolverGMRES solver(solver_control);
          solver.solve(nse_matrix->block(0, 0),
                       dst.block(0),
                       utmp,
                       a_preconditioner);
        }
      else
        a_preconditioner.vmult(dst.block(0), utmp);
    }

  private:
    const SmartPointer<const LA::BlockSparseMatrix> nse_matrix;
    const SmartPointer<const LA::BlockSparseMatrix> nse_preconditioner_matrix;
    const PreconditionerTypeMp &                    mp_preconditioner;
    const PreconditionerTypeA &                     a_preconditioner;
    SchurComplement<LA::BlockSparseMatrix,
                    LA::MPI::Vector,
                    PreconditionerTypeMp>
      schur_complement_Mp;

    const bool do_solve_A;

    const std::vector<IndexSet> &owned_partitioning;
    MPI_Comm                     mpi_communicator;
  };


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
                                   &_approx_nested_schur_complement_inverse)
      : nse_matrix(&S)
      , Mw_inverse(&Mw_inverse)
      , approx_Mu_minus_Sw_inverse(&_approx_Mu_minus_Sw_inverse)
      , approx_nested_schur_complement_inverse(
          &_approx_nested_schur_complement_inverse)
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
    }

  private:
    const SmartPointer<const LA::BlockSparseMatrix> nse_matrix;
    const SmartPointer<const InverseType>           Mw_inverse;
    const SmartPointer<const ApproxShiftedSchurComplementInverseType>
      approx_Mu_minus_Sw_inverse;
    const SmartPointer<const ApproxNestedSchurComplementInverseType>
      approx_nested_schur_complement_inverse;
  };

} // namespace LinearAlgebra


DYCOREPLANET_CLOSE_NAMESPACE
