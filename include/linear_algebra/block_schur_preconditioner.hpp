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

} // namespace LinearAlgebra


DYCOREPLANET_CLOSE_NAMESPACE
