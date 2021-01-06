#pragma once

// Deal.ii
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/numerics/vector_tools.h>

// STL
#include <memory>
#include <vector>

// My headers
#include <base/config.h>

DYCOREPLANET_OPEN_NAMESPACE

namespace LinearAlgebra
{
  /*!
   * @class NestedSchurComplement
   *
   * @brief Implements a MPI parallel nested Schur complement
   *
   * Implements a parallel nested Schur complement through the use of two inner
   * inverse matrices, i.e., if we want to solve
   * \f{eqnarray}{ \left(
   *	\begin{array}{ccc}
   *		M & R_w & 0 \\
   *		R_u & A & B^T \\
   *		0 & B & 0
   *	\end{array}
   *	\right)
   *	\left(
   *	\begin{array}{c}
   *		w \\
   *		u \\
   *		p
   *	\end{array}
   *	\right)
   *	=
   *	\left(
   *	\begin{array}{c}
   *		0 \\
   *		f \\
   *		0
   *	\end{array}
   *	\right)
   * \f}
   * and know that \f$M\f$ and \f$A\f$ are invertible. We do so by using nested
   * Schur complement solvers to first solve for \f$p\f$, then for \f$u\f$ and
   * finally for \f$w\f$. The inverse must be separately given to the class as
   *an input argument.
   */
  template <typename BlockMatrixType,
            typename VectorType,
            typename InverseMatrixType,
            typename DoFHandlerType>
  class NestedSchurComplement : public Subscriptor
  {
  private:
    using BlockType = typename BlockMatrixType::BlockType;

  public:
    /*!
     * Constructor. The user must take care to pass the correct inverse of the
     * upper left block of the system matrix.
     *
     * @param system_matrix
     * 	Block Matrix
     * @param relevant_inverse_matrix
     * 	Inverse of upper left block of the system matrix.
     * @param owned_partitioning
     * @param mpi_communicator
     */
    NestedSchurComplement(const BlockMatrixType &      system_matrix,
                          const InverseMatrixType &    relevant_inverse_matrix,
                          const std::vector<IndexSet> &owned_partitioning,
                          DoFHandlerType &             dof_handler,
                          MPI_Comm                     mpi_communicator);

    /*!
     * Matrix-vector product.
     *
     * @param dst
     * @param src
     */
    void
    vmult(VectorType &dst, const VectorType &src) const;

  private:
    /*!
     * Smart pointer to system matrix block 12.
     */
    const SmartPointer<const BlockType> block_12;

    /*!
     * Smart pointer to system matrix block 21.
     */
    const SmartPointer<const BlockType> block_21;

    /*!
     * Smart pointer to inverse upper left block of the system matrix.
     */
    const SmartPointer<const InverseMatrixType> relevant_inverse_matrix;

    /*!
     * Index set to initialize tmp vectors using only locally owned partition.
     */
    const std::vector<IndexSet> &owned_partitioning;

    /*
     * DofHandler object is necessary to compute the mean value
     */
    DoFHandlerType dof_handler;

    /*!
     * Current MPI communicator.
     */
    MPI_Comm mpi_communicator;

    /*!
     * Muatable types for temporary vectors.
     */
    mutable VectorType tmp1, tmp2;
  };


  ///////////////////////////////////////
  /// Implementation
  ///////////////////////////////////////


  template <typename BlockMatrixType,
            typename VectorType,
            typename InverseMatrixType,
            typename DoFHandlerType>
  NestedSchurComplement<BlockMatrixType,
                        VectorType,
                        InverseMatrixType,
                        DoFHandlerType>::
    NestedSchurComplement(const BlockMatrixType &      system_matrix,
                          const InverseMatrixType &    relevant_inverse_matrix,
                          const std::vector<IndexSet> &owned_partitioning,
                          DoFHandlerType &             _dof_handler,
                          MPI_Comm                     mpi_communicator)
    : block_12(&(system_matrix.block(1, 2)))
    , block_21(&(system_matrix.block(2, 1)))
    , relevant_inverse_matrix(&relevant_inverse_matrix)
    , owned_partitioning(owned_partitioning)
    , dof_handler(_dof_handler.get_triangulation())
    , mpi_communicator(mpi_communicator)
    , tmp1(owned_partitioning[1], mpi_communicator)
    , tmp2(owned_partitioning[1], mpi_communicator)
  {
    FEValuesExtractors::Scalar pressure_components(2 * 3);
    const auto &               nse_fe(_dof_handler.get_fe());
    ComponentMask pressure_mask(nse_fe.component_mask(pressure_components));
    const auto &  pressure_fe(nse_fe.get_sub_fe(pressure_mask));
    dof_handler.distribute_dofs(pressure_fe);
  }

  template <typename BlockMatrixType,
            typename VectorType,
            typename InverseMatrixType,
            typename DoFHandlerType>
  void
  NestedSchurComplement<BlockMatrixType,
                        VectorType,
                        InverseMatrixType,
                        DoFHandlerType>::vmult(VectorType &      dst,
                                               const VectorType &src) const
  {
    block_12->vmult(tmp1, src);
    relevant_inverse_matrix->vmult(tmp2, tmp1);
    block_21->vmult(dst, tmp2);
    const double mean_value =
      VectorTools::compute_mean_value(dof_handler, QGauss<3>(1), dst, 0);
    dst.add(-mean_value);
  }


  /*!
   * @class ApproxNestedSchurComplementInverse
   *
   * @tparam SchurComplementType
   * @tparam VectorType
   * @tparam DoFHandlerType
   */
  template <typename SchurComplementType,
            typename VectorType,
            typename DoFHandlerType>
  class ApproxNestedSchurComplementInverse : public Subscriptor
  {
  public:
    ApproxNestedSchurComplementInverse(
      const LA::BlockSparseMatrix &precon_matrix,
      const SchurComplementType &  schur_complement_matrix,
      const std::vector<IndexSet> &owned_partitioning,
      DoFHandlerType &             dof_handler,
      MPI_Comm                     mpi_communicator,
      const bool                   correct_to_zero_mean);

    /*!
     * Matrix-vector product.
     *
     * @param dst
     * @param src
     */
    void
    vmult(VectorType &dst, const VectorType &src) const;

  private:
    const SmartPointer<const LA::BlockSparseMatrix> precon_matrix;

    /*!
     * Smart pointer to system matrix block 00, pressure Laplace.
     */
    const SmartPointer<const SchurComplementType> approx_pressure_schur_compl;

    /*!
     * Index set to initialize tmp vectors using only locally owned partition.
     */
    const std::vector<IndexSet> &owned_partitioning;

    /*
     * DofHandler object is necessary to compute the mean value
     */
    DoFHandlerType dof_handler;

    /*!
     * Current MPI communicator.
     */
    MPI_Comm mpi_communicator;

    const bool correct_to_zero_mean;

    LA::PreconditionJacobi Mp_preconditioner;
  };


  ///////////////////////////////////////
  /// Implementation
  ///////////////////////////////////////


  template <typename SchurComplementType,
            typename VectorType,
            typename DoFHandlerType>
  ApproxNestedSchurComplementInverse<SchurComplementType,
                                     VectorType,
                                     DoFHandlerType>::
    ApproxNestedSchurComplementInverse(
      const LA::BlockSparseMatrix &_precon_matrix,
      const SchurComplementType &  approx_schur_complement_matrix,
      const std::vector<IndexSet> &owned_partitioning,
      DoFHandlerType &             _dof_handler,
      MPI_Comm                     mpi_communicator,
      const bool                   correct_to_zero_mean)
    : precon_matrix(&_precon_matrix)
    , approx_pressure_schur_compl(&approx_schur_complement_matrix)
    , owned_partitioning(owned_partitioning)
    , dof_handler(_dof_handler.get_triangulation())
    , mpi_communicator(mpi_communicator)
    , correct_to_zero_mean(correct_to_zero_mean)
  {
    Mp_preconditioner.initialize(precon_matrix->block(2, 2));

    if (correct_to_zero_mean)
      {
        FEValuesExtractors::Scalar pressure_components(2 * 3);
        const auto &               nse_fe(_dof_handler.get_fe());
        ComponentMask pressure_mask(nse_fe.component_mask(pressure_components));
        const auto &  pressure_fe(nse_fe.get_sub_fe(pressure_mask));
        dof_handler.distribute_dofs(pressure_fe);
      }
  }

  template <typename SchurComplementType,
            typename VectorType,
            typename DoFHandlerType>
  void
  ApproxNestedSchurComplementInverse<SchurComplementType,
                                     VectorType,
                                     DoFHandlerType>::vmult(VectorType &dst,
                                                            const VectorType
                                                              &src) const
  {
    try
      {
        SolverControl solver_control(
          /* maxiter */ 100,
          // std::max(static_cast<std::size_t>(src.size()),
          //                                     static_cast<std::size_t>(1000)),
          1e-6 * src.l2_norm(),
          /* log_history */ false,
          /* log_result */ true);
        SolverGMRES<VectorType> local_solver(solver_control);

        local_solver.solve(*approx_pressure_schur_compl,
                           dst,
                           src,
                           LA::PreconditionIdentity());
        //                  Mp_preconditioner);
      }
    catch (std::exception &exc)
      {
        // std::cout << "Applied 15 pressure preconditioning iterations"
        //           << std::endl;
      }

    if (correct_to_zero_mean)
      {
        const double mean_value =
          VectorTools::compute_mean_value(dof_handler, QGauss<3>(1), dst, 0);
        dst.add(-mean_value);
      }
  }
} // end namespace LinearAlgebra

DYCOREPLANET_CLOSE_NAMESPACE
