#pragma once

// Deal.ii
#include <deal.II/base/subscriptor.h>

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
            typename PreconditionerType>
  class NestedSchurComplement : public Subscriptor
  {
  private:
    using BlockType = typename BlockMatrixType::BlockType;

  public:
    /*!
     * Constructor. The user must take care to pass the correct inverse of the
     * upper left of the system matrix.
     */
    NestedSchurComplement(const BlockMatrixType &system_matrix,
                          const InverseMatrix<BlockType, PreconditionerType>
                            &inverse_matrix_block_00,
                          const std::vector<IndexSet> &owned_partitioning,
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
     * Smart pointer to system matrix.
     */
    const SmartPointer<const BlockMatrixType> block_system_matrix;

    /*!
     * Smart pointer to inverse upper left block of the system
     * matrix.
     */
    const SmartPointer<const InverseMatrix<BlockType, PreconditionerType>>
      inverse_matrix_block_00;

    /*!
     * Index set to initialize tmp vectors using only locally owned partition.
     */
    const std::vector<IndexSet> &owned_partitioning;

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
            typename PreconditionerType>
  NestedSchurComplement<BlockMatrixType, VectorType, PreconditionerType>::
    NestedSchurComplement(const BlockMatrixType &system_matrix,
                          const InverseMatrix<BlockType, PreconditionerType>
                            &                          relevant_inverse_matrix,
                          const std::vector<IndexSet> &owned_partitioning,
                          MPI_Comm                     mpi_communicator)
    : block_system_matrix(&system_matrix)
    , inverse_matrix_block_00(&inverse_matrix_block_00)
    , owned_partitioning(owned_partitioning)
    , mpi_communicator(mpi_communicator)
    , tmp1(owned_partitioning[0], mpi_communicator)
    , tmp2(owned_partitioning[0], mpi_communicator)
  {}

  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  void
  NestedSchurComplement<BlockMatrixType, VectorType, PreconditionerType>::vmult(
    VectorType &      dst,
    const VectorType &src) const
  {
    block_01->vmult(tmp1, src);
    relevant_inverse_matrix->vmult(tmp2, tmp1);
    block_10->vmult(dst, tmp2);
  }
} // end namespace LinearAlgebra

DYCOREPLANET_CLOSE_NAMESPACE