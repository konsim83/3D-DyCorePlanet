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
   * @class SchurComplement
   *
   * @brief Implements a MPI parallel Schur complement
   *
   * Implements a parallel Schur complement through the use of an inner inverse
   * matrix, i.e., if we want to solve
   * \f{eqnarray}{
   *	\left(
   *	\begin{array}{cc}
   *		A & B^T \\
   *		B & C
   *	\end{array}
   *	\right)
   *	\left(
   *	\begin{array}{c}
   *		\sigma \\
   *		u
   *	\end{array}
   *	\right)
   *	=
   *	\left(
   *	\begin{array}{c}
   *		0 \\
   *		u
   *	\end{array}
   *	\right)
   * \f}
   * and know that \f$A\f$ is invertible then we first define the inverse and
   *define the Schur complement as \f{eqnarray}{ \tilde S = C - BP_A^{-1}B^T \f}
   *to solve for \f$u\f$. The inverse must be separately given to the class as
   *an input argument.
   */
  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  class SchurComplement : public Subscriptor
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
    SchurComplement(const BlockMatrixType &system_matrix,
                    const InverseMatrix<BlockType, PreconditionerType>
                      &                          relevant_inverse_matrix,
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
    const SmartPointer<const BlockMatrixType> system_matrix;

    /*!
     * Smart pointer to inverse upper left block of the system matrix.
     */
    const SmartPointer<const InverseMatrix<BlockType, PreconditionerType>>
      relevant_inverse_matrix;

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
    mutable VectorType tmp1, tmp2, tmp3;
  };


  ///////////////////////////////////////
  /// Implementation
  ///////////////////////////////////////


  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  SchurComplement<BlockMatrixType, VectorType, PreconditionerType>::
    SchurComplement(const BlockMatrixType &system_matrix,
                    const InverseMatrix<BlockType, PreconditionerType>
                      &                          relevant_inverse_matrix,
                    const std::vector<IndexSet> &owned_partitioning,
                    MPI_Comm                     mpi_communicator)
    : system_matrix(&system_matrix)
    , relevant_inverse_matrix(&relevant_inverse_matrix)
    , owned_partitioning(owned_partitioning)
    , mpi_communicator(mpi_communicator)
    , tmp1(owned_partitioning[0], mpi_communicator)
    , tmp2(owned_partitioning[0], mpi_communicator)
    , tmp3(owned_partitioning[1], mpi_communicator)
  {}

  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  void
  SchurComplement<BlockMatrixType, VectorType, PreconditionerType>::vmult(
    VectorType &      dst,
    const VectorType &src) const
  {
    system_matrix->block(0, 1).vmult(tmp1, src);
    relevant_inverse_matrix->vmult(tmp2, tmp1);
    system_matrix->block(1, 0).vmult(dst, tmp2);
    system_matrix->block(1, 1).vmult(tmp3, src);
    dst -= tmp3;
  }
} // end namespace LinearAlgebra

DYCOREPLANET_CLOSE_NAMESPACE