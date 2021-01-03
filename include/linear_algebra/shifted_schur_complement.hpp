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
   * @class ShiftedSchurComplement
   *
   * @brief Implements a MPI parallel Schur complement
   *
   * Implements a parallel Schur complement through the use of an inner inverse
   * matrix, i.e., if we want to solve
   * \f{eqnarray}{
   *	\left(
   *	\begin{array}{cc}
   *		A & B^T \\
   *		B & 0
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
   *		f \\
   *		0
   *	\end{array}
   *	\right)
   * \f}
   * and know that \f$A\f$ is invertible then we first define the inverse and
   *define the Schur complement as \f{eqnarray}{ \tilde S = BP_A^{-1}B^T \f}
   *to solve for \f$u\f$. The inverse must be separately given to the class as
   *an input argument.
   */
  template <typename BlockMatrixType,
            typename VectorType,
            typename InverseMatrixType>
  class ShiftedSchurComplement : public Subscriptor
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
    ShiftedSchurComplement(const BlockMatrixType &      system_matrix,
                           const InverseMatrixType &    relevant_inverse_matrix,
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
     * Smart pointer to system matrix block 01.
     */
    const SmartPointer<const BlockType> block_01;

    /*!
     * Smart pointer to system matrix block 10.
     */
    const SmartPointer<const BlockType> block_10;

    /*!
     * Smart pointer to system matrix block 11.
     */
    const SmartPointer<const BlockType> block_11;

    /*!
     * Smart pointer to inverse upper left block of the system matrix.
     */
    const SmartPointer<const InverseMatrixType> relevant_inverse_matrix;

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
            typename InverseMatrixType>
  ShiftedSchurComplement<BlockMatrixType, VectorType, InverseMatrixType>::
    ShiftedSchurComplement(const BlockMatrixType &      system_matrix,
                           const InverseMatrixType &    relevant_inverse_matrix,
                           const std::vector<IndexSet> &owned_partitioning,
                           MPI_Comm                     mpi_communicator)
    : block_01(&(system_matrix.block(0, 1)))
    , block_10(&(system_matrix.block(1, 0)))
    , block_11(&(system_matrix.block(1, 1)))
    , relevant_inverse_matrix(&relevant_inverse_matrix)
    , owned_partitioning(owned_partitioning)
    , mpi_communicator(mpi_communicator)
    , tmp1(owned_partitioning[0], mpi_communicator)
    , tmp2(owned_partitioning[0], mpi_communicator)
  {}

  template <typename BlockMatrixType,
            typename VectorType,
            typename InverseMatrixType>
  void
  ShiftedSchurComplement<BlockMatrixType, VectorType, InverseMatrixType>::vmult(
    VectorType &      dst,
    const VectorType &src) const
  {
    block_11->vmult(dst, src);
    block_01->vmult(tmp1, src);
    relevant_inverse_matrix->vmult(tmp2, tmp1);
    tmp2 *= -1;
    block_10->vmult_add(dst, tmp2);
  }



  template <typename BlockMatrixType,
            typename VectorType,
            typename InverseMatrixType>
  class ApproxShiftedSchurComplementInverse : public Subscriptor
  {
  private:
    using BlockType = typename BlockMatrixType::BlockType;

  public:
    ApproxShiftedSchurComplementInverse(
      const BlockMatrixType &      system_matrix,
      const InverseMatrixType &    mass_w_inverse,
      const InverseMatrixType &    mass_u_inverse,
      const unsigned int           n_neumann_terms,
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
     * Smart pointer to system matrix block 01.
     */
    const SmartPointer<const BlockType> block_01;

    /*!
     * Smart pointer to system matrix block 10.
     */
    const SmartPointer<const BlockType> block_10;

    const SmartPointer<const InverseMatrixType> mass_w_inverse;
    const SmartPointer<const InverseMatrixType> mass_u_inverse;

    const ShiftedSchurComplement<BlockMatrixType, VectorType, InverseMatrixType>
      shifted_schur_complement;

    const unsigned int n_neumann_terms;

    /*!
     * Current MPI communicator.
     */
    MPI_Comm mpi_communicator;

    /*!
     * Muatable types for temporary vectors.
     */
    mutable VectorType tmp1, tmp2, tmp3, tmp4;
  };


  ///////////////////////////////////////
  /// Implementation
  ///////////////////////////////////////


  template <typename BlockMatrixType,
            typename VectorType,
            typename InverseMatrixType>
  ApproxShiftedSchurComplementInverse<BlockMatrixType,
                                      VectorType,
                                      InverseMatrixType>::
    ApproxShiftedSchurComplementInverse(
      const BlockMatrixType &      system_matrix,
      const InverseMatrixType &    mass_w_inverse,
      const InverseMatrixType &    mass_u_inverse,
      const unsigned int           n_neumann_terms,
      const std::vector<IndexSet> &owned_partitioning,
      MPI_Comm                     mpi_communicator)
    : block_01(&(system_matrix.block(0, 1)))
    , block_10(&(system_matrix.block(1, 0)))
    , mass_w_inverse(&mass_w_inverse)
    , mass_u_inverse(&mass_u_inverse)
    , shifted_schur_complement(system_matrix,
                               mass_w_inverse,
                               owned_partitioning,
                               mpi_communicator)
    , n_neumann_terms(n_neumann_terms)
    , mpi_communicator(mpi_communicator)
    , tmp1(owned_partitioning[1], mpi_communicator)
    , tmp2(owned_partitioning[0], mpi_communicator)
    , tmp3(owned_partitioning[0], mpi_communicator)
    , tmp4(owned_partitioning[1], mpi_communicator)
  {}

  template <typename BlockMatrixType,
            typename VectorType,
            typename InverseMatrixType>
  void
  ApproxShiftedSchurComplementInverse<BlockMatrixType,
                                      VectorType,
                                      InverseMatrixType>::vmult(VectorType &dst,
                                                                const VectorType
                                                                  &src) const
  {
    try
      {
        SolverControl solver_control(
          /* maxiter */ 30,
          1e-6 * src.l2_norm(),
          /* log_history */ false,
          /* log_result */ false);
        SolverGMRES<VectorType> local_solver(solver_control);
        local_solver.solve(shifted_schur_complement,
                           dst,
                           src,
                           LA::PreconditionIdentity());
      }
    catch (SolverControl::NoConvergence &)
      {
        // std::cout
        //   << "Applied 15 shifted_schur_complement preconditioning iterations"
        //   << std::endl;
      }
    // // zero order term
    // tmp4 = src;

    // for (unsigned int i = 0; i < n_neumann_terms - 1; ++i)
    //   {
    //     mass_u_inverse->vmult(tmp1, tmp4);
    //     block_01->vmult(tmp2, tmp1);
    //     mass_w_inverse->vmult(tmp3, tmp2);
    //     block_10->vmult_add(tmp4, tmp3);
    //   }

    // mass_u_inverse->vmult(dst, tmp4);
  }
} // end namespace LinearAlgebra

DYCOREPLANET_CLOSE_NAMESPACE
