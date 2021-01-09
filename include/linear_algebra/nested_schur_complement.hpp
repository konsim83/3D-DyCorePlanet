#pragma once

// Deal.ii
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/subscriptor.h>

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
   * @class ApproxNestedSchurComplementInverse
   *
   * @brief Implements a MPI parallel nested Schur complement
   *
   * Implements an approximate parallel nested Schur complement through the use
   * of two inner inverse matrices, i.e., if we want to solve
   * \f{eqnarray}{
   * \left( \begin{array}{ccc}
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
   * an input argument.
   *
   * @tparam SchurComplementType
   * @tparam VectorType
   */
  template <typename SchurComplementType, typename VectorType>
  class ApproxNestedSchurComplementInverse : public Subscriptor
  {
  public:
    ApproxNestedSchurComplementInverse(
      const LA::BlockSparseMatrix &precon_matrix,
      const SchurComplementType &  schur_complement_matrix);

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

    LA::PreconditionJacobi Mp_preconditioner;
  };


  ///////////////////////////////////////
  /// Implementation
  ///////////////////////////////////////


  template <typename SchurComplementType, typename VectorType>
  ApproxNestedSchurComplementInverse<SchurComplementType, VectorType>::
    ApproxNestedSchurComplementInverse(
      const LA::BlockSparseMatrix &_precon_matrix,
      const SchurComplementType &  approx_schur_complement_matrix)
    : precon_matrix(&_precon_matrix)
    , approx_pressure_schur_compl(&approx_schur_complement_matrix)
  {
    Mp_preconditioner.initialize(precon_matrix->block(2, 2));
  }

  template <typename SchurComplementType, typename VectorType>
  void
  ApproxNestedSchurComplementInverse<SchurComplementType, VectorType>::vmult(
    VectorType &      dst,
    const VectorType &src) const
  {
    try
      {
        SolverControl solver_control(
          /* maxiter */ 500,
          // std::max(static_cast<std::size_t>(src.size()),
          //                                     static_cast<std::size_t>(1000)),
          1e-6 * src.l2_norm(),
          /* log_history */ false,
          /* log_result */ false);
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
  }
} // end namespace LinearAlgebra

DYCOREPLANET_CLOSE_NAMESPACE
