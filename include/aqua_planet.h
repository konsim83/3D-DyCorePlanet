#ifndef INCLUDE_AQUA_PLANET_H_
#define INCLUDE_AQUA_PLANET_H_

// Deal.ii
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/numerics/data_out.h>


// C++ STL
#include <fstream>
#include <iostream>
#include <string>


// My headers


using namespace dealii;


/*!
 * Class to handle mesh for aqua planet. The mesh
 * is a 3D spherical shell.
 */
class AquaPlanet
{
public:
  /*!
   * Constructor of mesh handler for spherical shell.
   */
  AquaPlanet();
  ~AquaPlanet();

  void
  run();

private:
  void
  make_mesh();
  void
  output_mesh() const;
  void
  refine_grid();

  MPI_Comm mpi_communicator;

  ConditionalOStream pcout;
  TimerOutput        computing_timer;

  parallel::distributed::Triangulation<3> triangulation;

  DoFHandler<3> dof_handler;

  const Point<3> center;
  const double   inner_radius, outer_radius;

  const SphericalManifold<3> boundary_description;
};


#endif /* INCLUDE_AQUA_PLANET_H_ */
