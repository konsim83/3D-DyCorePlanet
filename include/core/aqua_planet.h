#pragma once

// Deal.ii
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

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


// AquaPlanet headers
#include <base/config.h>
#include <model_data/boussinesq_model_data.h>


AQUAPLANET_OPEN_NAMESPACE


/*!
 * Base class to handle mesh for aqua planet. The mesh
 * is a 3D spherical shell. Derived classes will implement different model
 * details.
 */
class AquaPlanet
{
public:
  /*!
   * Constructor of mesh handler for spherical shell.
   */
  AquaPlanet(double inner_radius, double outer_radius);
  ~AquaPlanet();

protected:
  void
  write_mesh_vtu() const;

  void
  refine_global(unsigned int n_refine);

  MPI_Comm mpi_communicator;

  ConditionalOStream pcout;
  TimerOutput        computing_timer;

  parallel::distributed::Triangulation<3> triangulation;

  const Point<3> center;
  const double   inner_radius, outer_radius;

  const SphericalManifold<3> boundary_description;
};

AQUAPLANET_CLOSE_NAMESPACE
