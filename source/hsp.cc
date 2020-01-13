#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/logstream.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/cell_id.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/base/timer.h>

// STL
#include <fstream>
#include <iostream>


using namespace dealii;


class AquaPlanet
{
public:
	AquaPlanet ();
	~AquaPlanet ();

	void run ();

private:
	void make_mesh ();
	void output_mesh () const;
	void refine_grid ();

	MPI_Comm mpi_communicator;

	ConditionalOStream 		pcout;
	TimerOutput        		computing_timer;

	parallel::distributed::Triangulation<3> triangulation;

	DoFHandler<3>      dof_handler;

	const Point<3> center;
	const double inner_radius, outer_radius;

	const SphericalManifold<3> boundary_description;
};



AquaPlanet::AquaPlanet ()
:
mpi_communicator(MPI_COMM_WORLD),
pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times),
triangulation(mpi_communicator,
			  typename Triangulation<3>::MeshSmoothing(
				Triangulation<3>::smoothing_on_refinement |
				Triangulation<3>::smoothing_on_coarsening)),
dof_handler (triangulation),
center (0,0,0),
inner_radius(1),
outer_radius(2),
boundary_description(center)
{
}



AquaPlanet::~AquaPlanet ()
{
}



void AquaPlanet::make_mesh ()
{
//	const Point<3> center (0,0,0);
//	const double inner_radius = 0.5,
//				 outer_radius = 1.0;

	GridGenerator::hyper_shell (triangulation,
								center, inner_radius, outer_radius);

//	static const SphericalManifold<3> boundary_description (center);

	triangulation.set_all_manifold_ids_on_boundary(0);
	triangulation.set_manifold (0, boundary_description);

	typename Triangulation<3>::active_cell_iterator
		cell = triangulation.begin_active(),
		endc = triangulation.end();
	for (; cell!=endc; ++cell)
		cell->set_all_manifold_ids (0);

	triangulation.refine_global (4);

	pcout << "   Number of active cells:       "
		<< triangulation.n_active_cells()
		<< std::endl;
}



void AquaPlanet::refine_grid ()
{
}



void AquaPlanet::output_mesh () const
{
//  Assert (cycle < 10, ExcNotImplemented());

	DataOut<3> data_out;

	data_out.attach_dof_handler (dof_handler);
//	data_out.add_data_vector (solution, "solution");
	data_out.build_patches ();

	std::ofstream output ("aqua_palnet_mesh.vtk");
	data_out.write_vtk (output);
}



void AquaPlanet::run ()
{
	make_mesh ();
	output_mesh ();
}



int main (int argc, char* argv[])
{

	dealii::Utilities::MPI::MPI_InitFinalize
		mpi_initialization(argc, argv, dealii::numbers::invalid_unsigned_int);

	try
	{
		AquaPlanet aqua_planet;
		aqua_planet.run ();
	}

	catch (std::exception &exc)
	{
		std::cerr << std::endl << std::endl
				<< "----------------------------------------------------"
				<< std::endl;

		std::cerr << "Exception on processing: " << std::endl
				<< exc.what() << std::endl
				<< "Aborting!" << std::endl
				<< "----------------------------------------------------"
				<< std::endl;

		return 1;
	}

	catch (...)
	{
		std::cerr << std::endl << std::endl
				<< "----------------------------------------------------"
				<< std::endl;

		std::cerr << "Unknown exception!" << std::endl
				<< "Aborting!" << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		return 1;
	}

	return 0;
}
