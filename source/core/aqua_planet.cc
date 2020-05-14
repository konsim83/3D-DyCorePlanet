#include <core/aqua_planet.h>

AQUAPLANET_OPEN_NAMESPACE

AquaPlanet::AquaPlanet(double inner_radius, double outer_radius)
  : mpi_communicator(MPI_COMM_WORLD)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
  , computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
  , triangulation(mpi_communicator,
                  typename Triangulation<3>::MeshSmoothing(
                    Triangulation<3>::smoothing_on_refinement |
                    Triangulation<3>::smoothing_on_coarsening))
  , center(0, 0, 0)
  , inner_radius(inner_radius)
  , outer_radius(outer_radius)
  , boundary_description(center)
{
  GridGenerator::hyper_shell(triangulation, center, inner_radius, outer_radius);

  triangulation.set_all_manifold_ids_on_boundary(0);
  triangulation.set_manifold(0, boundary_description);

  typename Triangulation<3>::active_cell_iterator cell = triangulation
                                                           .begin_active(),
                                                  endc = triangulation.end();
  for (; cell != endc; ++cell)
    cell->set_all_manifold_ids(0);

  pcout << "   Number of active cells:       " << triangulation.n_active_cells()
        << std::endl;
}



AquaPlanet::~AquaPlanet()
{}



void
AquaPlanet::refine_global(unsigned int n_refine)
{
  triangulation.refine_global(n_refine);

  pcout << "   Number of active cells after global refinement:       "
        << triangulation.n_active_cells() << std::endl;
}



void
AquaPlanet::write_mesh_vtu() const
{
  DataOut<3> data_out;
  data_out.attach_triangulation(triangulation);

  // Add data to indicate subdomain
  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    {
      subdomain(i) = triangulation.locally_owned_subdomain();
    }
  data_out.add_data_vector(subdomain, "subdomain");

  // Now build all data patches
  data_out.build_patches();

  const std::string filename_local =
    "boussinesq_palnet_mesh." +
    Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
    ".vtu";

  std::ofstream output(filename_local.c_str());
  data_out.write_vtu(output);

  // Write a pvtu record on master process
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i = 0;
           i < Utilities::MPI::n_mpi_processes(mpi_communicator);
           ++i)
        filenames.emplace_back("boussinesq_palnet_mesh." +
                               Utilities::int_to_string(i, 4) + ".vtu");

      std::string   master_file("boussinesq_palnet_mesh.pvtu");
      std::ofstream master_output(master_file.c_str());
      data_out.write_pvtu_record(master_output, filenames);
    }
}


AQUAPLANET_CLOSE_NAMESPACE
