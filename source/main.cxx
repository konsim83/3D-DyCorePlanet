// Deal.ii
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/utilities.h>

// C++ STL
#include <fstream>
#include <iostream>

// AquaPlanet
#include <core/boussinesq_model.h>


using namespace dealii;


int
main(int argc, char *argv[])
{
  // Very simple way of input handling.
  if (argc < 2)
    {
      std::cout << "You must provide an input file \"-p <filename>\""
                << std::endl;
      exit(1);
    }

  std::string input_file = "";

  std::list<std::string> args;
  for (int i = 1; i < argc; ++i)
    {
      args.push_back(argv[i]);
    }

  while (args.size())
    {
      if (args.front() == std::string("-p"))
        {
          if (args.size() == 1) /* This is not robust. */
            {
              std::cerr << "Error: flag '-p' must be followed by the "
                        << "name of a parameter file." << std::endl;
              exit(1);
            }
          else
            {
              args.pop_front();
              input_file = args.front();
              args.pop_front();
            }
        }
      else
        {
          std::cerr << "Unknown command line option: " << args.front()
                    << std::endl;
          exit(1);
        }
    } // end while

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, dealii::numbers::invalid_unsigned_int);
  //		  argc, argv, 1);

  try
    {
      dealii::deallog.depth_console(1);

      DyCorePlanet::CoreModelData::Parameters parameters_boussinesq(input_file);

      if (parameters_boussinesq.space_dimension == 2)
        {
          DyCorePlanet::Standard::BoussinesqModel<2> aqua_planet_boussinesq(
            parameters_boussinesq);
          aqua_planet_boussinesq.run();
        }
      else if (parameters_boussinesq.space_dimension == 3)
        {
          DyCorePlanet::Standard::BoussinesqModel<3> aqua_planet_boussinesq(
            parameters_boussinesq);
          aqua_planet_boussinesq.run();
        }
      else
        {
          std::cerr << "Invalid space dimension." << std::endl;
          exit(1);
        }
    }

  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
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
      std::cerr << std::endl
                << std::endl
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
