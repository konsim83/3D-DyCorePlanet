#include <base/utilities.h>

#include <base/utilities.tpp>

DYCOREPLANET_OPEN_NAMESPACE

namespace Tools
{
  void
  create_data_directory(std::string dir_name)
  {
    const int dir_err =
      mkdir(dir_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (-1 == dir_err)
      {
        throw std::runtime_error(
          "Error creating directory! It might already exist or you do not have write permissions in this folder.");
      }
  }


  void
  get_face_sign_change_raviart_thomas(
    const DoFHandler<3>::active_cell_iterator &cell,
    const FiniteElement<3> &                   fe,
    std::vector<double> &                      face_sign)
  {
    for (unsigned int face_index = 0;
         face_index < GeometryInfo<3>::faces_per_cell;
         ++face_index)
      {
        Triangulation<3>::face_iterator face = cell->face(face_index);

        if (!face->at_boundary())
          {
            for (unsigned int face_dof_index = 0;
                 face_dof_index < fe.n_dofs_per_face();
                 ++face_dof_index)
              {
                const unsigned int cell_dof_index =
                  fe.face_to_cell_index(face_dof_index, face_index);

                face_sign[cell_dof_index] =
                  (cell->face_orientation(face_index) ? 1.0 : -1.0);
              } // ++face_dof_index
          }
      } // ++face_index
  }

  /*
   * Template instantiations
   */
  template double
  compute_pressure_mean_value<2>(
    const DoFHandler<2> &       dof,
    const Quadrature<2> &       quadrature,
    const LA::MPI::BlockVector &distributed_pressure_vector);

  template double
  compute_pressure_mean_value<3>(
    const DoFHandler<3> &       dof,
    const Quadrature<3> &       quadrature,
    const LA::MPI::BlockVector &distributed_pressure_vector);

} // namespace Tools

DYCOREPLANET_CLOSE_NAMESPACE
