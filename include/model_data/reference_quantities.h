#pragma once

// Deal.ii
#include <deal.II/base/parameter_handler.h>

// AquaPlanet
#include <base/config.h>

DYCOREPLANET_OPEN_NAMESPACE

namespace CoreModelData
{
  /*!
   * @struct ReferenceQuantities
   *
   * Struct contains reference quantities for non-dimensionalization
   */
  struct ReferenceQuantities
  {
    ReferenceQuantities(const std::string &parameter_filename);

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);

    /*!
     * Earth reference pressure.
     */
    double pressure; /* Pa */

    /*!
     * Earth angular velocity.
     */
    double omega; /* 1/s */

    /*!
     * Reference density of air at bottom reference
     * temperature.
     */
    double density; /* kg / m^3 */

    /*!
     * Reference time is one hour.
     */
    double time; /* s */

    /*!
     * Reference velocity.
     */
    double velocity; /* m/s */

    /*!
     * Reference length.
     */
    double length; /* m */

    /*!
     * Reference temperature 273.15 K (0 degree Celsius) at bottom.
     */
    double temperature_bottom; /* K */

    /*!
     * Reference temperature at top of atmosphere.
     */
    double temperature_top; /* K */

    /*!
     * Reference temperature change.
     */
    double temperature_change; /* K */

  }; // struct ReferenceQuantities

} // namespace CoreModelData

DYCOREPLANET_CLOSE_NAMESPACE
