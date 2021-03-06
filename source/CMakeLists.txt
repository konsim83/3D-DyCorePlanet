###############################################################################
###############################################################################
#
# Include directory for sources
#
include_directories(${AquaPlanet_INCLUDE_DIR})
###############################################################################
###############################################################################


###############################################################################
###############################################################################
#
# Name all sources
#
#file(GLOB_RECURSE AquaPlanet_TARGET_LIB_SRC  "*.cc") # source files
#set(AquaPlanet_TARGET_LIB_SRC ${AquaPlanet_TARGET_LIB_SRC})
set(AquaPlanet_Boussinesq_TARGET_LIB_SRC
	base/utilities.cc
	model_data/reference_quantities.cc
	model_data/physical_constants.cc
	model_data/boussinesq_model_data.inst.cc
	model_data/boussinesq_model_parameters.cc
	core/dycore_planet.inst.cc
	model_data/core_model_data.cc
	model_data/core_model_data.inst.cc	
	core/boussinesq_model.inst.cc
	core/boussinesq_model_FEEC.inst.cc
	core/boussineq_model_assembly.inst.cc
	core/boussineq_model_assembly_FEEC.inst.cc)

print_all_args (
  ${AquaPlanet_Boussinesq_TARGET_LIB_SRC}
 )
###############################################################################
###############################################################################


###############################################################################
###############################################################################
#
# Compile and link the sources as SHARED
#
add_library (AquaPlanet_Boussinesq_LIBRARY SHARED ${AquaPlanet_Boussinesq_TARGET_LIB_SRC})
DEAL_II_SETUP_TARGET(AquaPlanet_Boussinesq_LIBRARY)

add_executable(BoussinesqPlanet "main.cxx")
DEAL_II_SETUP_TARGET(BoussinesqPlanet)
TARGET_LINK_LIBRARIES(BoussinesqPlanet AquaPlanet_Boussinesq_LIBRARY)

#
# Install into the DESTINATION provided by CMAKE_INSTALL_PREFIX
#
#install (TARGETS ${AquaPlanet_LIBRARY} DESTINATION ${CMAKE_INSTALL_PREFIX})
###############################################################################
###############################################################################
