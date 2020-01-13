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
set(AquaPlanet_TARGET_LIB_SRC
	aqua_planet.cc)

print_all_args (
  ${AquaPlanet_TARGET_LIB_SRC}
 )
###############################################################################
###############################################################################


###############################################################################
###############################################################################
#
# Compile and link the sources as SHARED
#
add_library (AquaPlanet_LIBRARY SHARED ${AquaPlanet_TARGET_LIB_SRC})
DEAL_II_SETUP_TARGET(AquaPlanet_LIBRARY)

add_executable(AquaPlanet "main.cxx")
DEAL_II_SETUP_TARGET(AquaPlanet)
TARGET_LINK_LIBRARIES(AquaPlanet AquaPlanet_LIBRARY)

#
# Install into the DESTINATION provided by CMAKE_INSTALL_PREFIX
#
#install (TARGETS ${AquaPlanet_LIBRARY} DESTINATION ${CMAKE_INSTALL_PREFIX})
###############################################################################
###############################################################################
