#include <core/boussineq_model_assembly_FEEC.h>

#include <core/boussineq_model_assembly_FEEC.tpp>

DYCOREPLANET_OPEN_NAMESPACE

/*
 * Template instantiations. Do not instantiate for dim=2 since the
 * meaning of curls is different there. This needs template specialization that
 * is not implemented yet..
 */
template class ExteriorCalculus::Assembly::Scratch::NSEPreconditioner<3>;
template class ExteriorCalculus::Assembly::Scratch::NSESystem<3>;
template class ExteriorCalculus::Assembly::Scratch::TemperatureMatrix<3>;
template class ExteriorCalculus::Assembly::Scratch::TemperatureRHS<3>;

template class ExteriorCalculus::Assembly::CopyData::NSEPreconditioner<3>;
template class ExteriorCalculus::Assembly::CopyData::NSESystem<3>;
template class ExteriorCalculus::Assembly::CopyData::TemperatureMatrix<3>;
template class ExteriorCalculus::Assembly::CopyData::TemperatureRHS<3>;

DYCOREPLANET_CLOSE_NAMESPACE
