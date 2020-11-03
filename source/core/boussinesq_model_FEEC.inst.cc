#include <core/boussineq_model_FEEC.h>

#include <core/boussineq_model_FEEC.tpp>

DYCOREPLANET_OPEN_NAMESPACE

/*
 * Template instantiations. Do not instantiate for dim=2 since the
 * meaning of curls is different there. This needs template specialization that
 * is not implemented yet..
 */
// template class ExteriorCalculus::BoussinesqModel<2>;
template class ExteriorCalculus::BoussinesqModel<3>;

DYCOREPLANET_CLOSE_NAMESPACE
