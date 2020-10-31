#include <model_data/core_model_data.h>

#include <model_data/core_model_data.tpp>

DYCOREPLANET_OPEN_NAMESPACE

template Tensor<1, 2>
CoreModelData::gravity_vector<2>(const Point<2> &p,
                                 const double    gravity_constant);
template Tensor<1, 3>
CoreModelData::gravity_vector<3>(const Point<3> &p,
                                 const double    gravity_constant);

template Tensor<1, 2>
CoreModelData::coriolis_vector(const Point<2> & /*p*/, const double omega);

template Tensor<1, 3>
CoreModelData::coriolis_vector(const Point<3> & /*p*/, const double omega);

DYCOREPLANET_CLOSE_NAMESPACE