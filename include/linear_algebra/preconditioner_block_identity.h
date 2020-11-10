#pragma once

// AquaPlanet
#include <base/config.h>

DYCOREPLANET_OPEN_NAMESPACE

namespace LinearAlgebra
{
  class PreconditionerBlockIdentity
  {
  public:
    PreconditionerBlockIdentity() = default;

    void
    vmult(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const
    {
      dst = src;
    }
  };

} // namespace LinearAlgebra


DYCOREPLANET_CLOSE_NAMESPACE
