#ifndef CONVERTAHANDAPOVRAYTOSTANDARD_H_INCLUDED
#define CONVERTAHANDAPOVRAYTOSTANDARD_H_INCLUDED

//#include "util/SophusUtil.h"
#include "sophus/se3.hpp"

Sophus::SE3f readPose(const char * filename);

#endif // CONVERTAHANDAPOVRAYTOSTANDARD_H_INCLUDED
