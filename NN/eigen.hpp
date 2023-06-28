#ifndef _EIGEN_HPP_
#define _EIGEN_HPP_

#include "global.hpp"

#include "eigen3/Eigen/Dense"

#if (EIGEN_WORLD_VERSION != 3 || \
     EIGEN_MAJOR_VERSION != 2)
#warning Eigen version is not 3.2.x.
#warning This can affect numerical precision and regression tests.
#endif

#endif /* !_EIGEN_HPP_ */

