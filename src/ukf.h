/* ukf.h
 * 
 * Copyright (C) 2011-2014 Jeremy Fix
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/**
 * \example example-001.cc
 * \example example-002.cc
 * \example example-003.cc
 * \example example-004.cc
 * \example example-005.cc
 * \example example-006.cc
 * \example example-007.cc
 * \example example-009.cc
 */

/*! \mainpage Kalman Filters
 *
 * \section intro_sec Introduction
 *
 * Two algorithms are implemented and all of them taken from the PhD of Van Der Merwe "Sigma-Point Kalman Filters for Probabilistic Inference in Dynamic State-Space Models" :<BR>
 * - UKF for parameter estimation, Algorithm 6, p. 93<BR>
 * - UKF for state or joint estimation, additive noise case, Algorithm 8,  p. 108 <BR>
 *
 * \subsection usage Usage
 *
 * To use the library, you simply need to :
 * - define a parameter and state structure, which depends on the algorithm you use : ukf_*_param, ukf_*_state
 * - call the proper function initializing the state : ukf_*_init()
 * - iterate with the proper function by providing a sample \f$ (x_i, y_i) \f$  : ukf_*_iterate()
 * - free the memory : ukf_*_free()
 *  To use these functions, you simply need to define you evolution/observation functions, provided to the ukf_*_iterate functions as well as the samples.<BR>
 *
 *  <b>Warning :</b> Be sure to always use gsl_vector_get and gsl_vector_set in your evolution/observation functions, never access the fiels of the vectors with the data array of the gsl_vectors.<BR>
 *
 * \subsection param_estimation UKF for parameter estimation
 *
 * For UKF for parameter estimation, two versions are implemented : in case of a scalar output or vectorial output. The vectorial version works also in the scalar case but is more expensive (memory and time) than the scalar version when the output is scalar.<BR>
 * For the <b>scalar</b> version :
 * - the structures to define are ukf::parameter::ukf_param and ukf::parameter::ukf_scalar_state
 * - the methods to initialize, iterate, evaluate and free are : ukf::parameter::ukf_scalar_init , ukf::parameter::ukf_scalar_iterate , ukf::parameter::ukf_scalar_evaluate , ukf::parameter::ukf_scalar_free<BR>
 *
 * Examples using the scalar version : <BR>
 *  - Training a simple MPL on the XOR problem : example-001.cc
 *  - Training a MLP on the extended XOR : example-002.cc
 *  - Training a RBF for fitting a sinc function : example-003.cc
 *  - Finding the minimum of the Rosenbrock banana function : example-006.cc
 *
 * For the <b>vectorial</b> version :
 * - the structures to define are ukf::parameter::ukf_param and ukf::parameter::ukf_state
 * - the methods to initialize, iterate, evaluate and free are : ukf::parameter::ukf_init , ukf::parameter::ukf_iterate , ukf::parameter::ukf_evaluate , ukf::parameter::ukf_free<BR>
 *
 * Examples using the vectorial version : <BR>
 *  - Training a 2-2-3 MLP to learn the OR, AND, XOR functions : example-004.cc
 *  - Training a 2-12-2 MLP to learn the Mackay Robot arm data : example-005.cc
 *
 * \subsection joint_ukf Joint UKF
 *
 * The Joint UKF tries to estimate both the state and the parameters of a system. The structures/methods related to Joint UKF are :
 * - ukf::state::ukf_param and ukf::state::ukf_state for the parameters and the state representations
 * - ukf::state::ukf_init, ukf::state::ukf_free, ukf::state::ukf_iterate, ukf::state::ukf_evaluate for respectively initializing the structures, freeing the memory, iterating on one sample and evaluating the observation from the sigma points.<BR>
 * To see how to use Joint UKF, have a look to the example example-007.cc where we seek the parameters and state of a Lorentz attractor.
 *
 * \section install_sec Installation and running
 *
 * \subsection tools_subsec Requirements:
 * In addition to a g++ compiler with the standard libraries, you also need to install :
 *          - GSL (Gnu Scientific Library), available here : <a href="http://www.gnu.org/software/gsl/">http://www.gnu.org/software/gsl/</a>
 *          - cmake for compilation
 *
 * \subsection compilation Compilation, Installation
 * The installation follows the standard, for example on Linux :
 * mkdir build
 * cd build
 * cmake .. -G"Unix Makefiles" -DCMAKE_INSTALL_PREFIX=<the prefix where you want the files to be installed>
 * make
 * make install
 *
 * It will compile the library, the examples, the documentation and install them.
 *
 * \section example Example outputs
 *
 * \subsection example1 Example 1 : Learning XOR with a 2-2-1 MLP
 *
 * Running (maybe several times if falling on a local minima) example-001-xor, you should get the following classification :
 *
 * \image html "example-001.png" "XOR classification"
 *
 * An example set of learned parameters is :
 *
 * x[0] -- (9.89157) --> y[0] <BR>
 * x[1] -- (4.18644) --> y[0]<BR>
 * Bias y[0] : 8.22042<BR>
 *
 * x[0] -- (10.7715) --> y[1]<BR>
 * x[1] -- (4.18047) --> y[1]<BR>
 * Bias y[1] : -8.70185<BR>
 *
 * y[0] -- (6.9837) --> z<BR>
 * y[1] -- (-6.83324) --> z<BR>
 * Bias z : -3.89682<BR>
 *  
 * The transfer function is a sigmoid : \f$ f(x) = \frac{2}{1 + exp(-x)}-1\f$
 *
 * \subsection example2 Example 2 : Learning the extended XOR with a 2-12-1 MLP and a parametrized transfer function
 *
 * Here we use a 2-12-1 MLP, with a sigmoidal transfer function, to learn the extended XOR problem. The transfer function has the shape : \f$f(x) = \frac{1}{1.0 + exp(-x)}\f$
 *
 * The classification should look like this :
 *
 * \image html "example-002.png" "Extended XOR classification"
 *
 * \subsection example3 Example 3 : Approximating the sinc function with a Radial Basis Function network
 *
 * In this example, we use a RBF network with 10 kernels to approximate the sinc function on [-5.0,5.0]
 * To make the life easier for the algorithm, we evenly spread the centers of the gaussians on [-5.0, 5.0].
 *
 * The results are saved in 'example-003.data', the first column contains the x-position, the second column the result given by the trained RBF and the last column the value of sinc(x)
 *
 * \image html "example-003.png" "RBF learning the sinc function"
 *
 * \subsection example4 Example 4 : Using a 2-2-3 MLP to learn three boolean functions : XOR, AND, OR
 *
 * \subsection example5 Example 5 : Using a 2-12-2 MLP to learn the Mackay-robot arm problem
 *
 * In this example, we learn the two outputs (x,y) from the inputs (theta, phi) of the Mackay-robot arm dataset. For this
 * we train a 2-12-2 MLP with a parametrized sigmoidal transfer function.
 *
 * \image html "example-005-x.png" "Learning the x-component" \image html "example-005-y.png" "Learning the y-component"
 *
 * \subsection example6 Example 6 : Finding the minimum of the Rosenbrock banana function
 *
 * We use here UKF for parameter estimation to find the minimum of the Rosenbrock banana function : \f$ f(x,y) = (1 - x)^2 + 100 ( y - x^2)^2 \f$<BR>
 *
 * \image html "example-006.png" "Minimisation of the Rosenbrock banana function"
 *
 * \subsection example7 Example 7 : Finding the parameters of a Lorentz attractor
 *
 * In this example, we try to find the parameters (initial condition, evolution parameters) of a noisy lorentz attractor. The dynamic of the lorentz attractor is defined by the three equations :
 *
 * \f$ \frac{dx}{dt} = \sigma ( y - x ) \f$ <BR>
 * \f$ \frac{dy}{dt} = x (\rho - z) - y \f$ <BR>
 * \f$ \frac{dz}{dt} = xy  - \beta z \f$ <BR>
 * While observing a noisy trajectory of such a Lorentz attractor, the algorithm tries to find the current state
 * and the evolution parameters \f$ (\sigma, \rho, \beta)\f$. The samples we provide are \f$ (t_i, {x(t_i), y(t_i), z(t_i)})\f$.
 *
 * To clearly see how UKF catches the true state, we initialized the estimated state of UKF to -15, -15 , -15<BR>
 *
 *  \image html "example-007-rms.png" "Learning RMS" \image html "example-007.png" "Estimated state with the true state and its noisy observation"
 *
 * <BR><BR>
 *
 */


#ifndef UKF_H
#define UKF_H

#include "ukf_math.h"
#include "ukf_types.h"
#include "ukf_samples.h"

#include "ukf_parameter_scalar.h"
#include "ukf_parameter_ndim.h"

#include "ukf_state_ndim.h"
#include "ukf_sr_state_ndim.h"

#include "ekf.h"
#include "ekf_types.h"

/**
  * @brief In this section we implement the Unscented Kalman Filter for parameter estimation and Joint UKF
  * involving the Scaled Unscented Transform detailed in Van der Merwe PhD Thesis
  *
*/
namespace ukf
{

}




#endif  // UKF_H
