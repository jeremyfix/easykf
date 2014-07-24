/* example-006.cc
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
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
*/


/* In this example, we want to find the minimum of the Rosenbrock's function */

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>

#include "ukf.h"

using namespace ukf::parameter;

#define VERBOSE true

/*****************************************************/
/*            Definition of the MLP                  */
/*****************************************************/

double my_func(gsl_vector * param, gsl_vector * input)
{
    double x1 = gsl_vector_get(param, 0);
    double x2 = gsl_vector_get(param, 1);

    // Rosenbrock's function
    return gsl_pow_2(1.0 - x1) + 100 * gsl_pow_2(x2 - gsl_pow_2(x1));

    // Himmelbau's function
    //return gsl_pow_2(gsl_pow_2(x1) + x2 - 11) + gsl_pow_2(x1 + gsl_pow_2(x2) - 7);

}

/*****************************************************/
/*                    main                           */
/*****************************************************/

int main(int argc, char* argv[]) {

   srand(time(NULL));

   // Definition of the parameters and state variables
   ukf_param p;
   ukf_scalar_state s;

   // Initialization of the parameters
   p.n = 2;
   p.kpa = 3.0-p.n;
   p.alpha = 1e-3;
   p.beta = 2.0;

   //EvolutionNoise * evolution_noise = new EvolutionRLS(1e-3,0.9995);
   EvolutionNoise * evolution_noise = new EvolutionRobbinsMonro(1e-4,1e-4);
   p.evolution_noise = evolution_noise;
   //p.evolution_noise = 1e-3;
   //p.evolution_noise_type = ukf::UKF_EVOLUTION_FIXED;

   p.observation_noise = 1e-1;
   p.prior_pi = 1e-1;

   // Initialization of the state and parameters
   ukf_scalar_init(p,s);

   // Initialize the parameter vector to some random values
   for(int i = 0 ; i < p.n ; i++)
      gsl_vector_set(s.w,i,10.0*(2.0*rand()/double(RAND_MAX-1)-1.0));

   // Allocate the input/output vectors
   gsl_vector * xi = gsl_vector_alloc(1);
   gsl_vector_set(xi, 0, 0.0);
   double yi=0.0;

   // Define some limit conditions for the learning
   double errorBound = 1e-5;
   int nbStepsLimit = 100000;
   double error = 2*errorBound;;

   /***********************************************/
   /***** Computing the error before learning *****/
   /***********************************************/

   error = 0.0;
   std::cout << "###########\n Before learning : " << std::endl;
   ukf_scalar_evaluate(p, s, my_func, xi, yi);
   std::cout << "Rosenbrock function value : " << yi << std::endl;
   std::cout << "###########\n " << std::endl;

   /***********************************************/
   /***** Iterate the learning on the samples *****/
   /***********************************************/

   int epoch = 0;
   error = 2 * errorBound;
   std::ofstream outfile("example-006.data");
   while( epoch <= nbStepsLimit && error > errorBound)
   {
       // Iterate on the samples
       ukf_scalar_iterate(p, s, my_func, xi, 0.0);
       ukf_scalar_evaluate(p, s, my_func,xi,yi);
       error = fabs(my_func(s.w, xi));
       if(epoch % (nbStepsLimit/100) == 0)
        std::cout << "Epoch " << epoch << " error = " << error << std::endl;
       outfile << s.w->data[0] << " " << s.w->data[1] << " " << my_func(s.w, xi) << std::endl;
       epoch++;
   }
   outfile.close();
   std::cout << " Run on " << epoch << " epochs ; RMS = " << error << "\n" << std::endl ;

   std::cout << " x = " << gsl_vector_get(s.w, 0) << " ; y = " << gsl_vector_get(s.w, 1) << " ; z = " << my_func(s.w, xi) << std::endl;
   std::cout << " You can plot the results by calling : python plot-example-006.py"<< std::endl;

   /***********************************************/
   /****            Free the memory            ****/
   /***********************************************/
   ukf_scalar_free(p,s);
}
