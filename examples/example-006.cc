/* Copyright (c) 2011-2012, Jérémy Fix. All rights reserved. */

/* Redistribution and use in source and binary forms, with or without */
/* modification, are permitted provided that the following conditions are met: */

/* * Redistributions of source code must retain the above copyright notice, */
/* this list of conditions and the following disclaimer. */
/* * Redistributions in binary form must reproduce the above copyright notice, */
/* this list of conditions and the following disclaimer in the documentation */
/* and/or other materials provided with the distribution. */
/* * None of the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission. */

/* THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS "AS IS" AND */
/* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED */
/* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE */
/* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE */
/* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL */
/* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR */
/* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER */
/* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, */
/* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE */
/* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */

/* In this example, we want to find the minimum of the Rosenbrock's function */

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>

#include <easykf.h>

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
