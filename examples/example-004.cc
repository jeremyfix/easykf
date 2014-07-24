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

/* In this example, we train a 2-2-3 MLP to learn the boolean functions AND, OR and XOR */
/* Using the vectorial implementation                                                   */

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>

#include <easykf.h>

using namespace ukf::parameter;

#define VERBOSE true

#define NB_INPUTS 2
#define NB_HIDDEN 2
#define NB_OUTPUTS 3
// We have : (NB_INPUTS + 1)*NB_HIDDEN + (NB_HIDDEN+1) * NB_OUTPUTS parameters

/*****************************************************/
/*            Definition of the MLP                  */
/*****************************************************/

double transfer(double x)
{
    return 2.0 / (1.0 + exp(-x))-1.0;
}

void my_func(gsl_vector * param, gsl_vector * input, gsl_vector * output)
{
    double y[NB_HIDDEN];
    double z=0.0;

    for(int i = 0 ; i < NB_HIDDEN ; i++)
        y[i] = 0.0;

    int i_param = 0;
    gsl_vector_set_zero(output);

    for(int j = 0 ; j < NB_HIDDEN ; j++)
    {
        for(int k = 0 ; k < NB_INPUTS ; k++)
        {
            y[j] += gsl_vector_get(param, i_param) * gsl_vector_get(input, k);
            i_param++;
        }
        y[j] += gsl_vector_get(param, i_param);
        i_param++;
        y[j] = transfer(y[j]);
    }
    for(int i = 0 ; i < NB_OUTPUTS ; i++)
    {
        z = 0.0;
        for(int j = 0 ; j < NB_HIDDEN ; j++)
        {
            z += gsl_vector_get(param,i_param) * y[j];
            i_param++;
        }
        z += gsl_vector_get(param, i_param);
        i_param++;

        z = transfer(z);
        gsl_vector_set(output, i, z);
    }
}

/*****************************************************/
/*                    main                           */
/*****************************************************/

int main(int argc, char* argv[]) {

   srand(time(NULL));

   // Definition of the parameters and state variables
   ukf_param p;
   ukf_state s;

   // Initialization of the parameters
   p.n = (NB_INPUTS + 1)*NB_HIDDEN + (NB_HIDDEN+1) * NB_OUTPUTS;
   p.no = 3;
   p.kpa = 3.0 - p.n;
   p.alpha = 1e-2;
   p.beta = 2.0;

   //EvolutionNoise * evolution_noise = new EvolutionAnneal(1e-4, 1.0, 1e-7);
   EvolutionNoise * evolution_noise = new EvolutionRobbinsMonro(1e-2, 1e-5);
   p.evolution_noise = evolution_noise;

   p.observation_noise = 1e-1;
   p.prior_pi = 1e-1;

   // Initialization of the state and parameters
   ukf_init(p,s);

   // Initialize the parameter vector to some random values
   for(int i = 0 ; i < p.n ; i++)
      gsl_vector_set(s.w,i,0.1*(2.0*rand()/double(RAND_MAX-1)-1.0));

   // Allocate the input/output vectors
   gsl_vector * xi = gsl_vector_alloc(NB_INPUTS);
   gsl_vector * yi = gsl_vector_alloc(NB_OUTPUTS);
   gsl_vector_set_zero(yi);

   // Define the training dataset
   double x[4] = {-1.0,1.0,-1.0,1.0};
   double y[4] = {-1.0,-1.0,1.0,1.0};
   double z0[4] = {-1.0,1.0,1.0,-1.0};
   double z1[4] = {-1.0, -1.0, -1.0, 1.0};
   double z2[4] = {-1.0, 1.0, 1.0, 1.0};

   // Define some limit conditions for the learning
   double errorBound = 1e-3;
   int nbStepsLimit = 500;
   double error = 2*errorBound;;

   /***********************************************/
   /***** Computing the error before learning *****/
   /***********************************************/

   error = 0.0;
   std::cout << "###########\n Before learning : " << std::endl;
   for(int j = 0 ; j < 4 ; j++)
   {
       gsl_vector_set(xi,0,x[j]);
       gsl_vector_set(xi,1,y[j]);

       my_func(s.w,xi,yi);
       std::cout << x[j] << " ; " << y[j] << " -> " << std::endl;
       std::cout << "               " << gsl_vector_get(yi,0) << " desired : " << z0[j] << std::endl;
       std::cout << "               " << gsl_vector_get(yi,1) << " desired : " << z1[j] << std::endl;
       std::cout << "               " << gsl_vector_get(yi,2) << " desired : " << z2[j] << std::endl;

       error += pow(gsl_vector_get(yi,0) - z0[j],2.0) + pow(gsl_vector_get(yi,1) - z1[j],2.0) + pow(gsl_vector_get(yi,2) - z2[j],2.0);
   }
   error = sqrt(error / 4.0);
   std::cout << "RMS : " << error << std::endl;
   std::cout << "###########\n " << std::endl;

   /***********************************************/
   /***** Iterate the learning on the samples *****/
   /***********************************************/

   int epoch = 0;
   error = 2 * errorBound;
   while( epoch <= nbStepsLimit && error > errorBound)
   {
       // Iterate on the samples
       for(int j = 0 ; j < 4 ; j++)
       {
           gsl_vector_set(xi,0,x[j]);
           gsl_vector_set(xi,1,y[j]);
           gsl_vector_set(yi,0,z0[j]);
           gsl_vector_set(yi,1,z1[j]);
           gsl_vector_set(yi,2,z2[j]);
           ukf_iterate(p, s, my_func,xi,yi);
       }

       // Evaluate the error on the data set
       error = 0.0;
       for(int j = 0 ; j < 4 ; j++)
       {
           gsl_vector_set(xi,0,x[j]);
           gsl_vector_set(xi,1,y[j]);

           // Evaluate the function using the sigma points
           //ukf_evaluate(p, s, my_func,xi,yi);

           // Or the function itself
           my_func(s.w,xi,yi);

           error += pow(gsl_vector_get(yi,0) - z0[j],2.0) + pow(gsl_vector_get(yi,1) - z1[j],2.0) + pow(gsl_vector_get(yi,2) - z2[j],2.0);
       }
       error = sqrt(error / 4.0);
       if(VERBOSE)
           std::cout << "Epoch " << epoch << " error = " << error << std::endl;
       epoch++;
   }
   std::cout << " Run on " << epoch << " epochs ; RMS = " << error << "\n" << std::endl ;

   /***********************************************/
   /**** Test the function on the training set ****/
   /***********************************************/

   std::cout << "###########\n After learning : " << std::endl;
   for(int j = 0 ; j < 4 ; j++)
   {
       gsl_vector_set(xi,0,x[j]);
       gsl_vector_set(xi,1,y[j]);

       //ukf_evaluate(p, s, my_func,xi,yi);
       my_func(s.w,xi,yi);
       std::cout << x[j] << " ; " << y[j] << " -> " << std::endl;
       std::cout << "               " << gsl_vector_get(yi,0) << " desired : " << z0[j] << std::endl;
       std::cout << "               " << gsl_vector_get(yi,1) << " desired : " << z1[j] << std::endl;
       std::cout << "               " << gsl_vector_get(yi,2) << " desired : " << z2[j] << std::endl;

   }

   /***********************************************/
   /****            Free the memory            ****/
   /***********************************************/
   ukf_free(p,s);
}

