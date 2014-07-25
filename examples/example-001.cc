/* example-001.cc
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

/* In this example, we learn the XOR with a 2-2-1 MLP */

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>

#include <ukf.h>

#define VERBOSE true

using namespace ukf::parameter;
#define NB_INPUTS 2
#define NB_HIDDEN 2
#define NB_OUTPUTS 1
// We have : (NB_INPUTS + 1)*NB_HIDDEN + (NB_HIDDEN+1) * NB_OUTPUTS parameters

/*****************************************************/
/*            Definition of the MLP                  */
/*****************************************************/

double transfer(double x)
{
    //return 2.0 / (1.0 + exp(-x))-1.0;
    //return tanh(x);
  return 1.0 / (1.0 + exp(-x));
}

double my_func(gsl_vector * param, gsl_vector * input)
{
    double y0,y1,z;
    y0 = 0.0;
    y1 = 0.0;
    int i_param = 0;
    for(int i = 0 ; i < NB_INPUTS ; i++)
      y0 += gsl_vector_get(param,i_param++) * gsl_vector_get(input, i);
    y0 += gsl_vector_get(param,i_param++);
    y0 = transfer(y0);

    for(int i = 0 ; i < NB_INPUTS ; i++)
      y1 += gsl_vector_get(param, i_param++) * gsl_vector_get(input, i);
    y1 += gsl_vector_get(param, i_param++);
    y1 = transfer(y1);

    z = gsl_vector_get(param,i_param++) * y0 + gsl_vector_get(param,i_param++)*y1 + gsl_vector_get(param,i_param++);
    z = transfer(z);

    return z;
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
   p.n = 9;
   p.kpa = 3.0 - p.n;
   p.alpha = 1e-1;
   p.beta = 2.0;

   //EvolutionNoise * evolution_noise = new EvolutionRLS(1e-4, 0.9995);
   EvolutionNoise * evolution_noise = new EvolutionRobbinsMonro(1e-2, 1e-5);
   p.evolution_noise = evolution_noise;

   p.observation_noise = 1e-1;
   p.prior_pi = 1e-1;

   // Initialization of the state and parameters
   ukf_scalar_init(p,s);

   // Initialize the parameter vector to some random values
   for(int i = 0 ; i < p.n ; i++)
      gsl_vector_set(s.w,i,0.01*(2.0*rand()/double(RAND_MAX-1)-1.0));


   // Allocate the input/output vectors
   gsl_vector * xi = gsl_vector_alloc(2);
   double yi=0.0;

   // Define the training dataset
   /*double x[4] = {-1.0,1.0,-1.0,1.0};
   double y[4] = {-1.0,-1.0,1.0,1.0};
   double z[4] = {-1.0,1.0,1.0,-1.0};*/
   double x[4] = {0.0,1.0,0.0,1.0};
   double y[4] = {0.0,0.0,1.0,1.0};
   double z[4] = {0.0,1.0,1.0,0.0};

   // Define some limit conditions for the learning
   double errorBound = 1e-4;
   int nbStepsLimit = 10000;
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
       ukf_scalar_evaluate(p, s, my_func,xi,yi);
       std::cout << x[j] << " ; " << y[j] << " -> " << yi << " mean = " << yi <<  " desired : " << z[j] << std::endl;

       error += pow(yi - z[j],2.0);
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
           ukf_scalar_iterate(p, s, my_func,xi,z[j]);
       }

       // Evaluate the error on the data set
       error = 0.0;
       for(int j = 0 ; j < 4 ; j++)
       {
           gsl_vector_set(xi,0,x[j]);
           gsl_vector_set(xi,1,y[j]);

           // Evaluate the function using the sigma points
           //ukf_scalar_evaluate(p, s, my_func,xi,yi);

           // Or the function itself, which is less expensive
           yi = my_func(s.w,xi);

           error += pow(yi - z[j],2.0);
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
       //ukf_scalar_evaluate(p, s, my_func,xi,yi);
       yi = my_func(s.w,xi);
       std::cout << x[j] << " ; " << y[j] << " -> " << yi << " desired : " << z[j] << std::endl;
   }

   /***********************************************/
   /****     Display the learned parameters    ****/
   /***********************************************/

   std::cout << std::endl;
   std::cout << "Parameters : " << std::endl;


   int param_index = 0;
   for(int i = 0 ; i < NB_HIDDEN ; ++i) {
     for(int j = 0 ; j < NB_INPUTS ; ++j)
       std::cout << "x[" << j << "] -- " << gsl_vector_get(s.w,param_index++) << " --> y[" << i << "]" << std::endl;
     std::cout << "Bias y[" << i << "]" << gsl_vector_get(s.w,param_index++) << std::endl;
   }
   for(int i = 0 ; i < NB_OUTPUTS ; ++i) {
     for(int j = 0 ; j < NB_HIDDEN ; ++j)
       std::cout << "y[" << j << "] -- " << gsl_vector_get(s.w,param_index++) << " --> z[" << i << "]" << std::endl;
     std::cout << "Bias z[" << i << "]" << gsl_vector_get(s.w,param_index++) << std::endl;
   }

   std::cout << "########### " << std::endl;


   /***********************************************/
   /**** Generate a PPM of the classification  ****/
   /***********************************************/

   std::cout << "Generating output image in example-001.ppm" << std::endl;
   double x_min, x_max, y_min, y_max;
   x_min = 0.0;
   x_max = 1.0;
   y_min = 0.0;
   y_max = 1.0;
   int N = 100;
   int color;
   std::ofstream image("example-001.ppm");
   image << "P3" << std::endl << "# example-001.ppm" << std::endl;
   image << N << " " << N << std::endl;
   image << "255" << std::endl;

   for(int i = 0 ; i < N ; i++)
     {
       for(int j = 0 ; j < N ; j++)
         {
           gsl_vector_set(xi, 0, x_min + j*(x_max - x_min)/double(N-1));
           gsl_vector_set(xi, 1, y_min + (N-1-i)*(y_max - y_min)/double(N-1));

           // Evaluate the output as the mean of the images of the sigma points
           //ukf_scalar_evaluate(p, s, my_func,xi,yi);
           yi = my_func(s.w, xi);
           color = int(255.0 * yi);
           image << color << " " << color << " " << color << " " ;
         }
       image << std::endl;
     }
   image.close();

   /***********************************************/
   /****            Free the memory            ****/
   /***********************************************/
   ukf_scalar_free(p,s);
}
