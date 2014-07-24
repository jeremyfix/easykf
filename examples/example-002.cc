/* example-002.cc
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



/* In this example, we learn the extended XOR with a 2-12-1 MLP */
/* Therefore, we have 12*(2+1) + (12+1) = 48 parameters to find */

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
#include <algorithm>

#include <ukf.h>

#define VERBOSE true
#define HIDDEN_LAYER_SIZE 12
#define VARIANCE 0.3 // Slightly overlapping

using namespace ukf::parameter;

/*****************************************************/
/*            Definition of the MLP                  */
/*****************************************************/

double transfer(double x)
{
    return 2.0 / (1.0 + exp(-x))-1.0;
    //return tanh(x);
}

double my_func(gsl_vector * param, gsl_vector * input)
{
    double y[HIDDEN_LAYER_SIZE];
    double z=0.0;
    int i_param = 0;

    for(int i = 0 ; i < HIDDEN_LAYER_SIZE ; i++)
        y[i] = 0.0;

    for(int i = 0 ; i < HIDDEN_LAYER_SIZE ; i++)
    {
        for(int j = 0 ; j < 2 ; j++)
        {
            y[i] += gsl_vector_get(param, i_param) * gsl_vector_get(input, j);
            i_param++;
        }
        y[i] += gsl_vector_get(param, i_param);
        i_param++;
        y[i] = transfer(y[i]);
    }

    for(int i = 0 ; i < HIDDEN_LAYER_SIZE ; i++)
    {
        z += gsl_vector_get(param,i_param) * y[i];
        i_param++;
    }
    z += gsl_vector_get(param, i_param);
    i_param++;

    z = transfer(z);

    return z;
}

/*****************************************************/
/*   Structure for the samples                       */
/*   And tool function generating samples according  */
/*   to a gaussian distribution                      */
/*****************************************************/
typedef struct {
    double x;
    double y;
    double z;
} Sample;

void gaussianDistribution(double theta, double mu_x, double std_x, double mu_y, double std_y, double z, int nb_samples, std::vector< Sample > & samples)
{
    double ux,uy,tx,ty;
    for(int i = 0 ; i < nb_samples ; i++)
    {
        ux = rand() / double(RAND_MAX);
        uy = rand() / double(RAND_MAX);
        tx = sqrt(-2.0 * log(ux)) * cos(2.0 * M_PI * uy);
        ty = sqrt(-2.0 * log(ux)) * sin(2.0 * M_PI * uy);
        Sample s;
        s.x = mu_x + (std_x * tx * cos(theta) - std_y * ty * sin(theta));
        s.y = mu_y + (std_x * tx * sin(theta) + std_y * ty * cos(theta));
        s.z = z;
        samples.push_back(s);
    }
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
   p.n = HIDDEN_LAYER_SIZE*3 + HIDDEN_LAYER_SIZE + 1;
   p.kpa = 3.0 - p.n;
   p.alpha = 1e-2;
   p.beta = 2.0;

   //EvolutionNoise * evolution_noise = new EvolutionRobbinsMonro(1e-4, 1e-4);
   EvolutionNoise * evolution_noise = new EvolutionRLS(1e-4,0.99995);
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

   /***********************************************/
   /*****          Training data set          *****/
   /***********************************************/
   std::vector< Sample > samples;
   for(int i = 0 ; i < 3 ; i ++)
       for(int j = 0 ; j < 3 ; j ++)
            gaussianDistribution(0.0,2*i+1,VARIANCE, 2*j+1,VARIANCE,pow(-1.0,i+j),100, samples);

   // Save the samples
   std::cout << "Saving the input samples in example-002-samples.data" << std::endl;
   std::ofstream outfile("example-002-samples.data");
   for(unsigned int j = 0 ; j < samples.size() ; j ++)
   {
       outfile << samples[j].x << "\t" << samples[j].y << "\t" << samples[j].z << std::endl;
   }
   outfile.close();

   /***********************************************/
   /***** Computing the error before learning *****/
   /***********************************************/

   double error = 0.0;
   std::cout << "###########\n Before learning : " << std::endl;
   Sample sample;
   int nbExamplesWellClassified = 0;
   for(unsigned int j = 0 ; j < samples.size() ; j++)
   {
       sample = samples[j];

       gsl_vector_set(xi,0,sample.x);
       gsl_vector_set(xi,1,sample.y);

       yi = my_func(s.w, xi);

       error += pow(my_func(s.w,xi) - sample.z,2.0);
       if(fabs(sample.z - yi) <= 0.1)
	 nbExamplesWellClassified++;
   }
   std::cout << " RMS = " << sqrt(error/ samples.size()) << std::endl;
   std::cout << nbExamplesWellClassified << " / " << samples.size() << " samples well classified " << std::endl;
   std::cout << "###########\n " <<  std::endl;
   std::cout << " I have " << p.n << " parameters to find " << std::endl;

   /***********************************************/
   /***** Iterate the learning on the samples *****/
   /***********************************************/

   double errorBound = 0.01;
   int nbStepsLimit = 1000;
   error = 2*errorBound;
   int epoch = 0;
   error = 2 * errorBound;
   while( epoch <= nbStepsLimit && error > errorBound)
   {
       // Iterate on the samples
       std::random_shuffle(samples.begin(), samples.end());
       for(unsigned int j = 0 ; j < samples.size() ; j++)
       {
           sample = samples[j];

           gsl_vector_set(xi,0,sample.x);
           gsl_vector_set(xi,1,sample.y);

           ukf_scalar_iterate(p, s, my_func,xi,sample.z);
       }

       // Compute the RMS on the data set
       error = 0.0;
       for(unsigned int j = 0 ; j < samples.size() ; j++)
       {
           sample = samples[j];

           gsl_vector_set(xi,0,sample.x);
           gsl_vector_set(xi,1,sample.y);

           yi = my_func(s.w, xi);
           //ukf_scalar_evaluate(p, s, my_func,xi,yi);

           error += pow(yi - sample.z,2.0);
       }
       error = sqrt(error / samples.size());
       if(VERBOSE)
           std::cout << "Epoch " << epoch << " error = " << error << std::endl;
       epoch++;
   }

   /***********************************************/
   /**** Test the function on the training set ****/
   /***********************************************/

   std::cout << "###########\n After learning : " << std::endl;
   nbExamplesWellClassified = 0;
   error = 0.0;
   for(unsigned int j = 0 ; j < samples.size() ; j++)
   {
       sample = samples[j];

       gsl_vector_set(xi,0,sample.x);
       gsl_vector_set(xi,1,sample.y);

       //yi = my_func(s.w, xi);
       ukf_scalar_evaluate(p, s, my_func,xi,yi);

       error += pow(my_func(s.w,xi) - sample.z,2.0);
       if(fabs(sample.z - yi) <= 0.1)
         nbExamplesWellClassified++;
   }
   std::cout << " RMS = " << sqrt(error/ samples.size()) << std::endl;
   std::cout << nbExamplesWellClassified << " / " << samples.size() << " samples well classified [" << double(nbExamplesWellClassified) * 100.0 / double(samples.size()) << "% ]" << std::endl;
   std::cout << "########### " << std::endl;

   /***********************************************/
   /**** Generate a PPM of the classification  ****/
   /***********************************************/

   double x_min, x_max, y_min, y_max;
   x_min = 0.0;
   x_max = 6.0;
   y_min = 0.0;
   y_max = 6.0;
   int N = 100;
   int color;
   std::cout << " Generating an output image in example-002.ppm" << std::endl;
   std::ofstream image("example-002.ppm");
   image << "P3" << std::endl << "# example-002.ppm" << std::endl;
   image << N << " " << N << std::endl;
   image << "255" << std::endl;

   for(int i = 0 ; i < N ; i++)
     {
       for(int j = 0 ; j < N ; j++)
	 {
	   gsl_vector_set(xi, 0, x_min + (x_max - x_min)/double(N-1)*j);
	   gsl_vector_set(xi, 1, y_min + (y_max - y_min)/double(N-1)*(N-1-i));

	   // Evaluate the output as the mean of the images of the sigma points
           //ukf_scalar_evaluate(p, s, my_func,xi,yi);
           yi = my_func(s.w, xi);
	   color = int(255.0*(1.0 + yi)/2.0);
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

