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

/* In this example, we use a RBF network to learn the sinc function on [-5;5] with 10 kernels. */
/* The results are saved in example-003.data, a 3 column datafile.                             */
/* The first column contains the x positions                                                   */
/* The second column contains the approximated values                                          */
/* The third column contains the desired output                                                */
/* There is 30 parameters to find                                                              */

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <time.h>

#include <easykf.h>

using namespace ukf::parameter;

#define VERBOSE true
#define NB_KERNEL 10
#define NOISE_AMPLITUDE 0.1

// Approximate the sinc(x) function between x_min and x_max
#define X_MIN -5.0
#define X_MAX 5.0

/*****************************************************/
/*            Definition of the RBF network          */
/*****************************************************/

double my_func(gsl_vector * param, gsl_vector * input)
{
    double z = 0.0;
    double A,s,dist;

    // Each kernel is parametrized by 3 parameters : Amplitude, Mean, Variance
    for(int i = 0 ; i < NB_KERNEL ; i++)
    {
        A = gsl_vector_get(param, 3 * i);

        dist = gsl_pow_2(gsl_vector_get(input, 0) - gsl_vector_get(param, 3 * i + 1));

        s = gsl_vector_get(param, 3 * i + 2);
        if(ukf::math::cmp_diff(s,0.0))
            z += A * exp(-dist/(2.0 * s * s));
        else if(ukf::math::cmp_equal(dist, 0.0))
            z+= 1.0;
	else
            z += 0.0;
    }
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

    p.n = 3*NB_KERNEL;
    p.kpa = 3.0 - p.n;
    p.alpha = 1.0;//1e-2;
    p.beta = 2.0;

    //EvolutionNoise * evolution_noise = new EvolutionAnneal(1e-3, 0.995, 1e-8);
    EvolutionNoise * evolution_noise = new EvolutionRLS(1e-4,0.99995);
    //EvolutionNoise * evolution_noise = new EvolutionRobbinsMonro(1e-4,1e-4);
    p.evolution_noise = evolution_noise;

    p.observation_noise = 1e-2 ; // Observation noise
    p.prior_pi = 1e-1; // Initializition of the covariance of the parameters

    // Initialization of the state and parameters
    ukf_scalar_init(p,s);

    // Initialize the parameter vector to some random values
    for(int i = 0 ; i < p.n ; i++)
        gsl_vector_set(s.w,i,(2.0 * rand()/double(RAND_MAX-1) - 1.0));
    for(int i = 1 ; i < p.n ; i+=3)
        gsl_vector_set(s.w,i,5.0*(2.0 * rand()/double(RAND_MAX-1) - 1.0));

    // Evenly spread the centers in x_min x_max
    // This works actually better than without spreading uniformely the centers
    for(int i = 0 ; i < NB_KERNEL ; i ++)
        gsl_vector_set(s.w, 3 * i     + 1, X_MIN + i * (X_MAX - X_MIN)/double(NB_KERNEL-1));

    // Allocate the input/output vectors
    gsl_vector * xi = gsl_vector_alloc(1);
    double yi=0.0;

    // Define some limit conditions for the learning
    double errorBound = 1e-2;
    int nbStepsLimit = 10000;
    int nbExamplesPerEpoch = 100;
    double error = 2*errorBound;

    /***********************************************/
    /***** Iterate the learning on the samples *****/
    /***********************************************/

    int epoch = 0;
    error = 2 * errorBound;
    double x,y;
    while( epoch <= nbStepsLimit && error > errorBound)
    {
        // Iterate on the samples
        for(int j = 0 ; j < nbExamplesPerEpoch ; j++)
        {
            x = X_MIN + (X_MAX - X_MIN) * rand()/double(RAND_MAX-1);
            y = (ukf::math::cmp_equal(x, 0) ? 1.0 : sin(M_PI * x)/(M_PI * x) );

            gsl_vector_set(xi,0,x);
            ukf_scalar_iterate(p, s, my_func,xi,y + NOISE_AMPLITUDE * (2.0*rand()/double(RAND_MAX) - 1.0));
        }

        // Evaluate the RMS error
        error = 0.0;
        for(int j = 0 ; j < nbExamplesPerEpoch ; j++)
        {
            x = X_MIN + j * (X_MAX - X_MIN) / double(nbExamplesPerEpoch - 1);
            y = (ukf::math::cmp_equal(x, 0) ? 1.0 : sin(M_PI * x)/(M_PI * x) );

            gsl_vector_set(xi,0,x);

            // We evaluate the output with the sigma points
            //ukf_scalar_evaluate(p, s, my_func,xi,yi);

            // Or directly with the function
            yi = my_func(s.w,xi);

            error += pow(yi - y,2.0);
        }

        error = sqrt(error /  double(nbExamplesPerEpoch));
        if(VERBOSE)
            std::cout << "Epoch " << epoch << " error = " << error << std::endl;
        epoch++;
    }

    /***********************************************/
    /**** Test the function on the training set ****/
    /***********************************************/

    std::cout << " Saving the output in example-003.data " << std::endl;
    //std::cout << " You can plot them using e.g. gnuplot : " << std::endl;
    //std::cout << " gnuplot Data/plot-example-003.gplot ; gv Output/example-003.ps " << std::endl;
    std::ofstream output("example-003.data");
    for(int i = 0 ; i < nbExamplesPerEpoch ; i++)
    {
        x = X_MIN + i * (X_MAX - X_MIN) / double(nbExamplesPerEpoch - 1);
        y = (ukf::math::cmp_equal(x, 0) ? 1.0 : sin(M_PI * x)/(M_PI * x) );
        gsl_vector_set(xi,0,x);

        // Evaluate the function with the function itself
        yi = my_func(s.w,xi);
        // Or using the mean of the sigma points images
        //ukf_scalar_evaluate(p, s, my_func,xi,yi);

        output << x << " " << y << " " << yi << " " << y+NOISE_AMPLITUDE*(2.0 * rand()/double(RAND_MAX)-1.0) << std::endl;
    }
    output.close();

    /***********************************************/
    /****     Display the learned parameters    ****/
    /***********************************************/

    std::cout << "Parameters " << std::endl;
    for(int i = 0 ; i < NB_KERNEL ; i++)
    {
	std::cout << "Kernel #" << i 
                << " Mean = " << gsl_vector_get(s.w, 3 * i + 1)
                << " Amplitude = " << gsl_vector_get(s.w, 3 * i)
                << " Variance = " << fabs(gsl_vector_get(s.w, 3 * i + 2))
                << std::endl;
    }

    /***********************************************/
    /****            Free the memory            ****/
    /***********************************************/
    ukf_scalar_free(p,s);
}
