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

/* In this example, we use joint UKF for state/parameter estimation in order to estimate the state and parameters of a Lorentz attractor */

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>

#include <easykf.h>

using namespace ukf::state;

#define SIGMA 10.0
#define RHO 28.0
#define BETA 8.0/3.0
#define DT 0.01

#define X0 0.1
#define Y0 1.2
#define Z0 0.4

#define NOISE_AMPLITUDE 2.0

#define VERBOSE true

/*************************************************************************************/
/*            Definition of the evolution and observation functions                  */
/*************************************************************************************/

// Evolution function
void evo_function(gsl_vector * params, gsl_vector * xk_1, gsl_vector * xk)
{
    double x = gsl_vector_get(xk_1,0);
    double y = gsl_vector_get(xk_1,1);
    double z = gsl_vector_get(xk_1,2);

    double sigma = gsl_vector_get(xk_1,3);
    double rho = gsl_vector_get(xk_1,4);
    double beta = gsl_vector_get(xk_1,5);

    double dt = gsl_vector_get(params, 0);

    gsl_vector_set(xk, 0, x + dt * (sigma*( y - x)));
    gsl_vector_set(xk, 1, y + dt * ( rho * x - y - x * z));
    gsl_vector_set(xk, 2, z + dt * (x * y - beta * z));
    gsl_vector_set(xk, 3, sigma);
    gsl_vector_set(xk, 4, rho);
    gsl_vector_set(xk, 5, beta);
}

// Observation function
void obs_function(gsl_vector * xk , gsl_vector * yk)
{
    for(int i = 0 ; i < yk->size ; ++i)
        gsl_vector_set(yk, i, gsl_vector_get(xk,i) + NOISE_AMPLITUDE*rand()/ double(RAND_MAX));
}

/*****************************************************/
/*                    main                           */
/*****************************************************/

int main(int argc, char* argv[]) {

    srand(time(NULL));

    // Definition of the parameters and state variables
    ukf_param p;
    ukf_state s;
    // The parameters for the evolution equation
    s.params = gsl_vector_alloc(1);
    s.params->data[0] = DT;

    // Initialization of the parameters
    p.n = 6;
    p.no = 3;
    p.kpa = 0.0;
    p.alpha = 0.9;
    p.beta = 2.0;

    //EvolutionNoise * evolution_noise = new EvolutionAnneal(1e-1, 1.0, 1e-2);
    //EvolutionNoise * evolution_noise = new EvolutionRLS(1e-5, 0.9995);
    EvolutionNoise * evolution_noise = new EvolutionRobbinsMonro(1e-4, 1e-3);
    p.evolution_noise = evolution_noise;

    p.measurement_noise = 1e-1;
    p.prior_x= 1.0;

    // Initialization of the state and parameters
    ukf_init(p,s);

    // Initialize the parameter vector to some random values
    for(int i = 0 ; i < p.n ; i++)
        gsl_vector_set(s.xi,i,5.0*(2.0*rand()/double(RAND_MAX-1)-1.0));
    s.xi->data[0] = -15.0;
    s.xi->data[1] = -15.0;
    s.xi->data[2] = -15.0;

    // Allocate the input/output vectors
    gsl_vector * xi = gsl_vector_alloc(p.n);
    gsl_vector * yi = gsl_vector_alloc(p.no);
    gsl_vector_set_zero(yi);

    /***********************************************/
    /***** Iterate the learning on the samples *****/
    /***********************************************/

    int epoch = 0;

    xi->data[0] = X0;
    xi->data[1] = Y0;
    xi->data[2] = Z0;
    xi->data[3] = SIGMA;
    xi->data[4] = RHO;
    xi->data[5] = BETA;

    std::ofstream outfile("example-007.data");
    std::ofstream outfile_rms("example-007-rms.data");
    double rms= 0.0;
    double errorBound = NOISE_AMPLITUDE/5.0;
    int count_time=0;
    int nb_steps_below_error=100;
    //int is_counting=0;
    while( epoch < 20000)
    {
        // Evaluate the true dynamical system
        evo_function(s.params, xi, xi);
        obs_function(xi, yi);
        // Provide the observation and iterate
        ukf_iterate(p,s,evo_function, obs_function,yi);

        epoch++;

        rms = 0.0;
        for(int i = 0 ; i < p.n ; ++i)
            rms += gsl_pow_2(xi->data[i] - s.xi->data[i]);
        rms /= double(p.n);
        rms = sqrt(rms);
        printf("%i : %e \n", epoch, rms);
        if(rms <= errorBound)
        {
            count_time++;
            if(count_time >= nb_steps_below_error)
                break;
        }
        else
        {
            count_time = 0;
        }

        outfile << epoch << " ";
        for(int i = 0 ; i < 6 ; ++i)
            outfile << xi->data[i] << " " ;
        for(int i = 0 ; i < 3 ; ++i)
            outfile << yi->data[i] << " " ;
        for(int i = 0 ; i < 6 ; ++i)
            outfile << s.xi->data[i] << " " ;
        outfile << std::endl;
        outfile_rms << rms << std::endl;
    }
    outfile.close();
    outfile_rms.close();
    std::cout << " Run on " << epoch << " epochs " << std::endl;
    printf("I found the following parameters : %e %e %e ; The true parameters being : %e %e %e \n", s.xi->data[3], s.xi->data[4],s.xi->data[5],SIGMA, RHO, BETA);

    std::cout << " Outputs are saved in example-007*.data " << std::endl;
    //std::cout << " You can plot them using e.g. gnuplot : " << std::endl;
    //std::cout << " gnuplot Data/plot-example-007.gplot ; gv Output/example-007-rms.ps ; gv Output/example-007-Lorentz.ps " << std::endl;

    /***********************************************/
    /****            Free the memory            ****/
    /***********************************************/
    ukf_free(p,s);
}




