/* example-007.cc
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

/* In this example, we use joint UKF for state/parameter estimation in order to estimate the state and parameters of a Lorentz attractor */

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include "ukf.h"

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




