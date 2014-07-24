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

/* In this example, we use a 2-12-2 MLP network to learn the Mackay Robot arm data  */
/* Here , we consider a vectorial output, it extends example-004                    */
/* In addition to the weights and biases, we also parametrize the transfer function */
/* The results are saved in example-005-results.data, a 4 column datafile.          */

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
#include <algorithm>

#include <easykf.h>

using namespace ukf::parameter;

#define VERBOSE true
#define NB_INPUTS 2
#define NB_HIDDEN 8
#define NB_OUTPUTS 2

/*****************************************************/
/*            Definition of the MLP                  */
/*****************************************************/
double transfer(double x, double a, double b, double u , double c)
{
    return a / (1.0 + exp(-b * (x-u)))+c;
}

/*double transfer(double x)
{
    return 2.0 / (1.0 + exp(-x)) - 1.0;
}*/

void my_func(gsl_vector * param, gsl_vector * input, gsl_vector * output)
{
    double y[NB_HIDDEN];
    double z=0.0;

    double a0 = gsl_vector_get(param,(NB_INPUTS+1) * NB_HIDDEN + (NB_HIDDEN+1)* NB_OUTPUTS);
    double b0 = gsl_vector_get(param,(NB_INPUTS+1) * NB_HIDDEN + (NB_HIDDEN+1)* NB_OUTPUTS+1);
    double u0 = gsl_vector_get(param,(NB_INPUTS+1) * NB_HIDDEN + (NB_HIDDEN+1)* NB_OUTPUTS+2);
    double c0 = gsl_vector_get(param,(NB_INPUTS+1) * NB_HIDDEN + (NB_HIDDEN+1)* NB_OUTPUTS+3);

    double a1 = gsl_vector_get(param,(NB_INPUTS+1) * NB_HIDDEN + (NB_HIDDEN+1)* NB_OUTPUTS+4);
    double b1 = gsl_vector_get(param,(NB_INPUTS+1) * NB_HIDDEN + (NB_HIDDEN+1)* NB_OUTPUTS+5);
    double u1 = gsl_vector_get(param,(NB_INPUTS+1) * NB_HIDDEN + (NB_HIDDEN+1)* NB_OUTPUTS+6);
    double c1 = gsl_vector_get(param,(NB_INPUTS+1) * NB_HIDDEN + (NB_HIDDEN+1)* NB_OUTPUTS+7);

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
        y[j] = transfer(y[j], 1.0, 1.0, 0.0, 0.0);
        //y[j] = transfer(y[j]);
    }

    // First output
    z = 0.0;
    for(int j = 0 ; j < NB_HIDDEN ; j++)
    {
        z += gsl_vector_get(param,i_param) * y[j];
        i_param++;
    }
    z += gsl_vector_get(param, i_param);
    i_param++;

    z = transfer(z,a0,b0,u0,c0);
    //z = transfer(z);
    gsl_vector_set(output, 0, z);

    // Second output
    z = 0.0;
    for(int j = 0 ; j < NB_HIDDEN ; j++)
    {
        z += gsl_vector_get(param,i_param) * y[j];
        i_param++;
    }
    z += gsl_vector_get(param, i_param);
    i_param++;

    z = transfer(z,a1,b1,u1,c1);
    //z = transfer(z);
    gsl_vector_set(output, 1, z);

}

// Structure holding a sample : the joint angles and the cartesian position
typedef struct {
    float theta;
    float phi;
    float x;
    float y;
} Sample;


/*****************************************************/
/*                    main                           */
/*****************************************************/

int main(int argc, char* argv[]) {

    srand(time(NULL));

    // Definition of the parameters and state variables
    ukf_param p;
    ukf_state s;

    p.n = (NB_INPUTS+1) * NB_HIDDEN + (NB_HIDDEN+1)* NB_OUTPUTS + 8 ;
    p.no = NB_OUTPUTS;
    p.kpa = 3.0 - p.n;
    p.alpha = 1e-2;
    p.beta = 2.0;

    EvolutionNoise * evolution_noise = new EvolutionAnneal(1e-3,0.999, 1e-8);
    //EvolutionNoise * evolution_noise = new EvolutionRLS(1e-4,0.9995);
    //EvolutionNoise * evolution_noise = new EvolutionRobbinsMonro(1e-3,1e-4);
    p.evolution_noise = evolution_noise;

    p.observation_noise = 1e-1; // Observation noise
    p.prior_pi = 1.0; // Initializition of the covariance of the parameters

    // Initialization of the state and parameters
    ukf_init(p,s);

    // Initialize the parameter vector to some random values
    for(int i = 0 ; i < p.n ; i++)
        gsl_vector_set(s.w,i,0.01*(2.0 * rand()/double(RAND_MAX-1) - 1.0));

    // Allocate the input/output vectors
    gsl_vector * xi = gsl_vector_alloc(NB_INPUTS);
    gsl_vector * yi = gsl_vector_alloc(NB_OUTPUTS);

    // Define some limit conditions for the learning
    double errorBound = 8e-3;
    int nbStepsLimit = 1000;
    double error = 2*errorBound;;

    /***********************************************/
    /*****        Load the training data set   *****/
    /***********************************************/

    std::vector< Sample > samples;
    FILE * inputs_file = fopen ("Data/Mackay-inputs.data","r");
    FILE * outputs_file = fopen ("Data/Mackay-outputs.data","r");
    char c;

    double mintheta=0.0;
    double maxtheta=0.0;
    double minphi=0.0;
    double maxphi =0.0;
    bool is_init = false;
    do
    {
        Sample s;
        fscanf(inputs_file,"%f",&s.theta);
        fscanf(inputs_file,"%f",&s.phi);
        fscanf(outputs_file,"%f",&s.x);
        fscanf(outputs_file,"%f",&s.y);
        if(!is_init)
        {
            mintheta = s.theta;
            maxtheta = s.theta;
            minphi = s.phi;
            maxphi = s.phi;
            is_init = true;
        }
        else
        {
            mintheta = s.theta < mintheta ? s.theta : mintheta;
            maxtheta = s.theta > maxtheta ? s.theta : maxtheta;
            minphi = s.phi < minphi ? s.phi : minphi;
            maxphi = s.phi > maxphi ? s.phi : maxphi;
        }
        samples.push_back(s);
        c = getc (inputs_file);
    } while(c != EOF);

    if(VERBOSE) std::cout << "I found : " << samples.size() << " samples " << std::endl;
    if(VERBOSE) std::cout << mintheta << " < theta < " << maxtheta << std::endl;
    if(VERBOSE) std::cout << minphi << " < phi < " << maxphi << std::endl;

    /***********************************************/
    /***** Iterate the learning on the samples *****/
    /***********************************************/

    int epoch = 0;
    error = 2 * errorBound;

    while( epoch <= nbStepsLimit && error > errorBound)
    {
        // Iterate on the samples
        std::random_shuffle(samples.begin(), samples.end());
        for(unsigned int j = 0 ; j < samples.size() ; j++)
        {
            gsl_vector_set(xi,0,samples[j].theta);
            gsl_vector_set(xi,1,samples[j].phi);

            gsl_vector_set(yi, 0, samples[j].x);
            gsl_vector_set(yi, 1, samples[j].y);

            ukf_iterate(p, s, my_func,xi,yi);
        }

        // Evaluate the RMS error
        error = 0.0;
        for(unsigned int j = 0 ; j < samples.size() ; j++)
        {
            gsl_vector_set(xi,0,samples[j].theta);
            gsl_vector_set(xi,1,samples[j].phi);

            my_func(s.w,xi,yi);

            error += pow(samples[j].x - gsl_vector_get(yi,0),2.0)+pow(samples[j].y - gsl_vector_get(yi,1),2.0);
        }

        error = sqrt(error /  double(samples.size()));
        if(VERBOSE)
            std::cout << "Epoch " << epoch << " error = " << error << std::endl;
        epoch++;
    }
    std::cout << "Run on " << epoch << " epochs ; RMS = " << error << "\n" << std::endl ;

    /***********************************************/
    /****     Save the results                  ****/
    /***********************************************/

    std::ofstream outfile("example-005.data");
    std::cout << " You can plot them using e.g. gnuplot : " << std::endl;
    //std::cout << " gnuplot Data/plot-example-005.gplot ; gv Output/example-005-y1.ps ; gv Output/example-005-y2.ps " << std::endl;
    for(unsigned int i = 0 ; i < samples.size() ; i++)
    {
        gsl_vector_set(xi, 0, samples[i].theta);
        gsl_vector_set(xi, 1, samples[i].phi);

        // Evaluate the output as the mean of the images of the sigma points
        //ukf_evaluate(p, s, my_func,xi,yi);
        // Or directly with the function
        my_func(s.w,xi, yi);
        outfile << gsl_vector_get(xi, 0) << " " << gsl_vector_get(xi, 1) << " " << gsl_vector_get(yi, 0) << " " << gsl_vector_get(yi,1) << std::endl;
    }

    outfile.close();

    /***********************************************/
    /****            Free the memory            ****/
    /***********************************************/

    ukf_free(p,s);
}
