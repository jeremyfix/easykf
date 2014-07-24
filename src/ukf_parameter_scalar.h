/* ukf_parameter_scalar.h
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

#ifndef UKF_PARAMETER_SCALAR_H
#define UKF_PARAMETER_SCALAR_H

#include <gsl/gsl_linalg.h> // For the Cholesky decomposition
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>

#include "ukf_types.h"
#include "ukf_math.h"

namespace ukf
{
    /**
      * @short UKF for parameter estimation.
      * The notations follow "Sigma-Point Kalman Filters for Probabilistic Inference in Dynamic State-Space Models",p93, PhD, van Der Merwe
      */
    namespace parameter
    {
        /**
         * @short Allocation of the vectors/matrices and initialization
         *
         */
        inline void ukf_scalar_init(ukf_param &p, ukf_scalar_state &s)
        {
            // Init the lambda
            p.lambda = p.alpha * p.alpha * (p.n + p.kpa) - p.n;
            p.gamma = sqrt(p.n + p.lambda);
            p.nbSamples = 2 * p.n + 1;

            // Init the matrices used to iterate
            s.Kk = gsl_vector_alloc(p.n); // Kalman gain
            gsl_vector_set_zero(s.Kk);

            s.Kk_mat = gsl_matrix_alloc(p.n,1);
            gsl_matrix_set_zero(s.Kk_mat);

            s.Kk_mat_T = gsl_matrix_alloc(1,p.n);
            gsl_matrix_set_zero(s.Kk_mat_T);

            s.Pwdk = gsl_vector_alloc(p.n); // Covariance of the parameters and output
            gsl_vector_set_zero(s.Pwdk);

            // Whatever the type of evolution noise, its covariance is set to evolution_noise
            s.Prrk = gsl_matrix_alloc(p.n,p.n);
            p.evolution_noise->init(p,s);

            s.Peek = p.observation_noise; // Covariance of the observation noise

            s.Pddk = 0.0; // Covariance of the output

            s.w = gsl_vector_alloc(p.n); // Parameter vector
            gsl_vector_set_zero(s.w);

            s.wk = gsl_vector_alloc(p.n); // Vector holding one sigma point
            gsl_vector_set_zero(s.wk);

            s.Pk = gsl_matrix_alloc(p.n,p.n); // Covariance matrix
            gsl_matrix_set_identity(s.Pk);
            gsl_matrix_scale(s.Pk,p.prior_pi);

            s.Sk = gsl_matrix_alloc(p.n,p.n); // Matrix holding the cholesky decomposition of Pk
            // Initialize Sk to the cholesky decomposition of Pk
            gsl_matrix_memcpy(s.Sk, s.Pk);
            gsl_linalg_cholesky_decomp(s.Sk);
            // Set all the elements of Lpi strictly above the diagonal to zero
            for(int k = 0 ; k < p.n ; k++)
                for(int j = 0 ; j < k ; j++)
                    gsl_matrix_set(s.Sk,j,k,0.0);

            s.cSk = gsl_vector_alloc(p.n); // Vector holding one column of Lpi
            gsl_vector_set_zero(s.cSk);

            s.wm  = gsl_vector_alloc(p.nbSamples); // Weights used to compute the mean of the sigma points images
            s.wc = gsl_vector_alloc(p.nbSamples); // Weights used to update the covariance matrices

            // Set the weights
            gsl_vector_set(s.wm, 0, p.lambda / (p.n + p.lambda));
            gsl_vector_set(s.wc, 0, p.lambda / (p.n + p.lambda) + (1.0 - p.alpha*p.alpha + p.beta));
            for(int j = 1 ; j < p.nbSamples; j ++)
            {
                gsl_vector_set(s.wm, j, 1.0 / (2.0 * (p.n + p.lambda)));
                gsl_vector_set(s.wc, j, 1.0 / (2.0 * (p.n + p.lambda)));
            }

            s.dk = gsl_vector_alloc(p.nbSamples); // Holds the image of the sigma points
            gsl_vector_set_zero(s.dk);

            s.d_mean = 0; // Holds the mean of the sigma points images

            s.sigmaPoints = gsl_matrix_alloc(p.n,p.nbSamples); // Holds the sigma points in the columns
            gsl_matrix_set_zero(s.sigmaPoints);

            s.temp_n = gsl_vector_alloc(p.n);
            gsl_vector_set_zero(s.temp_n);

            s.temp_n_n = gsl_matrix_alloc(p.n,p.n);
            gsl_matrix_set_zero(s.temp_n_n);
        }

        /**
         * @short Free of memory allocation
         *
         */
        inline void ukf_scalar_free(ukf_param &p, ukf_scalar_state &s)
        {
            gsl_vector_free(s.Kk);
            gsl_matrix_free(s.Kk_mat);
            gsl_matrix_free(s.Kk_mat_T);
            gsl_vector_free(s.Pwdk);
            gsl_matrix_free(s.Prrk);

            gsl_vector_free(s.w);
            gsl_vector_free(s.wk);

            gsl_matrix_free(s.Pk);
            gsl_matrix_free(s.Sk);
            gsl_vector_free(s.cSk);

            gsl_vector_free(s.wm);
            gsl_vector_free(s.wc);

            gsl_vector_free(s.dk);

            gsl_matrix_free(s.sigmaPoints);

            gsl_vector_free(s.temp_n);
            gsl_matrix_free(s.temp_n_n);
        }

        /**
         * @short Iteration for UKF for parameter estimation, in case of a scalar output
         *
         */
        template <typename FunctObj>
        inline void ukf_scalar_iterate(ukf_param &p, ukf_scalar_state &s, FunctObj g, gsl_vector * xk, double dk)
        {
            // Here, we implement the UKF for parameter estimation in the scalar case
            // The notations follow p93 of the PhD thesis of Van Der Merwe, "Sigma-Point Kalman Filters for Probabilistic Inference in Dynamic State-Space Models"

            // ************************************************** //
            // ************ Time update equations    ************ //
            // ************************************************** //
            // Add the evolution noise to the parameter covariance Eq 3.137
            gsl_matrix_add(s.Pk, s.Prrk);

            // ************************************************** //
            // ************ Compute the sigma points ************ //
            // ************************************************** //
            // Equations 3.138
            // w_k^j = w_(k-1)  <-- this is here denoted s.w
            // w_k^j = w_(k-1) + gamma Sk_j for 1 <= j <= n
            // w_k^j = w_(k-1) - gamma Sk_j for n+1 <= j <= 2n

            // Perform a cholesky decomposition of Pk
            gsl_matrix_memcpy(s.Sk, s.Pk);
            gsl_linalg_cholesky_decomp(s.Sk);
            // Set all the elements of Lpi strictly above the diagonal to zero
            for(int k = 0 ; k < p.n ; k++)
                for(int j = 0 ; j < k ; j++)
                    gsl_matrix_set(s.Sk,j,k,0.0);

            gsl_matrix_set_col(s.sigmaPoints,0, s.w);
            for(int j = 1 ; j < p.n+1 ; ++j)
                for(int i = 0 ; i < p.n ; ++i)
                    gsl_matrix_set(s.sigmaPoints, i, j, gsl_vector_get(s.w, i) + p.gamma * gsl_matrix_get(s.Sk,i,j-1));

            for(int j = p.n+1 ; j < p.nbSamples ; ++j)
                for(int i = 0 ; i < p.n ; ++i)
                    gsl_matrix_set(s.sigmaPoints, i, j, gsl_vector_get(s.w, i) - p.gamma * gsl_matrix_get(s.Sk,i,j-(p.n+1)));

            // ************************************************** //
            // ***** Compute the images of the sigma points ***** //
            // ************************************************** //

            // Compute the images of the sigma points
            // and the mean of the yj
            s.d_mean = 0.0;
            gsl_vector_set_zero(s.dk);
            for(int j = 0 ; j < p.nbSamples ; j++)
            {
                // Equation 3.139
                gsl_matrix_get_col(s.wk, s.sigmaPoints,j);
                s.dk->data[j] = g(s.wk,xk);

                // Update the mean : y_mean = sum_[j=0..2n] wm_j y_j
                // Equation 3.140
                s.d_mean += s.wm->data[j] * s.dk->data[j];
            }

            // ************************************************** //
            // ************** Update the statistics ************* //
            // ************************************************** //

            gsl_vector_set_zero(s.Pwdk);
            // The covariance of the output is initialized with the observation noise covariance
            // Eq 3.142
            s.Pddk = s.Peek;
            for(int j = 0 ; j < p.nbSamples ; j++)
            {
                // Eq 3.142
                s.Pddk += s.wc->data[j] * gsl_pow_2(s.dk->data[j] - s.d_mean);
                // Eq 3.143
                for(int i = 0 ; i < p.n ; ++i)
                {
                    s.Pwdk->data[i] = s.Pwdk->data[i] + s.wc->data[j] * (gsl_matrix_get(s.sigmaPoints,i,j) - s.w->data[i]) * (s.dk->data[j] - s.d_mean) ;
                }
            }

            // ************************************************** //
            // ******* Kalman gain and parameters update ******** //
            // ************************************************** //

            //if(s.Pddk == 0.0)
            //    printf("[Error] Output covariance is null !");
            // May not occur as soon as the observation noise covariance is set != 0.0

            // Eq. 3.144
            for(int i = 0 ; i < p.n ; ++i)
                s.Kk->data[i] = s.Pwdk->data[i] / s.Pddk;

            // Eq 3.145
            // wk = w_(k-1) + Kk * (dk - d_mean)
            s.ino_dk = dk - s.d_mean;
            for(int i = 0 ; i < p.n ; ++i)
                s.w->data[i] = s.w->data[i] + s.Kk->data[i] * s.ino_dk;

            // Eq. 3.146
            // Pk = P_(k-1) - Pddk . Kk Kk^T
            for(int i = 0 ; i < p.n ; ++i)
            {
                gsl_matrix_set(s.Kk_mat, i, 0, s.Kk->data[i]);
                gsl_matrix_set(s.Kk_mat_T, 0, i, s.Kk->data[i]);
            }
            gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,-s.Pddk, s.Kk_mat, s.Kk_mat_T,1.0,s.Pk);

            // Update of the evolution noise
            p.evolution_noise->updateEvolutionNoise(p,s);
        }

        /**
        * @short Evaluation of the output from the sigma points
        *
        */
        template <typename FunctObj>
        inline void ukf_scalar_evaluate(ukf_param &p, ukf_scalar_state &s, FunctObj g, gsl_vector * xk, double &dk)
        {
            // ************************************************** //
            // ************ Compute the sigma points ************ //
            // ************************************************** //

            // 1- Compute the cholesky decomposition of Pk
            gsl_matrix_memcpy(s.temp_n_n, s.Pk);
            gsl_linalg_cholesky_decomp(s.temp_n_n);

            // 2 - Set all the elements of Sk_temp strictly above the diagonal to zero
            for(int k = 0 ; k < p.n ; k++)
                for(int j = 0 ; j < k ; j++)
                    gsl_matrix_set(s.temp_n_n,j,k,0.0);
            // Now Sk_temp is a lower triangular matrix containing the cholesky decomposition of Pk

            // 3- Compute the sigma points
            // Equations 3.138
            // w_k^j = w_(k-1)  <-- this is here denoted s.w
            // w_k^j = w_(k-1) + gamma Sk_j for 1 <= j <= n
            // w_k^j = w_(k-1) - gamma Sk_j for n+1 <= j <= 2n
            gsl_matrix_set_col(s.sigmaPoints,0, s.w);
            for(int j = 1 ; j < p.n+1 ; j++)
                for(int i = 0 ; i < p.n ; ++i)
                    gsl_matrix_set(s.sigmaPoints, i, j, gsl_vector_get(s.w, i) + p.gamma * gsl_matrix_get(s.temp_n_n,i,j-1));

            for(int j = p.n+1 ; j < p.nbSamples ; j++)
                for(int i = 0 ; i < p.n ; ++i)
                    gsl_matrix_set(s.sigmaPoints, i, j, gsl_vector_get(s.w, i) - p.gamma * gsl_matrix_get(s.temp_n_n,i,j-(p.n+1)));

            // ************************************************** //
            // ***** Compute the images of the sigma points ***** //
            // ************************************************** //

            // Compute the images of the sigma points
            // and their mean
            dk = 0.0;
            for(int j = 0 ; j < p.nbSamples ; j++)
            {
                gsl_matrix_get_col(s.wk, s.sigmaPoints,j);
                // Update the mean : d_mean = sum_[j=0..2n] w_j y_j
                dk += gsl_vector_get(s.wm,j) * g(s.wk,xk);
            }
        }

    } // parameter
} // ukf

#endif // UKF_PARAMETER_SCALAR_H
