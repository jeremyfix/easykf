/* ukf_parameter_ndim.h
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

#ifndef UKF_PARAMETER_NDIM_H
#define UKF_PARAMETER_NDIM_H

#include <gsl/gsl_linalg.h> // For the Cholesky decomposition
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>

#include "ukf_types.h"

namespace ukf
{
    namespace parameter
    {

        /**
          * @short Allocation of the vectors/matrices and initialization
          *
          */
        inline void ukf_init(ukf_param &p, ukf_state &s)
        {
            // Init the lambda
            p.lambda = p.alpha * p.alpha * (p.n + p.kpa) - p.n;
            p.gamma = sqrt(p.n + p.lambda);
            p.nbSamples = 2 * p.n + 1;

            // Init the matrices used to iterate
            s.Kk = gsl_matrix_alloc(p.n,p.no); // Kalman gain
            gsl_matrix_set_zero(s.Kk);

            s.Kk_T = gsl_matrix_alloc(p.no,p.n);
            gsl_matrix_set_zero(s.Kk_T);

            s.Pwdk = gsl_matrix_alloc(p.n,p.no);
            gsl_matrix_set_zero(s.Pwdk);

            // Whatever the type of evolution noise, its covariance is set to evolution_noise
            s.Prrk = gsl_matrix_alloc(p.n,p.n);
            p.evolution_noise->init(p,s);

            // Whatever the type of observation noise, its covariance is set to observation_noise
            s.Peek = gsl_matrix_alloc(p.no,p.no);
            gsl_matrix_set_identity(s.Peek);
            gsl_matrix_scale(s.Peek, p.observation_noise);

            s.Pddk = gsl_matrix_alloc(p.no, p.no); // Covariance of the output
            gsl_matrix_set_zero(s.Pddk);

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

            s.dk = gsl_matrix_alloc(p.no, p.nbSamples); // Holds the image of the sigma points
            gsl_matrix_set_zero(s.dk);

            s.ino_dk = gsl_vector_alloc(p.no); // Holds the inovation
            gsl_vector_set_zero(s.ino_dk);

            s.d_mean = gsl_vector_alloc(p.no); // Holds the mean of the sigma points images
            gsl_vector_set_zero(s.d_mean);

            s.sigmaPoints = gsl_matrix_alloc(p.n,p.nbSamples); // Holds the sigma points in the columns
            gsl_matrix_set_zero(s.sigmaPoints);

            // Temporary vectors/matrices
            s.vec_temp_n = gsl_vector_alloc(p.n);
            s.vec_temp_output = gsl_vector_alloc(p.no);

            s.mat_temp_n_1 = gsl_matrix_alloc(p.n,1);
            s.mat_temp_n_output = gsl_matrix_alloc(p.n, p.no);
            s.mat_temp_output_n = gsl_matrix_alloc(p.no, p.n);
            s.mat_temp_1_output = gsl_matrix_alloc(1,p.no);
            s.mat_temp_output_1 = gsl_matrix_alloc(p.no, 1);
            s.mat_temp_output_output = gsl_matrix_alloc(p.no, p.no);
            s.mat_temp_n_n = gsl_matrix_alloc(p.n, p.n);
        }

        /**
          * @short Free of memory allocation
          *
          */
        inline void ukf_free(ukf_param &p, ukf_state &s)
        {
            gsl_matrix_free(s.Kk);
            gsl_matrix_free(s.Kk_T);
            gsl_matrix_free(s.Pwdk);
            gsl_matrix_free(s.Pddk);
            gsl_matrix_free(s.Peek);
            gsl_matrix_free(s.Prrk);

            gsl_vector_free(s.w);
            gsl_vector_free(s.wk);

            gsl_matrix_free(s.Pk);
            gsl_matrix_free(s.Sk);
            gsl_vector_free(s.cSk);

            gsl_vector_free(s.wm);
            gsl_vector_free(s.wc);

            gsl_matrix_free(s.dk);
            gsl_vector_free(s.ino_dk);
            gsl_vector_free(s.d_mean);

            gsl_matrix_free(s.sigmaPoints);

            gsl_vector_free(s.vec_temp_n);
            gsl_vector_free(s.vec_temp_output);

            gsl_matrix_free(s.mat_temp_n_1);
            gsl_matrix_free(s.mat_temp_n_output);
            gsl_matrix_free(s.mat_temp_output_n);
            gsl_matrix_free(s.mat_temp_1_output);
            gsl_matrix_free(s.mat_temp_output_1);

            gsl_matrix_free(s.mat_temp_output_output);
            gsl_matrix_free(s.mat_temp_n_n);
        }

        /**
          * @short Iteration for the statistical linearization
          *
          */
        template <typename FunctObj>
                inline void ukf_iterate(ukf_param &p, ukf_state &s, FunctObj g, gsl_vector * xk, gsl_vector* dk)
        {

            // Here, we implement the UKF for parameter estimation in the vectorial case
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
            for(int k = 0 ; k < p.n ; ++k)
                for(int j = 0 ; j < k ; ++j)
                    gsl_matrix_set(s.Sk,j,k,0.0);

            gsl_matrix_set_col(s.sigmaPoints,0, s.w);
            for(int j = 1 ; j < p.n+1 ; ++j)
                for(int i = 0 ; i < p.n ; ++i)
                    gsl_matrix_set(s.sigmaPoints, i, j, gsl_vector_get(s.w, i) + p.gamma * gsl_matrix_get(s.Sk,i,j-1));

            for(int j = p.n+1 ; j < p.nbSamples ; ++j)
                for(int i = 0 ; i < p.n ; ++i)
                    gsl_matrix_set(s.sigmaPoints, i, j, gsl_vector_get(s.w, i) - p.gamma * gsl_matrix_get(s.Sk,i,j-(p.n+1)));



            /**************************************************/
            /***** Compute the images of the sigma points *****/
            /**************************************************/

            // Compute the images of the sigma points
            // and the mean of the dk
            gsl_vector_set_zero(s.d_mean);
            for(int j = 0 ; j < p.nbSamples ; j++)
            {
                // Equation 3.129
                gsl_matrix_get_col(s.wk, s.sigmaPoints,j);
                g(s.wk,xk, s.vec_temp_output);
                gsl_matrix_set_col(s.dk, j, s.vec_temp_output);

                // Equation 3.140
                // Update the mean : y_mean = sum_[j=0..2n] w_j y_j
                gsl_vector_scale(s.vec_temp_output, gsl_vector_get(s.wm,j));
                gsl_vector_add(s.d_mean, s.vec_temp_output);
            }

            /**************************************************/
            /************** Update the statistics *************/
            /**************************************************/

            gsl_matrix_set_zero(s.Pwdk);
            gsl_matrix_memcpy(s.Pddk, s.Peek); // Add R^e_k to Pddk, Eq 3.142
            for(int j = 0 ; j < p.nbSamples ; ++j)
            {
                // Update of Pwdk
                // (wk - w)
                gsl_matrix_get_col(s.wk, s.sigmaPoints,j);
                gsl_vector_sub(s.wk, s.w);
                gsl_matrix_set_col(s.mat_temp_n_1, 0, s.wk);

                // (dk - d_mean)
                gsl_matrix_get_col(s.vec_temp_output, s.dk, j);
                gsl_vector_sub(s.vec_temp_output, s.d_mean);
                gsl_matrix_set_col(s.mat_temp_output_1, 0, s.vec_temp_output);

                // compute wc_j . (wk - w_mean) * (dk - d_mean)^T
                gsl_blas_dgemm(CblasNoTrans, CblasTrans, gsl_vector_get(s.wc,j) , s.mat_temp_n_1, s.mat_temp_output_1, 0.0, s.mat_temp_n_output);

                // Equation 3.142
                // And add it to Pwdk
                gsl_matrix_add(s.Pwdk, s.mat_temp_n_output);

                // Equation 3.143
                // Update of Pddk
                gsl_matrix_get_col(s.vec_temp_output, s.dk, j);
                gsl_vector_sub(s.vec_temp_output, s.d_mean);
                gsl_matrix_set_col(s.mat_temp_output_1, 0, s.vec_temp_output);
                gsl_blas_dgemm(CblasNoTrans, CblasTrans, gsl_vector_get(s.wc,j) , s.mat_temp_output_1, s.mat_temp_output_1, 0.0, s.mat_temp_output_output);

                gsl_matrix_add(s.Pddk, s.mat_temp_output_output);
            }

            // ************************************************** //
            // ******* Kalman gain and parameters update ******** //
            // ************************************************** //

            //*** Ki = Pwdk Pddk^-1
            // Compute the inverse of Pddk
            gsl_matrix_memcpy(s.mat_temp_output_output, s.Pddk);
            gsl_linalg_cholesky_decomp(s.mat_temp_output_output);
            gsl_linalg_cholesky_invert(s.mat_temp_output_output);

            // Compute the product : Pwdk . Pddk^-1
            // Equation 3.144
            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0 , s.Pwdk, s.mat_temp_output_output, 0.0, s.Kk);

            // Update of the parameters
            // wk = w_(k-1) + Kk * (dk - d_mean)
            // Equation 3.145

            // Set the inovations
            /*for(int i = 0 ; i < p.no; ++i)
                s.ino_dk->data[i] = dk->data[i] - s.d_mean->data[i];
            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0 , s.Kk, &gsl_matrix_view_array(s.ino_dk->data,p.no,1).matrix, 0.0, s.mat_temp_n_1);
            gsl_matrix_get_col(s.vec_temp_n, s.mat_temp_n_1, 0);
            gsl_vector_add(s.w, s.vec_temp_n);*/

            for(int i = 0 ; i < p.no; ++i)
                s.ino_dk->data[i] = dk->data[i] - s.d_mean->data[i];
            gsl_blas_dgemv(CblasNoTrans, 1.0, s.Kk, s.ino_dk, 1.0, s.w);

            // Update of the parameter covariance
            // Pk = P_(k-1) - Kk Pddk Kk^T
            // Equation 3.146
            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0 , s.Kk, s.Pddk, 0.0, s.mat_temp_n_output);
            gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, s.mat_temp_n_output , s.Kk, 0.0, s.mat_temp_n_n);
            gsl_matrix_sub(s.Pk, s.mat_temp_n_n);

            // Update of the evolution noise
            p.evolution_noise->updateEvolutionNoise(p, s);
        }

        /**
          * @short Evaluation of the output from the sigma points
          *
          */
        template <typename FunctObj>
                inline void ukf_evaluate(ukf_param &p, ukf_state &s, FunctObj g, gsl_vector * xk, gsl_vector * dk)
        {
            // ************************************************** //
            // ************ Compute the sigma points ************ //
            // ************************************************** //
            // Equations 3.138
            // w_k^j = w_(k-1)  <-- this is here denoted s.w
            // w_k^j = w_(k-1) + gamma Sk_j for 1 <= j <= n
            // w_k^j = w_(k-1) - gamma Sk_j for n+1 <= j <= 2n

            // Perform a cholesky decomposition of Pk
            gsl_matrix_memcpy(s.mat_temp_n_n, s.Pk);
            gsl_linalg_cholesky_decomp(s.mat_temp_n_n);
            // Set all the elements of Lpi strictly above the diagonal to zero
            for(int k = 0 ; k < p.n ; ++k)
                for(int j = 0 ; j < k ; ++j)
                    gsl_matrix_set(s.mat_temp_n_n,j,k,0.0);

            gsl_matrix_set_col(s.sigmaPoints,0, s.w);
            for(int j = 1 ; j < p.n+1 ; ++j)
                for(int i = 0 ; i < p.n ; ++i)
                    gsl_matrix_set(s.sigmaPoints, i, j, gsl_vector_get(s.w, i) + p.gamma * gsl_matrix_get(s.mat_temp_n_n,i,j-1));

            for(int j = p.n+1 ; j < p.nbSamples ; ++j)
                for(int i = 0 ; i < p.n ; ++i)
                    gsl_matrix_set(s.sigmaPoints, i, j, gsl_vector_get(s.w, i) - p.gamma * gsl_matrix_get(s.mat_temp_n_n,i,j-(p.n+1)));


            /**************************************************/
            /***** Compute the images of the sigma points *****/
            /**************************************************/

            // Compute the images of the sigma points
            // and the mean of the dk
            gsl_vector_set_zero(dk);
            for(int j = 0 ; j < p.nbSamples ; j++)
            {
                // Equation 3.129
                gsl_matrix_get_col(s.wk, s.sigmaPoints,j);
                g(s.wk,xk, s.vec_temp_output);
                gsl_matrix_set_col(s.dk, j, s.vec_temp_output);

                // Equation 3.140
                // Update the mean : y_mean = sum_[j=0..2n] w_j y_j
                gsl_vector_scale(s.vec_temp_output, gsl_vector_get(s.wm,j));
                gsl_vector_add(dk, s.vec_temp_output);
            }
        }

        /**
          * @short Returns a set of sigma points
          */
        void getSigmaPoints(ukf_param &p, ukf_state &s, gsl_matrix * sigmaPoints)
        {
            //gsl_matrix * sigmaPoints = gsl_matrix_alloc(p.n, p.nbSamples);

            // ************************************************** //
            // ************ Compute the sigma points ************ //
            // ************************************************** //
            // Equations 3.138
            // w_k^j = w_(k-1)  <-- this is here denoted s.w
            // w_k^j = w_(k-1) + gamma Sk_j for 1 <= j <= n
            // w_k^j = w_(k-1) - gamma Sk_j for n+1 <= j <= 2n

            // Perform a cholesky decomposition of Pk
            gsl_matrix_memcpy(s.mat_temp_n_n, s.Pk);
            gsl_linalg_cholesky_decomp(s.mat_temp_n_n);
            // Set all the elements of Lpi strictly above the diagonal to zero
            for(int k = 0 ; k < p.n ; ++k)
                for(int j = 0 ; j < k ; ++j)
                    gsl_matrix_set(s.mat_temp_n_n,j,k,0.0);

            gsl_matrix_set_col(sigmaPoints,0, s.w);
            for(int j = 1 ; j < p.n+1 ; ++j)
                for(int i = 0 ; i < p.n ; ++i)
                    gsl_matrix_set(sigmaPoints, i, j, gsl_vector_get(s.w, i) + p.gamma * gsl_matrix_get(s.mat_temp_n_n,i,j-1));

            for(int j = p.n+1 ; j < p.nbSamples ; ++j)
                for(int i = 0 ; i < p.n ; ++i)
                    gsl_matrix_set(sigmaPoints, i, j, gsl_vector_get(s.w, i) - p.gamma * gsl_matrix_get(s.mat_temp_n_n,i,j-(p.n+1)));

        }

    } // parameter
} // ukf

#endif // SL_PARAMETER_NDIM_H
