/* ukf_state_ndim.h
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

#ifndef UKF_NDIM_STATE_H
#define UKF_NDIM_STATE_H

#include <gsl/gsl_linalg.h> // For the Cholesky decomposition
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>

#include "ukf_types.h"

namespace ukf
{

    /**
      * @short UKF for state estimation, additive noise case
      * The notations follow "Sigma-Point Kalman Filters for Probabilistic Inference in Dynamic State-Space Models",p108, PhD, van Der Merwe
      */
    namespace state
    {
        /**
          * @short Allocation of the vectors/matrices and initialization
          *
          */
        inline void ukf_init(ukf_param &p, ukf_state &s)
        {
            // Parameters for the sigma points of the process equation
            p.nbSamples = 2 * p.n + 1;
            p.lambda = p.alpha * p.alpha * (p.n + p.kpa) - p.n;
            p.gamma = sqrt(p.n + p.lambda);

            // Parameters for the sigma points of the observation equation
            p.nbSamplesMeasure = 4 * p.n + 1;
            p.lambda_aug = p.alpha * p.alpha * (2*p.n + p.kpa) - 2*p.n;
            p.gamma_aug = sqrt(2*p.n + p.lambda_aug);

            // Init the matrices used to iterate
            s.xi = gsl_vector_alloc(p.n);
            gsl_vector_set_zero(s.xi);

            s.xi_prediction = gsl_matrix_alloc(p.n, p.nbSamples);
            gsl_matrix_set_zero(s.xi_prediction);

            s.xi_mean = gsl_vector_alloc(p.n);
            gsl_vector_set_zero(s.xi_mean);

            s.Pxxi = gsl_matrix_alloc(p.n,p.n);
            gsl_matrix_set_identity(s.Pxxi);
            gsl_matrix_scale(s.Pxxi, p.prior_x);

            s.cholPxxi = gsl_matrix_alloc(p.n,p.n);
            gsl_matrix_set_zero(s.cholPxxi);

            s.Pvvi = gsl_matrix_alloc(p.n,p.n);
            s.cholPvvi = gsl_matrix_alloc(p.n,p.n);
            p.evolution_noise->init(p,s);

            s.yi_prediction = gsl_matrix_alloc(p.no, p.nbSamplesMeasure);
            gsl_matrix_set_zero(s.yi_prediction);

            s.yi_mean = gsl_vector_alloc(p.no);
            gsl_vector_set_zero(s.yi_mean);

            s.ino_yi = gsl_vector_alloc(p.no);
            gsl_vector_set_zero(s.ino_yi);

            s.Pyyi = gsl_matrix_alloc(p.no, p.no);
            gsl_matrix_set_zero(s.Pyyi);

            s.Pnni = gsl_matrix_alloc(p.no,p.no);
            gsl_matrix_set_identity(s.Pnni);
            gsl_matrix_scale(s.Pnni, p.measurement_noise);

            s.Pxyi = gsl_matrix_alloc(p.n, p.no);
            gsl_matrix_set_zero(s.Pxyi);

            s.sigmaPoint = gsl_vector_alloc(p.n);
            gsl_vector_set_zero(s.sigmaPoint);

            s.sigmaPoints = gsl_matrix_alloc(p.n, p.nbSamples);
            gsl_matrix_set_zero(s.sigmaPoints);

            s.sigmaPointMeasure = gsl_vector_alloc(p.n);
            gsl_vector_set_zero(s.sigmaPoint);

            s.sigmaPointsMeasure = gsl_matrix_alloc(p.n, p.nbSamplesMeasure);
            gsl_matrix_set_zero(s.sigmaPointsMeasure);

            // Weights used to update the statistics
            s.wm_j  = gsl_vector_alloc(p.nbSamples); // Weights used to compute the mean of the sigma points images
            s.wc_j = gsl_vector_alloc(p.nbSamples); // Weights used to update the covariance matrices

            // Set the weights
            gsl_vector_set(s.wm_j, 0, p.lambda / (p.n + p.lambda));
            gsl_vector_set(s.wc_j, 0, p.lambda / (p.n + p.lambda) + (1.0 - p.alpha*p.alpha + p.beta));
            for(int j = 1 ; j < p.nbSamples; j ++)
            {
                gsl_vector_set(s.wm_j, j, 1.0 / (2.0 * (p.n + p.lambda)));
                gsl_vector_set(s.wc_j, j, 1.0 / (2.0 * (p.n + p.lambda)));
            }

            // Set the weights
            s.wm_aug_j  = gsl_vector_alloc(p.nbSamplesMeasure); // Weights used to compute the mean of the sigma points images
            s.wc_aug_j = gsl_vector_alloc(p.nbSamplesMeasure); // Weights used to update the covariance matrices
            gsl_vector_set(s.wm_aug_j, 0, p.lambda_aug / (2*p.n + p.lambda_aug));
            gsl_vector_set(s.wc_aug_j, 0, p.lambda_aug / (2*p.n + p.lambda_aug) + (1.0 - p.alpha*p.alpha + p.beta));
            for(int j = 1 ; j < p.nbSamplesMeasure; j ++)
            {
                gsl_vector_set(s.wm_aug_j, j, 1.0 / (2.0 * (2*p.n + p.lambda_aug)));
                gsl_vector_set(s.wc_aug_j, j, 1.0 / (2.0 * (2*p.n + p.lambda_aug)));
            }

            s.Ki = gsl_matrix_alloc(p.n, p.no);
            s.Ki_T = gsl_matrix_alloc(p.no, p.n);

            // Allocate temporary matrices
            s.temp_n = gsl_vector_alloc(p.n);

            s.temp_n_1 = gsl_matrix_alloc(p.n,1);
            s.temp_1_n = gsl_matrix_alloc(1,p.n);
            s.temp_n_n = gsl_matrix_alloc(p.n, p.n);
            s.temp_n_no = gsl_matrix_alloc(p.n, p.no);
            s.temp_no_1 = gsl_matrix_alloc(p.no,1);
            s.temp_1_no = gsl_matrix_alloc(1,p.no);
            s.temp_no_no = gsl_matrix_alloc(p.no, p.no);
        }

        /**
          * @short Free of memory allocation
          *
          */
        inline void ukf_free(ukf_param &p, ukf_state &s)
        {
            gsl_vector_free(s.xi);
            gsl_matrix_free(s.xi_prediction);
            gsl_vector_free(s.xi_mean);
            gsl_matrix_free(s.Pxxi);
            gsl_matrix_free(s.cholPxxi);
            gsl_matrix_free(s.Pvvi);
            gsl_matrix_free(s.cholPvvi);

            gsl_matrix_free(s.yi_prediction);
            gsl_vector_free(s.yi_mean);
            gsl_vector_free(s.ino_yi);
            gsl_matrix_free(s.Pyyi);
            gsl_matrix_free(s.Pnni);

            gsl_matrix_free(s.Pxyi);

            gsl_vector_free(s.sigmaPoint);
            gsl_matrix_free(s.sigmaPoints);

            gsl_vector_free(s.sigmaPointMeasure);
            gsl_matrix_free(s.sigmaPointsMeasure);

            gsl_vector_free(s.wm_j);
            gsl_vector_free(s.wc_j);

            gsl_vector_free(s.wm_aug_j);
            gsl_vector_free(s.wc_aug_j);

            gsl_matrix_free(s.Ki);
            gsl_matrix_free(s.Ki_T);

            gsl_vector_free(s.temp_n);
            gsl_matrix_free(s.temp_n_1);
            gsl_matrix_free(s.temp_1_n);
            gsl_matrix_free(s.temp_n_n);

            gsl_matrix_free(s.temp_n_no);
            gsl_matrix_free(s.temp_no_1);
            gsl_matrix_free(s.temp_1_no);
            gsl_matrix_free(s.temp_no_no);
        }

        /**
          * @short UKF-additive (zero-mean) noise case, "Kalman Filtering and Neural Networks", p.233
          *
          */
        template <typename FunctProcess, typename FunctObservation>
                inline void ukf_iterate(ukf_param &p, ukf_state &s, FunctProcess f, FunctObservation h, gsl_vector* yi)
        {
            int i,j,k;

            // ************************************************** //
            // ************ Compute the sigma points ************ //
            // ************************************************** //

            // 0 - Compute the Cholesky decomposition of s.Pxxi
            gsl_matrix_memcpy(s.cholPxxi, s.Pxxi);
            gsl_linalg_cholesky_decomp(s.cholPxxi);
            // Set all the elements of cholPvvi strictly above the diagonal to zero
            for(j = 0 ; j < p.n ; j++)
                for(k = j+1 ; k < p.n ; k++)
                    gsl_matrix_set(s.cholPxxi,j,k,0.0);

            // 1- Compute the sigma points,
            // Equation (3.170)
            // sigmapoint_j = x_(i-1)
            // sigmapoint_j = x_(i-1) + gamma * sqrt(P_i-1)_j for 1 <= j <= n
            // sigmapoint_j = x_(i-1) - gamma * sqrt(P_i-1)_(j-(n+1)) for n+1 <= j <= 2n
            gsl_matrix_set_col(s.sigmaPoints, 0, s.xi);
            for(j = 1 ; j < p.n+1 ; ++j)
                for(i = 0 ; i < p.n ; ++i)
                {
                    gsl_matrix_set(s.sigmaPoints,i,j, s.xi->data[i] + p.gamma * gsl_matrix_get(s.cholPxxi, i, j-1));
                    gsl_matrix_set(s.sigmaPoints,i,j+p.n, s.xi->data[i] - p.gamma * gsl_matrix_get(s.cholPxxi, i, j-1));
                }

            /**********************************/
            /***** Time update equations  *****/
            /**********************************/

            // Time update equations
            // 0 - Compute the image of the sigma points and the mean of these images
            gsl_vector_set_zero(s.xi_mean);
	    gsl_vector_view vec_view;
            for(j = 0 ; j < p.nbSamples ; ++j)
            {
                gsl_matrix_get_col(s.sigmaPoint, s.sigmaPoints, j);
		vec_view = gsl_matrix_column(s.xi_prediction,j);
                f(s.params, s.sigmaPoint, &vec_view.vector);

                // Update the mean, Eq (3.172)
                for(i = 0 ; i < p.n ; ++i)
                    s.xi_mean->data[i] += s.wm_j->data[j] * gsl_matrix_get(s.xi_prediction,i,j);
            }

            // 1 - Compute the covariance of the images and add the process noise,
            // Equation (3.173)
            // Warning, s.Pxxi will now hold P_xk^-
            gsl_matrix_set_zero(s.Pxxi);
            for(j = 0 ; j < p.nbSamples ; ++j)
            {
                for(i = 0 ; i < p.n ; ++i)
                    s.temp_n_1->data[i] = gsl_matrix_get(s.xi_prediction,i,j) - s.xi_mean->data[i];

                gsl_blas_dgemm(CblasNoTrans, CblasTrans, s.wc_j->data[j] , s.temp_n_1, s.temp_n_1, 0, s.temp_n_n);
                gsl_matrix_add(s.Pxxi, s.temp_n_n);
            }
            // Add the covariance of the evolution noise
            gsl_matrix_add(s.Pxxi, s.Pvvi);

            // Augment sigma points
            // Equation 3.174
            // First put the images of the initial sigma points
	    gsl_matrix_view mat_view;
	    mat_view = gsl_matrix_submatrix(s.sigmaPointsMeasure, 0, 0, p.n, p.nbSamples);
            gsl_matrix_memcpy(&mat_view.matrix, s.xi_prediction);
            // And add the additional sigma points eq. (7.56)
            for(j = 0 ; j < p.n ; ++j)
            {
                for(i = 0 ; i < p.n ; ++i)
                {
                    gsl_matrix_set(s.sigmaPointsMeasure, i, j+p.nbSamples, gsl_matrix_get(s.xi_prediction,i,0)+p.gamma_aug*gsl_matrix_get(s.cholPvvi,i,j));
                    gsl_matrix_set(s.sigmaPointsMeasure, i, j+p.nbSamples+p.n, gsl_matrix_get(s.xi_prediction,i,0)-p.gamma_aug*gsl_matrix_get(s.cholPvvi,i,j));
                }
            }

            // Compute the image of the sigma points through the observation equation
            // eq (3.175)
            gsl_vector_set_zero(s.yi_mean);
            for(j = 0 ; j < p.nbSamplesMeasure ; ++j)
            {
                gsl_matrix_get_col(s.sigmaPointMeasure, s.sigmaPointsMeasure, j);
		vec_view = gsl_matrix_column(s.yi_prediction,j);
                h(s.sigmaPointMeasure, &vec_view.vector);

                // Update the mean , eq (3.176)
                for(i = 0 ; i < p.no ; ++i)
                    s.yi_mean->data[i] += s.wm_aug_j->data[j] * gsl_matrix_get(s.yi_prediction,i,j);
            }

            /*****************************************/
            /***** Measurement update equations  *****/
            /*****************************************/

            // Compute the covariance of the observations
            // Eq. (3.177)
            // Initialize with the observation noise covariance
            gsl_matrix_memcpy(s.Pyyi, s.Pnni);
            for(j = 0 ; j < p.nbSamplesMeasure ; ++j)
            {
                for(i = 0 ; i < p.no ; ++i)
                    s.temp_no_1->data[i] = gsl_matrix_get(s.yi_prediction,i,j) - s.yi_mean->data[i];

                gsl_blas_dgemm(CblasNoTrans, CblasTrans, s.wc_aug_j->data[j] , s.temp_no_1, s.temp_no_1, 0, s.temp_no_no);
                gsl_matrix_add(s.Pyyi, s.temp_no_no);
            }

            // Compute the state/observation covariance
            // Eq (3.178)
            gsl_matrix_set_zero(s.Pxyi);
            for(j = 0 ; j < p.nbSamplesMeasure ; ++j)
            {
                for(i = 0 ; i < p.n ; ++i)
                    s.temp_n_1->data[i] = gsl_matrix_get(s.sigmaPointsMeasure,i,j) - s.xi_mean->data[i];

                for(i = 0 ; i < p.no ; ++i)
                    s.temp_1_no->data[i] = gsl_matrix_get(s.yi_prediction,i,j) - s.yi_mean->data[i];

                gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, s.wc_aug_j->data[j] , s.temp_n_1, s.temp_1_no, 0, s.temp_n_no);
                gsl_matrix_add(s.Pxyi, s.temp_n_no);
            }

            // Compute the Kalman gain, eq (3.179)
            // 0- Compute the inverse of Pyyi
            gsl_matrix_memcpy(s.temp_no_no, s.Pyyi);
            gsl_linalg_cholesky_decomp(s.temp_no_no);
            gsl_linalg_cholesky_invert(s.temp_no_no);

            // 1- Compute the Kalman gain
            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0 , s.Pxyi, s.temp_no_no, 0, s.Ki);

            // Correction : correct the estimation of the state
            // Eq. 3.180
            // Compute the innovations
            for(i = 0 ; i < p.no ; ++i)
                s.ino_yi->data[i] = gsl_vector_get(yi, i) - gsl_vector_get(s.yi_mean, i);
            gsl_vector_memcpy(s.xi, s.xi_mean);
            gsl_blas_dgemv(CblasNoTrans, 1.0 , s.Ki, s.ino_yi, 1.0, s.xi);

            // Correction : Update the covariance matrix Pk
            // Eq. 3.181
            gsl_matrix_transpose_memcpy(s.Ki_T, s.Ki);
            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0 , s.Ki, s.Pyyi, 0, s.temp_n_no);
            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0 , s.temp_n_no, s.Ki_T, 1.0, s.Pxxi);

            // Update of the process noise
            p.evolution_noise->updateEvolutionNoise(p,s);
//            switch(p.process_noise_type)
//            {
//            case ukf::UKF_PROCESS_FIXED:
//                //nothing to do
//                break;
//            case ukf::UKF_PROCESS_RLS:
//                gsl_matrix_memcpy(s.Pvvi, s.Pxxi);
//                gsl_matrix_scale(s.Pvvi, 1.0/p.process_noise-1.0);
//                gsl_matrix_memcpy(s.cholPvvi, s.Pvvi);
//                gsl_linalg_cholesky_decomp(s.cholPvvi);
//                for(j = 0 ; j < p.n ; j++)
//                    for(k = j+1 ; k < p.n ; k++)
//                        gsl_matrix_set(s.cholPvvi,j,k,0.0);
//                break;
//            default:
//                printf("Warning : Unrecognized process noise type\n");
//            }


        }

        /**
          * @short Evaluation of the output from the sigma points
          *
          */
        template <typename FunctProcess, typename FunctObservation>
                inline void ukf_evaluate(ukf_param &p, ukf_state &s, FunctProcess f, FunctObservation h, gsl_vector* yi)
        {

	  int i,j,k;
            // ************************************************** //
            // ************ Compute the sigma points ************ //
            // ************************************************** //

            // 0 - Compute the Cholesky decomposition of s.Pxxi
            gsl_matrix_memcpy(s.cholPxxi, s.Pxxi);
            gsl_linalg_cholesky_decomp(s.cholPxxi);
            // Set all the elements of cholPvvi strictly above the diagonal to zero
            for(j = 0 ; j < p.n ; j++)
                for(k = j+1 ; k < p.n ; k++)
                    gsl_matrix_set(s.cholPxxi,j,k,0.0);

            // 1- Compute the sigma points,
            // Equation (3.170)
            // sigmapoint_j = x_(i-1)
            // sigmapoint_j = x_(i-1) + gamma * sqrt(P_i-1)_j for 1 <= j <= n
            // sigmapoint_j = x_(i-1) - gamma * sqrt(P_i-1)_(j-(n+1)) for n+1 <= j <= 2n
            gsl_matrix_set_col(s.sigmaPoints, 0, s.xi);
            for(j = 1 ; j < p.n+1 ; ++j)
                for(i = 0 ; i < p.n ; ++i)
                {
                    gsl_matrix_set(s.sigmaPoints,i,j, s.xi->data[i] + p.gamma * gsl_matrix_get(s.cholPxxi, i, j-1));
                    gsl_matrix_set(s.sigmaPoints,i,j+p.n, s.xi->data[i] - p.gamma * gsl_matrix_get(s.cholPxxi, i, j-1));
                }

            /**********************************/
            /***** Time update equations  *****/
            /**********************************/

            // Time update equations
            // 0 - Compute the image of the sigma points and the mean of these images
            gsl_vector_set_zero(s.xi_mean);
            for(j = 0 ; j < p.nbSamples ; ++j)
            {
                gsl_matrix_get_col(s.sigmaPoint, s.sigmaPoints, j);
                f(s.params, s.sigmaPoint, &gsl_matrix_column(s.xi_prediction,j).vector);

                // Update the mean, Eq (3.172)
                for(i = 0 ; i < p.n ; ++i)
                    s.xi_mean->data[i] += s.wm_j->data[j] * gsl_matrix_get(s.xi_prediction,i,j);
            }

            // 1 - Compute the covariance of the images and add the process noise,
            // Equation (3.173)
            // Warning, s.Pxxi will now hold P_xk^-
            gsl_matrix_set_zero(s.Pxxi);
            for(j = 0 ; j < p.nbSamples ; ++j)
            {
                for(i = 0 ; i < p.n ; ++i)
                    s.temp_n_1->data[i] = gsl_matrix_get(s.xi_prediction,i,j) - s.xi_mean->data[i];

                gsl_blas_dgemm(CblasNoTrans, CblasTrans, s.wc_j->data[j] , s.temp_n_1, s.temp_n_1, 0, s.temp_n_n);
                gsl_matrix_add(s.Pxxi, s.temp_n_n);
            }
            // Add the covariance of the evolution noise
            gsl_matrix_add(s.Pxxi, s.Pvvi);

            // Augment sigma points
            // Equation 3.174
            // First put the images of the initial sigma points
            gsl_matrix_memcpy(&gsl_matrix_submatrix(s.sigmaPointsMeasure, 0, 0, p.n, p.nbSamples).matrix, s.xi_prediction);
            // And add the additional sigma points eq. (7.56)
            for(j = 0 ; j < p.n ; ++j)
            {
                for(i = 0 ; i < p.n ; ++i)
                {
                    gsl_matrix_set(s.sigmaPointsMeasure, i, j+p.nbSamples, gsl_matrix_get(s.xi_prediction,i,0)+p.gamma_aug*gsl_matrix_get(s.cholPvvi,i,j));
                    gsl_matrix_set(s.sigmaPointsMeasure, i, j+p.nbSamples+p.n, gsl_matrix_get(s.xi_prediction,i,0)-p.gamma_aug*gsl_matrix_get(s.cholPvvi,i,j));
                }
            }

            // Compute the image of the sigma points through the observation equation
            // eq (3.175)
            gsl_vector_set_zero(yi);
            for(j = 0 ; j < p.nbSamplesMeasure ; ++j)
            {
                gsl_matrix_get_col(s.sigmaPointMeasure, s.sigmaPointsMeasure, j);
                h(s.sigmaPointMeasure, &gsl_matrix_column(s.yi_prediction,j).vector);

                // Update the mean , eq (3.176)
                for(i = 0 ; i < p.no ; ++i)
                    yi->data[i] += s.wm_aug_j->data[j] * gsl_matrix_get(s.yi_prediction,i,j);
            }
        }
    } // state
} // ukf


#endif // UKF_NDIM_STATE_H
