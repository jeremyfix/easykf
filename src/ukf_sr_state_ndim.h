/* ukf_sr_state_ndim.h
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

#define DEBUGMODE true

#ifndef UKF_SR_STATE_NDIM_H
#define UKF_SR_STATE_NDIM_H

#include <gsl/gsl_linalg.h> // For the Cholesky decomposition
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>

#include "ukf_types.h"
#include "ukf_math.h"

void print_mat(const gsl_matrix * A)
{
    for(unsigned int i = 0 ; i < A->size1 ; ++i)
    {
        for(unsigned int j = 0 ; j < A ->size2 ; ++j)
            printf("%e ", gsl_matrix_get(A,i,j));

        printf("\n");
    }
}

void print_vec(const gsl_vector * A)
{
    for(unsigned int i = 0 ; i < A->size ; ++i)
    {
        printf("%e ", gsl_vector_get(A,i));
    }
    printf("\n");
}

namespace ukf
{

    /**
      * @short MUST NOT BE USED !!!!!!! Square root UKF for state estimation, additive noise case 
      * The notations follow "Sigma-Point Kalman Filters for Probabilistic Inference in Dynamic State-Space Models",p115, PhD, van Der Merwe
      */
    namespace srstate
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

            // Images of the sigma points
            s.xi_prediction = gsl_matrix_alloc(p.n, p.nbSamples);
            gsl_matrix_set_zero(s.xi_prediction);

            s.xi_mean = gsl_vector_alloc(p.n);
            gsl_vector_set_zero(s.xi_mean);

            // Cholesky factor of the variance/covariance matrix Pxxi
            s.Sxi = gsl_matrix_alloc(p.n,p.n);
            gsl_matrix_set_identity(s.Sxi);
            gsl_matrix_scale(s.Sxi, sqrt(p.prior_x));

            // Cholesky factor of the variance/covariance matrix of the process noise
            s.Svi = gsl_matrix_alloc(p.n,p.n);
            gsl_matrix_set_identity(s.Svi);
            gsl_matrix_scale(s.Svi, sqrt(p.process_noise));

            s.yi_prediction = gsl_matrix_alloc(p.no, p.nbSamplesMeasure);
            gsl_matrix_set_zero(s.yi_prediction);

            s.yi_mean = gsl_vector_alloc(p.no);
            gsl_vector_set_zero(s.yi_mean);

            s.ino_yi = gsl_vector_alloc(p.no);
            gsl_vector_set_zero(s.ino_yi);

            // Cholesky factor of the variance/covariance matrix Pyyi
            s.Syi = gsl_matrix_alloc(p.no, p.no);
            gsl_matrix_set_zero(s.Syi);

            s.Sni = gsl_matrix_alloc(p.no,p.no);
            gsl_matrix_set_identity(s.Sni);
            gsl_matrix_scale(s.Sni, sqrt(p.measurement_noise));

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

            s.U = gsl_matrix_alloc(p.n, p.no);
            gsl_matrix_set_zero(s.U);

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
            s.temp_no = gsl_vector_alloc(p.no);

            s.temp_n_1 = gsl_matrix_alloc(p.n,1);
            s.temp_1_n = gsl_matrix_alloc(1,p.n);
            s.temp_n_n = gsl_matrix_alloc(p.n, p.n);
            s.temp_n_no = gsl_matrix_alloc(p.n, p.no);
            s.temp_no_1 = gsl_matrix_alloc(p.no,1);
            s.temp_1_no = gsl_matrix_alloc(1,p.no);
            s.temp_no_no = gsl_matrix_alloc(p.no, p.no);

            s.temp_3n_n = gsl_matrix_alloc(3*p.n, p.n);
            s.temp_2nno_no = gsl_matrix_alloc(2*p.n + p.no, p.no);
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
            gsl_matrix_free(s.Sxi);
            gsl_matrix_free(s.Svi);

            gsl_matrix_free(s.yi_prediction);
            gsl_vector_free(s.yi_mean);
            gsl_vector_free(s.ino_yi);
            gsl_matrix_free(s.Syi);
            gsl_matrix_free(s.Sni);

            gsl_matrix_free(s.Pxyi);

            gsl_vector_free(s.sigmaPoint);
            gsl_matrix_free(s.sigmaPoints);

            gsl_vector_free(s.sigmaPointMeasure);
            gsl_matrix_free(s.sigmaPointsMeasure);

            gsl_matrix_free(s.U);

            gsl_vector_free(s.wm_j);
            gsl_vector_free(s.wc_j);

            gsl_vector_free(s.wm_aug_j);
            gsl_vector_free(s.wc_aug_j);

            gsl_matrix_free(s.Ki);
            gsl_matrix_free(s.Ki_T);

            gsl_vector_free(s.temp_n);
            gsl_vector_free(s.temp_no);

            gsl_matrix_free(s.temp_n_1);
            gsl_matrix_free(s.temp_1_n);
            gsl_matrix_free(s.temp_n_n);

            gsl_matrix_free(s.temp_n_no);
            gsl_matrix_free(s.temp_no_1);
            gsl_matrix_free(s.temp_1_no);
            gsl_matrix_free(s.temp_no_no);

            gsl_matrix_free(s.temp_3n_n);
            gsl_matrix_free(s.temp_2nno_no);
        }

        /**
          * @short UKF-additive (zero-mean) noise case, "Kalman Filtering and Neural Networks", p.233
          *
          */
        template <typename FunctProcess, typename FunctObservation>
                inline void ukf_iterate(ukf_param &p, ukf_state &s, FunctProcess f, FunctObservation h, gsl_vector* yi)
        {
            int i,j;
	    gsl_matrix_view mat_view;
	    gsl_vector_view vec_view;
            // ************************************************** //
            // ************ Compute the sigma points ************ //
            // ************************************************** //

            // 1- Compute the sigma points,
            // Equation (3.209)
            // sigmapoint_j = x_(i-1)
            // sigmapoint_j = x_(i-1) + gamma * Sxi_j for 1 <= j <= n
            // sigmapoint_j = x_(i-1) - gamma * Sxi_(j-(n+1)) for n+1 <= j <= 2n
            gsl_matrix_set_col(s.sigmaPoints, 0, s.xi);
            for(j = 1 ; j < p.n + 1 ; ++j)
                for(i = 0 ; i < p.n ; ++i)
                {
                gsl_matrix_set(s.sigmaPoints,i,j, s.xi->data[i] + p.gamma * gsl_matrix_get(s.Sxi, i, j-1));
                gsl_matrix_set(s.sigmaPoints,i,j+p.n, s.xi->data[i] - p.gamma * gsl_matrix_get(s.Sxi, i, j-1));
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
                // Compute the image of the sigma points
                // Eq 3.210
		vec_view = gsl_matrix_column(s.xi_prediction,j);
                f(s.params, s.sigmaPoint, &vec_view.vector);

                // Update the mean, Eq (3.211)
                for(i = 0 ; i < p.n ; ++i)
                    s.xi_mean->data[i] += s.wm_j->data[j] * gsl_matrix_get(s.xi_prediction,i,j);
            }

            // 1- Compute the QR decomposition and keeps only the upper triangular part of R
            // Eq. 3.212
            // We first fill the matrix Sxk^-
            // with the transpose of what is in the brackets of qr { ... }
            for(j = 1 ; j < p.nbSamples; ++j)
            {
                for(i = 0 ; i < p.n ; ++i)
                {
                    gsl_matrix_set(s.temp_3n_n, j-1, i, sqrt(s.wc_j->data[1]) * (gsl_matrix_get(s.xi_prediction,i,j) - s.xi_mean->data[i]));
                }
            }
            // The last rows are filled with the cholesky factor of the process noise covariance
            for(j = 0 ; j < p.n ; ++j)
            {
	      mat_view = gsl_matrix_submatrix(s.temp_3n_n, p.nbSamples-1, 0, p.n, p.n);
                gsl_matrix_memcpy(&mat_view.matrix, s.Svi);
            }
            // We now perform the QR decomposition of s.temp_3n_n;
            gsl_linalg_QR_decomp(s.temp_3n_n, s.temp_n);
            // From this QR decomposition, we keep only the upper triangular part of R and copy it in s.Sxi
            for( j = 0 ; j < p.n ; ++j)
            {
                for(i = 0 ; i < j+1 ; ++i)
                    gsl_matrix_set(s.Sxi, i, j, gsl_matrix_get(s.temp_3n_n, i, j));

                for(i = j+1 ; i < p.n ; ++i)
                    gsl_matrix_set(s.Sxi, i, j, 0.0);
            }
            //gsl_matrix_transpose(s.Sxi);
            if(DEBUGMODE) print_mat(s.Sxi);
            if(DEBUGMODE) printf("\n");

            // 2- Perform the cholupdate, which is a rank one update
            // Eq. 3.213
            for(i = 0 ; i < p.n ; ++i)
                s.temp_n->data[i] = gsl_matrix_get(s.xi_prediction,i,0) - s.xi_mean->data[i];

            if(DEBUGMODE) printf("Before cholupdate 1 \n");
            gsl_matrix_transpose(s.Sxi);
            ukf::math::choleskyUpdate(s.Sxi, ukf::math::signof(s.wc_j->data[0]) * sqrt(fabs(s.wc_j->data[0])), s.temp_n);
            // Warning : Here, we keep Sxi as a triangular inferior matrix

            //gsl_matrix_transpose(s.Sxi);
            if(DEBUGMODE) printf("ok\n");
            if(DEBUGMODE) printf("Sxi after cholupdate : \n");
            if(DEBUGMODE) print_mat(s.Sxi);
            // ** Augment the sigma points ** //
            // Eq 3.124

            // 1 - Copy the images of the original sigma points
	    mat_view = gsl_matrix_submatrix(s.sigmaPointsMeasure,0,0, p.n, p.nbSamples);
            gsl_matrix_memcpy(&mat_view.matrix, s.xi_prediction);

            // 2 - Compute the additional sigma points
            for(j = 0 ; j < p.n ; ++j)
            {
                for(i = 0 ; i < p.n ; ++i)
                {
                    gsl_matrix_set(s.sigmaPointsMeasure, i ,j + p.nbSamples , gsl_matrix_get(s.xi_prediction, i, 0) + p.gamma_aug * gsl_matrix_get(s.Svi, i,j));
                    gsl_matrix_set(s.sigmaPointsMeasure, i ,j + p.nbSamples + p.n , gsl_matrix_get(s.xi_prediction, i, 0) - p.gamma_aug * gsl_matrix_get(s.Svi, i,j));
                }
            }

            // Compute the image of the sigma points, i.e. the associated predicted observations
            // and their mean
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

            // Compute the Cholesky factor of the variance-covariance matrix of the observations
            // Eq. 3.217
            // First put in s.temp_2nno_no the transpose of what is in the brackets of qr{...} ,
            for(j = 0 ; j < 2*p.n ; ++j)
                for(i = 0 ; i < p.no ; ++i)
                    gsl_matrix_set(s.temp_2nno_no, j, i, sqrt(s.wc_aug_j->data[1])*(gsl_matrix_get(s.yi_prediction, i, j+1) - s.yi_mean->data[i]) );

            // Copy the Cholesky factor of the covariance of the observation noise
	    mat_view = gsl_matrix_submatrix(s.temp_2nno_no,2*p.n, 0, p.no, p.no);
            gsl_matrix_memcpy(&mat_view.matrix, s.Sni);

            // We now perform the QR decomposition of s.temp_2nno_no;
            gsl_linalg_QR_decomp(s.temp_2nno_no, s.temp_no);
            // From this QR decomposition, we keep only the upper triangular part of R and copy it in s.Syi
            for( j = 0 ; j < p.no ; ++j)
            {
                for(i = 0 ; i < j+1 ; ++i)
                    gsl_matrix_set(s.Syi, i, j, gsl_matrix_get(s.temp_2nno_no, i, j));

                for(i = j+1 ; i < p.no ; ++i)
                    gsl_matrix_set(s.Syi, i, j, 0.0);
            }
            if(DEBUGMODE) printf("Syi : \n");
            if(DEBUGMODE) print_mat(s.Syi);
            if(DEBUGMODE) printf("\n");
            //gsl_matrix_transpose(s.Syi);

            // 2- Perform the cholupdate, which is a rank one update
            // Eq. 3.218
            // Syi is triangular superior
            for(i = 0 ; i < p.no ; ++i)
                s.temp_no->data[i] = gsl_matrix_get(s.yi_prediction,i,0) - s.yi_mean->data[i];
            if(DEBUGMODE) print_vec(s.temp_no);
            if(DEBUGMODE) printf("Before cholupdate 2 \n");
            gsl_matrix_transpose(s.Syi);
            ukf::math::choleskyUpdate(s.Syi, ukf::math::signof(s.wc_aug_j->data[0])*sqrt(fabs(s.wc_aug_j->data[0])), s.temp_no);
            if(DEBUGMODE) printf("ok \n");
            // Now, Syi is the lower triangular cholesky factor
            //gsl_matrix_transpose(s.Syi);
            if(DEBUGMODE) printf("Syi after cholupdate : \n");
            if(DEBUGMODE) print_mat(s.Syi);


            // Compute the state/observation covariance
            // Eq (3.219)
            gsl_matrix_set_zero(s.Pxyi);
            for(j = 0 ; j < p.nbSamplesMeasure ; ++j)
            {
                for(i = 0 ; i < p.n ; ++i)
                    s.temp_n_1->data[i] = gsl_matrix_get(s.sigmaPointsMeasure,i,j) - s.xi_mean->data[i];

                for(i = 0 ; i < p.no ; ++i)
                    s.temp_1_no->data[i] = gsl_matrix_get(s.yi_prediction,i,j) - s.yi_mean->data[i];

                gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, s.wc_aug_j->data[j] , s.temp_n_1, s.temp_1_no, 1.0, s.Pxyi);
                //gsl_matrix_add(s.Pxyi, s.temp_n_no);
            }

            /*gsl_matrix_set_zero(s.Pxyi);
            for(j = 0 ; j < p.nbSamples ; ++j)
            {
                for(i = 0 ; i < p.n ; ++i)
                    s.temp_n_1->data[i] = gsl_matrix_get(s.xi_prediction,i,j) - s.xi_mean->data[i];

                for(i = 0 ; i < p.no ; ++i)
                    s.temp_1_no->data[i] = gsl_matrix_get(s.yi_prediction,i,j) - s.yi_mean->data[i];

                gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, s.wc_j->data[j] , s.temp_n_1, s.temp_1_no, 1.0, s.Pxyi);
                //gsl_matrix_add(s.Pxyi, s.temp_n_no);
            }*/

            // Compute the Kalman gain,
            // Eq (3.220)

            // We want to solve : Kxy . Sy Sy^T = Pxy
            // Which can be solved using : Sy Sy^T Kxy^T = Pxy^T
            // We have a number of no systems to solve of the form S x = Pxy^T[:,i] ; S^T Kxy[:,i] = x with S being triangular
            // To solve this, we make use of gsl_blas_dtrsv
            for(int i = 0 ; i < p.n ; ++i)
            {
	      vec_view = gsl_matrix_row(s.Pxyi, i);
                gsl_vector_memcpy(s.temp_no, &vec_view.vector);
                gsl_blas_dtrsv(CblasLower,CblasNoTrans,CblasNonUnit,s.Syi,s.temp_no);
                gsl_blas_dtrsv(CblasLower,CblasTrans,CblasNonUnit,s.Syi,s.temp_no);
                // And then copy the result in the ith row of Ki
		vec_view = gsl_matrix_row(s.Ki, i);
                gsl_vector_memcpy(&vec_view.vector, s.temp_no);
            }
            if(DEBUGMODE) printf("Ki : \n");
            if(DEBUGMODE) print_mat(s.Ki);

            // Correction : correct the estimation of the state
            // Eq. 3.221
            // Compute the innovations
            for(i = 0 ; i < p.no ; ++i)
                s.ino_yi->data[i] = yi->data[i] - s.yi_mean->data[i];
            // And correct the current state estimate
            gsl_blas_dgemv(CblasNoTrans, 1.0, s.Ki, s.ino_yi, 1.0, s.xi);

            // Compute the matrix U = Kk Syk
            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0 , s.Ki, s.Syi, 0, s.U);
            if(DEBUGMODE) printf("Matrix U : \n");
            if(DEBUGMODE) print_mat(s.U);

            // Perform the cholesky updates of Sxi
            // We perform p.no updates rank-1 update with the columns of U
            if(DEBUGMODE) printf("Before cholupdate 3 \n");
            for(i = 0 ; i < p.no ; ++i)
            {
	      vec_view = gsl_matrix_column(s.U, i);
                gsl_vector_memcpy(s.temp_n, &vec_view.vector);
                ukf::math::choleskyUpdate(s.Sxi, -1.0, s.temp_n);
                if(DEBUGMODE) printf("After cholupdate %i\n", i);
                if(DEBUGMODE) print_mat(s.Sxi);
            }

            //printf("After cholupdate %i\n", i);
            //print_mat(s.Sxi);
            //if(DEBUGMODE) printf("\n");

            //
            //            // Update of the process noise
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

            //          int i,j,k;
            //            // ************************************************** //
            //            // ************ Compute the sigma points ************ //
            //            // ************************************************** //
            //
            //            // 0 - Compute the Cholesky decomposition of s.Pxxi
            //            gsl_matrix_memcpy(s.cholPxxi, s.Pxxi);
            //            gsl_linalg_cholesky_decomp(s.cholPxxi);
            //            // Set all the elements of cholPvvi strictly above the diagonal to zero
            //            for(j = 0 ; j < p.n ; j++)
            //                for(k = j+1 ; k < p.n ; k++)
            //                    gsl_matrix_set(s.cholPxxi,j,k,0.0);
            //
            //            // 1- Compute the sigma points,
            //            // Equation (3.170)
            //            // sigmapoint_j = x_(i-1)
            //            // sigmapoint_j = x_(i-1) + gamma * sqrt(P_i-1)_j for 1 <= j <= n
            //            // sigmapoint_j = x_(i-1) - gamma * sqrt(P_i-1)_(j-(n+1)) for n+1 <= j <= 2n
            //            gsl_matrix_set_col(s.sigmaPoints, 0, s.xi);
            //            for(j = 1 ; j < p.n+1 ; ++j)
            //                for(i = 0 ; i < p.n ; ++i)
            //                {
            //                    gsl_matrix_set(s.sigmaPoints,i,j, s.xi->data[i] + p.gamma * gsl_matrix_get(s.cholPxxi, i, j-1));
            //                    gsl_matrix_set(s.sigmaPoints,i,j+p.n, s.xi->data[i] - p.gamma * gsl_matrix_get(s.cholPxxi, i, j-1));
            //                }
            //
            //            /**********************************/
            //            /***** Time update equations  *****/
            //            /**********************************/
            //
            //            // Time update equations
            //            // 0 - Compute the image of the sigma points and the mean of these images
            //            gsl_vector_set_zero(s.xi_mean);
            //            for(j = 0 ; j < p.nbSamples ; ++j)
            //            {
            //                gsl_matrix_get_col(s.sigmaPoint, s.sigmaPoints, j);
            //                f(s.params, s.sigmaPoint, &gsl_matrix_column(s.xi_prediction,j).vector);
            //
            //                // Update the mean, Eq (3.172)
            //                for(i = 0 ; i < p.n ; ++i)
            //                    s.xi_mean->data[i] += s.wm_j->data[j] * gsl_matrix_get(s.xi_prediction,i,j);
            //            }
            //
            //            // 1 - Compute the covariance of the images and add the process noise,
            //            // Equation (3.173)
            //            // Warning, s.Pxxi will now hold P_xk^-
            //            gsl_matrix_set_zero(s.Pxxi);
            //            for(j = 0 ; j < p.nbSamples ; ++j)
            //            {
            //                for(i = 0 ; i < p.n ; ++i)
            //                    s.temp_n_1->data[i] = gsl_matrix_get(s.xi_prediction,i,j) - s.xi_mean->data[i];
            //
            //                gsl_blas_dgemm(CblasNoTrans, CblasTrans, s.wc_j->data[j] , s.temp_n_1, s.temp_n_1, 0, s.temp_n_n);
            //                gsl_matrix_add(s.Pxxi, s.temp_n_n);
            //            }
            //            // Add the covariance of the evolution noise
            //            gsl_matrix_add(s.Pxxi, s.Pvvi);
            //
            //            // Augment sigma points
            //            // Equation 3.174
            //            // First put the images of the initial sigma points
            //            gsl_matrix_memcpy(&gsl_matrix_submatrix(s.sigmaPointsMeasure, 0, 0, p.n, p.nbSamples).matrix, s.xi_prediction);
            //            // And add the additional sigma points eq. (7.56)
            //            for(j = 0 ; j < p.n ; ++j)
            //            {
            //                for(i = 0 ; i < p.n ; ++i)
            //                {
            //                    gsl_matrix_set(s.sigmaPointsMeasure, i, j+p.nbSamples, gsl_matrix_get(s.xi_prediction,i,0)+p.gamma_aug*gsl_matrix_get(s.cholPvvi,i,j));
            //                    gsl_matrix_set(s.sigmaPointsMeasure, i, j+p.nbSamples+p.n, gsl_matrix_get(s.xi_prediction,i,0)-p.gamma_aug*gsl_matrix_get(s.cholPvvi,i,j));
            //                }
            //            }
            //
            //            // Compute the image of the sigma points through the observation equation
            //            // eq (3.175)
            //            gsl_vector_set_zero(yi);
            //            for(j = 0 ; j < p.nbSamplesMeasure ; ++j)
            //            {
            //                gsl_matrix_get_col(s.sigmaPointMeasure, s.sigmaPointsMeasure, j);
            //                h(s.sigmaPointMeasure, &gsl_matrix_column(s.yi_prediction,j).vector);
            //
            //                // Update the mean , eq (3.176)
            //                for(i = 0 ; i < p.no ; ++i)
            //                    yi->data[i] += s.wm_aug_j->data[j] * gsl_matrix_get(s.yi_prediction,i,j);
            //            }
        }
    } // srstate
} // ukf

#endif // UKF_SR_STATE_NDIM_H
