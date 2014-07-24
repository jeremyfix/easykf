/* ukf_types.h
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

#ifndef UKF_TYPES_H
#define UKF_TYPES_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>
#include "ukf_math.h"

namespace ukf
{

    /**
      * @short The different types of implemented process noise for UKF state estimation
      */
    typedef enum
    {
        /**
          * @brief The covariance of the evolution noise is fixed to \f$\textbf{P}_{\theta\theta_i} = \alpha . \textbf{I}\f$
          */
        UKF_PROCESS_FIXED,
        /**
          * @brief The covariance of the evolution noise is defined as \f$\textbf{P}_{\theta\theta_i} = (\alpha^{-1} - 1)\textbf{P}_i\f$
          */
        UKF_PROCESS_RLS
    } ProcessNoise;      

    namespace parameter
    {
        class EvolutionNoise;
        /**
          * @short Structure holding the parameters of the Unscented Kalman Filter
          *
          */
        typedef struct
        {
            /**
              * @short \f$\kappa \geq 0\f$, \f$\kappa = 0\f$ is a good choice. According to van der Merwe, its value is not critical
              */
            double kpa;

            /**
              * @short \f$0 \leq \alpha \leq 1\f$ : "Size" of sigma-point distribution. Should be small if the function is strongly non-linear
              */
            double alpha;

            /**
              * @short Non negative weights used to introduce knowledge about the higher order moments of the distribution. For gaussian distributions, \f$\beta = 2\f$ is a good choice
              */
            double beta;

            /**
              * @short \f$\lambda = \alpha^2 (n + \kappa) - n\f$
              */
            double lambda;

            /**
              * @short \f$\gamma = \sqrt{\lambda + n}\f$
              */
            double gamma;

            /**
              * @short Parameter used for the evolution noise
              */
            //double evolution_noise_parameter;

            /**
              * @short Initial value of the evolution noise
              */
            //double evolution_noise;

            /**
              * @short Evolution noise type
              */
            //EvolutionNoise evolution_noise_type;
            EvolutionNoise * evolution_noise;

            /**
              * @short Covariance of the observation noise
              */
            double observation_noise;

            /**
              * @short Prior estimate of the covariance matrix
              */
            double prior_pi;

            /**
              * @short Number of parameters to estimate
              */
            int n;

            /**
              * @short \f$nbSamples = (2 n + 1)\f$ Number of sigma-points
              */
            int nbSamples;

            /**
              * @short Dimension of the output
              */
            int no;

        } ukf_param;

        /**
          * @short Pointer to the function to approximate in the scalar case
          *
          */
        //typedef double (*ukf_function_scalar) (gsl_vector * param, gsl_vector * input);

        /**
          * @short Structure holding the matrices manipulated by the statistical linearization
          * in the scalar case
          *
          */
        typedef struct
        {
            /**
              * @short Kalman gain, a vector of size \f$n\f$
              */
            gsl_vector * Kk;

            /**
              * @short Temporary matrix for the kalman gain, a matrix of size \f$(n,1)\f$
              */
            gsl_matrix * Kk_mat;

            /**
              * @short Temporary matrix for the transpose of the kalman gain, a matrix of size \f$(1,n)\f$
              */
            gsl_matrix * Kk_mat_T;

            /**
             * @short Covariance of \f$(w, d)\f$: \f$P_{w_k d_k}\f$, of size \f$n\f$
             */
            gsl_vector * Pwdk;

            /**
              * @short Covariance of the evolution noise
              */
            gsl_matrix * Prrk;

            /**
              * @short Covariance of the observation noise
              */
            double Peek;

            /**
              * @short Covariance of the output
              */
            double Pddk;

            /**
              * @short Parameter vector, of size \f$n\f$
              */
            gsl_vector * w;

            /**
              * @short Temporary vector, holding a sigma-point
              */
            gsl_vector * wk;

            /**
              * @short Covariance matrix of the parameters, of size \f$(n,n)\f$
              */
            gsl_matrix * Pk;

            /**
              * @short Matrix holding the Cholesky decomposition of \f$P_k\f$, of size \f$(n,n)\f$
              */
            gsl_matrix * Sk;

            /**
              * @short Vector holding one column of Sk, of size \f$n\f$
              */
            gsl_vector * cSk;

            /**
              * @short Weights used to compute the mean of the sigma points' images
              * @brief \f$wm_0 = \frac{\lambda}{n + \lambda}\f$
              *        \f$wm_i = \frac{1}{2(n + \lambda)}\f$
              */
            gsl_vector * wm;

            /**
              * @short Weights used to update the covariance matrices
              * @brief \f$wc_0 = \frac{\lambda}{n + \lambda} + (1 - \alpha^2 + \beta)\f$
              *        \f$wc_i = \frac{1}{2(n + \lambda)}\f$
              */
            gsl_vector * wc;

            /**
              * @short Temporary vector holding the image of the sigma points, of size \f$nbSamples\f$
              */
            gsl_vector * dk;

            /**
              * @short Innovation
              */
            double ino_dk;

            /**
              * @short Variable holding the mean of the sigma points image
              */
            double d_mean;

            /**
              * @short Matrix holding the sigma points in the columns, of size \f$(n,nbSamples)\f$
              */
            gsl_matrix * sigmaPoints;

            /**
              * @short Temporary vector
              */
            gsl_vector * temp_n;

            /**
              * @short Temporary matrix
              */
            gsl_matrix * temp_n_n;

        } ukf_scalar_state;

        /**
          * @short Structure holding the matrices manipulated by the unscented kalman filter
          *  in the vectorial case, for Parameter estimation
          *
          */
        typedef struct
        {
            /**
              * @short Kalman gain, a matrix of size \f$n \times no\f$
              */
            gsl_matrix * Kk;

            /**
              * @short The tranposed Kalman gain, a vector of size \f$no \times n\f$
              */
            gsl_matrix * Kk_T;

            /**
             * @short Covariance of \f$(w, d)\f$: \f$P_{w_k d_k}\f$, of size \f$n \times no\f$
             */
            gsl_matrix * Pwdk;

            /**
              * @short Covariance of the output, a matrix of size \f$ no \times no \f$
              */
            gsl_matrix * Pddk;

            /**
              * @short Covariance of the observation noise, of size \f$ no \times no \f$
              */
            gsl_matrix * Peek;

            /**
              * @short Covariance of the evolution noise, of size \f$ n \times n \f$
              */
            gsl_matrix * Prrk;

            /**
              * @short Parameter vector, of size \f$n\f$
              */
            gsl_vector * w;

            /**
              * @short Temporary vector holding one sigma point, of size \f$n\f$
              */
            gsl_vector * wk;

            /**
              * @short Covariance matrix of the parameters, of size \f$(n,n)\f$
              */
            gsl_matrix * Pk;

            /**
              * @short Matrix holding the Cholesky decomposition of \f$P_k\f$, of size \f$(n,n)\f$
              */
            gsl_matrix * Sk;

            /**
              * @short Vector holding one column of Sk, of size \f$n\f$
              */
            gsl_vector * cSk;

            /**
              * @short Weights used to compute the mean of the sigma points' images
              * @brief \f$wm_0 = \frac{\lambda}{n + \lambda}\f$
              *        \f$wm_i = \frac{1}{2(n + \lambda)}\f$
              */
            gsl_vector * wm;

            /**
              * @short Weights used to update the covariance matrices
              * @brief \f$wc_0 = \frac{\lambda}{n + \lambda} + (1 - \alpha^2 + \beta)\f$
              *        \f$wc_i = \frac{1}{2(n + \lambda)}\f$
              */
            gsl_vector * wc;

            /**
              * @short Temporary matrix holding the image of the sigma points, of size \f$ no \times nbSamples\f$
              */
            gsl_matrix * dk;

            /**
              * @short Vector holding the mean of the sigma points image, of size \f$ no \f$ŝ
              */
            gsl_vector * d_mean;

            /**
              * @short Vector holding the inovation, of size \f$ no \f$ŝ
              */
            gsl_vector * ino_dk;

            /**
              * @short Matrix holding the sigma points in the columns, of size \f$(n,nbSamples)\f$
              */
            gsl_matrix * sigmaPoints;

            /**
              * @short Temporary vector of size \f$ n \f$
              */
            gsl_vector * vec_temp_n;

            /**
              * @short Temporary vector of size \f$ no \f$
              */
            gsl_vector * vec_temp_output;

            /**
              * @short Temporary matrix of size \f$ n \times 1 \f$
              */
            gsl_matrix * mat_temp_n_1;

            /**
              * @short Temporary matrix of size \f$ n \times no \f$
              */
            gsl_matrix * mat_temp_n_output;

            /**
              * @short Temporary matrix of size \f$ no \times n \f$
              */
            gsl_matrix * mat_temp_output_n;

            /**
              * @short Temporary matrix of size \f$ 1 \times no \f$
              */
            gsl_matrix * mat_temp_1_output;

            /**
              * @short Temporary matrix of size \f$ no \times 1 \f$
              */
            gsl_matrix * mat_temp_output_1;

            /**
              * @short Temporary matrix of size \f$ no \times no \f$
              */
            gsl_matrix * mat_temp_output_output;

            /**
              * @short Temporary matrix of size \f$ n \times n \f$
              */
            gsl_matrix * mat_temp_n_n;
        } ukf_state;

        /**
          * @short Mother class from which the evolution noises inherit
          */
        class EvolutionNoise
        {
        protected:
            double _initial_value;
        public:
            EvolutionNoise(double initial_value) : _initial_value(initial_value) {};
            void init(ukf_param &p, ukf_state &s)
            {
                gsl_matrix_set_identity(s.Prrk);
                gsl_matrix_scale(s.Prrk, _initial_value);
            }
            void init(ukf_param &p, ukf_scalar_state &s)
            {
                gsl_matrix_set_identity(s.Prrk);
                gsl_matrix_scale(s.Prrk, _initial_value);
            }

            virtual void updateEvolutionNoise(ukf_param &p, ukf_state &s) = 0;
            virtual void updateEvolutionNoise(ukf_param &p, ukf_scalar_state &s) = 0;
        };

        /**
          * @short Annealing type evolution noise
          */
        class EvolutionAnneal : public EvolutionNoise
        {
            double _decay, _lower_bound;
        public:
            EvolutionAnneal(double initial_value, double decay, double lower_bound) : EvolutionNoise(initial_value),
            _decay(decay),
            _lower_bound(lower_bound)
            { };

            void updateEvolutionNoise(ukf_param &p, ukf_state &s)
            {
                for(int i = 0 ; i < p.n ; ++i)
                {
                    for(int j = 0 ; j < p.n ; ++j)
                    {
                        if(i == j)
                            gsl_matrix_set(s.Prrk, i, i, ukf::math::max(_decay * gsl_matrix_get(s.Prrk,i,i),_lower_bound));
                        else
                            gsl_matrix_set(s.Prrk, i, j, 0.0);
                    }
                }
            }
            void updateEvolutionNoise(ukf_param &p, ukf_scalar_state &s)
            {
                for(int i = 0 ; i < p.n ; ++i)
                {
                    for(int j = 0 ; j < p.n ; ++j)
                    {
                        if(i == j)
                            gsl_matrix_set(s.Prrk, i, i, ukf::math::max(_decay * gsl_matrix_get(s.Prrk,i,i),_lower_bound));
                        else
                            gsl_matrix_set(s.Prrk, i, j, 0.0);
                    }
                }
            }
        };

        /**
          * @short Forgetting type evolution noise
          */
        class EvolutionRLS : public EvolutionNoise
        {
            double _decay;
        public:
            EvolutionRLS(double initial_value, double decay) : EvolutionNoise(initial_value), _decay(decay)
            {
                if(ukf::math::cmp_equal(_decay, 0.0))
                    printf("Forgetting factor should not be null !!\n");
            };

            void updateEvolutionNoise(ukf_param &p, ukf_state &s)
            {
                gsl_matrix_memcpy(s.Prrk, s.Pk);
                gsl_matrix_scale(s.Prrk, 1.0 / _decay - 1.0);
            }
            void updateEvolutionNoise(ukf_param &p, ukf_scalar_state &s)
            {
                gsl_matrix_memcpy(s.Prrk, s.Pk);
                gsl_matrix_scale(s.Prrk, 1.0 / _decay - 1.0);
            }
        };

        /**
          * @short Robbins-Monro evolution noise
          */
        class EvolutionRobbinsMonro : public EvolutionNoise
        {
            double _alpha;
        public:
            EvolutionRobbinsMonro(double initial_value, double alpha) : EvolutionNoise(initial_value),
            _alpha(alpha)
            { };

            void updateEvolutionNoise(ukf_param &p, ukf_state &s)
            {
                // Compute Kk * ino_yk
	      gsl_matrix_view mat_view = gsl_matrix_view_array(s.ino_dk->data, p.no, 1);
	      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, s.Kk, &mat_view.matrix, 0.0, s.mat_temp_n_1);
	      // Compute : Prrk = (1 - alpha) Prrk + alpha . (Kk * ino_dk) * (Kk * ino_dk)^T
	      gsl_blas_dgemm(CblasNoTrans, CblasTrans, _alpha, s.mat_temp_n_1, s.mat_temp_n_1, (1.0 - _alpha),s.Prrk);
            }
            void updateEvolutionNoise(ukf_param &p, ukf_scalar_state &s)
            {
                // Compute  Prrk = (1 - alpha) Prrk + alpha . ino_dk^2 Kk Kk^T
	      gsl_matrix_view mat_view = gsl_matrix_view_array(s.Kk->data,p.n,1);
                gsl_blas_dgemm(CblasNoTrans, CblasTrans, gsl_pow_2(s.ino_dk) * _alpha, &mat_view.matrix, &mat_view.matrix, 1.0 - _alpha, s.Prrk);
            }
        };


    } // parameter

    namespace state
    {

        class EvolutionNoise;
        /**
          * @short Structure holding the parameters of the statistical linearization
          *
          */
        typedef struct
        {
            /**
              * @short \f$\kappa \geq 0\f$, \f$\kappa = 0\f$ is a good choice. According to van der Merwe, its value is not critical
              */
            double kpa;

            /**
              * @short \f$0 \leq \alpha \leq 1\f$ : "Size" of sigma-point distribution. Should be small if the function is strongly non-linear
              */
            double alpha;

            /**
              * @short Non negative weights used to introduce knowledge about the higher order moments of the distribution. For gaussian distributions, \f$\beta = 2\f$ is a good choice
              */
            double beta;

            /**
              * @short \f$\lambda = \alpha^2 (n + \kappa) - n\f$
              */
            double lambda;
            double lambda_aug;

            /**
              * @short \f$\gamma = \sqrt{\lambda + n}\f$
              */
            double gamma;
            double gamma_aug;

            /**
              * @short Size of the state vector
              */
            int n;

            /**
              * @short \f$nbSamples = (2 n + 1)\f$ Number of sigma-points
              */
            int nbSamples;

            /**
              * @short \f$nbSamplesMeasure = (4 n + 1)\f$ Number of sigma-points
              */
            int nbSamplesMeasure;

            /**
              * @short Dimension of the output : the measurements
              */
            int no;

            /**
              * @short Prior estimate of the covariance matrix of the state
              */
            double prior_x;

            /**
              * @short Type of process noise
              */
            EvolutionNoise * evolution_noise;

            /**
              * @short Parameter used for the evolution noise
              */
            //double process_noise;

            /**
              * @short Covariance of the observation noise
              */
            double measurement_noise;

        } ukf_param;

        /**
          * @short Structure holding the matrices manipulated by the statistical linearization
          *  in the vectorial case for state estimation
          *
          */
        typedef struct
        {
            gsl_vector * params; // The parameters of the process equation or anything you need to give to the process function

            gsl_vector * xi; // State vector
            gsl_matrix * xi_prediction; // Prediction of the states for each sigma-point
            gsl_vector * xi_mean; // Mean state vector
            gsl_matrix * Pxxi; // State covariance matrix
            gsl_matrix * cholPxxi; // Cholesky decomposition of Pxxi
            gsl_matrix * Pvvi; // Process noise covariance
            gsl_matrix * cholPvvi; // Cholesky decomposition of the process noise

            gsl_matrix * yi_prediction; // Prediction of the measurements for each sigma-point
            gsl_vector * yi_mean; // Mean measurement vector
            gsl_vector * ino_yi; // Innovations
            gsl_matrix * Pyyi; // Measurement covariance matrix
            gsl_matrix * Pnni; // Measurement noise covariance

            gsl_matrix * Pxyi; // State-Measurement covariance matrix

            gsl_vector * sigmaPoint; // Vector holding one sigma point
            gsl_matrix * sigmaPoints;  // Matrix holding all the sigma points

            gsl_vector * sigmaPointMeasure; // one sigma point for the observation equation
            gsl_matrix * sigmaPointsMeasure; // Sigma points for the observation equation

            gsl_vector * wm_j; // Weights used to compute the mean of the sigma points images
            gsl_vector * wc_j; // Weights used to update the covariance matrices

            gsl_vector * wm_aug_j; // Augmented set of Weights used to compute the mean of the sigma points images for the observations
            gsl_vector * wc_aug_j; // Augmented set of Weights used to update the covariance matrices of the observations

            gsl_matrix * Ki; // Kalman gain
            gsl_matrix * Ki_T; // Its transpose

            gsl_vector * temp_n;
            gsl_matrix * temp_n_n;
            gsl_matrix * temp_n_1;
            gsl_matrix * temp_1_n;

            gsl_matrix * temp_n_no;

            gsl_matrix * temp_no_no;
            gsl_matrix * temp_no_1;
            gsl_matrix * temp_1_no;
        } ukf_state;

        /**
          * @short Mother class from which the evolution noises inherit
          */
        class EvolutionNoise
        {
        protected:
            double _initial_value;
        public:
            EvolutionNoise(double initial_value) : _initial_value(initial_value) {};
            void init(ukf_param &p, ukf_state &s)
            {
                gsl_matrix_set_identity(s.Pvvi);
                gsl_matrix_scale(s.Pvvi, _initial_value);
                gsl_matrix_set_identity(s.cholPvvi);
                gsl_matrix_scale(s.cholPvvi, sqrt(_initial_value));
            }

            virtual void updateEvolutionNoise(ukf_param &p, ukf_state &s) = 0;
        };

        /**
          * @short Annealing type evolution noise
          */
        class EvolutionAnneal : public EvolutionNoise
        {
            double _decay, _lower_bound;
        public:
            EvolutionAnneal(double initial_value, double decay, double lower_bound) : EvolutionNoise(initial_value),
            _decay(decay),
            _lower_bound(lower_bound)
            { };

            void updateEvolutionNoise(ukf_param &p, ukf_state &s)
            {
                double value;
                for(int i = 0 ; i < p.n ; ++i)
                {
                    for(int j = 0 ; j < p.n ; ++j)
                    {
                        if(i == j)
                        {
                            value = ukf::math::max(_decay * gsl_matrix_get(s.Pvvi,i,i),_lower_bound);
                            gsl_matrix_set(s.Pvvi, i, i, value);
                            gsl_matrix_set(s.cholPvvi, i ,i, sqrt(value));
                        }
                        else
                        {
                            gsl_matrix_set(s.Pvvi, i, j, 0.0);
                            gsl_matrix_set(s.cholPvvi, i, j, 0.0);
                        }
                    }
                }
            }
        };

        /**
          * @short Forgetting type evolution noise
          */
        class EvolutionRLS : public EvolutionNoise
        {
            double _decay;
        public:
            EvolutionRLS(double initial_value, double decay) : EvolutionNoise(initial_value), _decay(decay)
            {
                if(ukf::math::cmp_equal(_decay, 0.0))
                    printf("Forgetting factor should not be null !!\n");
            };

            void updateEvolutionNoise(ukf_param &p, ukf_state &s)
            {
                gsl_matrix_memcpy(s.Pvvi, s.Pxxi);
                gsl_matrix_scale(s.Pvvi, 1.0 / _decay - 1.0);
                gsl_matrix_memcpy(s.cholPvvi, s.Pvvi);
                gsl_linalg_cholesky_decomp(s.cholPvvi);
                // Set all the elements of cholPvvi strictly above the diagonal to zero
                for(int j = 0 ; j < p.n ; j++)
                    for(int k = j+1 ; k < p.n ; k++)
                         gsl_matrix_set(s.cholPvvi,j,k,0.0);
            }
        };

        /**
          * @short Robbins-Monro evolution noise
          */
        class EvolutionRobbinsMonro : public EvolutionNoise
        {
            double _alpha;
        public:
            EvolutionRobbinsMonro(double initial_value, double alpha) : EvolutionNoise(initial_value),
            _alpha(alpha)
            { };

            void updateEvolutionNoise(ukf_param &p, ukf_state &s)
            {
                // Compute Kk * ino_yk
                gsl_blas_dgemv(CblasNoTrans, 1.0, s.Ki, s.ino_yi, 0.0, s.temp_n);
                // Compute : Prrk = (1 - alpha) Prrk + alpha . (Kk * ino_dk) * (Kk * ino_dk)^T
		gsl_matrix_view mat_view = gsl_matrix_view_array(s.temp_n->data,p.n,1);
                gsl_blas_dgemm(CblasNoTrans, CblasTrans, _alpha, &mat_view.matrix, &mat_view.matrix, (1.0 - _alpha),s.Pvvi);
            }
        };


    } // namespace for state estimation

    namespace srstate
    {


        /**
          * @short Structure holding the parameters of the statistical linearization
          *
          */
        typedef struct
        {
            /**
              * @short \f$\kappa \geq 0\f$, \f$\kappa = 0\f$ is a good choice. According to van der Merwe, its value is not critical
              */
            double kpa;

            /**
              * @short \f$0 \leq \alpha \leq 1\f$ : "Size" of sigma-point distribution. Should be small if the function is strongly non-linear
              */
            double alpha;

            /**
              * @short Non negative weights used to introduce knowledge about the higher order moments of the distribution. For gaussian distributions, \f$\beta = 2\f$ is a good choice
              */
            double beta;

            /**
              * @short \f$\lambda = \alpha^2 (n + \kappa) - n\f$
              */
            double lambda;
            double lambda_aug;

            /**
              * @short \f$\gamma = \sqrt{\lambda + n}\f$
              */
            double gamma;
            double gamma_aug;

            /**
              * @short Size of the state vector
              */
            int n;

            /**
              * @short \f$nbSamples = (2 n + 1)\f$ Number of sigma-points
              */
            int nbSamples;

            /**
              * @short \f$nbSamplesMeasure = (4 n + 1)\f$ Number of sigma-points
              */
            int nbSamplesMeasure;

            /**
              * @short Dimension of the output : the measurements
              */
            int no;

            /**
              * @short Prior estimate of the covariance matrix of the state
              */
            double prior_x;

            /**
              * @short Type of process noise
              */
            ProcessNoise process_noise_type;

            /**
              * @short Parameter used for the evolution noise
              */
            double process_noise;

            /**
              * @short Covariance of the observation noise
              */
            double measurement_noise;

        } ukf_param;

        /**
          * @short Structure holding the matrices manipulated by the statistical linearization
          *  in the vectorial case for state estimation
          *
          */
        typedef struct
        {
            gsl_vector * params; // The parameters of the process equation or anything you need to give to the process function

            gsl_vector * xi; // State vector
            gsl_matrix * xi_prediction; // Prediction of the states for each sigma-point
            gsl_vector * xi_mean; // Mean state vector
            gsl_matrix * Sxi; // Cholesky factor of variance covariance of the state
            gsl_matrix * Svi; // Cholesky factor of the process noise

            gsl_matrix * yi_prediction; // Prediction of the measurements for each sigma-point
            gsl_vector * yi_mean; // Mean measurement vector
            gsl_vector * ino_yi; // Innovations
            gsl_matrix * Syi; // Cholesky factor of the variance covariance of the output
            gsl_matrix * Sni; // Cholesky factor of the measurement noise

            gsl_matrix * Pxyi; // State-Measurement covariance matrix

            gsl_vector * sigmaPoint; // Vector holding one sigma point
            gsl_matrix * sigmaPoints;  // Matrix holding all the sigma points

            gsl_vector * sigmaPointMeasure; // one sigma point for the observation equation
            gsl_matrix * sigmaPointsMeasure; // Sigma points for the observation equation

            gsl_matrix * U;

            gsl_vector * wm_j; // Weights used to compute the mean of the sigma points images
            gsl_vector * wc_j; // Weights used to update the covariance matrices

            gsl_vector * wm_aug_j; // Augmented set of Weights used to compute the mean of the sigma points images for the observations
            gsl_vector * wc_aug_j; // Augmented set of Weights used to update the covariance matrices of the observations

            gsl_matrix * Ki; // Kalman gain
            gsl_matrix * Ki_T; // Its transpose

            gsl_vector * temp_n;
            gsl_vector * temp_no;

            gsl_matrix * temp_n_n;
            gsl_matrix * temp_n_1;
            gsl_matrix * temp_1_n;

            gsl_matrix * temp_n_no;

            gsl_matrix * temp_no_no;
            gsl_matrix * temp_no_1;
            gsl_matrix * temp_1_no;

            gsl_matrix * temp_3n_n;
            gsl_matrix * temp_2nno_no;
        } ukf_state;        


    } // namespace for state estimation, square root implementation

}


#endif // UKF_TYPES_H
