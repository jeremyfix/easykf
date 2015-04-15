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

#ifndef EKF_H
#define EKF_H

#include "ekf_types.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

namespace ekf
{

  void ekf_init(ekf_param &p, ekf_state &s)
  {

    // Matrices for the state
    s.xk = gsl_vector_alloc(p.n);
    gsl_vector_set_zero(s.xk);

    s.xkm = gsl_vector_alloc(p.n);
    gsl_vector_set_zero(s.xkm);

    s.Pxk = gsl_matrix_alloc(p.n, p.n);
    gsl_matrix_set_zero(s.Pxk);

    s.Fxk = gsl_matrix_alloc(p.n, p.n);
    gsl_matrix_set_zero(s.Fxk);

    s.Rv = gsl_matrix_alloc(p.n, p.n);
    gsl_matrix_set_zero(s.Rv);

    // Matrices for the observations
    s.yk = gsl_vector_alloc(p.no);
    gsl_vector_set_zero(s.yk);

    s.ino_yk = gsl_vector_alloc(p.no);
    gsl_vector_set_zero(s.ino_yk);

    s.Hyk = gsl_matrix_alloc(p.no, p.n);
    gsl_matrix_set_zero(s.Hyk);

    s.Rn = gsl_matrix_alloc(p.no, p.no);
    gsl_matrix_set_zero(s.Rn);

    // Matrices for the kalman gain and the updates
    s.Kk = gsl_matrix_alloc(p.n, p.no);
    gsl_matrix_set_zero(s.Kk);

    // Temporary matrices
    s.temp_n_n = gsl_matrix_alloc(p.n, p.n);
    gsl_matrix_set_zero(s.temp_n_n);

    s.temp_n_1 = gsl_matrix_alloc(p.n, 1);
    gsl_matrix_set_zero(s.temp_n_1);

    s.temp_no_no = gsl_matrix_alloc(p.no, p.no);
    gsl_matrix_set_zero(s.temp_no_no);

    s.temp_n_no = gsl_matrix_alloc(p.n, p.no);
    gsl_matrix_set_zero(s.temp_n_no);

    s.temp_2_n_n = gsl_matrix_alloc(p.n, p.n);
    gsl_matrix_set_zero(s.temp_2_n_n);

    s.temp_no = gsl_vector_alloc(p.no);
    gsl_vector_set_zero(s.temp_no);

    // Initialize the noises
    //gsl_matrix_set_identity(s.Rv);
    //gsl_matrix_scale(s.Rv, p.evolution_noise);
    p.evolution_noise->init(p, s);
    //printf("Evolution noise : max = %e , min = %e \n", gsl_matrix_max(s.Rv), gsl_matrix_min(s.Rv));

    gsl_matrix_set_identity(s.Rn);
    gsl_matrix_scale(s.Rn, p.observation_noise);

    // Initialize the covariance of the parameters
    gsl_matrix_set_identity(s.Pxk);
    gsl_matrix_scale(s.Pxk, p.prior_pk);

  }

  void ekf_free(ekf_param &p, ekf_state &s)
  {
    gsl_vector_free(s.xk);
    gsl_vector_free(s.xkm);
    gsl_matrix_free(s.Pxk);
    gsl_matrix_free(s.Fxk);
    gsl_matrix_free(s.Rv);

    gsl_vector_free(s.yk);
    gsl_vector_free(s.ino_yk);
    gsl_matrix_free(s.Hyk);
    gsl_matrix_free(s.Rn);

    gsl_matrix_free(s.Kk);

    gsl_matrix_free(s.temp_n_n);
    gsl_matrix_free(s.temp_n_1);
    gsl_matrix_free(s.temp_no_no);
    gsl_matrix_free(s.temp_n_no);
    gsl_matrix_free(s.temp_2_n_n);
    gsl_vector_free(s.temp_no);
  }

    void ekf_iterate(ekf_param &p, ekf_state &s, 
		     void(*f)(gsl_vector*, gsl_vector*, gsl_vector*), 
		     void(*df)(gsl_vector*, gsl_vector*, gsl_matrix*), 
		     void(*h)(gsl_vector*, gsl_vector*, gsl_vector*), 
		     void(*dh)(gsl_vector*, gsl_vector*, gsl_matrix*), 
		     gsl_vector* yk)
  {
    /****************************/
    /***** Prediction step  *****/
    /****************************/

    // Compute the Jacobian of the evolution
    // Eq. 2.34
    df(s.params, s.xk, s.Fxk);

    // Compute the predicted state mean and covariance
    // Eq. 2.36
    // s.xk will now hold the predicted state !
    f(s.params, s.xk, s.xkm);

    // Eq. 2.37
    // s.Pxk will now hold Pxk^-
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, s.Pxk, s.Fxk, 0.0, s.temp_n_n);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, s.Fxk, s.temp_n_n, 0.0, s.Pxk);
    gsl_matrix_add(s.Pxk, s.Rv);

    /****************************/
    /***** Correction step  *****/
    /****************************/

    // Compute the Jacobian of the observation model
    // Eq. 2.38
    dh(s.params, s.xkm, s.Hyk);

    if(!p.observation_gradient_is_diagonal)
      {
	// Update the estimates
	// Eq 2.40
	// 1 - Compute H.P^-.H^T + Rn
	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, s.Pxk, s.Hyk, 0.0, s.temp_n_no);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, s.Hyk, s.temp_n_no, 0.0, s.temp_no_no);
	gsl_matrix_add(s.temp_no_no, s.Rn);
	// 2 - Compute its inverse
	gsl_linalg_cholesky_decomp(s.temp_no_no);
	gsl_linalg_cholesky_invert(s.temp_no_no);

	// 3 - Compute P^-.H ^T.( H P H^T + R)^-1
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, s.Hyk, s.temp_no_no, 0.0, s.temp_n_no);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, s.Pxk, s.temp_n_no, 0.0, s.Kk);
      }
    else
      {
	// We can make some simplifications when computing H P H^T = H P H
	// and also when computing P . H . (H P H^T + R) ^-1

	// Update the estimates
	// Eq 2.40
	// 1 - Compute H.P^-.H^T + Rn
	for(int i = 0 ; i < p.no ; ++i)
	  {
	    for(int j = 0 ; j < p.no ; ++j)
	      {
		gsl_matrix_set(s.temp_no_no, i, j, gsl_matrix_get(s.Hyk, i, i) * gsl_matrix_get(s.Hyk, j, j) * gsl_matrix_get(s.Pxk, i, j) + gsl_matrix_get(s.Rn, i, j));
	      }
	  }
	// Compute its inverse : U = (H P H^T + R) ^-1
	gsl_linalg_cholesky_decomp(s.temp_no_no);
	gsl_linalg_cholesky_invert(s.temp_no_no);

	// 3 - Compute P^- H^T . U

	// Compute H^T . U
	for(int i = 0 ; i < p.no; ++i)
	  for(int j = 0 ; j < p.no ; ++j)
	    gsl_matrix_set(s.temp_n_no, i, j, gsl_matrix_get(s.Hyk, i,i) * gsl_matrix_get(s.temp_no_no, i, j));
	for(int i = p.no ; i < p.n; ++i)
	  for(int j = 0 ; j < p.no ; ++j)
	    gsl_matrix_set(s.temp_n_no, i, j, 0.0);
	// Compute P^- . (H^T . U)
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, s.Pxk, s.temp_n_no, 0.0, s.Kk);

      }
    // Update the current estimate
    // Eq 2.41
    // 1 - We need the observations
    h(s.params, s.xkm, s.yk);
    gsl_vector_memcpy(s.ino_yk, yk);
    gsl_vector_sub(s.ino_yk, s.yk);
    gsl_vector_memcpy(s.xk, s.xkm);
    gsl_blas_dgemv(CblasNoTrans, 1.0, s.Kk, s.ino_yk, 1.0,s.xk);

    // Update the variance/covariance matrix
    // Compute  -Kk * Hk
    if(!p.observation_gradient_is_diagonal)
      {
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, s.Kk, s.Hyk, 0.0, s.temp_n_n);

	// Add identity : I - Kk * Hk
	for(int i = 0 ; i < p.n ; ++i)
	  gsl_matrix_set(s.temp_n_n, i, i,gsl_matrix_get(s.temp_n_n,i,i) + 1.0);
      }
    else
      {
	// We can make some simplifications when H is diagonal
	for(int i = 0 ; i < p.n ; ++i)
	  {
	    for(int j = 0 ; j < p.no ; ++j)
	      {
		gsl_matrix_set(s.temp_n_n, i, j, (i==j?1.0:0.0) - gsl_matrix_get(s.Kk, i,j) * gsl_matrix_get(s.Hyk, j,j) );
	      }
	    for(int j = p.no ; j < p.n ; ++j)
	      {
		gsl_matrix_set(s.temp_n_n, i, j, (i == j ? 1.0 : 0.0));
	      }
	  }
      }

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, s.temp_n_n, s.Pxk, 0.0, s.temp_2_n_n);
    gsl_matrix_memcpy(s.Pxk, s.temp_2_n_n);

    /***********************************/
    /***** Evolution noise update  *****/
    /***********************************/

    p.evolution_noise->updateEvolutionNoise(p, s);
  }
}

#endif // EKF_H
