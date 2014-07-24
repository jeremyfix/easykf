/* ekf_types.h
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

#ifndef EKF_TYPES_H
#define EKF_TYPES_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include "ukf_math.h"

/**
 * @short Extended Kalman Filter in the case of additive noise, the notations follow Van Der Merwe, phD, p. 36
 */
namespace ekf
{

  class EvolutionNoise;

  /**
   * @short Structure holding the parameters of the statistical linearization
   *
   */
  typedef struct
  {

    /**
     * @short Evolution noise type
     */
    EvolutionNoise *evolution_noise;

    /**
     * @short Covariance of the observation noise
     */
    double observation_noise;

    /**
     * @short Prior estimate of the covariance matrix
     */
    double prior_pk;

    /**
     * @short Number of parameters to estimate
     */
    int n;

    /**
     * @short Dimension of the output
     */
    int no;

    /**
     * @short Is the observation gradient diagonal ? In that case, simplifications can be introduced
     */
    bool observation_gradient_is_diagonal;

  } ekf_param;

  typedef struct
  {
    /**
     * @short Current estimate of the state
     */
    gsl_vector * xk;

    /**
     * @short Predicted state
     */
    gsl_vector * xkm;

    /**
     * @short Variance-Covariance of the state
     */
    gsl_matrix * Pxk;

    /**
     * @short Jacobian of the evolution function
     */
    gsl_matrix *Fxk;

    /**
     * @short Variance covariance of the evolution noise
     */
    gsl_matrix * Rv;

    /**
     * @short Current observation
     */
    gsl_vector * yk;

    /**
     * @short Current innovations
     */
    gsl_vector * ino_yk;

    /**
     * @short Jacobian of the observation function
     */
    gsl_matrix *Hyk;

    /**
     * @short Variance covariance of the observation noise
     */
    gsl_matrix * Rn;

    /**
     * @short Kalman gain
     */
    gsl_matrix *Kk;

    /**
     * @short Temporary matrices
     */
    gsl_matrix * temp_n_n;
    gsl_matrix * temp_n_1;
    gsl_matrix * temp_no_no;
    gsl_matrix * temp_n_no;
    gsl_matrix * temp_2_n_n;
    gsl_vector * temp_no;

    /**
     * @short Optional parameters (this must be allocated and initialized from the user side!
     */
    gsl_vector * params;



  } ekf_state;

  /**
   * @short Mother class from which the evolution noises inherit
   */
  class EvolutionNoise
  {
  protected:
    double _initial_value;
  public:
  EvolutionNoise(double initial_value) : _initial_value(initial_value) {};
    void init(ekf_param &p, ekf_state &s)
    {
      gsl_matrix_set_identity(s.Rv);
      gsl_matrix_scale(s.Rv, _initial_value);
    }

    virtual void updateEvolutionNoise(ekf_param &p, ekf_state &s) = 0;
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

    void updateEvolutionNoise(ekf_param &p, ekf_state &s)
    {
      for(int i = 0 ; i < p.n ; ++i)
	{
	  for(int j = 0 ; j < p.n ; ++j)
	    {
	      if(i == j)
		gsl_matrix_set(s.Rv, i, i, ukf::math::max(_decay * gsl_matrix_get(s.Rv,i,i),_lower_bound));
	      else
		gsl_matrix_set(s.Rv, i, j, 0.0);
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

    void updateEvolutionNoise(ekf_param &p, ekf_state &s)
    {
      gsl_matrix_memcpy(s.Rv, s.Pxk);
      gsl_matrix_scale(s.Rv, 1.0 / _decay - 1.0);
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

    void updateEvolutionNoise(ekf_param &p, ekf_state &s)
    {
      // Compute Kk * ino_yk
      gsl_matrix_view mat_view = gsl_matrix_view_array(s.ino_yk->data, p.no, 1);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, s.Kk, &mat_view.matrix, 0.0, s.temp_n_1);
      // Compute : Rv = (1 - alpha) Rv + alpha . (Kk * ino_yk) * (Kk * ino_yk)^T
      gsl_blas_dgemm(CblasNoTrans, CblasTrans, _alpha, s.temp_n_1, s.temp_n_1, (1.0 - _alpha),s.Rv);
    }
  };

}

#endif // EKF_TYPES_H
