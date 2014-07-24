/* ukf_math.h
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

#ifndef UKF_MATH_H
#define UKF_MATH_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#include <iostream>
#include <cassert>
#include <cmath>

namespace ukf
{
    namespace math
    {

        inline double min(double a, double b)
        {
            return a < b ? a : b;
        }

        inline double max(double a, double b)
        {
            return a < b ? b : a;
        }

        inline double signof(double x)
        {
            return x <= 0.0 ? -1.0 : 1.0;
        }

        inline bool cmp_equal(double x, double y)
        {
            return fabs(x - y) <= 1e-10;
        }
        inline bool cmp_diff(double x, double y)
        {
            return fabs(x - y) >= 1e-10;
        }

        /**
         * @brief This function performs a cholesky update according to Strange et al.(2007)
         * @author <a href="mailto:Mathieu.Geist@Supelec.Fr">Mathieu.Geist@Supelec.fr</a>
         *
         */
        void choleskyUpdate(gsl_matrix * sigmaTheta, double alpha, gsl_vector *x) {

            /*
                This function performs a cholesky update of the
                cholesky factorization sigmaTheta, that is it replaces
                the Cholesky factorization sigmaTheta by the cholesky factorization of

                            sigmaTheta*sigmaTheta^T + alpha * x * x^T

                The algorithm is an adaptation of a LU factorization rank one update.

                Reference is :

                Peter Strange, Andreas Griewank and Matthias Bollhöfer.
                On the Efficient Update of Rectangular LU Factorizations subject to Low Rank Modifications.
                Electronic Transactions on Numerical Analysis, 26:161-177, 2007.
                alg. is given in left part of fig.2.1.

                Perhaps a more efficient algorithm exists, however it should do the work for now. And the code is probably not optimal...
            */

            if (sigmaTheta->size1 != sigmaTheta->size2)
                std::cout<<"ERROR ukf::math::choleskyUpdate Cannot use CholeskyUpdate on non-squared matrices"<<std::endl ;

            int i,j,n= sigmaTheta->size1 ;
            double tmp ;
            gsl_matrix *U = gsl_matrix_alloc(n,n) ;
            gsl_vector *D = gsl_vector_alloc(n) ;
            gsl_vector *y = gsl_vector_alloc(n) ;

            // A first thing is to set SS' (chol factor) in a LU form, L being unitriangular
            // Compute U = L^T and D = diag(L)
            gsl_matrix_set_zero(U) ;
            for (i=0; i<n; i++){
                gsl_vector_set(D,i,gsl_matrix_get(sigmaTheta,i,i));
                for (j=0; j<=i; j++){
                    gsl_matrix_set(U,j,i,gsl_matrix_get(sigmaTheta,i,j));
                }
            }
            // Replace L by L*D^{-1} and U by D*U
            for (i=0; i<n; i++){
                for (j=0; j<=i; j++){
                    tmp = gsl_matrix_get(sigmaTheta,i,j);
                    tmp /= gsl_vector_get(D,j);
                    gsl_matrix_set(sigmaTheta,i,j,tmp);
                    tmp = gsl_matrix_get(U,j,i);
                    tmp *= gsl_vector_get(D,j);
                    gsl_matrix_set(U,j,i,tmp);
                }
            }

            // compute the y = alpha x vector
            gsl_vector_memcpy(y,x) ;
            gsl_vector_scale(y,alpha) ;

            // perform the rank 1 LU modification
            for (i=0; i<n; i++){

                // diagonal update
                tmp = gsl_matrix_get(U,i,i) +
                      gsl_vector_get(x,i)*gsl_vector_get(y,i);
                gsl_matrix_set(U,i,i,tmp);
                tmp = gsl_vector_get(y,i);
                tmp /= gsl_matrix_get(U,i,i);
                gsl_vector_set(y,i,tmp);

                for (j=i+1; j<n; j++){
                    // L (that is sigmaTheta) update
                    tmp = gsl_vector_get(x,j) -
                          gsl_vector_get(x,i)*gsl_matrix_get(sigmaTheta,j,i);
                    gsl_vector_set(x,j,tmp) ;
                    tmp = gsl_matrix_get(sigmaTheta,j,i) + gsl_vector_get(y,i) * gsl_vector_get(x,j);
                    gsl_matrix_set(sigmaTheta,j,i,tmp);
                }

                for (j=i+1; j<n; j++) {
                    // U update
                    tmp = gsl_matrix_get(U,i,j) +
                          gsl_vector_get(x,i)*gsl_vector_get(y,j);
                    gsl_matrix_set(U,i,j,tmp) ;
                    tmp = gsl_vector_get(y,j) -
                          gsl_vector_get(y,i) * gsl_matrix_get(U,i,j);
                    gsl_vector_set(y,j,tmp) ;
                }
            }

            // Now we want the chol decomposition
            // first D = sqrt(diag(U)) ;
            for (i=0; i<n; i++){
                tmp =  gsl_matrix_get(U,i,i) ;

                if (tmp<=0){std::cout<< "WARNING ukf::math::choleskyUpdate::matrix not positive definite  : " << tmp << std::endl;}
                gsl_vector_set(D,i,sqrt(tmp)) ;
            }
            // then L = L*D ;
            for (i=0; i<n; i++) {
                for (j=0; j<n; j++) {
                    tmp = gsl_matrix_get(sigmaTheta,i,j) * gsl_vector_get(D,j);
                    gsl_matrix_set(sigmaTheta,i,j,tmp);
                }
            }
            // that's all folks !

            //free memory
            gsl_matrix_free(U);
            gsl_vector_free(y);
            gsl_vector_free(D);
        }
    }  // math
}  // ukf


#endif  // UKF_MATH_H
