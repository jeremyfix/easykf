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
