/* ukf_samples.h
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

#ifndef UKF_SAMPLES_H
#define UKF_SAMPLES_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sort_vector_double.h>
#include <gsl/gsl_permute_vector.h>
#include <gsl/gsl_randist.h>

#include <time.h>

#include <iostream>

namespace ukf
{
    namespace samples
    {
        /**
          * @short Generate 1D samples according to a uniform distribution :
          * @brief In fact, gsl_ran_choose would do the job!!
          */
        class RandomSample_1D
        {
        public:
            static void generateSample(gsl_vector * samples, unsigned int nb_samples, double min, double max, double delta)
            {
                if(samples->size != nb_samples)
                {
                    std::cerr << "[ERROR] RandomSample_1D : samples should be allocated to the size of the requested number of samples" << std::endl;
                    return;
                }


                for(unsigned int i = 0 ; i < nb_samples ; i++)
                    gsl_vector_set(samples, i, min + floor((max - min)*rand()/double(RAND_MAX)/delta)    * delta  );
            }
        };

        /**
          * @short Generate 2D samples according to a uniform distribution
          * and put them alternativaly at the odd/even positions
          */
        class RandomSample_2D
        {
        public:
            static void generateSample(gsl_vector * samples, unsigned int nb_samples, double minx, double maxx, double deltax, double miny, double maxy, double deltay)
            {
                if(samples->size != 2*nb_samples)
                {
                    std::cerr << "[ERROR] RandomSample_2D : samples should be allocated to the size of the 2 times the requested number of samples" << std::endl;
                    return;
                }
		gsl_vector_view vec_view = gsl_vector_subvector_with_stride (samples, 0, 2, nb_samples);
                RandomSample_1D::generateSample(&vec_view.vector, nb_samples, minx, maxx, deltax);
		vec_view = gsl_vector_subvector_with_stride (samples, 1, 2, nb_samples);
                RandomSample_1D::generateSample(&vec_view.vector, nb_samples, miny, maxy, deltay);
            }
        };

        /**
          * @short Generate 3D samples according to a uniform distribution
          * and put them alternativaly at the odd/even positions
          */
        class RandomSample_3D
        {
        public:
            static void generateSample(gsl_vector * samples, unsigned int nb_samples, double minx, double maxx, double deltax, double miny, double maxy, double deltay, double minz, double maxz, double deltaz)
            {
                if(samples->size != 3*nb_samples)
                {
                    std::cerr << "[ERROR] RandomSample_2D : samples should be allocated to the size of the 2 times the requested number of samples" << std::endl;
                    return;
                }
		gsl_vector_view vec_view = gsl_vector_subvector_with_stride (samples, 0, 3, nb_samples);
                RandomSample_1D::generateSample(&vec_view.vector, nb_samples, minx, maxx, deltax);
		vec_view = gsl_vector_subvector_with_stride (samples, 1, 3, nb_samples);
                RandomSample_1D::generateSample(&vec_view.vector, nb_samples, miny, maxy, deltay);
		vec_view = gsl_vector_subvector_with_stride (samples, 2, 3, nb_samples);
                RandomSample_1D::generateSample(&vec_view.vector, nb_samples, minz, maxz, deltaz);
            }
        };

        /**
          * @short Extract the indexes of the nb_samples highest values of a vector
          * Be carefull, this function is extracting the indexes as, the way it is used, it doesn't not to which sample a vector index corresponds
          */
        class MaximumIndexes_1D
        {
        public:
            static void generateSample(gsl_vector * v, gsl_vector * indexes_samples, unsigned int nb_samples)
            {
                if(indexes_samples->size != nb_samples)
                {
                    std::cerr << "[ERROR] MaximumIndexes_1D : samples should be allocated to the size of the requested number of samples" << std::endl;
                    return;
                }
                if(nb_samples > v->size)
                {
                    std::cerr << "[ERROR] MaximumIndexes_1D : the vector v must hold at least nb_samples elements" << std::endl;
                    return;
                }

                gsl_permutation * p = gsl_permutation_alloc(v->size);
                gsl_sort_vector_index(p,v);
                for(unsigned int i = 0 ; i < nb_samples ; i++)
                    gsl_vector_set(indexes_samples, i, gsl_permutation_get(p, p->size - i - 1));
                gsl_permutation_free(p);
            }
        };

        /**
          * @short Extract the indexes of the nb_samples highest values of a matrix
          * Be carefull, this function is extracting the indexes as, the way it is used, it doesn't not to which sample a vector index corresponds
          */
        class MaximumIndexes_2D
        {
        public:
            static void generateSample(gsl_matrix * m, gsl_vector * indexes_samples, unsigned int nb_samples)
            {

                if(indexes_samples->size != 2*nb_samples)
                {
                    std::cerr << "[ERROR] MaximumIndexes_2D : samples should be allocated to the size of 2 times the requested number of samples" << std::endl;
                    return;
                }
                if(nb_samples > m->size1 * m->size2 )
                {
                    std::cerr << "[ERROR] MaximumIndexes_2D : the vector v must hold at least nb_samples elements" << std::endl;
                    return;
                }

                gsl_permutation * p = gsl_permutation_alloc(m->size1 * m->size2);
                // We cast the matrix in a vector ordered in row-major and get the permutation to order the data by increasing value
		gsl_vector_view vec_view = gsl_vector_view_array(m->data,m->size1 * m->size2);
                gsl_sort_vector_index(p,&vec_view.vector);

                // The indexes stored in p are in row-major order
                // we then convert the nb_samples first indexes into matrix indexes and copy them in indexes_samples
                int k,l;
                for(unsigned int i = 0 ; i < nb_samples ; i++)
                {
                    // Line
                    k = gsl_permutation_get(p,p->size - i - 1) / m->size2;
                    // Column
                    l = gsl_permutation_get(p,p->size - i - 1) % m->size2;
                    gsl_vector_set(indexes_samples, 2 * i, k);
                    gsl_vector_set(indexes_samples, 2 * i + 1, l);
                }

                gsl_permutation_free(p);
            }
        };

        /**
          * @short Generate 1D samples according to a discrete vectorial distribution
          */
        class DistributionSample_1D
        {
        public:
            static void generateSample(gsl_vector * v, gsl_vector * indexes_samples, unsigned int nb_samples)
            {
                if(indexes_samples->size != nb_samples)
                {
                    std::cerr << "[ERROR] RandomSample_1D : samples should be allocated to the size of the requested number of samples" << std::endl;
                    return;
                }

                // Generating nb_samples samples from an array of weights for each index is easily done in the gsl with gsl_ran_discrete_* methods
                // the weights v do not need to add up to one
                gsl_rng * r = gsl_rng_alloc(gsl_rng_default);
                // Init the random seed
                gsl_rng_set(r, time(NULL));
                gsl_ran_discrete_t * ran_pre = gsl_ran_discrete_preproc(v->size, v->data);

                for(unsigned int i = 0 ; i < nb_samples ; i++)
                    gsl_vector_set(indexes_samples, i , gsl_ran_discrete(r, ran_pre));

                gsl_ran_discrete_free(ran_pre);
                gsl_rng_free(r);
            }
        };

        /**
          * @short Generate 2D samples according to a discrete matricial distribution
          */
        class DistributionSample_2D
        {
        public:
            static void generateSample(gsl_matrix * m, gsl_vector * indexes_samples, unsigned int nb_samples)
            {
                if(indexes_samples->size != 2*nb_samples)
                {
                    std::cerr << "[ERROR] RandomSample_1D : samples should be allocated to the size of the requested number of samples" << std::endl;
                    return;
                }

                // To generate 2D samples according to a matrix of weights
                // we simply cast the matrix into a vector, ask DistributionSample_1D to do the job
                // and then convert the linear row-major indexes back into 2D indexes

                gsl_vector * samples_tmp = gsl_vector_alloc(nb_samples);
		gsl_vector_view vec_view = gsl_vector_view_array(m->data,m->size1 * m->size2);
                DistributionSample_1D::generateSample(&vec_view.vector, samples_tmp, nb_samples);

                int k,l;
                for(unsigned int i = 0 ; i < nb_samples ; i++)
                {
                    k = int(gsl_vector_get(samples_tmp,i)) / m->size2;
                    l = int(gsl_vector_get(samples_tmp,i)) % m->size2;
                    gsl_vector_set(indexes_samples, 2*i, k);
                    gsl_vector_set(indexes_samples, 2*i+1, l);
                }

                gsl_vector_free(samples_tmp);
            }
        };
    }
}


#endif // UKF_SAMPLES_H
