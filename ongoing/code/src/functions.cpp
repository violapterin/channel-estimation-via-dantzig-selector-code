#include <iostream>
#include <cstdio>      /* printf, scanf, puts, NULL */
#include <cstdlib>     /* srand, rand */
#include <ctime>       /* time */

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include "constants.h"

// stackoverflow.com/questions/29877760/boostublas-how-to-get-determinant-of-int-matrix?rq=1
Comp determinant(Mat m)
{
    ublas::permutation_matrix<std::size_t> pivots( m.size1() );
    auto isSingular = ublas::lu_factorize(m, pivots);
    if (isSingular) {return 0;}

    Comp det = 1;
    for (std::size_t i = 0; i < pivots.size(); ++i) 
    {
        if (pivots(i) != i) {det *= -1;}
        det *= m(i,i);
    }
    return det;
}

double frob_norm(Mat& m)
{
   double ret =0;
   for(std::size_t i=0; i<=nn-1; i++)
      for(std::size_t j=0; j<=nn-1; j++)
      {
         double hold_entry =std::abs( m(i,j) );
         ret =ret +hold_entry *hold_entry; // hold
      }
   ret =std::sqrt(ret);
   return ret;
}

Mat get_array_response(double psi, std::size_t len)
{
   Mat ret(len,1);
   for(std::size_t i=0; i <=len -1; i++)
      { ret(i,1) =std::exp(i*psi); }
   return ret;
}

void set_rand_channel( Mat& h )
{
   h =ublas::zero_matrix <Comp> (nn, nn);
   for(std::size_t n_c=0; n_c <nn_c-1; n_c++)
   {
      double mu =rand() /RAND_MAX;
      mu =2*mu -1;// [-1,1]
      for(std::size_t n_s=0; n_s <nn_s-1; n_s++)
      {
         double delta=rand() /RAND_MAX;
         delta =spread *delta *delta *delta;// [0,spread]
         double psi= mu+delta;// [-1-spread, 1+spread]
         if( psi >1 ){ psi =psi-2; }
         if( psi <1 ){ psi =psi+2; }
         psi =pi*(psi+1);
         h= h +ublas::prod( get_array_response(a_t_phase *psi,nn), ublas::trans(get_array_response(a_r_phase *psi,nn)) );
      }
   }
}

void accept_random_step(Map_mat* map_f, Map_mat* map_dd)
{
   for(std::size_t i=0; i<=ff-1; i++)
      { (*map_f)[i] +=(*map_f)[i]; }
}

void set_random_step(Map_mat* p_map_d)
{
   for(std::size_t f=0; f<=ff-1; f++)
   {
      Mat& m =(*p_map_d)[f];
      for(std::size_t i=0; i<=nn-1; i++)
         for(std::size_t j=0; j<=nn-1; j++)
            m(i,j) =rand() /RAND_MAX;
      m =m /frob_norm(m);
   }
}

double find_new_sum_rate(Map_mat* p_map_h, Map_mat* p_map_f, Map_mat* p_map_d)
{
   double ret =0;
   for(std::size_t k=0; k<=kk-1; k++)
   {
      double sq_norm[uu];
      double sum_sq_norm =0;
      for(std::size_t u=0; u<=uu-1; u++)
      {
         std::size_t f =k /s_blk_kk +(u /s_blk_uu) *nn_blk_kk;
         Mat hold_eff_h =ublas::prod( (*p_map_h)[k], (*p_map_f)[f] );
         sq_norm[u] =std::abs( determinant(hold_eff_h) );// hold
         sq_norm[u] =sq_norm[u] * sq_norm[u];
         sum_sq_norm +=sq_norm[u];
         ret =ret +std::log( 1 +(sq_norm[u]) /(1 +sum_sq_norm -sq_norm[u]) );
      }
   }
   return ret;
}

