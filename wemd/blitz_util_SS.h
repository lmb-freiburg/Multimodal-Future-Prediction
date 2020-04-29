/** @file 
 * Useful global functions for blitz Arrays
 * @author Sameer Sheorey
*/

#ifndef _BLITZ_UTIL_SS_H
#define _BLITZ_UTIL_SS_H

#include  <algorithm>
#include  <cassert>

#include  <blitz/array.h>
#include  <blitz/array/cartesian.h>
#include  <blitz/array/indirect.h>

BZ_NAMESPACE(blitz)

  /** Allows array addressing using arbitrary dimensional Cartesian products.
   * @see CartesianProduct
   */
/*  template <typename T_tuple, typename T_container, int N_containers>
  class CartesianProductN : public CartesianProduct<T_tuple, T_container, 
  N_containers> {
    public:
      CartesianProductN(const TinyVector<T_container, N_containers> & 
          containers)
      {
        for( int i=0; i<N_containers; ++i)
          containers_[i] = &containers(i);
      }
  };
*/
  /** Allows array addressing using arbitrary dimensional Cartesian products.
   * This is a wrapper for the CartesianProductN constructor
   * @see CartesianProduct
   */
/*template <typename T_container, int N_containers>
  CartesianProductN<TinyVector<int, N_containers>, T_container, N_containers>
indexSet(const T_container& containers)
{
  return CartesianProductN<TinyVector<int, N_containers>, T_container,
         N_containers> (const_cast<T_container&>(containers));
}
*/

/** Shrink array by different amounts from both sides in each dimension.
 * @param left start shrink amount
 * @param right end shrink amount
 */ 
template <typename T, size_t ndims>
inline void shrink(Array<T, ndims> &X, TinyVector<int, ndims> left, 
    TinyVector<int, ndims> right) {

  assert(all(X.extent() >= left+right));
  assert(all(left >= 0));
  assert(all(right >= 0));
  if(all(left + right == 0)) return;

  Array<T, ndims> Y(X.extent() - left - right);
  Y = X( RectDomain<ndims>(X.lbound() + left, X.ubound() - right) );
  swap(X, Y);
}

/** Shrink array equally from both sides in each dimension.
 * @param newsize final array size
 */ 
template <typename T, size_t ndims>
inline void shrink_center(Array<T, ndims> &X, TinyVector<int, ndims> newsize) {

  assert(all(X.extent() >= newsize));
  assert(all((X.extent() - newsize) % 2 == 0));

  if(all(X.extent() == newsize)) return;

  Array<T, ndims> Y(newsize);
  Y = X( RectDomain<ndims>(X.lbound() + (X.extent()-newsize)/2, 
        X.ubound() - (X.extent()-newsize)/2) );
  swap(X, Y);
}


/** Extend array by different amounts from both sides in each dimension.
 * @param left start extend amount
 * @param right end extend amount
 * @param extMode Currently only zero padding (zpd) is implemented
 */ 
template <typename T, size_t ndims>
inline void extend(Array<T, ndims> &X, TinyVector<int, ndims> left, 
    TinyVector<int, ndims> right, TinyVector<extModeEnum, ndims> extMode) {

  assert(all(left >= 0));
  assert(all(right >= 0));
  if(all(left + right == 0)) return;

  Array<T, ndims> Y(X.extent() + left + right);
  RectDomain<ndims> rd(Y.lbound() + left, Y.ubound() - right), rdd(rd), rds(rd);
  Y(rd) = X;    // Copy original array
  for (int d=0; d<ndims; ++d) {
    if (d>0) {
      rds.lbound(d-1) = rdd.lbound(d-1) = Y.lbound(d-1);
      rds.ubound(d-1) = rdd.ubound(d-1) = Y.ubound(d-1);
    }
    switch(extMode(d)) {
      /*case periodic: 
        for(int k = rd.ubound(d) + 1; k <= Y.ubound(d); k += X.extent(d)) {
        rdd.lbound(d) = k;
        rdd.ubound(d) = std::min(k + X.extent(d), Y.ubound(d));
        rds.ubound(d) = rds.lbound(d) + rdd.ubound(d) - rdd.lbound(d);
        Y(rdd) = Y(rds);
        }
        rds.ubound(d) = rd.ubound(d);
        for(int k = rd.lbound(d) - 1; k >= Y.lbound(d); k -= X.extent(d)) {
        rdd.ubound(d) = k;
        rdd.lbound(d) = std::max(k - X.extent(d), Y.lbound(d));
        rds.lbound(d) = rds.ubound(d) - (rdd.ubound(d) - rdd.lbound(d));
        Y(rdd) = Y(rds);
        }
        break;*/
      case periodic://Only need to add one zero to the right to make size even
      case zpd: 
        rdd.lbound(d) = rd.ubound(d) + 1; rdd.ubound(d) = Y.ubound(d);
        if (rdd.lbound(d) <= rdd.ubound(d))  Y(rdd) = 0;
        rdd.ubound(d) = rd.lbound(d) - 1; rdd.lbound(d) = Y.lbound(d);
        if (rdd.lbound(d) <= rdd.ubound(d))  Y(rdd) = 0;
        break;
      default: throw std::invalid_argument("Extension mode not implemented.");
    }
  }
  swap(Y,X);
}


/** Extend array by equal amounts from both sides in each dimension.
 * @param newsize final array size
 * @param extMode Extension mode. Currently only zero padding (zpd) is 
 * implemented
 */ 
template <typename T, size_t ndims>
inline void extend_center(Array<T, ndims> &X, TinyVector<int, ndims> newsize, 
    TinyVector<extModeEnum, ndims> extMode) {

  assert(all(X.extent() <= newsize));
  assert(all((X.extent() - newsize) % 2 == 0));

  if(all(X.extent() == newsize)) return;

  Array<T, ndims> Y(newsize);
  RectDomain<ndims> rd(Y.lbound() + (newsize-X.extent())/2, 
      Y.ubound() - (newsize-X.extent())/2), rdd(rd), rds(rd);
  Y(rd) = X;    // Copy original array
  for (int d=0; d<ndims; ++d) {
    if (d>0) {
      rds.lbound(d-1) = rdd.lbound(d-1) = Y.lbound(d-1);
      rds.ubound(d-1) = rdd.ubound(d-1) = Y.ubound(d-1);
    }
    switch(extMode(d)) {
      case periodic: 
        for(int k = rd.ubound(d) + 1; k <= Y.ubound(d); k += X.extent(d)) {
          rdd.lbound(d) = k;
          rdd.ubound(d) = std::min(k + X.extent(d), Y.ubound(d));
          rds.ubound(d) = rds.lbound(d) + rdd.ubound(d) - rdd.lbound(d);
          Y(rdd) = Y(rds);
        }
        rds.ubound(d) = rd.ubound(d);
        for(int k = rd.lbound(d) - 1; k >= Y.lbound(d); k -= X.extent(d)) {
          rdd.ubound(d) = k;
          rdd.lbound(d) = std::max(k - X.extent(d), Y.lbound(d));
          rds.lbound(d) = rds.ubound(d) - (rdd.ubound(d) - rdd.lbound(d));
          Y(rdd) = Y(rds);
        }
        break;
      case zpd: 
        rdd.lbound(d) = rd.ubound(d) + 1; rdd.ubound(d) = Y.ubound(d);
        Y(rdd) = 0;
        rdd.ubound(d) = rd.lbound(d) - 1; rdd.lbound(d) = Y.lbound(d);
        Y(rdd) = 0;
        break;
      default: assert(false);
    }
  }
  swap(Y,X);
}

/** Returns v1(i) if bexpr(i) is true, else v2(i) 
 */ 
template <typename T, int N>
inline TinyVector<T, N> where_else(const TinyVector<bool, N> &bexpr, 
    const TinyVector<T, N> &v1, const TinyVector<T, N> &v2) {
  TinyVector<T, N> v;
  for (int i=0; i < N; ++i)
    v(i) = bexpr(i) ? v1(i) : v2(i);
  return v;
}

/** Returns v1(i) if bexpr(i) is true, else v2(i) 
 */ 
template <typename T, int N>
inline TinyVector<T, N> where_else(const TinyVector<bool, N> &bexpr, T v1, T v2) {
  TinyVector<T, N> v;
  for (int i=0; i < N; ++i)
    v(i) = bexpr(i) ? v1 : v2;
  return v;
}

/** Returns v1(i) if bexpr(i) is true, else v2 
 */ 
template <typename T, int N>
inline TinyVector<T, N> where_else(const TinyVector<bool, N> &bexpr, 
    const TinyVector<T, N> &v1, T v2) {
  TinyVector<T, N> v;
  for (int i=0; i < N; ++i)
    v(i) = bexpr(i) ? v1(i) : v2;
  return v;
}

/** Returns v1 if bexpr(i) is true, else v2(i) 
 */ 
template <typename T, int N>
inline TinyVector<T, N> where_else(const TinyVector<bool, N> &bexpr, T v1, 
    const TinyVector<T, N> &v2) {
  TinyVector<T, N> v;
  for (int i=0; i < N; ++i)
    v(i) = bexpr(i) ? v1 : v2(i);
  return v;
}

/** Cumulative sum along dimension d
 * @param d sum along this dimension (default firstDim)
 */ 
  template <typename T, int N>
inline Array<T, N> & cumsum(Array<T, N> &X, const size_t d=firstDim)
{
  RectDomain<N> r1 = X.domain(), r2 = r1;
  for(int i=X.lbound(d); i<X.ubound(d); ++i) {
    r1.lbound(d) = r1.ubound(d) = i;
    r2.lbound(d) = r2.ubound(d) = i+1;
    X(r2) += X(r1);
  }
  return X;
}

/* Convolve N d X array with tensor product of 1D filter array f.
 * If Range(X) = [lx, ux] and range(f) = [lf, uf], then range(Y) = [lx+lf, 
 * ux+uf]
 * @param X N d array
 * @param f 1D filter
 * @param em extension mode. Only periodic and zero padding supported. Use 
 * cicular convolution if periodic.
 */
  template <typename T, size_t N>
inline Array<T, N> & convolveTP(Array<T, N> &X, const Array<T, 1> &f, 
    TinyVector<extModeEnum, N> &em)
{
  RectDomain<N> rx = X.domain(), ry = rx;

  Array<T, N> Y(X.extent() + f.rows() - 1);
  Y.reindexSelf(rx.lbound() + f.lbound(firstDim));

  extend<T, N>(X, TinyVector<int, N>(0), TinyVector<int, N>(f.rows()-1), 
      TinyVector<extModeEnum, N>(zpd));
  X.reindexSelf(Y.base());

  for(int d=0; d<N; ++d) 
  {
    Y = 0;
    for(int j=Y.lbound(d); j<=Y.ubound(d); ++j)
      for(int i = std::max(j-Y.ubound(d), f.lbound(firstDim)); 
          i <= std::min(j-Y.lbound(d), f.ubound(firstDim)); ++i)
      {
        ry.lbound(d) = ry.ubound(d) = j;
        rx.lbound(d) = rx.ubound(d) = j - i;
        Y(ry) += f(i) * X(rx);
      }
    rx.lbound(d) = ry.lbound(d) = Y.lbound(d);
    rx.ubound(d) = ry.ubound(d) = Y.ubound(d);
    if (em(d)==periodic)    // Circular convolution along dimension d
    {
      RectDomain<N> rYstart = Y.domain(), rYend = Y.domain();
      // ub-lb = extra - 1 = (lf - 1) - 1 = lf - 2;
      rYstart.ubound(d) = rYstart.lbound(d) + f.rows() - 2;
      rYend.lbound(d) = rYend.ubound(d) - f.rows() + 2;
      Y(rYstart) += Y(rYend);
    }
    swap(X,Y);
  }
  TinyVector<int, N> Xsize = X.extent();
  // Remove trailing extra part for periodic dimensions
  for(int d=0; d<N; ++d) 
    if(em(d)==periodic)
      Xsize(d) -= f.rows() - 1;
  X.resizeAndPreserve(Xsize);

  return X;
}

BZ_NAMESPACE_END

#endif // _BLITZ_UTIL_SS_H
