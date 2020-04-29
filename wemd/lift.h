/** @file 
 * Lifting wavelet transform
 * @author Sameer Sheorey
 */ 
#ifndef _LIFT_H
#define _LIFT_H

#ifdef _DEBUG
#define D(desc, a) std::cerr << desc##" "###a << " = " << a <<  std::endl
#define DV(a) std::cerr << #a << " = " << a <<  std::endl
#else
#define D(desc, a)
#define DV(a)
#endif

//TODO: Change vector to list later

#include  <blitz/array.h>
#include  <blitz/array/domain.h>
#include  <blitz/tinyvec-et.h>

#include  <cassert>     // For debugging
#include  <stdexcept>   // For runtime errors
using std::invalid_argument; using std::out_of_range;
#include  <vector>
using std::vector;
#include  <iostream>
using std::cout; using std::cin; using std::cerr; using std::endl;
using std::ostream;
#include  <iomanip>
using std::setw;
#include  <algorithm>
#include  <utility>
using std::pair;
#include  <memory>
using std::auto_ptr;

using namespace blitz;

/** Forward or inverse transform */
enum LWTtype {forward, inverse};
/** Elimentary lifting step type: primal (update), dual (predict) or simple
 * normalize */
enum ELStype {primal, dual, normalize};
/** Extension mode: No extension, zero padding or periodic extension.
 * Currently, periodic can only be used with arrays of length k*2^J, for a
 * level J transform. */ 
enum extModeEnum {none, zpd, periodic};

#include  "blitz_util_SS.h"

/** Elementary lifting step
 * The Laurent polynomial P is of the form 
 * P(z) = C(1)*z^d + C(2)*z^(d-1) + ... + C(m)*z^(d-m+1)
 * Last ELS in LS is always of type normalize coeff(first) = primal norm,
 * coeff(second) = dual norm.
 * usually pnorm*dnorm = 1;
 */ 

// Forward declarations for template friend
template <typename T> class ELS;
template <typename T> ostream & operator<<(ostream &, const ELS<T> &);

template <typename T = double>
class ELS {

  private:
  ELStype type;
  Array<T, 1> coefficients; ///< C, m = length(C);
  int max_degree; ///< d

  public:
  /// Constructor
  ELS(ELStype type_, Array<T, 1> coefficients_, int max_degree_ = 0);

  /** Apply ELS
   * @param X input array
   * @param t forward or inverse LWT
   */ 
  template<size_t ndims>
  void els_predict_update(Array<T, ndims> X, const LWTtype t = forward) const;

  // get info about the ELS
  inline ELStype gettype() const { return type; }
  inline int maxdeg() const {return max_degree;}
  inline int mindeg() const {return max_degree-coefficients.rows()+1; }
  inline int length() const {return coefficients.rows(); }

  /// Display ELS
  friend ostream & operator<< <>(ostream &os, const ELS<T> &els);
  ostream & display(ostream &os) const;

  private:
  /** Apply ELS without extension. This is also used (after extending the
   * array) for extension modes where extension cannot be performed by
   * modified addressing, for example if the output will need to be larger in
   * size.
   * A1 += filter*A2 (forward) or A1 -= filter*A2 (inverse) 
   * @param A1 Unchanged signal component
   * @param A2 Changing signal component
   * @param t forward or inverse LWT
   */ 
  template <size_t ndims> inline void pu_none(Array<T, ndims> A1, const
      Array<T, ndims> A2, const size_t dim, const LWTtype t = forward) const;

  /** Apply ELS with periodic extension using modified addressing.
   * A1 += filter*A2 (forward) or A1 -= filter*A2 (inverse) 
   * @param A1 Unchanged signal component
   * @param A2 Changing signal component
   * @param t forward or inverse LWT
   */   
  template <size_t ndims> inline void pu_periodic(Array<T, ndims> A1, const
      Array<T, ndims> A2, const size_t dim, const LWTtype t = forward) const;

};
// (2^ndims) - 1
#define ncomps ((1<<ndims)-1)
/** Lifting scheme.
 * Currently, this only works correctly with Lifting schemes output by
 * Matlab's filt2ls function with orthoonrmal filters.
 * Not tested with biorthogonal lifting schemes.
 */ 

// Forward declarations for template friend
template <typename T> class LS;
template <typename T> ostream & operator<<(ostream &, const LS<T> &);

template <typename T = double>
class LS {
  private:
    string wname;   ///< Name of wavelet or lifting scheme
    vector<ELS<T> > LSteps;     ///< Vector of elimentary lifting steps
    int lf;    ///< net filter length
    TinyVector<int, 2> ext_range;
    bool initialize; ///< initialize fine scale WT coeffs before LWT ?
    /// Convolve with alpha to initialize fine scale coefficients
    Array<T, 1> alpha; 

    typedef typename vector<ELS<T> >::const_iterator lift_citer;
    typedef typename vector<ELS<T> >::const_reverse_iterator lift_creviter;

  public:
    /** static Template function to mimic the behaviour of a static template
     * data member. Stores extension mode for different dimensions. 
     * Using a function instead of a class has slightly shorter syntax. */
    template <size_t ndims>
      static TinyVector<extModeEnum, ndims> & extMode() {
        static TinyVector<extModeEnum, ndims> extMode(none);
        return extMode;
      }

  public:

    /*************** Constructors ************************/
    LS() : lf(0), ext_range(0), wname(""), initialize(false) {} 
    LS(string wname);  ///< Use known lifting scheme from lifting-schemes.h
    /// TODO: construct lifting scheme from filter quadruplet
    LS(const Array<T, 1> LoD, const Array<T, 1> HiD, const Array<T, 1> LoR, 
        const Array<T, 1> HiR);
    /// TODO: Orthonormal LS constructor D = R
    LS(const Array<T, 1> Lo, const Array<T, 1> Hi); 

    /** Use initialization for fine scale coefficients ? 
     * Calculate #alpha if initialization required.  
     * @return size of signal extension due to initialization. There is no
     * change in signal size if extMode() is periodic 
     */ 
    unsigned set_init(bool init = true);   

    /// Add lifting step at the end, can't add after a normalize step
    void addlift(ELS<T> els); 

    /************** Forward & Inverse transforms **********/
    /** Forward Lifted wavelet transform (inplace) only if extMode()==none
     * @param X input array. This also contains the result in Swelden format
     * @param J LWT level
     */ 
    template<size_t ndims> 
      void lwt_inplace(Array<T, ndims> &X, const int J) const;

    /** Forward Lifted wavelet transform (out of place)
     * @param X input array. This contains the approximation coeffs. on
     * return. 
     * @param Y vector of detail coefficients. Length (J) specifies the number 
     * of levels to compute, but each Y(j) should be empty.
     * @return Y(j)[k] contains the level j detail coefficients with the
     * wavelet \psi_k. \psi^k is the tensor product of \psi or \phi along each
     * dimension, according to whether the dth bit of k is 1 or 0
     * respectively. 
     * By Meyer's convention, Y(j) is actually scale J-j-1.
     */ 
    template<size_t ndims>
      void lwt(Array<T, ndims> &X, vector<Array<TinyVector<T, ncomps>, ndims> >
          &Y) const;

    /** Inverse Lifted wavelet transform (inplace) only if extMode()==none
     * @param X LWT in Swelden format
     * @param Jnow current WT level
     * @param Jfinal required WT level (0 for original data)
     */ 
    template<size_t ndims>    
      void ilwt(Array<T, ndims> &X, const int Jnow, const int Jfinal = 0)
      const;

    /** Inverse Lifted wavelet transform (not inplace)
     * @param Xapp Approximation coeffs.
     * @param Y vector of detail coefficients. @see lwt() for format.
     * @param szodd Is the final array size odd ?
     * @param Jfinal required WT level (0 for original data)
     */ 
    template<size_t ndims>
      void ilwt(Array<T, ndims> &Xapp, const vector<Array<TinyVector<T, 
          ncomps>, ndims> > &Y, const TinyVector<bool, ndims> szodd, 
          const int Jfinal = 0) const;

/*    template<size_t ndims>    // Undecimated/stationery Forward LWT
        void ulwt(Array<T, ndims> X, vector<Array<TinyVector<T, ncomps>, ndims>
            > &Y) const;

    template<size_t ndims>    // Undecimated/stationery Inverse LWT
        Array<T, ndims> uilwt(Array<T, ndims> &Xapp, const 
            vector<Array<TinyVector<T, ncomps>, ndims> > &Y, const 
            TinyVector<bool, ndims> szodd, const int Jfinal = 0) const;
    // levels J = size(Y, last);
*/
    /**************** Miscellaneous ************************/
    /** Compute wavelet and scaling functions
     * @param iters Grid size = 2^(-iters)
     * @return vector of size 2 with scaling function and wavelet
     * respectively. 
     */ 
    vector<Array<T, 1> > wavefun(const size_t iters) const;

    template <size_t ndims> auto_ptr<pair<Array<T, ndims>, Array<T, ndims> > > 
      wsfun(const size_t J) const;

    /** Maximum wavelet transform level
     * the rule is the last level for which at least one coefficient 
     * is correct : (lf-1)*(2^lev) < lx
     * @param lx minimum array size
     */ 
    int wmaxlev(const size_t lx) const 
    {  return std::max( floor( log(double(lx) / double(lf-1)) / log(2.0) ), 0.0); }

    /** Display lifting scheme */
    friend ostream & operator<< <>(ostream &os, const LS<T> &ls);
    template <size_t ndims>
      ostream & display(ostream &) const;
};

/** Display wavelet transform
 * @param A Approximation coeffs. @see LS::lwt()
 * @param D detail coeffs. @see LS::lwt()
 */ 
template <typename T, size_t ndims> 
ostream & display(ostream &os, const Array<T, ndims> &A, const 
    vector<Array<TinyVector<T, ncomps>, ndims> > &D);


#include  "lifting-schemes.h"
#include  "lift_impl.h"

#endif  // _LIFT_H
