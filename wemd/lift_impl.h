/** @file
 * Implementation of functions declared in lift.h
 * @author Sameer Sheorey
 */ 

#ifndef _LIFT_IMPL_H
#define _LIFT_IMPL_H

#include  "blitz_util_SS.h"


/*********** ELS constructor  ***********************************/
template <typename T>
ELS<T>::ELS(ELStype type_, Array<T, 1> coefficients_, int max_degree_) : 
type(type_), coefficients(coefficients_.copy()), max_degree(max_degree_) {
  if(type==normalize) {
    if(coefficients.size() != 2)
      throw invalid_argument("2 normalization constants expected.\n");
    if(fabs(product(coefficients)-1) > 10*numeric_limits<T>::epsilon())
      throw invalid_argument("Normalization constants should have product 1.\n");
  }
}

/*************** Display ELS ************************************/
template <typename T>
ostream & operator<<(ostream &os, const ELS<T> &els) {

  return els.display(os);
}

template <typename T>
ostream & ELS<T>::display(ostream &os) const {

  switch(type) {
    case primal:  os << setw(11) << "primal" << " : "; break;
    case dual: os << setw(11) << "dual" << " : "; break;
    case normalize: os << setw(11) << "normalize" << " : "; break;
  }

  if (type!=normalize)  {
    for(int i=0; i<coefficients.rows(); ++i) {
      os << setw(10) << showpos << coefficients(i);
      if (max_degree-i!=0) {
        if (max_degree-i!=1) os << " z^" << noshowpos << max_degree-i << ' ';
        else os << " z ";
      }
    }
   /* TinyVector<int, 2> els_range(mindeg(), maxdeg());
    els_range = 2*els_range + (gettype()==primal ? 1 : -1);
    if (els_range(0) > 0) els_range(0) = 0;
    if (els_range(1) < 0) els_range(1) = 0;
    os << els_range;*/
  } else
    os << noshowpos << setw(10) << coefficients(0) << setw(10) 
      << coefficients(1);
  os << endl;
  return os;
}

/*************** Display Lifting Scheme ****************************/
template <typename T>
ostream & operator<<(ostream &os, const LS<T> &ls) {
  return ls.display(os);
}

template <typename T> template <size_t ndims>
ostream & LS<T>::display(ostream &os) const {
  os << wname << endl;
  for (lift_citer lsit = LSteps.begin(); 
      lsit != LSteps.end(); ++lsit)
    lsit->display(os);
  os << "Length: " << lf << '\t' << "Extension range: " << ext_range << endl 
    << "Extension modes: [ ";
  for (unsigned k=0; k<ndims; ++k) {
    switch(extMode<ndims>()(k)) {
      case none: os << setw(11) << "none"; break;
      case periodic: os << setw(11) << "periodic"; break;
      case zpd: os << setw(11) << "zpd"; break;
    }
  }
  os << " ]" << endl;
  return os;
}

/*********** ELS predict / update / normalize *********************/
template <typename T> template<size_t ndims>
void ELS<T>::els_predict_update(Array<T, ndims> X, LWTtype t) const {

#ifdef TEST_ELS
  static bool test_els = true;
  Array<T, ndims> Yin;
  if(test_els) {
    Yin.resize(X.shape());
    Yin = X;
  }
#endif

  StridedDomain<ndims> idxAPP(X.lbound(), X.ubound(), TinyVector<int, 
      ndims>(1)), idxDET(X.lbound(), X.ubound(), TinyVector<int, ndims>(1));
  TinyVector<int, ndims> splitstride, shiftlbound, shiftubound;

  for(int dim=0; dim < ndims; ++dim) {
    // Stride must evenly divide range
    splitstride = 1; splitstride(dim) = 2;
    shiftubound = X.ubound() - (X.ubound() - X.lbound()) % splitstride;  
    idxAPP = StridedDomain<ndims>(X.lbound(), shiftubound, splitstride);
    shiftlbound = X.lbound();    ++shiftlbound(dim);
    shiftubound = X.ubound() - (X.ubound() - shiftlbound) % splitstride;  
    idxDET = StridedDomain<ndims>(shiftlbound, shiftubound, splitstride);
    // gamma = odd/detail, lambda = even/approx
    Array<T, ndims> gamma = X(idxAPP), lambda = X(idxDET);  //references
#ifdef TEST_ELS
    cerr << "idxAPP: \t\t\t idxDET:" << endl << idxAPP.lbound() << '\t' << 
      idxDET.lbound() << endl << idxAPP.ubound() << '\t' << idxDET.ubound() <<
      endl << idxAPP.stride() << '\t' << idxDET.stride() << endl;
#endif

    switch(type) {
      case primal:
        switch(LS<T>::template extMode<ndims>()(dim)) {
          case zpd:     // zero padding already done in LS::lwt
          case none: pu_none<ndims>(gamma, lambda, dim, t); break;
          case periodic: pu_periodic<ndims>(gamma, lambda, dim, t); break;
          default: throw invalid_argument("Unknown extension mode.\n");
        }
        break;
      case dual:
        switch(LS<T>::template extMode<ndims>()(dim)) {
          case zpd:     // zero padding already done in LS::lwt
          case none: pu_none<ndims>(lambda, gamma, dim, t); break;
          case periodic: pu_periodic<ndims>(lambda, gamma, dim, t); break;
          default: throw invalid_argument("Unknown extension mode.\n");
        }
        break;
      case normalize:
        if(t==forward) {
          gamma *= coefficients(0);
          lambda *= coefficients(1);
        } else {
          gamma /= coefficients(0);
          lambda /= coefficients(1);
        }
    }

#ifdef TEST_ELS
    cerr << "Dim " << dim << '\t';
    if(type==primal || t==normalize)
      cerr << "gamma (after els) " << gamma << endl;
    if(type==dual || t==normalize)
      cerr << "lambda (after els) " << lambda << endl;
#endif
  }

#ifdef TEST_ELS // Verify perfect reconstruction by applying the inverse ELS
  Array<T, ndims> Yout;
  if (test_els) {
    Yout.resize(X.shape()); Yout = X;
    test_els = false;    // Don't do this recursively !!
    LWTtype tinv = (t==forward ? inverse : forward);
    els_predict_update(Yout, tinv);
    test_els = true;
    double err = sqrt(sum(sqr(Yout-Yin))/(Yout.size()-1));    // RMS error
    assert(err < 1e-6);
  }
#endif
}

template <typename T> template <size_t ndims> 
inline void ELS<T>::pu_none(Array<T, ndims> A1, const Array<T, ndims> A2,
    const size_t dim, const LWTtype t) const {

  RectDomain<ndims> A1DomShift = A1.domain(), A2DomShift = A2.domain();
  assert(all(A1.extent()==A2.extent()));

  int bstart = max_degree + 1 - coefficients.rows(),  //min_degree
      bend = max_degree;
  const int &gl = A1.lbound(dim),     &gu = A1.ubound(dim),
      &ll = A2.lbound(dim),    &lu = A2.ubound(dim);
  assert(ll==gl);

  // A1 -= convolve(coefficients, A2)
  for(int b = bstart; b <= bend; ++b) {

    A2DomShift.lbound(dim) = max(ll, gl + b); 
    A1DomShift.lbound(dim) = max(gl, ll - b);
    A2DomShift.ubound(dim) = min(lu, gu + b);
    A1DomShift.ubound(dim) = min(gu, lu - b);
    // Make sure domain is not empty
    // TODO: This probably means LWT level is too high for data size
    if (A1DomShift.lbound(dim) <= A1DomShift.ubound(dim))
    {
      if(t==forward)
        A1(A1DomShift) += coefficients(max_degree - b) * A2(A2DomShift);
      else
        A1(A1DomShift) -= coefficients(max_degree - b) * A2(A2DomShift);
    }
  }
}

template <typename T> template <size_t ndims> 
inline void ELS<T>::pu_periodic(Array<T, ndims> A1, const Array<T, ndims> A2,
    const size_t dim, const LWTtype t) const {

  RectDomain<ndims> A1DomShift = A1.domain(), A2DomShift = A2.domain();
  assert(all(A1.extent()==A2.extent()));

  int bstart = max_degree + 1 - coefficients.rows(),  //min_degree
      bend = max_degree;

  // A1 -= convolve(coefficients, A2)
  for(int b = bstart; b <= bend; ++b) {
    for(int i = A1.lbound(dim); i <= A1.ubound(dim); ++i) {

      A1DomShift.lbound(dim) = A1DomShift.ubound(dim) = i;   //  Pick a slice
      A2DomShift.lbound(dim) = i + b;
      // Wrap around (periodic BC)
      while(A2DomShift.lbound(dim) > A2.ubound(dim))
        A2DomShift.lbound(dim) -= A2.extent(dim);
      while(A2DomShift.lbound(dim) < A2.lbound(dim))
        A2DomShift.lbound(dim) += A2.extent(dim);
      A2DomShift.ubound(dim) = A2DomShift.lbound(dim);

      if(t==forward)
        A1(A1DomShift) += coefficients(max_degree - b) * A2(A2DomShift);
      else
        A1(A1DomShift) -= coefficients(max_degree - b) * A2(A2DomShift);

    }
  }
}


/************** LS constructor ********************************/

/************** Set initialization and compute alpha **********/

template <typename T>
unsigned LS<T>::set_init(bool init)
{
  initialize = init;
  if(initialize && alpha.size()==0)
  {
    if(wname=="haar" || lf==2)
    {
      alpha.resize(1);
      alpha = 1;
    } else {
      const int iter = 10;
      vector<Array<T, 1> > phipsi(wavefun(iter));
      cumsum(phipsi[0], firstDim) /= (1<<iter); // integrate
      alpha.resize(phipsi[0].rows() >> iter);
      alpha = phipsi[0](Range(1<<iter, toEnd, 1<<iter));  
      // skip first zero and 0!=fromStart
      for(int i=alpha.ubound(firstDim); i>alpha.lbound(firstDim); --i)
        alpha(i) -= alpha(i-1);
      alpha.reverseSelf(firstDim); // Need correlation instead of conv
    }
    return alpha.size()-1;
  }
  else return 0;
}

/************** Add ELS ***************************************/
template <typename T>
  void LS<T>::addlift(ELS<T> els) {
    if (!LSteps.empty() && LSteps.back().gettype()==normalize)
      throw domain_error("Can't add ELS after normalization step.\n");
    LSteps.push_back(els);
    if (els.gettype()!=normalize) {
      lf += els.length();
      TinyVector<int, 2> els_range(els.mindeg(), els.maxdeg());
      els_range = 2*els_range + (els.gettype()==primal ? 1 : -1);
      if (els_range(0) > 0) els_range(0) = 0;
      if (els_range(1) < 0) els_range(1) = 0;
      ext_range += els_range;
    }
  }

/************ Forward LWT (inplace) ************************************/

template <typename T> template<size_t ndims>
void LS<T>::lwt_inplace(Array<T, ndims> &X, const int J) const {

  if (!all(extMode<ndims>() == none)) // inplace LWT iff no extensions required
    throw invalid_argument("Inplace LWT not compatible with extension.\n");

  // Compute low level wavelet coefficients by convolution
  // TODO: This assumes zero padding
  if (initialize)
    convolveTP(X, alpha);

  TinyVector<int, ndims> APPstride = 1, APPlbound = X.lbound(),
    APPubound = X.ubound();
  StridedDomain<ndims> idxAPP(APPlbound, APPubound, APPstride);
  // Compute level j coefficients
  for(int j=0; j<J; ++j) {

    // For each elementary lifting step
    for(lift_citer els_it = LSteps.begin(); 
        els_it != LSteps.end(); ++els_it) {
      els_it->template els_predict_update<ndims>(X(idxAPP), forward);
    }

    APPstride <<= 1;
    APPubound -= (APPubound - APPlbound) % APPstride;
    idxAPP = StridedDomain<ndims>(APPlbound, APPubound, APPstride);
  }
}

/************ Forward LWT  (out of place) ************************************/
// Compute level J LWT. Detail coefficients are returned in the vector while
// the input array contains the approx coefficients.
template <typename T> template<size_t ndims>
void LS<T>::lwt(Array<T, ndims> &X, vector<Array<TinyVector<T, ncomps>, ndims> > &Y) const 
{
  const int J = Y.size();
  Array<T, ndims> Xapp;
  // Compute low level wavelet coefficients by convolution
  if (initialize) {
    convolveTP<T, ndims>(X, alpha, extMode<ndims>());
    D("Low level coeff estimate :", X);
  }

  TinyVector<int, ndims> COMPlbound, COMPubound, sx, lext, rext, newsize;
  TinyVector<bool, ndims> bexp;
  const int perSizeDiv = (1<<J);
  for (size_t d=0; d<ndims; ++d)
  {
    bexp(d) = extMode<ndims>()(d) == periodic || extMode<ndims>()(d) == none;
    if (extMode<ndims>()(d) == periodic && !(X.extent(d)%perSizeDiv==0))
      throw invalid_argument("For periodic extension, 2^J should divide length.\
          \n");
  }

  // Compute level j coefficients
  for(int j=0; j<J; ++j) {

    // Extend
    sx = X.extent();
    // final size = 2*floor((sx+lf-1)/2) = even
    // lext + rext = (lf-1) if sx+lf is odd, else (lf-2)
    lext = lf-2;  lext = where_else<int>(bexp, 0, lext);
    rext = lf-2 + (sx+lf)%2; rext = where_else<int>(bexp, 0, rext);
    extend<T, ndims>(X, lext, rext, extMode<ndims>());

    // Lift (for each elementary lifting step)
    for(lift_citer els_it = LSteps.begin(); 
        els_it != LSteps.end(); ++els_it) {
      els_it->template els_predict_update<ndims>(X, forward);
    }

    // Dis-assemble components
    // ceil(lx/2), floor((lx+lf-1)/2)
    newsize = where_else(bexp, TinyVector<int, ndims>(sx/2), 
        TinyVector<int, ndims>((sx+lf-1)/2) );  
    Y[j].resize(newsize); Xapp.resize(newsize);
    newsize *= 2;
    for (int nc = ncomps; nc >= 0; --nc) {
      unsigned rsnc = nc;
      for (int d = 0; d < ndims; ++d, rsnc >>= 1)
        COMPlbound(d) = (rsnc & 1) ? X.ubound(d)-(newsize(d)-2) : X.lbound(d);
      COMPubound = COMPlbound + (newsize-2);
      if(nc>0)
        Y[j][nc-1] = X(StridedDomain<ndims>(COMPlbound, COMPubound, 2));
      else
        Xapp = X(StridedDomain<ndims>(COMPlbound, COMPubound, 2));
    }
    swap(X, Xapp); Xapp.free();
  }
}

/************ Inverse LWT (out of place) ***********************************/
// Xapp = input approx coeffs, Y = detail coeffs.
// TODO: Doesn't work with initialization
template <typename T> template<size_t ndims> 
void LS<T>::ilwt(Array<T, ndims> &Xapp, const vector<Array<TinyVector<T, 
    ncomps>, ndims> > &Y, const TinyVector<bool, ndims> szodd, 
    const int Jfinal) const
{ 
  const int Jnow = Y.size();
  if (Jnow <= Jfinal)
    throw invalid_argument("Final level should be < current level.\n");

  TinyVector<int, ndims> COMPlbound, COMPubound, newsize, lext, rext,
    newsize_ext, newpersize;
  //WARNING: szdiff may be used uninitialized in this function
  TinyVector<bool, ndims> szdiff(false), bexp;
  Array<T, ndims> X;

  for (size_t d=0; d<ndims; ++d)
    bexp(d) = extMode<ndims>()(d) == periodic || extMode<ndims>()(d) == none;

  for(int j=Jnow-1; j>=Jfinal; --j)
  {
    // Reassemble level j LWT
    newpersize = 2*Xapp.extent();
    if (j>0) {
      newsize = Y[j-1].extent();
      { // size Consistency tests
        TinyVector<bool, ndims> persize = (newsize == 2*Xapp.extent()),
          nonpersize = (newsize + lf-1 == 2*Xapp.extent() || 
              newsize + lf-1 == 2*Xapp.extent()+1), isnotperext;
        for(int d=0; d<ndims; ++d)
          isnotperext(d) = (extMode<ndims>()(d) != periodic);
        assert(all(persize ^ isnotperext));
        assert(all(nonpersize ^ persize));
      }
      szdiff = ((newsize + lf)%2==1);
    } else {
      // Test if (sx+lf-1) is odd
      newsize = 2*Xapp.extent() + (szodd+lf-1)%2 - lf + 1; 
      newsize = where_else(bexp, newpersize, newsize);
    }

    newsize_ext = newsize + 2*lf-4 + (newsize+lf)%2;
    newsize_ext = where_else(bexp, newpersize, newsize_ext);
    X.resize(newsize_ext);
    for (int nc=0; nc <= ncomps; ++nc) {
      unsigned rsnc = nc;
      for (int d = 0; d < ndims; ++d, rsnc >>= 1)
        COMPlbound(d) = (rsnc & 1) ? X.ubound(d)-(2*Xapp.extent(d)-2) :
          X.lbound(d);
      COMPubound = COMPlbound + (2*Xapp.extent()-2);
      X(StridedDomain<ndims>(COMPlbound, COMPubound, 2)) 
        = (nc==0 ? Xapp : Y[j][nc-1]);
    }
    // Lift
    // For each elementary lifting step
    for(lift_creviter els_it = LSteps.rbegin(); 
        els_it != LSteps.rend(); ++els_it)
      els_it->template els_predict_update<ndims>(X, inverse);

    // Trim extra parts
//    newsize = X.extent() + 1 - lf;
    lext = lf-2; rext = lf-2;
    if (j>0)  rext += szdiff;
    else      rext += szodd;
    lext = where_else(bexp, 0, lext);
    rext = where_else(bexp, 0, rext);

    // shrink_center(X, newsize);
    shrink<T, ndims>(X, lext, rext);
    swap(X, Xapp);
  }
}


/************ Inverse LWT (inplace) ***********************************/
// TODO: Won't work with initialization
template <typename T> template<size_t ndims>
void LS<T>::ilwt(Array<T, ndims> &X, const int Jnow, const int Jfinal) const 
{ 
  for(size_t d=0; d<ndims; ++d)
    if (extMode<ndims>()(d) != none)
      throw invalid_argument("In place LWT works only without signal \
          extension.\n");
  if (Jnow <= Jfinal)
    throw invalid_argument("Final level should be < current level.\n");

  TinyVector<int, ndims> APPstride = 1<<Jnow, APPlbound = X.lbound(),
    APPubound = X.ubound();
  StridedDomain<ndims> idxAPP(APPlbound, APPubound, APPstride);
  // Compute level j coefficients
  for(int j=Jnow-1; j>=Jfinal; --j) {

    APPstride >>= 1;
    APPubound = X.ubound() - (X.ubound() - APPlbound) % APPstride;
    idxAPP = StridedDomain<ndims>(APPlbound, APPubound, APPstride);

    // For each elementary lifting step
    for(lift_creviter els_it = LSteps.rbegin(); 
        els_it != LSteps.rend(); ++els_it)
      els_it->template els_predict_update<ndims>(X(idxAPP), inverse);

  }
}

/************** Undecimated/stationery Forward LWT *********************/
/* Chang Soo Lee, Chang Kil Lee, and Kyung Yul Yoo, “New lifting based 
 * structure for undecimated wavelet transform,” Electronics Letters,  
 * vol. 36, 2000, pp. 1894-1895. */
/*template<size_t ndims>    
void ulwt(Array<T, ndims> X, vector<Array<TinyVector<T, ncomps>, ndims> > &Y)
  const 
{
}*/


/************** Compute wavelet and scaling functions ****************/
// TODO: USe inplace LWT, return with std::pair
// The cascade algorithm is used (take inverse LWT starting from only 1
// non-zero coefficient.)
// Only works properly for orthogonal wavelets
// The returned values are over intervals of 2^-J, with fun[x](0) the value at
// 0.
template <typename T>
vector<Array<T, 1> > LS<T>::wavefun(const size_t J) const
{
  //  double pas = pow(2, -J), nbpts = (lf-1)/pas + 1;
  double coef = (1<<(J/2));
  if (J%2==1)
    coef *= M_SQRT2;  
  vector<Array<T, 1> > fun(2);
  vector<Array<TinyVector<T, 1>, 1> > Y(J);
  //save old extension mode
  TinyVector<extModeEnum, 1> oldextmode = extMode<1>();
  extMode<1>() = zpd;

  // Set size so that all non-zeros survive despite trimming in ilwt
  // Size is matched to outpt size of Matlab's wavefun
  int sz = 2*lf-3, pos = (sz-1)/2;
  fun[0].resize(sz);        // phi
  fun[1].resize(sz);        // psi
  fun[0] = 0; fun[1] = 0;
  fun[0](pos) = 1;

  for (int j=J-1; j>=0; --j, sz=2*sz-(lf-2)) {
    Y[j].resize(sz);
    Y[j] = TinyVector<T, 1>(T(0));
  }

  ilwt<1>(fun[0], Y, TinyVector<bool, 1>(false));

  Y[J-1](pos) = 1;
  ilwt<1>(fun[1], Y, TinyVector<bool, 1>(false));

  fun[0] *= coef;
  fun[1] *= coef;
  // shift everything by lf-3 elements to make fun[x](0) correspond to zero
  fun[0].reindexSelf(TinyVector<int, 1>(-lf+3));
  fun[1].reindexSelf(TinyVector<int, 1>(-lf+3));

  extMode<1>() = oldextmode;    //reset extension mode
  return fun;
}

/************** Compute wavelet and scaling functions ****************/
// Test LWT by computing approximations to the wavelet and scaling functions.
// These can be compared with those obtained from Matlab.
// The cascade algorithm is used (take inverse LWT starting from only 1
// non-zero coefficient.)
// Only works properly for orthogonal wavelets
// The returned values are over intervals of 2^-J, with fun[x](0) the value at
// 0.

template <typename T> template <size_t ndims>
auto_ptr<pair<Array<T, ndims>, Array<T, ndims> > > LS<T>::wsfun(const size_t J)
  const
{
  // Normalization
  double coef = (1<<(J*ndims/2));
  if ((J*ndims)%2==1)
    coef *= M_SQRT2; 

  auto_ptr<pair<Array<T, ndims>, Array<T, ndims> > > wsf(new 
    pair<Array<T, ndims>, Array<T, ndims> >);
  //save old extension mode
  TinyVector<extModeEnum, ndims> oldextmode = extMode<ndims>();
  extMode<ndims>() = none;
  
  wsf->first.resize(TinyVector<size_t, ndims>((lf-1)*(1<<J)));
  wsf->first(TinyVector<size_t, ndims>(size_t(0))) = coef;
  wsf->second.resize(TinyVector<size_t, ndims>((lf-1)*(1<<J)));
  wsf->second(TinyVector<size_t, ndims>((lf-1)*(1<<(J-1)))) = 
    ndims%2==0 ? coef : -coef;
  //TODO: Why is the wavelet inverted ?

  ilwt<ndims>(wsf->first, J);
  ilwt<ndims>(wsf->second, J);
  
  extMode<ndims>() = oldextmode;    //reset extension mode

  return wsf;
}

/*************** Display wavelet transform ********************/
template <typename T, size_t ndims> 
ostream & display(ostream &os, const Array<T, ndims> &A, const 
    vector<Array<TinyVector<T, ncomps>, ndims> > &D)
{
  os << "Approximation coefficients: " << A << endl << "Detail coefficients: " 
    << endl;
  for(int j=D.size()-1; j>=0; --j)
  {
    os << "Scale " << D.size()-j-1 << ": ";
    for(unsigned nc=0; nc<ncomps; ++nc)
      os << D[j][nc] << endl;
  }
  return os;
}


#endif //_LIFT_IMPL_H

// CaresianProduct indirection implementation of pu_periodic
// doesn't work because indirectArray does not provide needed functionality
//
// template <typename T> template <size_t ndims> 
// inline void ELS<T>::pu_periodic(Array<T, ndims> A1, const Array<T, ndims> A2,
//     const size_t dim, const LWTtype t = forward) const {
// 
//   static TinyVector<vector<int>, ndims> A1DomShift, A2DomShift;
//   if(A1DomShift(0).empty()) {   // initialize
//     for(int d=0; d<ndims; ++d) {
//       A1DomShift(d).reserve(A1.extent(d));
//       for(int i=0; i<A1.extent(d); ++i)
//         A1DomShift(d)[i] = i + A1.lbound(d);
//       A2DomShift(d).reserve(A2.extent(d));
//       for(int i=0; i<A2.extent(d); ++i)
//         A2DomShift(d)[i] = i + A2.lbound(d);
//     }
//   }
// 
//   int bstart = max_degree + 1 - coefficients.extent(firstDim),  //min_degree
//   bend = max_degree;
// 
//   if (A1.extent(dim)==A2.extent(dim)) {     // in-place WT possible
// 
//     // A1 -= convolve(coefficients, A2)
//     for(int b = bstart; b <= bend; ++b) {
// 
//       for(int i=0; i<A1.extent(dim); ++i) {
//         A1DomShift(dim)[i] = i + A1.lbound(dim) - b;
//         if (A1DomShift(dim)[i] < A1.lbound(dim))
//           A1DomShift(dim)[i] += A1.extent(dim);
//         else if (A1DomShift(dim)[i] > A1.ubound(dim))
//           A1DomShift(dim)[i] -= A1.extent(dim);
//       }
//       for(int i=0; i<A2.extent(dim); ++i) {
//         A2DomShift(dim)[i] = i + A2.lbound(dim) + b;
//         if (A2DomShift(dim)[i] < A2.lbound(dim))
//           A2DomShift(dim)[i] += A2.extent(dim);
//         else if (A2DomShift(dim)[i] > A2.ubound(dim))
//           A2DomShift(dim)[i] -= A2.extent(dim);
//       }
// 
//       if(t==forward)
//         A1[indexSet(A1DomShift)] = A1 + coefficients(max_degree - b + 
//             coefficients.base(firstDim)) * A2[indexSet(A2DomShift)];
//       else
//         A1[indexSet(A1DomShift)] = A1 - coefficients(max_degree - b + 
//             coefficients.base(firstDim)) * A2[indexSet(A2DomShift)];
//     }
//   } else {                                  // in-place WT NOT possible
// 
// 
//   }
// 
//   for(int i=0; i<A1.extent(dim); ++i)
//     A1DomShift(dim)[i] = i + A1.lbound(dim);
//   for(int i=0; i<A2.extent(dim); ++i)
//     A2DomShift(dim)[i] = i + A2.lbound(dim);
// }

