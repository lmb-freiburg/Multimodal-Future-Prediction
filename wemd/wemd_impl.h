/** @file
 * Implementation of functions declared in wemd.h
 * @author Sameer Sheorey
 */ 

#ifndef _WEMD_IMPL_H
#define _WEMD_IMPL_H

#include  <limits>
#include  <stdexcept>   // For runtime errors
using std::invalid_argument; using std::out_of_range;
#include  <algorithm>
using std::sort;
#include  <cmath>
#include  <vector>
using std::vector;
#include  <map>
using std::map;
#include  <string>
using std::string;
#include  <utility>
using std::pair;

#include  <blitz/array.h>
#include  "sparse.h"
#include  "lift.h"

using namespace blitz;

// Set elements to zero if they are less than tper * (mean absolute value)
template <typename T, size_t ndims>
unsigned wemddes(vector<Array<T, ndims> > &H,
    vector<pair<vector<unsigned>, vector<T> > > &wd,     
    const TinyVector<bool, ndims> isperiodic, const float s, const T C0,
    const float tper, const string wname)
{

  if(numeric_limits<T>::epsilon() >= 1)
    throw invalid_argument("For floating point data types only.\n");

  if(s<=0 || s>1)
    throw out_of_range("s must be in the range (0,1].\n");
  if(C0<0)
    throw out_of_range("C0 >= 0\n");

  wd.clear();
  wd.resize(H.size());

  LS<T>::template extMode<ndims>() = where_else(isperiodic, periodic, zpd);
  LS<T> wtls(wname);
  unsigned init_ext = wtls.set_init(true);
  // TODO: Make sure that lifting scheme is orthonormal wavelet with
  // regularity atleast 1 ?

  T wl1norm, thresh;
  unsigned numel(0);

  TinyVector<int, ndims> sz = H[0].shape();
  // max level
  int J = (int)ceil(log((double)min(H[0].shape())+init_ext)/log((double)2));
  //cout << "J is: " << J << endl;
  //cout << "init_ext is: " << init_ext << endl;
  // Restriction due to periodic extension mode ( 2^J | H.shape() )
  for (int d=0; d<ndims; ++d)
  {
    if(!isperiodic(d)) continue;
    unsigned t;
    int lev;    // lev = log2(H[0].extent(d)), J = min(J, lev)
    for(lev=0, t=H[0].extent(d); (t&1)==0; t>>=1, ++lev);
    if (J>lev)
      J = lev;
  }
  //cout << "J is: " << J << endl;
  //cout << "ndims: " << ndims << endl;
  //cout << "ncomps: " << ncomps << endl;

  vector<Array<TinyVector<T, ncomps>, ndims> > Y(J);

  numel = 0;
  for( int i=0; i<(int)H.size(); ++i)
  {
    wl1norm = 0;

    if( any(H[i].shape() != sz) )
      throw invalid_argument("All histograms must be the same size.\n");

    //cout << H[i] << endl;
    wtls.template lwt<ndims>(H[i], Y);
    //display(cout, H[i], Y);
    for(int j=0; j<J; ++j) // Scale of Y[j] is actually j-J+1
    {
      Y[j] *= pow(2, (j-J+1)*(s+double(ndims)/2.0));
      for(int nc=0; nc<ncomps; ++nc)
        wl1norm += (T)sum(abs(Y[j][nc]));
      if(i==0)
        numel += Y[j].size() * ncomps;
    }
    H[i] *= C0;
    wl1norm += (T)sum(abs(H[i]));
    if(i==0)
      numel += H[i].size();
    //display(cout, H[i], Y);

    thresh = tper * wl1norm / numel;
    //cout << "numel: " << numel << endl;
    //cout << "thresh: " << thresh << endl;
    
    /*
    for (int j=0; j<J; ++j)
    {
        cout << "level j" << j << ": " << Y[j] << endl;
    }*/

    // Convert to 1D sparse array = pair<vector<unsigned> idx, vector<T> value>
    unsigned idx = 0;
    const T* vp;
    const TinyVector<T, ncomps>* tvp;

    // H[i], Y[j] should have contiguous storage, not reversed
    if(C0>0)
    {
      vp = H[i].data();
      for(int k=0; k<H[i].size(); ++k, ++idx)
        if (abs(vp[k]) > thresh)
        {
          wd[i].first.push_back(idx);
          wd[i].second.push_back(vp[k]);
        }
    }
    else
      idx = H[i].size();

    for (int j=0; j<J; ++j) {
      tvp = Y[j].data();
      for (int nc=0; nc<ncomps; ++nc)
        for(int k=0; k<Y[j].size(); ++k, ++idx)
          if (abs(tvp[k](nc)) > thresh)
          {
            wd[i].first.push_back(idx);
            wd[i].second.push_back(tvp[k](nc));
          }
    }
  }
  return numel;
}

// Best partial match : \min_{\alpha>=0} ||u \alpha - v||_1
// TODO Should we cap \alpha at 1 ?
template <typename T> T bpm(const map<unsigned, T> &u, 
    const map<unsigned, T> &v)
{
  Array<T, 1> ud, vd, alpha;
  copy_common(u, v, ud, vd);
  alpha.resize(u.extent());
  alpha = v/u;

  for (int i=alpha.lbound(0); i<alpha.ubound(0); ++i)
    if (isinf(alpha(i)) || isnan(alpha(i)) || alpha(i)<0)
      alpha(i) = 0;
    else if (alpha(i) > 1)
      alpha(i) = 1;

  sort(alpha.begin(), alpha.end());

  int i=alpha.lbound(0), j=alpha.ubound(0), k;
  T fi = sum(abs(alpha(i)*u-v)), fj = sum(abs(alpha(j)*u-v)), fk, fkp1;

  while(fi != fj)
  {
    k = (i+j)/2;
    fk = sum(abs(alpha(k)*u-v));
    fkp1 = sum(abs(alpha(k+1)*u-v));

    if (fkp1==fk)
      return fk;
    if (fkp1<fk) {
      i = k+1;
      fi = fkp1;
    } else {
      j = k;
      fj = fk;
    }
  }
  return fi;
}


// Best partial match : \min_{\alpha>=0} ||u \alpha - v||_1
// TODO Should we cap \alpha at 1 ?
template <typename T> T bpm(const Array<T, 1> &u, const Array<T, 1> &v)
{
  Array<T, 1> alpha = v/u;

  for (int i=alpha.lbound(0); i<alpha.ubound(0); ++i)
    if (isinf(alpha(i)) || isnan(alpha(i)) || alpha(i)<0)
      alpha(i) = 0;
    else if (alpha(i) > 1)
      alpha(i) = 1;

  sort(alpha.begin(), alpha.end());

  int i=alpha.lbound(0), j=alpha.ubound(0), k;
  T fi = sum(abs(alpha(i)*u-v)), fj = sum(abs(alpha(j)*u-v)), fk, fkp1;

  while(fi != fj)
  {
    k = (i+j)/2;
    fk = sum(abs(alpha(k)*u-v));
    fkp1 = sum(abs(alpha(k+1)*u-v));

    if (fkp1==fk)
      return fk;
    if (fkp1<fk) {
      i = k+1;
      fi = fkp1;
    } else {
      j = k;
      fj = fk;
    }
  }
  return fi;
}

#endif //_WEMD_IMPL_H
