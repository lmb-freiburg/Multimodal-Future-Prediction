/** @author Sameer Sheorey */
#ifndef _SPARSE_H
#define _SPARSE_H

/* Some functions for sparse vector manipulation using a map<unsigned, T> for
 * data storage
 */

#include  <iostream>
using std::ostream;
#include  <blitz/array.h>
#include  <map>
using std::map;
#include  <utility>
using std::pair;

using namespace blitz;

template <typename T>
inline map<unsigned, T> & operator-=(map<unsigned, T> &u, const map<unsigned, T> &v)
{
  typename map<unsigned, T>::iterator iu = u.begin();
  typename map<unsigned, T>::const_iterator iv = v.begin();

  for(;iv!=v.end(); ++iu) {
    if (iu->first == iv->first) {
      iu->second -= iv->second;
      ++iv;
    }
    else if (iu->first > iv->first) {
      iu = u.insert(iu, pair<unsigned, T>(iv->first, -iv->second));
      ++iv;
    }
  }
  return u;
}

template <typename T>
inline map<unsigned, T> & operator+=(map<unsigned, T> &u, const map<unsigned, T> &v)
{
  typename map<unsigned, T>::iterator iu = u.begin();
  typename map<unsigned, T>::const_iterator iv = v.begin();

  for(;iv!=v.end(); ++iu) {
    if (iu->first == iv->first) {
      iu->second += iv->second;
      ++iv;
    }
    else if (iu->first > iv->first) {
      iu = u.insert(iu, pair<unsigned, T>(iv->first, iv->second));
      ++iv;
    }
  }
  return u;
}


template <typename T>
inline map<unsigned, T> & operator*=(map<unsigned, T> &u, const map<unsigned, T> &v)
{
  typename map<unsigned, T>::iterator iu = u.begin();
  typename map<unsigned, T>::const_iterator iv = v.begin();

  for(;iv!=v.end(); ++iu) {
    if (iu->first == iv->first) {
      iu->second *= iv->second;
      ++iv;
    }
  }
  return u;
}

// Ignores v=0 (no inf in output)
template <typename T>
inline map<unsigned, T> & operator/=(map<unsigned, T> &u, const map<unsigned, T> &v)
{
  typename map<unsigned, T>::iterator iu = u.begin();
  typename map<unsigned, T>::const_iterator iv = v.begin();

  for(;iv!=v.end(); ++iu) {
    if (iu->first == iv->first) {
      iu->second /= iv->second;
      ++iv;
    }
  }
  return u;
}

// unary -
template <typename T>
inline map<unsigned, T> & operator-(map<unsigned, T> & v) 
{
  typename map<unsigned, T>::iterator iv = v.begin();
  for(;iv!=v.end(); ++iv)
    iv->second = -iv->second;
  return v;
}

// Scalar multiplication
template <typename T>
inline map<unsigned, T> & operator*=(map<unsigned, T> & v, const T a) 
{
  typename map<unsigned, T>::iterator iv = v.begin();
  if (a!=1)
    for(;iv!=v.end(); ++iv)
      iv->second *= a;
  return v;
}

template <typename T>
inline T norm(const map<unsigned, T> & v, int p) {

  T np = 0;
  typename map<unsigned, T>::const_iterator iv = v.begin();
  switch(p) {
    case 0: return v.size();
    case 1: for(;iv!=v.end(); ++iv)
              np += abs(iv->second);
            return np;
    case 2: for(;iv!=v.end(); ++iv)
              np += (iv->second)*(iv->second);
            return sqrt(np);
    default: for(;iv!=v.end(); ++iv)
               np += pow(abs(iv->second), p);
             return pow(np, 1/p);
  }
}

// ax + y
template <typename T>
inline map<unsigned, T> axpy(const T a, const map<unsigned, T> &x, 
    const map<unsigned, T> & y)
{
  map<unsigned, T> u;
  if(a==0) 
    return y;
  else {
    u = x;
    u *= a;
    u += y;
    return u;
  }
}

// Convert u,v into dense arrays; If an element is 0 in both, it is ignored.
template <typename T>
inline void copy_common(const map<unsigned, T> &u, const map<unsigned, T> &v,
    blitz::Array<T, 1> &ud, blitz::Array<T, 1> &vd)
{
  // find number of keys
  typename map<unsigned, T>::const_iterator iu = u.begin(), iv = v.begin();
  int common = 0, sz;
  for(;iv!=v.end(); ++iu) {
    if (iu->first == iv->first) {
      ++common;
      ++iv;
    }
  }
  sz = u.size() + v.size() - common;
  ud.resize(TinyVector<int, 1>(sz));
  vd.resize(TinyVector<int, 1>(sz));

  int i=0;
  for(iu = u.begin(), iv = v.begin(); iv!=v.end() || iu!=u.end(); ++i) {
    if ((iu->first) < (iv->first)) {
      ud(i) = iu->second;
      vd(i) = 0;
      ++iu;
    } else if  (iu->first > iv->first) {
      vd(i) = iv->second;
      ud(i) = 0;
      ++iv;
    } else {
      ud(i) = iu->second;
      vd(i) = iv->second;
      ++iu;
      ++iv;
    }
  }
}

template <typename T>
ostream & operator<<(ostream &os, const map<unsigned, T> &u)
{
  for(typename map<unsigned, T>::const_iterator iu = u.begin(); iu!= u.end(); 
      ++iu)
    os << iu->first << ": " << iu->second << endl;
  return os;
}

#endif // _SPARSE_H
