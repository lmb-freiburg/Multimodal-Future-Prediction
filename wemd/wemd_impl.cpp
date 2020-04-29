#include  <vector>
using std::vector;
using std::pair;

#include  <blitz/array.h>
#include "wemd_impl.h"
#include <stdio.h>

using namespace blitz;


extern "C" float WEMD_1D(float* hist1, float* hist2, int size)
{
  const int ndims = 1;
  float tper(0.0), s(1);
  float C0(0);
  TinyVector<bool, ndims> isper(false);

  vector<Array<float, ndims> > H;
  vector<pair<vector<unsigned>, vector<float> > > wd;

  Array<float, ndims> A(hist1, shape(size), neverDeleteData);
  Array<float, ndims> B(hist2, shape(size), neverDeleteData);

  H.push_back(A);
  H.push_back(B);

  size_t numel = wemddes<float, ndims>(H, wd, isper, s, C0, tper, "sym5");

  float distance = 0.0;
  for(int j=0; j<wd[0].first.size(); ++j)
  {
      int index = wd[0].first[j];
      for(int k=0; k<wd[1].first.size(); ++k)
      {
        if(index == wd[1].first[k])
        {
            distance += abs(float(wd[0].second[j])-float(wd[1].second[k]));
            break;
        }
      }
  }
  return distance;
}

extern "C" float WEMD_2D(float* hist1, float* hist2, int width, int height)
{
  const int ndims = 2;
  float tper(0.01), s(1);
  float C0(0);
  TinyVector<bool, ndims> isper(false);

  vector<Array<float, ndims> > H;
  vector<pair<vector<unsigned>, vector<float> > > wd;

  Array<float, ndims> A(hist1, shape(width, height), neverDeleteData);
  Array<float, ndims> B(hist2, shape(width, height), neverDeleteData);

  H.push_back(A);
  H.push_back(B);

  size_t numel = wemddes<float, ndims>(H, wd, isper, s, C0, tper, "sym5");

  float distance = 0.0;
  for(int j=0; j<wd[0].first.size(); ++j)
  {
      int index = wd[0].first[j];
      for(int k=0; k<wd[1].first.size(); ++k)
      {
        if(index == wd[1].first[k])
        {
            distance += abs(float(wd[0].second[j])-float(wd[1].second[k]));
            break;
        }
      }
  }
  return distance;
}