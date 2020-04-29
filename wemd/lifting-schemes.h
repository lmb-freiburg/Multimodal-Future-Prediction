/** @author Sameer Sheorey
 */ 
#ifndef _LIFTING_SCHEMES_H
#define _LIFTING_SCHEMES_H

#include <cmath>
#include <string>
using std::string;
#include  <stdexcept>   // For runtime errors
using std::invalid_argument;

const double M_SQRT3 = sqrt(double(3));

/************** Pre-defined lifting schemes ********************
 * Currently known wavelets: lazy, haar, db1-4, sym2-5, coif1-2
 * The others do not work YET.

 The valid values for WNAME are:
 'lazy'
 'haar', 
 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8'
 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8'
 Cohen-Daubechies-Feauveau wavelets:
 'cdf1.1','cdf1.3','cdf1.5' - 'cdf2.2','cdf2.4','cdf2.6'
 'cdf3.1','cdf3.3','cdf3.5' - 'cdf4.2','cdf4.4','cdf4.6'
 'cdf5.1','cdf5.3','cdf5.5' - 'cdf6.2','cdf6.4','cdf6.6'
 'biorX.Y' , see WAVEINFO
 'rbioX.Y' , see WAVEINFO
 'bs3'  : identical to 'cdf4.2'
 'rbs3' : reverse of 'bs3'
 '9.7'  : identical to 'bior4.4' 
 'r9.7' : reverse of '9.7'

Note:
'cdfX.Y' == 'biorX.Y' except for bior4.4 and bior5.5.
'rbioX.Y'  is the reverse of 'biorX.Y'
'haar' == 'db1' == 'bior1.1' == 'cdf1.1'
'db2'  == 'sym2'  and  'db3' == 'sym4'

 ****************************************************************/
template <typename T>
LS<T>::LS(string wname_) : wname(wname_), LSteps(), lf(0), ext_range(0), 
  initialize(false), alpha() 
{

  bool reverse_steps = false; // reverse lifting steps ?
  bool unknown_wname = false;
  if(wname == "haar")   wname = "db1";
  if(wname == "bs3")    wname = "cdf4.2";
  if(wname == "9.7")    wname = "bior4.4";
  if(wname == "rbs3") { wname = "cdf4.2"; reverse_steps = true; }
  if(wname == "r9.7") { wname = "bior4.4"; reverse_steps = true; }

  Array<T, 1> coeffs1(1), coeffs2(2), coeffs3(3);

  switch(wname[0]) {
    case 'l': // Lazy wavelets
      coeffs2 = 1, 1; addlift(ELS<T>(normalize, coeffs2));
      break;

    case 'd': // Daubechies db1 ... db8
      switch(wname[2]) {

        case '1':
          coeffs1 = -1; 
          addlift(ELS<T>(dual, coeffs1, 0));
          coeffs1 = 0.5; 
          addlift(ELS<T>(primal, coeffs1, 0));
          coeffs2 = T(M_SQRT2), T(M_SQRT1_2); 
          addlift(ELS<T>(normalize, coeffs2));
          break;

        case '2':
          coeffs1 = T(-M_SQRT3); 
          addlift(ELS<T>(dual, coeffs1, 0));
          coeffs2 = T((M_SQRT3-2)/4), T(M_SQRT3/4); 
          addlift(ELS<T>(primal, coeffs2, 1));
          coeffs1 = 1; 
          addlift(ELS<T>(dual, coeffs1, -1));
          coeffs2 = T((M_SQRT3+1)/M_SQRT2), T((M_SQRT3-1)/M_SQRT2); 
          addlift(ELS<T>(normalize, coeffs2));
          break;

        case '3':
          coeffs1 = T(-2.4254972439123361); 
          addlift(ELS<T>(dual, coeffs1, 0));
          coeffs2 = T(-0.0793394561587384), T(0.3523876576801823); 
          addlift(ELS<T>(primal, coeffs2, 1));
          coeffs2 = T(2.8953474543648969), T(-0.5614149091879961); 
          addlift(ELS<T>(dual, coeffs2, -1));
          coeffs1 = T(0.0197505292372931); 
          addlift(ELS<T>(primal, coeffs1, 2));
          coeffs2 = T(2.3154580432421348), T(0.4318799914853075); 
          addlift(ELS<T>(normalize, coeffs2));
          break;

        case '4':    // From Matlab filt2ls
          coeffs1 = T(-3.10293148586056); 
          addlift(ELS<T>(dual, coeffs1, 0));
          coeffs2 = T(-0.0763000865717434), T(0.291953126000889); 
          addlift(ELS<T>(primal, coeffs2, 1));
          coeffs2 = T(5.19949157304526), T(-1.66252835342021); 
          addlift(ELS<T>(dual, coeffs2, -1));
          coeffs2 = T(-0.00672237263275628), T(0.0378927481264294); 
          addlift(ELS<T>(primal, coeffs2, 3));
          coeffs1 = T(0.314106493410408); 
          addlift(ELS<T>(dual, coeffs1, -3));
          coeffs2 = T(2.61311836984573), T(0.382684539491043); 
          addlift(ELS<T>(normalize, coeffs2));


          break;
        case '5':unknown_wname = true; break;
//          coeffs1 = T(-0.2651451428113514); 
//          addlift(ELS<T>(dual, coeffs1, 1));
//          coeffs2 = T(0.9940591341382633),  T(0.2477292913288009); 
//          addlift(ELS<T>(primal, coeffs2, 0));
//          coeffs2 = T(-0.5341246460905558),  T(0.2132742982207803); 
//          addlift(ELS<T>(dual, coeffs2, 0));
//          coeffs2 = T(0.2247352231444452), T(-0.7168557197126235); 
//          addlift(ELS<T>(primal, coeffs2, 2));
//          coeffs2 = T(-0.0775533344610336),  T(0.0121321866213973); 
//          addlift(ELS<T>(dual, coeffs2, -2));
//          coeffs1 = T(-0.0357649246294110); 
//          addlift(ELS<T>(primal, coeffs1, 3));
//          coeffs2 = T(0.7632513182465389), T(1.3101844387211246); 
//          addlift(ELS<T>(normalize, coeffs2));
          break;
        case '6':
        case '7':
        case '8':
        default: unknown_wname = true;
      }
      break;

    case 's':   // Symmetric Daubechies sym2 ... sym8
      switch(wname[3]) {

        case '2':   // Same as db2
          coeffs1 = T(-M_SQRT3); 
          addlift(ELS<T>(dual, coeffs1, 0));
          coeffs2 = T((M_SQRT3-2)/4), T(M_SQRT3/4); 
          addlift(ELS<T>(primal, coeffs2, 1));
          coeffs1 = 1; 
          addlift(ELS<T>(dual, coeffs1, -1));
          coeffs2 = T((M_SQRT3+1)/M_SQRT2), T((M_SQRT3-1)/M_SQRT2); 
          addlift(ELS<T>(normalize, coeffs2));
          break;

        case '3':   //filt2ls
          coeffs1 = T(2.42549724391234); 
          addlift(ELS<T>(primal, coeffs1, 0));
          coeffs2 = T(-0.352387657680182),  T(0.0793394561587384);
          addlift(ELS<T>(dual, coeffs2, 0));
          coeffs2 = T(0.561414909187996), T(-2.8953474543649);
          addlift(ELS<T>(primal, coeffs2, 2));
          coeffs1 = T(-0.0197505292372931);
          addlift(ELS<T>(dual, coeffs1, -2));
          coeffs2 = T(0.431879991485308), T(2.31545804324213);
          addlift(ELS<T>(normalize, coeffs2));
          break;

        case '4':  //filt2ls
          coeffs1 = T(0.39114694197004);
          addlift(ELS<T>(dual, coeffs1, 0));
          coeffs2 = T(6.04725100901425), T(-0.339243991864945); 
          addlift(ELS<T>(primal, coeffs2, 1));
          coeffs2 = T(-0.162031452039304), T(-0.0291990284237228); 
          addlift(ELS<T>(dual, coeffs2, -1));
          coeffs2 = T(-39.2360111543618), T(20.9669036111182); 
          addlift(ELS<T>(primal, coeffs2, 3));
          coeffs1 = T(0.0215828896069427);
          addlift(ELS<T>(dual, coeffs1, -3));
          coeffs2 = T(0.0958080709409193), T(10.4375340217074); 
          addlift(ELS<T>(normalize, coeffs2));
          break;

        case '5': //filt2ls
          coeffs1 = T(-1.07999184552398); 
          addlift(ELS<T>(primal, coeffs1, 0));
          coeffs2 = T(0.498523184228117),  T(1.88386645463912); 
          addlift(ELS<T>(dual, coeffs2, 0));
          coeffs2 = T(0.101688696479947),  T(-0.50075842493123); 
          addlift(ELS<T>(primal, coeffs2, 2));
          coeffs2 = T(-4.00415801263283), T(-17.2380322599159); 
          addlift(ELS<T>(dual, coeffs2, -2));
          coeffs2 = T(0.0210887838523621 ),  T(0.0537859694583165); 
          addlift(ELS<T>(primal, coeffs2, 4));
          coeffs1 = T(-31.3822009310036); 
          addlift(ELS<T>(dual, coeffs1, -4));
          coeffs2 = T(4.73638905584892), T(0.211131304504031); 
          addlift(ELS<T>(normalize, coeffs2));
          break;

        case '6':
        case '7':
        case '8':
        default:  unknown_wname = true;
      }
      break;

    case 'c': 
      switch(wname[1]) {
        case 'o':   // Coiflets
          switch(wname[4]) {
            case '1':
              coeffs1 = T(4.6457513110481772); 
              addlift(ELS<T>(dual, coeffs1, 0));
              coeffs2 = T(-0.1171567416519999), T(-0.2057189138840000); 
              addlift(ELS<T>(primal, coeffs2, 1));
              coeffs2 = T(7.4686269664352070), T(-0.6076252184992341); 
              addlift(ELS<T>(dual, coeffs2, -1));
              coeffs1 = T(0.0728756555332089); 
              addlift(ELS<T>(primal, coeffs1, 2));
              coeffs2 = T(-1.7186236496830642), T(-0.5818609561112537); 
              addlift(ELS<T>(normalize, coeffs2));
              break;

            case '2':
              coeffs1 = T(2.5303036209828274); 
              addlift(ELS<T>(dual, coeffs1, 0));
              coeffs2 = T(0.2401406244344829),  T(-0.3418203790296641); 
              addlift(ELS<T>(primal, coeffs2, 1));
              coeffs2 = T(-3.1631993897610227), T(-15.2683787372529950); 
              addlift(ELS<T>(dual, coeffs2, -1));
              coeffs2 = T(-0.0057171329709620),   T(0.0646171619180252); 
              addlift(ELS<T>(primal, coeffs2, 3));
              coeffs2 = T(63.9510482479880200), T(-13.5911725693075900); 
              addlift(ELS<T>(dual, coeffs2, -3));
              coeffs2 = T(-0.0005087264425263),   T(0.0018667030862775); 
              addlift(ELS<T>(primal, coeffs2, 5));
              coeffs1 = T(3.7930423341992774); 
              addlift(ELS<T>(dual, coeffs1, -5));
              coeffs2 = T(9.2878701738310099),  T(0.1076673102965570); 
              addlift(ELS<T>(normalize, coeffs2));
              break;

            case '3':

              break;

            default:
              unknown_wname = true;
          }
          break;

        case 'd':   // CDF wavelets
          unknown_wname = true;
          break;
      }
      break;
    case 'b':   // biorthogonal spline wavelets
    case 'r':   // Reverse biorthogonal spline or CDF wavelets
    default:  unknown_wname = true;
  }

  if(unknown_wname)
    throw invalid_argument("Unknown wavelet " + wname + ".\n");
}

#endif // _LIFTING_SCHEMES_H
