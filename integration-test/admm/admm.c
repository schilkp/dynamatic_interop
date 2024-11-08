 GNU nano 6.2                                                          origin.txt                                                                    
This kernel is adapted from code provided to us by Min Jeong, and was used in
When FPGAs Meet ADMM with High-level Synthesis (HLS):
A Real-time Implementation of Long-Horizon MPC for Power Electronic Systems
ICPE'23

It is based off
OSQP: An Operator Splitting Solver for Quadratic Programs
MPC'20

While we do not distribute any of the OSQP code itself, it is

Copyright (c) 2019 Bartolomeo Stellato, Goran Banjac, Paul Goulart, Stephen Boyd

#include "admm.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

void admm(in_float_t vdc, in_float_t inp[30], in_float_t KKT_inv[30][30],
          out_float_t out[2], out_float_t x[30], out_float_t z[30],
          out_float_t y[30], out_float_t rhs[30],
          out_float_t temp_x_tilde[30]) {

  float one_alpha = -0.6000000000000001e+0;
  float rho_inv = 0.1000000000000000e+2;
  float rho = 0.1000000000000000e+0;
  float u_bound = vdc * 0.5f;
  float l_bound = -u_bound;

  // Initialize rhs
  for (int k = 0; k < 30; ++k) {
    rhs[k] = -inp[k];
    x[k] = 0;
    z[k] = 0;
    y[k] = 0;
  }

  for (int iter = 0; iter < 7; ++iter) {
    for (int k = 0; k < 30; ++k) {
      temp_x_tilde[k] = 0;
    }
    for (int k = 0; k < 30; ++k) {
      for (int j = 0; j < 30; ++j) {
        temp_x_tilde[k] += KKT_inv[k][j] * rhs[j];
      }
      x[k] = temp_x_tilde[k] + one_alpha * x[k];
      float helper_var = temp_x_tilde[k] + one_alpha * z[k];
      float temp_z = helper_var + rho_inv * y[k];
      temp_z = (temp_z < l_bound) ? l_bound
                                  : ((temp_z > u_bound) ? u_bound : temp_z);
      z[k] = temp_z;
      y[k] = y[k] + rho * (helper_var - temp_z);
    }
    for (int k = 0; k < 30; ++k) {
      rhs[k] = x[k] + rho * z[k] - y[k];
    }
  }
  out[0] = rhs[0];
  out[1] = rhs[1];
}

int main(void) {
  in_float_t vdc = 300;
  out_float_t out[2];
  out_float_t x[30];
  out_float_t z[30];
  out_float_t y[30];
  out_float_t rhs[30];
  out_float_t temp_x_tilde[30];

  float inp[30] = {
      0.8386580931964525e+0,  -0.4936583289182020e+0, -0.4557200085496808e+0,
      -0.1826728119221966e+0, 0.7892242671009013e+0,  -0.9862663592685157e+0,
      -0.1123442347551846e+0, -0.6070686060996733e+0, -0.2644226570277375e+0,
      0.1007972461227849e+1,  -0.1035177913618547e+1, 0.7786033836815691e+0,
      -0.2194636638045441e+0, -0.7117193487451676e+0, 0.8027830583161479e+0,
      0.7630990677274904e+0,  -0.1017775372656082e+1, -0.1592668456060188e+0,
      -0.4923940635075482e+0, -0.2430991169546847e+0, 0.7893431649326816e+0,
      -0.2170098450099754e+0, -0.9399680909397738e+0, 0.4682180285388441e+0,
      0.2081183167405252e+0,  0.9768797233935672e+0,  0.3266079593729807e+0,
      0.1402547290327087e-1,  -0.1994313019609008e+0, -0.4441969911283863e+0};
  float KKT_inv[30][30] = {
      {0.1013497059390120e+2,  0.1521411769337743e+1,  -0.3141032386289819e+0,
       -0.7194544561636715e+0, 0.5025915790122476e+0,  0.1262635202002143e+0,
       0.2931634799838759e+1,  0.1521411769337741e+1,  0.6282064772579631e+0,
       0.3597272280818353e+0,  0.5025915790122465e+0,  -0.2525270404004286e+0,
       0.2931634799838759e+1,  -0.3042823538675484e+1, -0.3141032386289817e+0,
       0.3597272280818355e+0,  -0.1005183158024496e+1, 0.1262635202002142e+0,
       -0.3042823538675485e+1, 0.1942965356256217e+1,  0.3838197697256101e+0,
       0.5147194303398804e+0,  0.1944843687912656e-1,  0.1279897097552657e+0,
       0.1521411769337744e+1,  0.1942965356256219e+1,  -0.7676395394512213e+0,
       -0.2573597151699407e+0, 0.1944843687912719e-1,  -0.2559794195105312e+0},
      {0.1521411769337743e+1,  0.1211230948106627e+2,  0.3838197697256107e+0,
       -0.2573597151699406e+0, -0.3889687375825268e-1, 0.1279897097552656e+0,
       -0.3749079378934465e+0, 0.1367831262472767e+1,  0.1388263446063955e+1,
       0.4440964792427002e+0,  -0.3258971193543375e+0, -0.1140075580330939e-1,
       0.1874539689467241e+0,  0.1367831262472766e+1,  -0.2760576660550837e+1,
       -0.2220482316781822e+0, -0.3258971193543379e+0, 0.2280151160661852e-1,
       0.1874539689467234e+0,  -0.2735662524945532e+1, 0.1388263446063954e+1,
       -0.2220482316781826e+0, 0.6517942387245294e+0,  -0.1140075580330917e-1,
       0.6282064772579622e+0,  0.3838197697256099e+0,  0.1839651786063314e+1,
       -0.9734345007939788e+0, -0.1597274800578676e+0, -0.1482109344398407e+0},
      {-0.3141032386289819e+0, 0.3838197697256107e+0,  0.1228863276199441e+2,
       0.4867252023819535e+0,  -0.1597274800578678e+0, 0.2964218688796971e+0,
       -0.3141032386289806e+0, -0.7676395394512213e+0, 0.1839651786063315e+1,
       0.4867252023819535e+0,  0.3194549759878168e+0,  -0.1482109344398407e+0,
       0.1268083650399181e+0,  -0.2404795140358145e+0, 0.1384376081912961e+1,
       -0.2679945033120402e+1, 0.5294265996499400e+0,  -0.1081218048070971e+0,
       -0.6340418251995983e-1, -0.2404795140358149e+0, -0.2752816242789115e+1,
       0.1347940477071477e+1,  0.5294265996499403e+0,  0.2162436096300347e+0,
       -0.6340418251995974e-1, 0.4809590280716289e+0,  0.1384376081912959e+1,
       0.1347940477071478e+1,  -0.1058837295329964e+1, -0.1081218048070973e+0},
      {-0.7194544561636715e+0, -0.2573597151699406e+0, 0.4867252023819535e+0,
       0.1258351718534109e+2,  0.1302455156090113e+1,  0.1539387046912903e+0,
       0.3597272280818352e+0,  -0.2573597151699410e+0, -0.9734345007939823e+0,
       0.1692209567249043e+1,  0.1302455156090113e+1,  -0.3078773935104979e+0,
       0.3597272280818355e+0,  0.5147194303398802e+0,  0.4867252023819541e+0,
       0.1692209567249041e+1,  -0.2588974391157672e+1, 0.1539387046912901e+0,
       -0.1005183158024494e+1, 0.1944843687912634e-1,  -0.1597274800578672e+0,
       -0.2588974391157670e+1, 0.1567967244984078e+1,  0.4698691404746331e+0,
       0.5025915790122474e+0,  0.1944843687912705e-1,  0.3194549759878168e+0,
       0.1302455156090112e+1,  0.1567967244984079e+1,  -0.9397223769793354e+0},
      {0.5025915790122476e+0,  -0.3889687375825268e-1, -0.1597274800578678e+0,
       0.1302455156090113e+1,  0.1283200182987102e+2,  0.4698691404746335e+0,
       -0.6898370190688369e+0, 0.1830002349162223e+0,  -0.2961566962742989e+0,
       -0.1231753996936859e+1, 0.1141985593166730e+1,  0.5694929374790280e+0,
       0.3449185095344184e+0,  0.1830002349162227e+0,  0.5923133925644388e+0,
       0.6158849504533879e+0,  0.1141985593166730e+1,  -0.1123049953919663e+1,
       0.3449185095344186e+0,  -0.3660004698324459e+0, -0.2961566962742992e+0,
       0.6158849504533871e+0,  -0.2268035265310904e+1, 0.5694929374790284e+0,
       -0.2525270404004286e+0, 0.1279897097552656e+0,  -0.1482109344398406e+0,
       -0.3078773935104977e+0, 0.4698691404746335e+0,  0.3515638169927387e+0},
      {0.1262635202002143e+0,  0.1279897097552656e+0,  0.2964218688796971e+0,
       0.1539387046912903e+0,  0.4698691404746335e+0,  0.1526480870172578e+2,
       0.1262635202002143e+0,  -0.2559794195105311e+0, -0.1482109344398406e+0,
       0.1539387046912903e+0,  -0.9397223769793354e+0, 0.3515638169927384e+0,
       -0.3460314223963996e-1, 0.2325181517907963e-1,  -0.2331271247484195e-1,
       -0.3029917753103120e-1, 0.6998630706684139e-1,  0.6367503181429962e-1,
       0.1730157111981996e-1,  0.2325181517907961e-1,  0.4662542494968391e-1,
       0.1514958877344363e-1,  0.6998630706684140e-1,  -0.1113982386361302e+0,
       0.1730157111981999e-1,  -0.4650363035815917e-1, -0.2331271247484193e-1,
       0.1514958877344362e-1,  -0.1399725982457596e+0, 0.6367503181429954e-1},
      {0.2931634799838759e+1,  -0.3749079378934465e+0, -0.3141032386289806e+0,
       0.3597272280818352e+0,  -0.6898370190688369e+0, 0.1262635202002143e+0,
       0.1013497059390120e+2,  0.1874539689467235e+0,  -0.3141032386289814e+0,
       -0.7194544561636725e+0, 0.3449185095344184e+0,  0.1262635202002143e+0,
       0.2931634799838762e+1,  0.1874539689467234e+0,  0.6282064772579627e+0,
       0.3597272280818360e+0,  0.3449185095344185e+0,  -0.2525270404004285e+0,
       0.1521411769337743e+1,  -0.2735662524945534e+1, 0.3838197697256097e+0,
       -0.2573597151699408e+0, -0.3660004698324448e+0, 0.1279897097552656e+0,
       -0.3042823538675485e+1, 0.1367831262472767e+1,  0.3838197697256119e+0,
       0.5147194303398812e+0,  0.1830002349162225e+0,  0.1279897097552656e+0},
      {0.1521411769337741e+1,  0.1367831262472767e+1,  -0.7676395394512213e+0,
       -0.2573597151699410e+0, 0.1830002349162223e+0,  -0.2559794195105311e+0,
       0.1874539689467235e+0,  0.1221826082974478e+2,  0.1388263446063955e+1,
       -0.2220482316781826e+0, 0.2875658825185018e+0,  -0.1140075580330933e-1,
       -0.3749079378934482e+0, 0.1882007397271920e+1,  0.1388263446063955e+1,
       0.4440964792427007e+0,  -0.1437829412592433e+0, -0.1140075580330922e-1,
       0.1874539689467241e+0,  0.1882007397271920e+1,  -0.2760576660550833e+1,
       -0.2220482316781832e+0, -0.1437829412592432e+0, 0.2280151160661847e-1,
       -0.3141032386289815e+0, -0.2760576660550836e+1, 0.1839651786063316e+1,
       0.4867252023819537e+0,  0.5923133925644378e+0,  -0.1482109344398405e+0},
      {0.6282064772579631e+0,  0.1388263446063955e+1,  0.1839651786063315e+1,
       -0.9734345007939823e+0, -0.2961566962742989e+0, -0.1482109344398406e+0,
       -0.3141032386289814e+0, 0.1388263446063955e+1,  0.1228863276199440e+2,
       0.4867252023819560e+0,  -0.2961566962742985e+0, 0.2964218688796968e+0,
       -0.6340418251995956e-1, -0.8698960307747665e+0, 0.1384376081912960e+1,
       0.1347940477071478e+1,  0.2860104702678754e-1,  -0.1081218048070974e+0,
       0.1268083650399182e+0,  0.4349559745132711e+0,  0.1384376081912959e+1,
       -0.2679945033120402e+1, -0.1430051557735217e-1, -0.1081218048070974e+0,
       -0.6340418251995956e-1, 0.4349559745132719e+0,  -0.2752816242789110e+1,
       0.1347940477071476e+1,  -0.1430051557735226e-1, 0.2162436096300348e+0},
      {0.3597272280818353e+0,  0.4440964792427002e+0,  0.4867252023819535e+0,
       0.1692209567249043e+1,  -0.1231753996936859e+1, 0.1539387046912903e+0,
       -0.7194544561636725e+0, -0.2220482316781826e+0, 0.4867252023819560e+0,
       0.1258351718534109e+2,  0.6158849504533874e+0,  0.1539387046912898e+0,
       0.3597272280818356e+0,  -0.2220482316781826e+0, -0.9734345007939794e+0,
       0.1692209567249042e+1,  0.6158849504533869e+0,  -0.3078773935104984e+0,
       0.5025915790122473e+0,  0.6517942387245285e+0,  -0.1597274800578666e+0,
       0.1302455156090113e+1,  -0.2268035265310904e+1, 0.4698691404746326e+0,
       -0.1005183158024496e+1, -0.3258971193543376e+0, -0.1597274800578676e+0,
       -0.2588974391157672e+1, 0.1141985593166729e+1,  0.4698691404746329e+0},
      {0.5025915790122465e+0,  -0.3258971193543375e+0, 0.3194549759878168e+0,
       0.1302455156090113e+1,  0.1141985593166730e+1,  -0.9397223769793354e+0,
       0.3449185095344184e+0,  0.2875658825185018e+0,  -0.2961566962742985e+0,
       0.6158849504533874e+0,  0.1374945594196751e+2,  0.5694929374790280e+0,
       -0.6898370190688365e+0, -0.1437829412592432e+0, -0.2961566962742995e+0,
       -0.1231753996936861e+1, 0.1109240188935837e+1,  0.5694929374790283e+0,
       0.3449185095344180e+0,  -0.1437829412592432e+0, 0.5923133925644385e+0,
       0.6158849504533892e+0,  0.1109240188935837e+1,  -0.1123049953919662e+1,
       0.1262635202002143e+0,  0.2280151160661856e-1,  -0.1482109344398405e+0,
       0.1539387046912903e+0,  -0.1123049953919662e+1, 0.3515638169927386e+0},
      {-0.2525270404004286e+0, -0.1140075580330939e-1, -0.1482109344398407e+0,
       -0.3078773935104979e+0, 0.5694929374790280e+0,  0.3515638169927384e+0,
       0.1262635202002143e+0,  -0.1140075580330933e-1, 0.2964218688796968e+0,
       0.1539387046912898e+0,  0.5694929374790280e+0,  0.1526480870172577e+2,
       0.1730157111981998e-1,  -0.5264607920000963e-2, -0.2331271247484194e-1,
       0.1514958877344363e-1,  -0.1894834504444906e+0, 0.6367503181429960e-1,
       -0.3460314223964001e-1, 0.2632303960000493e-2,  -0.2331271247484196e-1,
       -0.3029917753103126e-1, 0.9474968514324423e-1,  0.6367503181429954e-1,
       0.1730157111982000e-1,  0.2632303960000493e-2,  0.4662542494968389e-1,
       0.1514958877344361e-1,  0.9474968514324428e-1,  -0.1113982386361301e+0},
      {0.2931634799838759e+1,  0.1874539689467241e+0,  0.1268083650399181e+0,
       0.3597272280818355e+0,  0.3449185095344184e+0,  -0.3460314223963996e-1,
       0.2931634799838762e+1,  -0.3749079378934482e+0, -0.6340418251995956e-1,
       0.3597272280818356e+0,  -0.6898370190688365e+0, 0.1730157111981998e-1,
       0.1013497059390120e+2,  0.1874539689467257e+0,  -0.6340418251995978e-1,
       -0.7194544561636712e+0, 0.3449185095344183e+0,  0.1730157111982000e-1,
       0.1521411769337742e+1,  0.1367831262472769e+1,  0.4809590280716284e+0,
       -0.2573597151699406e+0, 0.1830002349162225e+0,  -0.4650363035815927e-1,
       0.1521411769337740e+1,  -0.2735662524945533e+1, -0.2404795140358144e+0,
       -0.2573597151699407e+0, -0.3660004698324459e+0, 0.2325181517907964e-1},
      {-0.3042823538675484e+1, 0.1367831262472766e+1,  -0.2404795140358145e+0,
       0.5147194303398802e+0,  0.1830002349162227e+0,  0.2325181517907963e-1,
       0.1874539689467234e+0,  0.1882007397271920e+1,  -0.8698960307747665e+0,
       -0.2220482316781826e+0, -0.1437829412592432e+0, -0.5264607920000963e-2,
       0.1874539689467257e+0,  0.1221826082974478e+2,  0.4349559745132718e+0,
       -0.2220482316781828e+0, 0.2875658825185021e+0,  0.2632303960000487e-2,
       -0.3749079378934479e+0, 0.1882007397271921e+1,  0.4349559745132721e+0,
       0.4440964792427002e+0,  -0.1437829412592429e+0, 0.2632303960000476e-2,
       -0.3141032386289809e+0, 0.1388263446063955e+1,  -0.2752816242789116e+1,
       0.4867252023819532e+0,  -0.2961566962742988e+0, 0.4662542494968389e-1},
      {-0.3141032386289817e+0, -0.2760576660550837e+1, 0.1384376081912961e+1,
       0.4867252023819541e+0,  0.5923133925644388e+0,  -0.2331271247484195e-1,
       0.6282064772579627e+0,  0.1388263446063955e+1,  0.1384376081912960e+1,
       -0.9734345007939794e+0, -0.2961566962742995e+0, -0.2331271247484194e-1,
       -0.6340418251995978e-1, 0.4349559745132718e+0,  0.1238257943317523e+2,
       0.1347940477071478e+1,  -0.1430051557735244e-1, 0.4715937547249911e-1,
       -0.6340418251996003e-1, -0.8698960307747680e+0, 0.1792678443331979e+1,
       0.1347940477071476e+1,  0.2860104702678790e-1,  -0.2357968773624168e-1,
       0.1268083650399179e+0,  0.4349559745132732e+0,  0.1792678443331979e+1,
       -0.2679945033120404e+1, -0.1430051557735244e-1, -0.2357968773624164e-1},
      {0.3597272280818355e+0,  -0.2220482316781822e+0, -0.2679945033120402e+1,
       0.1692209567249041e+1,  0.6158849504533879e+0,  -0.3029917753103120e-1,
       0.3597272280818360e+0,  0.4440964792427007e+0,  0.1347940477071478e+1,
       0.1692209567249042e+1,  -0.1231753996936861e+1, 0.1514958877344363e-1,
       -0.7194544561636712e+0, -0.2220482316781828e+0, 0.1347940477071478e+1,
       0.1258351718534109e+2,  0.6158849504533885e+0,  0.1514958877344362e-1,
       0.5025915790122474e+0,  -0.3258971193543374e+0, -0.1058837295329962e+1,
       0.1302455156090113e+1,  0.1141985593166730e+1,  -0.1399725982457596e+0,
       0.5025915790122463e+0,  0.6517942387245298e+0,  0.5294265996499398e+0,
       0.1302455156090112e+1,  -0.2268035265310905e+1, 0.6998630706684148e-1},
      {-0.1005183158024496e+1, -0.3258971193543379e+0, 0.5294265996499400e+0,
       -0.2588974391157672e+1, 0.1141985593166730e+1,  0.6998630706684139e-1,
       0.3449185095344185e+0,  -0.1437829412592433e+0, 0.2860104702678754e-1,
       0.6158849504533869e+0,  0.1109240188935837e+1,  -0.1894834504444906e+0,
       0.3449185095344183e+0,  0.2875658825185021e+0,  -0.1430051557735244e-1,
       0.6158849504533885e+0,  0.1374945594196751e+2,  0.9474968514324429e-1,
       -0.6898370190688365e+0, -0.1437829412592430e+0, -0.1430051557735172e-1,
       -0.1231753996936860e+1, 0.1109240188935837e+1,  0.9474968514324433e-1,
       0.1262635202002143e+0,  -0.1140075580330928e-1, 0.2162436096300350e+0,
       0.1539387046912907e+0,  0.5694929374790285e+0,  -0.1113982386361304e+0},
      {0.1262635202002142e+0,  0.2280151160661852e-1,  -0.1081218048070971e+0,
       0.1539387046912901e+0,  -0.1123049953919663e+1, 0.6367503181429962e-1,
       -0.2525270404004285e+0, -0.1140075580330922e-1, -0.1081218048070974e+0,
       -0.3078773935104984e+0, 0.5694929374790283e+0,  0.6367503181429960e-1,
       0.1730157111982000e-1,  0.2632303960000487e-2,  0.4715937547249911e-1,
       0.1514958877344362e-1,  0.9474968514324429e-1,  0.1595999789368230e+2,
       0.1730157111981998e-1,  -0.5264607920001042e-2, -0.2357968773624165e-1,
       0.1514958877344359e-1,  -0.1894834504444906e+0, 0.1193717358971243e-1,
       -0.3460314223964001e-1, 0.2632303960000532e-2,  -0.2357968773624164e-1,
       -0.3029917753103122e-1, 0.9474968514324434e-1,  0.1193717358971244e-1},
      {-0.3042823538675485e+1, 0.1874539689467234e+0,  -0.6340418251995983e-1,
       -0.1005183158024494e+1, 0.3449185095344186e+0,  0.1730157111981996e-1,
       0.1521411769337743e+1,  0.1874539689467241e+0,  0.1268083650399182e+0,
       0.5025915790122473e+0,  0.3449185095344180e+0,  -0.3460314223964001e-1,
       0.1521411769337742e+1,  -0.3749079378934479e+0, -0.6340418251996003e-1,
       0.5025915790122474e+0,  -0.6898370190688365e+0, 0.1730157111981998e-1,
       0.1211230948106627e+2,  0.1367831262472768e+1,  -0.2404795140358150e+0,
       -0.3889687375825304e-1, 0.1830002349162224e+0,  0.2325181517907964e-1,
       0.1942965356256216e+1,  0.1367831262472766e+1,  0.4809590280716280e+0,
       0.1944843687912645e-1,  0.1830002349162227e+0,  -0.4650363035815919e-1},
      {0.1942965356256217e+1,  -0.2735662524945532e+1, -0.2404795140358149e+0,
       0.1944843687912634e-1,  -0.3660004698324459e+0, 0.2325181517907961e-1,
       -0.2735662524945534e+1, 0.1882007397271920e+1,  0.4349559745132711e+0,
       0.6517942387245285e+0,  -0.1437829412592432e+0, 0.2632303960000493e-2,
       0.1367831262472769e+1,  0.1882007397271921e+1,  -0.8698960307747680e+0,
       -0.3258971193543374e+0, -0.1437829412592430e+0, -0.5264607920001042e-2,
       0.1367831262472768e+1,  0.1221826082974477e+2,  0.4349559745132719e+0,
       -0.3258971193543374e+0, 0.2875658825185024e+0,  0.2632303960000537e-2,
       -0.7676395394512195e+0, 0.1388263446063953e+1,  0.1384376081912961e+1,
       0.3194549759878165e+0,  -0.2961566962742987e+0, -0.2331271247484195e-1},
      {0.3838197697256101e+0,  0.1388263446063954e+1,  -0.2752816242789115e+1,
       -0.1597274800578672e+0, -0.2961566962742992e+0, 0.4662542494968391e-1,
       0.3838197697256097e+0,  -0.2760576660550833e+1, 0.1384376081912959e+1,
       -0.1597274800578666e+0, 0.5923133925644385e+0,  -0.2331271247484196e-1,
       0.4809590280716284e+0,  0.4349559745132721e+0,  0.1792678443331979e+1,
       -0.1058837295329962e+1, -0.1430051557735172e-1, -0.2357968773624165e-1,
       -0.2404795140358150e+0, 0.4349559745132719e+0,  0.1238257943317523e+2,
       0.5294265996499394e+0,  -0.1430051557735261e-1, 0.4715937547249917e-1,
       -0.2404795140358143e+0, -0.8698960307747701e+0, 0.1792678443331981e+1,
       0.5294265996499389e+0,  0.2860104702678754e-1,  -0.2357968773624167e-1},
      {0.5147194303398804e+0,  -0.2220482316781826e+0, 0.1347940477071477e+1,
       -0.2588974391157670e+1, 0.6158849504533871e+0,  0.1514958877344363e-1,
       -0.2573597151699408e+0, -0.2220482316781832e+0, -0.2679945033120402e+1,
       0.1302455156090113e+1,  0.6158849504533892e+0,  -0.3029917753103126e-1,
       -0.2573597151699406e+0, 0.4440964792427002e+0,  0.1347940477071476e+1,
       0.1302455156090113e+1,  -0.1231753996936860e+1, 0.1514958877344359e-1,
       -0.3889687375825304e-1, -0.3258971193543374e+0, 0.5294265996499394e+0,
       0.1283200182987101e+2,  0.1141985593166730e+1,  0.6998630706684137e-1,
       0.1944843687912665e-1,  -0.3258971193543374e+0, -0.1058837295329965e+1,
       0.1567967244984077e+1,  0.1141985593166729e+1,  -0.1399725982457596e+0},
      {0.1944843687912656e-1,  0.6517942387245294e+0,  0.5294265996499403e+0,
       0.1567967244984078e+1,  -0.2268035265310904e+1, 0.6998630706684140e-1,
       -0.3660004698324448e+0, -0.1437829412592432e+0, -0.1430051557735217e-1,
       -0.2268035265310904e+1, 0.1109240188935837e+1,  0.9474968514324423e-1,
       0.1830002349162225e+0,  -0.1437829412592429e+0, 0.2860104702678790e-1,
       0.1141985593166730e+1,  0.1109240188935837e+1,  -0.1894834504444906e+0,
       0.1830002349162224e+0,  0.2875658825185024e+0,  -0.1430051557735261e-1,
       0.1141985593166730e+1,  0.1374945594196751e+2,  0.9474968514324431e-1,
       -0.2559794195105312e+0, -0.1140075580330935e-1, -0.1081218048070972e+0,
       -0.9397223769793354e+0, 0.5694929374790286e+0,  0.6367503181429959e-1},
      {0.1279897097552657e+0,  -0.1140075580330917e-1, 0.2162436096300347e+0,
       0.4698691404746331e+0,  0.5694929374790284e+0,  -0.1113982386361302e+0,
       0.1279897097552656e+0,  0.2280151160661847e-1,  -0.1081218048070974e+0,
       0.4698691404746326e+0,  -0.1123049953919662e+1, 0.6367503181429954e-1,
       -0.4650363035815927e-1, 0.2632303960000476e-2,  -0.2357968773624168e-1,
       -0.1399725982457596e+0, 0.9474968514324433e-1,  0.1193717358971243e-1,
       0.2325181517907964e-1,  0.2632303960000537e-2,  0.4715937547249917e-1,
       0.6998630706684137e-1,  0.9474968514324431e-1,  0.1595999789368230e+2,
       0.2325181517907962e-1,  -0.5264607920001008e-2, -0.2357968773624167e-1,
       0.6998630706684131e-1,  -0.1894834504444906e+0, 0.1193717358971244e-1},
      {0.1521411769337744e+1,  0.6282064772579622e+0,  -0.6340418251995974e-1,
       0.5025915790122474e+0,  -0.2525270404004286e+0, 0.1730157111981999e-1,
       -0.3042823538675485e+1, -0.3141032386289815e+0, -0.6340418251995956e-1,
       -0.1005183158024496e+1, 0.1262635202002143e+0,  0.1730157111982000e-1,
       0.1521411769337740e+1,  -0.3141032386289809e+0, 0.1268083650399179e+0,
       0.5025915790122463e+0,  0.1262635202002143e+0,  -0.3460314223964001e-1,
       0.1942965356256216e+1,  -0.7676395394512195e+0, -0.2404795140358143e+0,
       0.1944843687912665e-1,  -0.2559794195105312e+0, 0.2325181517907962e-1,
       0.1211230948106627e+2,  0.3838197697256104e+0,  -0.2404795140358150e+0,
       -0.3889687375825286e-1, 0.1279897097552656e+0,  0.2325181517907961e-1},
      {0.1942965356256219e+1,  0.3838197697256099e+0,  0.4809590280716289e+0,
       0.1944843687912705e-1,  0.1279897097552656e+0,  -0.4650363035815917e-1,
       0.1367831262472767e+1,  -0.2760576660550836e+1, 0.4349559745132719e+0,
       -0.3258971193543376e+0, 0.2280151160661856e-1,  0.2632303960000493e-2,
       -0.2735662524945533e+1, 0.1388263446063955e+1,  0.4349559745132732e+0,
       0.6517942387245298e+0,  -0.1140075580330928e-1, 0.2632303960000532e-2,
       0.1367831262472766e+1,  0.1388263446063953e+1,  -0.8698960307747701e+0,
       -0.3258971193543374e+0, -0.1140075580330935e-1, -0.5264607920001008e-2,
       0.3838197697256104e+0,  0.1228863276199441e+2,  0.1384376081912961e+1,
       -0.1597274800578671e+0, 0.2964218688796968e+0,  -0.2331271247484192e-1},
      {-0.7676395394512213e+0, 0.1839651786063314e+1,  0.1384376081912959e+1,
       0.3194549759878168e+0,  -0.1482109344398406e+0, -0.2331271247484193e-1,
       0.3838197697256119e+0,  0.1839651786063316e+1,  -0.2752816242789110e+1,
       -0.1597274800578676e+0, -0.1482109344398405e+0, 0.4662542494968389e-1,
       -0.2404795140358144e+0, -0.2752816242789116e+1, 0.1792678443331979e+1,
       0.5294265996499398e+0,  0.2162436096300350e+0,  -0.2357968773624164e-1,
       0.4809590280716280e+0,  0.1384376081912961e+1,  0.1792678443331981e+1,
       -0.1058837295329965e+1, -0.1081218048070972e+0, -0.2357968773624167e-1,
       -0.2404795140358150e+0, 0.1384376081912961e+1,  0.1238257943317523e+2,
       0.5294265996499397e+0,  -0.1081218048070975e+0, 0.4715937547249914e-1},
      {-0.2573597151699407e+0, -0.9734345007939788e+0, 0.1347940477071478e+1,
       0.1302455156090112e+1,  -0.3078773935104977e+0, 0.1514958877344362e-1,
       0.5147194303398812e+0,  0.4867252023819537e+0,  0.1347940477071476e+1,
       -0.2588974391157672e+1, 0.1539387046912903e+0,  0.1514958877344361e-1,
       -0.2573597151699407e+0, 0.4867252023819532e+0,  -0.2679945033120404e+1,
       0.1302455156090112e+1,  0.1539387046912907e+0,  -0.3029917753103122e-1,
       0.1944843687912645e-1,  0.3194549759878165e+0,  0.5294265996499389e+0,
       0.1567967244984077e+1,  -0.9397223769793354e+0, 0.6998630706684131e-1,
       -0.3889687375825286e-1, -0.1597274800578671e+0, 0.5294265996499397e+0,
       0.1283200182987102e+2,  0.4698691404746338e+0,  0.6998630706684135e-1},
      {0.1944843687912719e-1,  -0.1597274800578676e+0, -0.1058837295329964e+1,
       0.1567967244984079e+1,  0.4698691404746335e+0,  -0.1399725982457596e+0,
       0.1830002349162225e+0,  0.5923133925644378e+0,  -0.1430051557735226e-1,
       0.1141985593166729e+1,  -0.1123049953919662e+1, 0.9474968514324428e-1,
       -0.3660004698324459e+0, -0.2961566962742988e+0, -0.1430051557735244e-1,
       -0.2268035265310905e+1, 0.5694929374790285e+0,  0.9474968514324434e-1,
       0.1830002349162227e+0,  -0.2961566962742987e+0, 0.2860104702678754e-1,
       0.1141985593166729e+1,  0.5694929374790286e+0,  -0.1894834504444906e+0,
       0.1279897097552656e+0,  0.2964218688796968e+0,  -0.1081218048070975e+0,
       0.4698691404746338e+0,  0.1526480870172578e+2,  0.6367503181429958e-1},
      {-0.2559794195105312e+0, -0.1482109344398407e+0, -0.1081218048070973e+0,
       -0.9397223769793354e+0, 0.3515638169927387e+0,  0.6367503181429954e-1,
       0.1279897097552656e+0,  -0.1482109344398405e+0, 0.2162436096300348e+0,
       0.4698691404746329e+0,  0.3515638169927386e+0,  -0.1113982386361301e+0,
       0.2325181517907964e-1,  0.4662542494968389e-1,  -0.2357968773624164e-1,
       0.6998630706684148e-1,  -0.1113982386361304e+0, 0.1193717358971244e-1,
       -0.4650363035815919e-1, -0.2331271247484195e-1, -0.2357968773624167e-1,
       -0.1399725982457596e+0, 0.6367503181429959e-1,  0.1193717358971244e-1,
       0.2325181517907961e-1,  -0.2331271247484192e-1, 0.4715937547249914e-1,
       0.6998630706684135e-1,  0.6367503181429958e-1,  0.1595999789368229e+2}};

  CALL_KERNEL(admm, vdc, inp, KKT_inv, out, x, z, y, rhs, temp_x_tilde);
  return 0;
}
