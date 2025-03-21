/*
  This file is part of MADNESS.

  Copyright (C) 2007,2010 Oak Ridge National Laboratory

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

  For more information please contact:

  Robert J. Harrison
  Oak Ridge National Laboratory
  One Bethel Valley Road
  P.O. Box 2008, MS-6367

  email: harrisonrj@ornl.gov
  tel:   865-241-3937
  fax:   865-572-0680


  $Id$
*/

#include <madness/constants.h>
#include<madness/chem/atomutil.h>
#include <madness/misc/misc.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

/// \file atomutil.cc
/// \brief implementation of utility functions for atom

namespace madness {
static const unsigned int NUMBER_OF_ATOMS_IN_TABLE = 110;

/// Atomic weights are taken from IUPAC 2005 (M. E. Wieser, “Atomic weights of the elements
/// 2005 (IUPAC Technical Report),” Pure and Applied Chemistry, vol. 78, no. 11.)
/// Negative masses refer to the longest-living isotope.
/// Note that the masses refer to isotopic averaging, not to the specified isotope!
static const AtomicData atomic_data[NUMBER_OF_ATOMS_IN_TABLE] = {
    // symbol    number          nuclear_radius                    nuclear_gaussian_exponent mass in amu
    //     symbol      isotope                    nuclear_half_charge_radius       covalent_radius
    {"Bq",  "bq",   0  ,  0   ,  0.0               , 0.0           ,0.0             , 0.0   , 0.0},
    {"H",   "h",    1  ,  1   ,  2.6569547399e-05  , 1.32234e-05   ,2.1248239171e+09, 0.30  , 1.00794  },
    {"He",  "he",   2  ,  4   ,  3.5849373401e-05  , 2.63172e-05   ,1.1671538870e+09, 1.22  , 4.002602 },
    {"Li",  "li",   3  ,  7   ,  4.0992133976e-05  , 2.34051e-05   ,8.9266848806e+08, 1.23  , 6.941    },
    {"Be",  "be",   4  ,  9   ,  4.3632829651e-05  , 3.03356e-05   ,7.8788802914e+08, 0.89  , 9.012182 },
    {"B",   "b",    5  ,  11  ,  4.5906118608e-05  , 3.54894e-05   ,7.1178709563e+08, 0.88  , 10.811   },
    {"C",   "c",    6  ,  12  ,  4.6940079496e-05  , 3.76762e-05   ,6.8077502929e+08, 0.77  , 12.0107  },
    {"N",   "n",    7  ,  14  ,  4.8847128967e-05  , 4.15204e-05   ,6.2865615725e+08, 0.70  , 14.0067  },
    {"O",   "o",    8  ,  16  ,  5.0580178957e-05  , 4.48457e-05   ,5.8631436655e+08, 0.66  , 15.9994  },
    {"F",   "f",    9  ,  19  ,  5.2927138943e-05  , 4.91529e-05   ,5.3546911034e+08, 0.58  , 18.9984032},
    {"Ne",  "ne",  10  ,  20  ,  5.3654104231e-05  , 5.04494e-05   ,5.2105715255e+08, 1.60  , 20.1797  },
    {"Na",  "na",  11  ,  23  ,  5.5699159416e-05  , 5.40173e-05   ,4.8349721509e+08, 1.66  , 22.98976928},
    {"Mg",  "mg",  12  ,  24  ,  5.6341070732e-05  , 5.51157e-05   ,4.7254270882e+08, 1.36  , 24.3050},
    {"Al",  "al",  13  ,  27  ,  5.8165765928e-05  , 5.81891e-05   ,4.4335984491e+08, 1.25  , 26.9815386},
    {"Si",  "si",  14  ,  28  ,  5.8743802504e-05  , 5.91490e-05   ,4.3467748823e+08, 1.17  , 28.0855},
    {"P",   "p",   15  ,  31  ,  6.0399312923e-05  , 6.18655e-05   ,4.1117553148e+08, 1.10  , 30.973762},
    {"S",   "s",   16  ,  32  ,  6.0927308666e-05  , 6.27224e-05   ,4.0407992047e+08, 1.04  , 32.065},
    {"Cl",  "cl",  17  ,  35  ,  6.2448101115e-05  , 6.51676e-05   ,3.8463852873e+08, 0.99  , 35.453},
    {"Ar",  "ar",  18  ,  40  ,  6.4800211825e-05  , 6.88887e-05   ,3.5722217300e+08, 1.91  , 39.948},
    {"K",   "k",   19  ,  39  ,  6.4346167051e-05  , 6.81757e-05   ,3.6228128110e+08, 2.03  , 39.0983},
    {"Ca",  "ca",  20  ,  40  ,  6.4800211825e-05  , 6.88887e-05   ,3.5722217300e+08, 1.74  , 40.078},
    {"Sc",  "sc",  21  ,  45  ,  6.6963627201e-05  , 7.22548e-05   ,3.3451324570e+08, 1.44  , 44.955912},
    {"Ti",  "ti",  22  ,  48  ,  6.8185577480e-05  , 7.41350e-05   ,3.2263108827e+08, 1.32  , 47.867},
    {"V",   "v",   23  ,  51  ,  6.9357616830e-05  , 7.59254e-05   ,3.1181925878e+08, 1.22  , 50.9415},
    {"Cr",  "cr",  24  ,  52  ,  6.9738057221e-05  , 7.65040e-05   ,3.0842641793e+08, 1.19  , 51.9961},
    {"Mn",  "mn",  25  ,  55  ,  7.0850896638e-05  , 7.81897e-05   ,2.9881373610e+08, 1.17  , 54.938045},
    {"Fe",  "fe",  26  ,  56  ,  7.1212829817e-05  , 7.87358e-05   ,2.9578406371e+08, 1.165 , 55.845},
    {"Co",  "co",  27  ,  59  ,  7.2273420879e-05  , 8.03303e-05   ,2.8716667270e+08, 1.16  , 58.933195},
    {"Ni",  "ni",  28  ,  58  ,  7.1923970253e-05  , 7.98058e-05   ,2.8996391416e+08, 1.15  , 58.6934},
    {"Cu",  "cu",  29  ,  63  ,  7.3633018675e-05  , 8.23625e-05   ,2.7665979354e+08, 1.17  , 63.546},
    {"Zn",  "zn",  30  ,  64  ,  7.3963875193e-05  , 8.28551e-05   ,2.7419021043e+08, 1.25  , 65.409},
    {"Ga",  "ga",  31  ,  69  ,  7.5568424848e-05  , 8.52341e-05   ,2.6267002737e+08, 1.25  , 69.723},
    {"Ge",  "ge",  32  ,  74  ,  7.7097216161e-05  , 8.74862e-05   ,2.5235613399e+08, 1.22  , 72.64},
    {"As",  "as",  33  ,  75  ,  7.7394645153e-05  , 8.79228e-05   ,2.5042024280e+08, 1.21  , 74.92160},
    {"Se",  "se",  34  ,  80  ,  7.8843427408e-05  , 9.00427e-05   ,2.4130163719e+08, 1.17  , 78.96},
    {"Br",  "br",  35  ,  79  ,  7.8558604038e-05  , 8.96268e-05   ,2.4305454351e+08, 1.14  , 79.904},
    {"Kr",  "kr",  36  ,  84  ,  7.9959560033e-05  , 9.16684e-05   ,2.3461213272e+08, 1.98  , 83.798},
    {"Rb",  "rb",  37  ,  85  ,  8.0233033713e-05  , 9.20658e-05   ,2.3301551109e+08, 2.22  , 85.4678},
    {"Sr",  "sr",  38  ,  88  ,  8.1040799081e-05  , 9.32375e-05   ,2.2839354730e+08, 1.92  , 87.62},
    {"Y",   "y",   39  ,  89  ,  8.1305968993e-05  , 9.36215e-05   ,2.2690621893e+08, 1.62  , 88.90585},
    {"Zr",  "zr",  40  ,  90  ,  8.1569159980e-05  , 9.40022e-05   ,2.2544431039e+08, 1.45  , 91.224},
    {"Nb",  "nb",  41  ,  93  ,  8.2347219223e-05  , 9.51261e-05   ,2.2120420724e+08, 1.34  , 92.90638},
    {"Mo",  "mo",  42  ,  98  ,  8.3607614434e-05  , 9.69412e-05   ,2.1458511597e+08, 1.29  , 95.94},
    {"Tc",  "tc",  43  ,  98  ,  8.3607614434e-05  , 9.69412e-05   ,2.1458511597e+08, 1.27  , -97.9072},
    {"Ru",  "ru",  44  , 102  ,  8.4585397905e-05  , 9.83448e-05   ,2.0965270287e+08, 1.24  , 101.07},
    {"Rh",  "rh",  45  , 103  ,  8.4825835954e-05  , 9.86893e-05   ,2.0846586999e+08, 1.25  , 102.90550},
    {"Pd",  "pd",  46  , 106  ,  8.5537941156e-05  , 9.97084e-05   ,2.0500935221e+08, 1.28  , 106.42},
    {"Ag",  "ag",  47  , 107  ,  8.5772320442e-05  , 1.00043e-04   ,2.0389047621e+08, 1.34  , 107.8682},
    {"Cd",  "cd",  48  , 114  ,  8.7373430179e-05  , 1.02327e-04   ,1.9648639618e+08, 1.41  , 112.411},
    {"In",  "in",  49  , 115  ,  8.7596760865e-05  , 1.02644e-04   ,1.9548577691e+08, 1.50  , 114.818},
    {"Sn",  "sn",  50  , 120  ,  8.8694413774e-05  , 1.04204e-04   ,1.9067718154e+08, 1.40  , 118.710},
    {"Sb",  "sb",  51  , 121  ,  8.8910267995e-05  , 1.04510e-04   ,1.8975246242e+08, 1.41  , 121.760},
    {"Te",  "te",  52  , 130  ,  9.0801452955e-05  , 1.07185e-04   ,1.8193056289e+08, 1.37  , 127.60},
    {"I",   "i",   53  , 127  ,  9.0181040290e-05  , 1.06309e-04   ,1.8444240538e+08, 1.33  , 126.90447},
    {"Xe",  "xe",  54  , 132  ,  9.1209776425e-05  , 1.07762e-04   ,1.8030529331e+08, 2.09  , 131.293},
    {"Cs",  "cs",  55  , 133  ,  9.1412392742e-05  , 1.08047e-04   ,1.7950688281e+08, 2.35  , 132.9054519},
    {"Ba",  "ba",  56  , 138  ,  9.2410525664e-05  , 1.09453e-04   ,1.7565009043e+08, 1.98  , 137.327},
    {"La",  "la",  57  , 139  ,  9.2607247118e-05  , 1.09730e-04   ,1.7490463170e+08, 1.69  , 138.90547},
    {"Ce",  "ce",  58  , 140  ,  9.2803027311e-05  , 1.10006e-04   ,1.7416744147e+08, 1.65  , 140.116},
    {"Pr",  "pr",  59  , 141  ,  9.2997877424e-05  , 1.10279e-04   ,1.7343837120e+08, 1.65  , 140.90765},
    {"Nd",  "nd",  60  , 144  ,  9.3576955934e-05  , 1.11093e-04   ,1.7129844956e+08, 1.64  , 144.242},
    {"Pm",  "pm",  61  , 145  ,  9.3768193375e-05  , 1.11361e-04   ,1.7060044589e+08, 1.65  , -144.9127},
    {"Sm",  "sm",  62  , 152  ,  9.5082839751e-05  , 1.13204e-04   ,1.6591550422e+08, 1.66  , 150.36},
    {"Eu",  "eu",  63  , 153  ,  9.5267329183e-05  , 1.13462e-04   ,1.6527352089e+08, 1.65  , 151.964},
    {"Gd",  "gd",  64  , 158  ,  9.6177915369e-05  , 1.14735e-04   ,1.6215880671e+08, 1.61  , 157.25},
    {"Tb",  "tb",  65  , 159  ,  9.6357719009e-05  , 1.14986e-04   ,1.6155419421e+08, 1.59  , 158.92535},
    {"Dy",  "dy",  66  , 162  ,  9.6892647152e-05  , 1.15733e-04   ,1.5977529080e+08, 1.59  , 162.500},
    {"Ho",  "ho",  67  , 162  ,  9.6892647152e-05  , 1.15733e-04   ,1.5977529080e+08, 1.58  , 164.93032},
    {"Er",  "er",  68  , 168  ,  9.7943009317e-05  , 1.17198e-04   ,1.5636673634e+08, 1.57  , 167.259},
    {"Tm",  "tm",  69  , 169  ,  9.8115626740e-05  , 1.17438e-04   ,1.5581702004e+08, 1.56  , 168.93421},
    {"Yb",  "yb",  70  , 174  ,  9.8968651305e-05  , 1.18625e-04   ,1.5314257850e+08, 1.56  , 173.04},
    {"Lu",  "lu",  71  , 175  ,  9.9137288835e-05  , 1.18859e-04   ,1.5262201512e+08, 1.56  , 174.967},
    {"Hf",  "hf",  72  , 180  ,  9.9970978172e-05  , 1.20018e-04   ,1.5008710340e+08, 1.44  , 178.49},
    {"Ta",  "ta",  73  , 181  ,  1.0013585755e-04  , 1.20246e-04   ,1.4959325643e+08, 1.34  , 180.94788},
    {"W",   "w",   74  , 184  ,  1.0062688070e-04  , 1.20928e-04   ,1.4813689532e+08, 1.30  , 183.84},
    {"Re",  "re",  75  , 187  ,  1.0111259523e-04  , 1.21601e-04   ,1.4671710337e+08, 1.28  , 186.207},
    {"Os",  "os",  76  , 192  ,  1.0191070333e-04  , 1.22706e-04   ,1.4442808782e+08, 1.26  , 190.23},
    {"Ir",  "ir",  77  , 193  ,  1.0206865731e-04  , 1.22925e-04   ,1.4398142103e+08, 1.26  , 192.217},
    {"Pt",  "pt",  78  , 195  ,  1.0238293593e-04  , 1.23360e-04   ,1.4309883584e+08, 1.29  , 195.084},
    {"Au",  "au",  79  , 197  ,  1.0269507292e-04  , 1.23792e-04   ,1.4223027307e+08, 1.34  , 196.966569},
    {"Hg",  "hg",  80  , 202  ,  1.0346628039e-04  , 1.24857e-04   ,1.4011788914e+08, 1.44  , 200.59},
    {"Tl",  "tl",  81  , 205  ,  1.0392291259e-04  , 1.25488e-04   ,1.3888925203e+08, 1.55  , 204.3833},
    {"Pb",  "pb",  82  , 208  ,  1.0437511130e-04  , 1.26112e-04   ,1.3768840081e+08, 1.54  , 207.2},
    {"Bi",  "bi",  83  , 209  ,  1.0452487744e-04  , 1.26318e-04   ,1.3729411599e+08, 1.52  , 208.98040},
    {"Po",  "po",  84  , 209  ,  1.0452487744e-04  , 1.26318e-04   ,1.3729411599e+08, 1.53  , -208.9824},
    {"At",  "at",  85  , 210  ,  1.0467416660e-04  , 1.26524e-04   ,1.3690277000e+08, 1.50  , -209.9871},
    {"Rn",  "rn",  86  , 222  ,  1.0642976299e-04  , 1.28942e-04   ,1.3242350205e+08, 2.20  , -220.0176},
    {"Fr",  "fr",  87  , 223  ,  1.0657317899e-04  , 1.29139e-04   ,1.3206733609e+08, 3.24  , -223.0197},
    {"Ra",  "ra",  88  , 226  ,  1.0700087100e-04  , 1.29727e-04   ,1.3101367628e+08, 2.68  , -226.0254},
    {"Ac",  "ac",  89  , 227  ,  1.0714259349e-04  , 1.29922e-04   ,1.3066730974e+08, 2.25  , -227.0278},
    {"Th",  "th",  90  , 232  ,  1.0784503195e-04  , 1.30887e-04   ,1.2897067480e+08, 2.16  , 232.03806 },
    {"Pa",  "pa",  91  , 231  ,  1.0770535752e-04  , 1.30695e-04   ,1.2930539512e+08, 1.93  , 321.03588 },
    {"U",   "u",   92  , 238  ,  1.0867476102e-04  , 1.32026e-04   ,1.2700881714e+08, 3.00  , 238.02891 },
    {"Np",  "np",  93  , 237  ,  1.0853744903e-04  , 1.31838e-04   ,1.2733038109e+08, 1.57  , -239.0482},
    {"Pu",  "pu",  94  , 244  ,  1.0949065967e-04  , 1.33145e-04   ,1.2512299012e+08, 1.81  , -244.0642},
    {"Am",  "am",  95  , 243  ,  1.0935561268e-04  , 1.32960e-04   ,1.2543221826e+08, 2.21  , -243.0614},
    {"Cm",  "cm",  96  , 247  ,  1.0989359973e-04  , 1.33697e-04   ,1.2420711085e+08, 1.43  , -247.0704},
    {"Bk",  "bk",  97  , 247  ,  1.0989359973e-04  , 1.33697e-04   ,1.2420711085e+08, 1.42  , -247.0703},
    {"Cf",  "cf",  98  , 251  ,  1.1042580946e-04  , 1.34426e-04   ,1.2301273547e+08, 1.40  , -251.0796},
    {"Es",  "es",  99  , 252  ,  1.1055797721e-04  , 1.34607e-04   ,1.2271879740e+08, 1.39  , -252.0830},
    {"Fm",  "fm",  100 , 257  ,  1.1121362374e-04  , 1.35504e-04   ,1.2127611477e+08, 1.38  , -257.0951},
    {"Md",  "md",  101 , 258  ,  1.1134373034e-04  , 1.35682e-04   ,1.2099285491e+08, 1.37  , -258.0984},
    {"No",  "no",  102 , 259  ,  1.1147350119e-04  , 1.35859e-04   ,1.2071131346e+08, 1.36  , -259.1010},
    {"Lr",  "lr",  103 , 262  ,  1.1186082063e-04  , 1.36389e-04   ,1.1987683191e+08, 1.34  , -262.1096},
    {"Db",  "db",  104 , 261  ,  1.1173204420e-04  , 1.36213e-04   ,1.2015331850e+08, 1.40  , -1.0},        // ??
    {"Jl",  "jl",  105 , 262  ,  1.1186082063e-04  , 1.36389e-04   ,1.1987683191e+08, 1.40  , -1.0},        // ??
    {"Rf",  "rf",  106 , 263  ,  1.1198926979e-04  , 1.36565e-04   ,1.1960199758e+08, 1.40  , -1.0},        // ??
    {"Bh",  "bh",  107 , 262  ,  1.1186082063e-04  , 1.36389e-04   ,1.1987683191e+08, 1.40  , -272.1380},
    {"Hn",  "hn",  108 , 265  ,  1.1224519460e-04  , 1.36914e-04   ,1.1905722195e+08, 1.40  , -1.0},        // ??
    {"Mt",  "mt",  109 , 266  ,  1.1237267433e-04  , 1.37088e-04   ,1.1878724932e+08, 1.40  , -276.1512}
};

const AtomicData& get_atomic_data(unsigned int atomic_number) {
    if (atomic_number >= NUMBER_OF_ATOMS_IN_TABLE) throw "I am not an alchemist";
    return atomic_data[atomic_number];
}


unsigned int symbol_to_atomic_number(const std::string& symbol) {
    //first check if pseudo-atom (i.e. starts with ps)
    std::string rsymbol(symbol);
    if (symbol.size()>1){
        if (symbol[0]=='p' && symbol[1]=='s') rsymbol.erase(0,2);}

    std::string tlow = madness::lowercase(rsymbol);    

    for (unsigned int i=0; i<NUMBER_OF_ATOMS_IN_TABLE; ++i) {
        if (tlow.compare(atomic_data[i].symbol_lowercase) == 0) return i;
    }
    throw "unknown atom";
}

/// return the lower-case element symbol corresponding to the atomic number
std::string atomic_number_to_symbol(const unsigned int atomic_number) {
    return atomic_data[atomic_number].symbol_lowercase;
}


bool check_if_pseudo_atom(const std::string& symbol) {
    //check if pseudo-atom (i.e. starts with ps)
    if (symbol.size()>1) {
        if (symbol[0]=='p' && symbol[1]=='s') return true;}
    return false;

}


/// Returns radius for smoothing nuclear potential with energy precision eprec
double smoothing_parameter(double Z, double eprec) {
    // The min is since asymptotic form not so good at low acc.
    // The /2 is from two electrons in 1s closed shell.
    if (Z == 0.0) return 1.0;
    eprec = std::min(1e-3,eprec/2.0);
    //eprec = std::min(1e-2,eprec/2.0);
    double Z5 = Z*Z*Z*Z*Z;
    double c = pow(eprec/(0.65*Z5),1.0/3.0);
    //std::cout << "SMOOTHING " << Z << " " << c << std::endl;
    return c;
}


/// Smoothed 1/r potential

/// Invoke as \c u(r/c)/c where \c c is the radius of the
/// smoothed volume.
double smoothed_potential(double r) {
    //     if (r > 7) {
    //         return 1.0/r;
    //     }
    //     else if (r > 1e-2) {
    //         double rsq = r*r;
    //         return erf(r)/r + exp(-rsq)/sqrt(madness::constants::pi);
    //     }
    //     else {
    //         double rsq = r*r;
    //         return 1.6925687506432689+(-.94031597257959385+(.39493270848342941-.12089776790309064*rsq)*rsq)*rsq;
    //     }

    // Below code is about 3x faster than the above and accurate under gcc 4.7 is accurate to 3*epsilon (6.66e-16)
    static const double lo0=0., hi0=.75, m0=(hi0+lo0)*0.5;
    static const double q0[16] = {1.5678214965991468, -.62707838966047510, -.64081087092780663, .47615447123508785, .17661021898450753, -.19684394977153049, -0.28659921463622429e-1, 0.55915374519467232e-1, 0.14360220805627533e-2, -0.12055628477910659e-1, 0.64426691086411555e-3, 0.20868614450806855e-2, -0.23638877395492455e-3, -0.30013965988191071e-3, 0.45381875950933199e-4, 0.34681863672368034e-4};

    static const double lo1=0.75, hi1=1.5, m1=(hi1+lo1)*0.5;
    static const double q1[16] = {.94881540742968045, -.77708439858316963, .29787227010512777, .15217642747453712, -.21398168552777233, 0.46031655538422367e-1, 0.47949184476371850e-1, -0.28318961737780952e-1, -0.32540636504803776e-2, 0.68978584780210998e-2, -0.95383257324881606e-3, -0.10234901787860559e-2, 0.34967650634270816e-3, 0.91583510656898485e-4, -0.59911753487282495e-4};

    static const double lo2=1.5, hi2=2.3, m2=(hi2+lo2)*0.5;
    static const double q2[16] = {.53778364876259625, -.31694255320073229, .20069433198898583, -.10393741805917812, 0.20334064004004232e-1, 0.21235241487756926e-1, -0.20518056794680800e-1, 0.58265985152125050e-2, 0.20871477843956435e-2, -0.21669668079311651e-2, 0.43633711821764852e-3, 0.23242367731229579e-3, -0.14633201591612828e-3, 0.81202239221096509e-5, 0.17796616558877600e-4, -0.52312378743479744e-5};

    static const double lo3=2.3, hi3=3.1, m3=(hi3+lo3)*0.5;
    static const double q3[16] = {.37070557988365330, -.13894942379378032, 0.55150689152956653e-1, -0.25223306695010866e-1, 0.13102816917145070e-1, -0.62578895129075508e-2, 0.18797639251795907e-2, 0.22594681129385247e-3, -0.63097453855253135e-3, 0.34542603237417486e-3, -0.67467846042544434e-4, -0.28565855991132860e-4, 0.24299523362660657e-4, -0.52291058128176869e-5, -0.12789313539110192e-5};

    static const double lo4=3.1, hi4=4.1, m4=(hi4+lo4)*0.5;
    static const double q4[16] = {.27777900622536875, -0.77169285529226839e-1, 0.21463679730008438e-1, -0.60198321985860536e-2, 0.17566374490424399e-2, -0.57950539050391789e-3, 0.23569755765811705e-3, -0.11059754002626651e-3, 0.49084348022523836e-4, -0.16493922349103034e-4, 0.22821100336745665e-5, 0.15678232491648044e-5, -0.13699711510193177e-5, 0.50039859344562296e-6, -0.59433859103763842e-7};

    static const double lo5=4.1, hi5=5.6, m5=(hi5+lo5)*0.5;
    static const double q5[16] = {.20618556704321355, -0.42512488361955708e-1, 0.87654629615324323e-2, -0.18073162972250135e-2, 0.37265209018141361e-3, -0.76851802150694868e-4, 0.15868227305029941e-4, -0.32974415757731337e-5, 0.70444925352388208e-6, -0.16523539352856092e-6, 0.47936573168344514e-7, -0.18010609222020871e-7, 0.77011461658052001e-8, -0.32809439744789984e-8, 0.12461858963532225e-8, -0.28690455490168773e-9};

    static const double lo6=5.6, hi6=7.0, m6=(hi6+lo6)*0.5;
    static const double q6[16] = {.15873015873015873, -0.25195263290501128e-1, 0.39992481413494903e-2, -0.63480129229574217e-3, 0.10076210989447071e-3, -0.15993985425859338e-4, 0.25387278290368119e-5, -0.40297464132256025e-6, 0.63964378952634106e-7, -0.10146488064153093e-7, 0.16103633738263416e-8, -0.26645848419611212e-9, 0.42505693478320828e-10, 0.0, 0.0, 0.0};


//     double rsq = r*r;
//     double formula = erf(r)/r + exp(-rsq)/sqrt(madness::constants::pi);

    const double* a;

    if (r > hi6) {              // Most common case
        return 1.0/r;
    }
    else if (r > hi3) {
        if (r > hi5) {
            r -= m6;
            a = q6;
        }
        else if (r > hi4) {
            r -= m5;
            a = q5;
        }
        else {
            r -= m4;
            a = q4;
        }
    }
    else if (r > hi1) {
        if (r > hi2) {
            r -= m3;
            a = q3;
        }
        else {
            r -= m2;
            a = q2;
        }
    }
    else if (r > hi0) {
        r -= m1;
        a = q1;
    }
    else {
        r -= m0;                // Least common case
        a = q0;
    }


    double b0 = a[ 0] + r*a[ 1];
    double b1 = a[ 2] + r*a[ 3];
    double b2 = a[ 4] + r*a[ 5];
    double b3 = a[ 6] + r*a[ 7];
    double b4 = a[ 8] + r*a[ 9];
    double b5 = a[10] + r*a[11];
    double b6 = a[12] + r*a[13];
    double b7 = a[14] + r*a[15];

    double r2 = r*r;
    double c0 = b0 + r2*b1;
    double c1 = b2 + r2*b3;
    double c2 = b4 + r2*b5;
    double c3 = b6 + r2*b7;

    double r4 = r2*r2;
    double d0 = c0 + r4*c1;
    double d1 = c2 + r4*c3;

    double r8 = r4*r4;
    double result = d0 + r8*d1;

//     if (abs(result-formula) > 1e-12) {
//         printf("ERROR in potn: r=%.10f formula=%.10f result=%.10f err=%.1e\n",
//                r, formula, result, formula-result);
//         throw "bad";
//     }

    return result;
}


/// Derivative of the regularized 1/r potential

/// dV/dx = (x/r) * du(r/c)/(c*c)
double dsmoothed_potential(double r) {
//      double rsq = r*r;
//      if (r > 7.0) {
//          return -1.0/rsq;
//      }
//      else if (r > 1e-2) {
//          return -erf(r)/rsq + exp(-rsq)*(2/r - 2*r)/sqrt(madness::constants::pi);
//      }
//      else {
//          return (-1.8806319451591876+(1.5797308339337176-.72538660741854381*rsq)*rsq)*r;
//      }

    // Below we have 16-term polynomial approximations generated from Chebyshev expansions
    // computed by Maple, accurate to about 1e-14.  These are over 5x faster than
    // the above code using gcc.  Note the use of a tree algorithm to compute the polynomials
    // with lots of parallelism and use of FMA.

    static const double lo0=0.0, hi0=0.65, m0=(hi0+lo0)*0.5;
    static const double q0[16] = {-.55952054067648194, -1.4186837724703172, 1.3079855069574490, .89801173653480100, -.92664998851771673, -.28801593449022579, .37975455626369975, 0.55119107345331705e-1, -.10877141173737492, -0.53157122124920191e-2, 0.23855987780906309e-1, -0.39335550662218725e-3, -0.42224803990545490e-2, 0.27757150363394877e-3, 0.59449662015255330e-3, -0.61851566882100875e-4};

    static const double lo1=.65, hi1=1.3, m1=(hi1+lo1)*0.5;
    static const double q1[16] = {-.85319294453145976, .39871916999716314, .86161440627190766, -.91664151410845898, -0.47179381608173164e-1, .44169815463839599, -.13041276737306307, -.10308252655802193, 0.61601684937724164e-1, 0.11350111203259745e-1, -0.15588916300580124e-1, 0.58815593818761124e-3, 0.26813418239064462e-2, -0.50269597194697906e-3, -0.32324952206733884e-3, 0.10747689363541249e-3};

    static const double lo2=1.3, hi2=2.05, m2=(hi2+lo2)*0.5;
    static const double q2[16] = {-.42361918942412451, .54752071970870675, -.31914917137271829, -0.83673298657446777e-1, .26525917360889740, -.14704606647385246, -0.15489414991976576e-1, 0.54789663251404254e-1, -0.19295808218984080e-1, -0.59946709656351250e-2, 0.62453013421289702e-2, -0.75889771423321585e-3, -0.90288108595876188e-3, 0.35950533255739200e-3, 0.53102646864226617e-4, -0.58410284838976865e-4};

    static const double lo3=2.05, hi3=2.85, m3=(hi3+lo3)*0.5;
    static const double q3[16] = {-.17220555920881688, .16011983697724617, -.12825554624056501, 0.89384680084907880e-1, -0.40233745605529377e-1, -0.6582373116861300e-3, 0.16338415642492739e-1, -0.11773063688286990e-1, 0.28825867688662255e-2, 0.12891024069952489e-2, -0.12813889708133769e-2, 0.32788790594626357e-3, 0.95157444762995566e-4, -0.90416802462172157e-4, 0.14868602678620544e-4, 0.72371956855016211e-5};

    static const double lo4=2.85, hi4=3.9, m4=(hi4+lo4)*0.5;
    static const double q4[16] = {-0.87830594664260867e-1, 0.52274501692140091e-1, -0.23878700103915117e-1, 0.10566809233968182e-1, -0.52605680459469043e-2, 0.29752045061396727e-2, -0.16104234494416077e-2, 0.66084450440893938e-3, -0.11555273542331601e-3, -0.79814006315265824e-4, 0.82495635391798472e-4, -0.36112562130960013e-4, 0.56745127440475224e-5, 0.30778452492334783e-5, -0.23005717864408269e-5, 0.53498531579838538e-6};

    static const double lo5=3.9, hi5=5.0, m5=(hi5+lo5)*0.5;
    static const double q5[16] = {-0.50498686366854397e-1, 0.22696136873300272e-1, -0.76507833315391100e-2, 0.22934220273682421e-2, -0.64621983684159045e-3, 0.17717650067309256e-3, -0.49822348110737912e-4, 0.15950369754100344e-4, -0.64382111428954669e-5, 0.30935528834043852e-5, -0.14868198880550113e-5, 0.61815723812849196e-6, -0.19561205283581128e-6, 0.29092516398105484e-7, 0.18203712532317880e-7, -0.12991676945664683e-7};

    static const double lo6=5.0, hi6=7.0, m6=(hi6+lo6)*0.5;
    static const double q6[16] = {-0.27777777777779237e-1, 0.92592592592770810e-2, -0.23148148149286158e-2, 0.51440329259377589e-3, -0.10716735349262763e-3, 0.21433472880730380e-4, -0.41676252668392218e-5, 0.79384016105355812e-6, -0.14884727258382283e-6, 0.27571110992976271e-7, -0.50751367707046113e-8, 0.93366213261041495e-9, -0.16069628589065609e-9, 0.29406617613306847e-10, -0.15659360054098260e-10, 0.58338713932903765e-11};

//     double rsq = r*r;
//     double formula= -erf(r)/rsq + exp(-rsq)*(2/r - 2*r)/sqrt(madness::constants::pi);

    const double* a;

    if (r > hi6) {              // Most common case
        return -1.0/(r*r);
    }
    else if (r > hi3) {
        if (r > hi5) {
            r -= m6;
            a = q6;
        }
        else if (r > hi4) {
            r -= m5;
            a = q5;
        }
        else {
            r -= m4;
            a = q4;
        }
    }
    else if (r > hi1) {
        if (r > hi2) {
            r -= m3;
            a = q3;
        }
        else {
            r -= m2;
            a = q2;
        }
    }
    else if (r > hi0) {
        r -= m1;
        a = q1;
    }
    else {
        r -= m0;                // Least common case
        a = q0;
    }

    double b0 = a[ 0] + r*a[ 1];
    double b1 = a[ 2] + r*a[ 3];
    double b2 = a[ 4] + r*a[ 5];
    double b3 = a[ 6] + r*a[ 7];
    double b4 = a[ 8] + r*a[ 9];
    double b5 = a[10] + r*a[11];
    double b6 = a[12] + r*a[13];
    double b7 = a[14] + r*a[15];

    double r2 = r*r;
    double c0 = b0 + r2*b1;
    double c1 = b2 + r2*b3;
    double c2 = b4 + r2*b5;
    double c3 = b6 + r2*b7;

    double r4 = r2*r2;
    double d0 = c0 + r4*c1;
    double d1 = c2 + r4*c3;

    double r8 = r4*r4;
    double result = d0 + r8*d1;

//     if (abs(result-formula) > 1e-12) {
//         printf("ERROR in dpotn: r=%.10f formula=%.10f result=%.10f err=%.1e\n",
//                r, formula, result, formula-result);
//         throw "bad";
//     }

    return result;
}

/// second radial derivative of the regularized 1/r potential

/// invoke as d2smoothed_potential(r*rc) * rc*rc*rc
/// with rc the reciprocal smoothing radius
/// d2u[r*rc]*rc*rc*rc \approx 2/r^3
double d2smoothed_potential(double r) {
    double rsq = r*r;
    const double sqrtpi=sqrt(madness::constants::pi);
    if (r > 7.0) {
        return 2.0/(rsq*r);
    }
    else if (r > 1e-2) {
        double er2=exp(-rsq);
        double e4r2=exp(-4*rsq);
        return -(4.* er2)*sqrtpi - (4.* er2)/(sqrtpi* rsq)
                + (-2.* er2 + 4.* er2* rsq +
                 16.* (-8.* e4r2 + 64.* e4r2*rsq))/(3*sqrtpi)
                 + (2.* erf(r))/(r*rsq);
    }
    else {
        return
          -134./(3. *sqrtpi) + (2582 *rsq)/(5.* sqrtpi)
          - (35905* rsq*rsq)/(21. *sqrtpi)
          +(86051. *rsq*rsq*rsq)/(27.0*sqrtpi);
    }
}


/// Charge density corresponding to smoothed `1/r` potential

/// To obtain the desired density as a function of `r`,
///// \f$
/////  \frac{\exp(-\frac{r^2}{c^2}) \left(\frac{5}{2}-\frac{r^2}{c^2}\right)}{\pi ^{3/2} c^3}
///// \f$,
/// invoke as \c smoothed_density(r/c)/c^3 where \c c is the radius of the
/// smoothed volume.
/// \param rs effective distance, \f$ r_s \equiv r/c \f$ , from the origin of the density
/// \return \f$ \frac{\exp(-r_s^2) \left(\frac{5}{2}- r_s^2 \right)}{\pi^{3/2}} \f$
double smoothed_density(double rs) {
    static const double rpithreehalf = std::pow(madness::constants::pi, -1.5);
    double rs2 = rs*rs;
    return exp(-rs2)*(2.5 - rs2) * rpithreehalf;
}


// static double smoothing_parameter_original(double Z, double eprec) {
//     // The min is since asymptotic form not so good at low acc.
//     // The 2 is from two electrons in 1s closed shell.
//     if (Z == 0.0) return 1.0;
//     double Z5 = Z*Z*Z*Z*Z;
//     double c = pow(std::min(1e-3,eprec)/2.0/0.00435/Z5,1.0/3.0);
//     return c;
// }


// static double smoothed_potential_original(double r) {
//     // This eliminated the first 3 moments ... not such
//     // a good idea ... 1 moment is good enough.
//     const double THREE_SQRTPI = 5.31736155271654808184;
//     double r2 = r*r, pot;
//     if (r > 6.5){
//         pot = 1.0/r;
//     } else if (r > 1e-8){
//         pot = erf(r)/r + (exp(-r2) + 16.0*exp(-4.0*r2))/(THREE_SQRTPI);
//     } else{
//         pot = (2.0 + 17.0/3.0)/sqrt(PI);
//     }

//     return pot;
// }

// static double dsmoothed_potential_original(double r)
// {
//     if (r < 1e-3) {
//         const double t1 = sqrt(0.31415926535897932385e1);
//         const double t2 = 0.1e1 / t1;
//         const double t5 = r * r;
//         return -0.134e3 / 0.3e1 * r * t2 + 0.2582e4 / 0.15e2 * t5 * r * t2;
//     }
//     else {
//         const double t1 = r * r;
//         const double t2 = exp(-t1);
//         const double t5 = erf(r);
//         const double t6 = sqrt(PI);
//         const double t9 = t1 * r;
//         const double t13 = exp(-0.4e1 * t1);
//         return -(-0.6e1 * r * t2 + 0.3e1 * t5 * t6 + 0.2e1 * t9 * t2 + 0.128e3 * t9 * t13) / t6 / t1 / 0.3e1;
//     }
// }


// /// Charge density corresponding to smoothed 1/r potential

// /// Invoke as \c rho(r/c)/c^3 where \c c is the radius of the
// /// smoothed volume.

// static double smoothed_density_original(double r) {
//     const double RPITO1P5 = 0.1795871221251665617; // 1.0/Pi^1.5
//     return ((-3.0/2.0+(1.0/3.0)*r^2)*exp(-r^2)+(-32.0+(256.0/3.0)*r^2)*exp(-4.0*r^2))*RPITO1P5;
// }

}
