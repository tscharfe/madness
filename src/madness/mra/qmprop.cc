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
#include <madness/mra/mra.h>
#include <madness/mra/operator.h>
#include <madness/mra/adquad.h>
#include <madness/misc/cfft.h>
#include <madness/misc/interpolation_1d.h>
#include <cmath>
#include <complex>
#include <madness/constants.h>
#include <iostream>
#include <vector>
#include <algorithm>

/// \file qmprop.cc
/// \brief Implements BandlimitedPropagator and qm_free_particle_propagator


namespace madness {
    /// Class to evaluate the filtered Schrodinger free-particle propagator in real space

    /// Follows the corresponding Maple worksheet and the implementation notes.
    class BandlimitedPropagator {
    private:
        double width;
        double xmax;
        CubicInterpolationTable<double_complex> fit;

        double_complex g0_filtered(double k, double c, double t) {
            double r = k/c;
            if (fabs(r) > 4.0) return 0.0;
            double_complex arg(0.0, -k*k*t*0.5);
            return exp(arg)/(1.0 + pow(r,30.0));
        }

    public:
        typedef double_complex returnT;

        BandlimitedPropagator(double c, double t, double width) : width(width) {
            // 4.0 is range of window in Fourier space, 32x for
            // oversampling in real space for accurate cubic
            // interpolation.  

            // Extra factor of 4 from accurate tdse1d testing

            const double kmax = 32.0*4.0*c * 4;
            const double fac = kmax/constants::pi;
            
            // NEW
            double hk = 0.0625;
            const double dN = 2.0*kmax/hk;
            const int N = std::pow(2.0,double(std::ceil(std::log(dN)/std::log(2.0))));
            hk = 2.0*kmax/N;

            // ORIGINAL
            // const double fac = kmax/constants::pi;
            // const double c2 = std::pow(2.0,double(std::ceil(std::log(c)/std::log(2.0))));
            // const int N = 131072 * std::max(2*c2,1.0);
            // const double hk = 2.0*kmax/N;

            const double hx = constants::pi/kmax;

            //print("c", c,"t", t, "width", width, "kmax", kmax, "hk", hk, "N", N, "hx", hx);
            
            std::vector<double_complex> s(N);

            for (int i=0; i<N/2; ++i) {
                double k = i*hk;
                s[i] = g0_filtered(k, c, t)*fac;
                if (i) s[N-i] = g0_filtered(-k, c, t)*fac;
            }

            CFFT::Inverse(&s[0], N);

            // for (int i=0; i<N; i++) {
            //     print(i, real(s[i]), imag(s[i]));
            // }
            // throw "done";
            
            MADNESS_ASSERT(abs(s[N/2]) <1e-14);
            int n;
            for (n=N/2; n>=0 && std::abs(s[n])<1e-14; --n);
            ++n;

            s.resize(n);

            xmax = (n-1)*hx;
            fit = CubicInterpolationTable<double_complex>(0.0, xmax, n, s);
        }

        Level natural_level() const {return 13;}

        std::complex<double> operator()(double x) const {
            x = fabs(x)*width;
            if (x >= xmax) return 0.0;
            else return fit(x)*width;
        }

        static void test() {
            std::complex<double> maple(1.138514411208581,-0.986104972271240);
            BandlimitedPropagator bp(31.4, 0.07, 1.0);
            if (std::abs(bp(0.1)-maple) > 1e-11) {
                std::cout.precision(14);
                std::cout << bp(0.1) << " " << maple << " " << bp(0.1)-maple << std::endl;
                throw "BandlimitedPropagator: failed test";
            }
            return;
        }

        static void plot() {
            test();
            std::cout.precision(12);
            for (int j=0; j<=4; j++) {
                double width = 100.0;
                double c = 10.0;
                double tcrit  = 2.0*3.14159/(c*c);
                double t = tcrit * (1<<j);
                BandlimitedPropagator bp(c, t, width);

                print("QM: c", c, "tcrit", tcrit, "*", (1<<j));
                for (int i=0; i<10001; ++i) {
                    double x = i/10000.0; // SIMULATION coords so don't need width
                    double_complex value = bp(x);
                    print(x*width,value.real(),value.imag());
                }
            }

        }
    };

    void bandlimited_propagator_plot(){BandlimitedPropagator::plot();}


    Convolution1D<double_complex>*
    qm_1d_free_particle_propagator(int k, double bandlimit, double timestep, double width) {
        return new GenericConvolution1D<double_complex,BandlimitedPropagator>(k,BandlimitedPropagator(bandlimit,timestep,width),0);
    }

    template <std::size_t NDIM>
    SeparatedConvolution<double_complex,NDIM>
    qm_free_particle_propagator(World& world, int k, double bandlimit, double timestep) {
        double width = FunctionDefaults<NDIM>::get_cell_min_width(); // Assuming cubic so all dim equal
        std::vector< std::shared_ptr< Convolution1D<double_complex> > > q(1);
        q[0].reset(qm_1d_free_particle_propagator(k, bandlimit, timestep, width));
        return SeparatedConvolution<double_complex,NDIM>(world, q, k, true);
    }

    template <std::size_t NDIM>
    SeparatedConvolution<double_complex,NDIM>*
    qm_free_particle_propagatorPtr(World& world, int k, double bandlimit, double timestep) {
        double width = FunctionDefaults<NDIM>::get_cell_min_width(); // Assuming cubic so all dim equal
        std::vector< std::shared_ptr< Convolution1D<double_complex> > > q(1);
        q[0].reset(qm_1d_free_particle_propagator(k, bandlimit, timestep, width));
        return new SeparatedConvolution<double_complex,NDIM>(world, q, k, true);
    }

#ifdef FUNCTION_INSTANTIATE_1
    template SeparatedConvolution<double_complex,1> qm_free_particle_propagator(World& world, int k, double bandlimit, double timestep);
    template SeparatedConvolution<double_complex,1>* qm_free_particle_propagatorPtr(World& world, int k, double bandlimit, double timestep);
#endif

#ifdef FUNCTION_INSTANTIATE_2
    template SeparatedConvolution<double_complex,2> qm_free_particle_propagator(World& world, int k, double bandlimit, double timestep);
#endif

#ifdef FUNCTION_INSTANTIATE_3
    template SeparatedConvolution<double_complex,3> qm_free_particle_propagator(World& world, int k, double bandlimit, double timestep);
#endif

#ifdef FUNCTION_INSTANTIATE_4
    template SeparatedConvolution<double_complex,4> qm_free_particle_propagator(World& world, int k, double bandlimit, double timestep);
#endif

#ifdef FUNCTION_INSTANTIATE_5
    template SeparatedConvolution<double_complex,5> qm_free_particle_propagator(World& world, int k, double bandlimit, double timestep);
#endif

#ifdef FUNCTION_INSTANTIATE_6
    template SeparatedConvolution<double_complex,6> qm_free_particle_propagator(World& world, int k, double bandlimit, double timestep);
#endif
}
