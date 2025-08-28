#include <madness/mra/mra.h>
#include <madness/mra/operator.h>
#include <tuple>
#include <string>


using namespace madness;


const double L = 50.0; // box size
const long k = 8; // wavelet order
const double thresh = 1e-6; // precision

const std::string path_to_plots = "/Users/timo/workspace/up_to_date_madness/madness/cmake-build-debug/src/examples/";
// Convenience routine for plotting
template<int NDIM>
void plane_plot(World& world, std::string filename, Function<double, NDIM> f, const std::string plane = "xy", double zoom = 1, int datapoints = 251,
                                std::vector<double> origin = {0.0,0.0,0.0}) {
    PlotParameters params;
    if (plane == "xy" || plane == "yx") {
        params.set_plane({"x1", "x2"});
    } else if (plane == "yz" || plane == "zy") {
        params.set_plane({"x2", "x3"});
    } else if (plane == "zx" || plane == "xz") {
        params.set_plane({"x1", "x3"});
    } else {
        std::cout << "Plane to plot not recognized.";
    }
    params.set_zoom(zoom);
    params.set_origin(origin);

    params.set_npoints(datapoints);
    std::string full_path = path_to_plots + filename;
    plot_plane<NDIM>(world, f, filename, params);
}


// Convenience routine for plotting
template<int NDIM>
void plot(const char *filename, const Function<double, NDIM> &f) {
    Vector<double, NDIM> lo(0.0), hi(0.0);
    lo[NDIM - 1] = -L / 2;
    hi[NDIM - 1] = L / 2;
    std::string full_path = path_to_plots + filename;
    plot_line(full_path.c_str(), 50000, lo, hi, f);
}

template<std::size_t NDIM>
std::pair<Tensor<double>,Tensor<double>> gauss_fit_coulomb(double lo, double hi, double eps, bool prnt = false, bool fix_interval = false) {
    //original implementation in madness/mra/gfit.h (function bsh_fit)
	eps=eps/(4.0*constants::pi);
    double TT;
    double slo, shi;

    if (eps >= 1e-2) TT = 5;
    else if (eps >= 1e-4) TT = 10;
    else if (eps >= 1e-6) TT = 14;
    else if (eps >= 1e-8) TT = 18;
    else if (eps >= 1e-10) TT = 22;
    else if (eps >= 1e-12) TT = 26;
    else TT = 30;

    slo = log(eps / hi) - 1.0;

    shi = 0.5 * log(TT / (lo * lo));
    if (shi <= slo) throw "bsh_fit: logic error in slo,shi";

    // Resolution required for quadrature over s
    double h = 1.0 / (0.2 - .50 * log10(eps)); // was 0.5 was 0.47

    // Truncate the number of binary digits in h's mantissa
    // so that rounding does not occur when performing
    // manipulations to determine the quadrature points and
    // to limit the number of distinct values in case of
    // multiple precisions being used at the same time.
    h = floor(64.0 * h) / 64.0;

    // Round shi/lo up/down to an integral multiple of quadrature points
    shi = ceil(shi / h) * h;
    slo = floor(slo / h) * h;

    long npt = long((shi - slo) / h + 0.5);

    Tensor<double> coeff(npt), expnt(npt);

    for (int i = 0; i < npt; ++i) {
        double s = slo + h * (npt - i); // i+1
        coeff[i] = h * 2.0 / sqrt(constants::pi) * exp(s);
        coeff[i] = coeff[i]/(4.0*constants::pi);
        expnt[i] = exp(2.0 * s);
    }

#if ONE_TERM
    npt=1;
    double s=1.0;
    coeff[0]=1.0;
    expnt[0] = exp(2.0*s);
    coeff=coeff(Slice(0,0));
    expnt=expnt(Slice(0,0));
    print("only one term in gfit",s,coeff[0],expnt[0]);


#endif

    // Prune large exponents from the fit ... never necessary due to construction

    // Prune small exponents from Coulomb fit.  Evaluate a gaussian at
    // the range midpoint, and replace it there with the next most
    // diffuse gaussian.  Then examine the resulting error at the two
    // end points ... if this error is less than the desired
    // precision, can discard the diffuse gaussian.

    if (not fix_interval) {
        //		if (restrict_interval) {
        GFit<double, NDIM>::prune_small_coefficients(eps, lo, hi, coeff, expnt);
    }

    //in the original implementation there is a bunch of code here which gets executed when a variable nmom is larger than 0,
    //however nmom is always set to zero in the original implementation, so I removed it
    coeff.scale(4.0*constants::pi);
    return {coeff,expnt};
}

template<std::size_t NDIM>
SeparatedConvolution<double,NDIM>* CoulombOperatorNDPtr(World& world,
                                                       double lo,
                                                       double eps,
                                                       const array_of_bools<NDIM>& lattice_summed = FunctionDefaults<NDIM>::get_bc().is_periodic(),
                                                       int k=FunctionDefaults<NDIM>::get_k())
{
    const Tensor<double> &cell_width =
              FunctionDefaults<NDIM>::get_cell_width();
    double hi = cell_width.normf(); // Diagonal width of cell
    // Extend kernel range for lattice summation
    // N.B. if have periodic boundaries, extend range just in case will be using periodic domain
    const auto lattice_summed_any = lattice_summed.any();
    if (lattice_summed.any() || FunctionDefaults<NDIM>::get_bc().is_periodic_any()) {
        hi *= 100;
    }
    auto [coeffs, expnts]=gauss_fit_coulomb<NDIM>(lo,hi,eps);
    return new SeparatedConvolution<double,NDIM>(world,coeffs,expnts, lo, eps, lattice_summed, k);
}

template<int NDIM>
Vector<double, NDIM> origin(0.0);

template<int NDIM>
class SumOfGaussians : public FunctionFunctorInterface<double, NDIM> {
public:
    double a;
    double Q;
    std::vector<Vector<double, NDIM> > charge_locations;

    double get_a() {
        return a;
    }
    SumOfGaussians(double a = 100, double Q = 1,
                   std::vector<Vector<double, NDIM> > charge_locations = {origin<NDIM>}) : a(a), Q(Q),
        charge_locations(charge_locations) {
    }

    double operator()(const Vector<double, NDIM> &r) const override {
        double result = 0.0;
        for (Vector<double, NDIM> ChargeLoc: charge_locations) {
            result += exp(-a * std::pow((r - ChargeLoc).normf(), 2));
        }
        return result;
    }
};


void run2d(int argc, char **argv) {
    World &world = initialize(argc, argv);
    startup(world, argc, argv);
    std::cout.precision(6);


    const double a = 1000000;
    const double Q = 1.0;
    double coulomb_lo = 0.001;
    double coulomb_eps = 1e-6;
    std::vector<Vector<double, 2> > charge_locations = {origin<2>};

    FunctionDefaults<2>::set_k(k);
    FunctionDefaults<2>::set_thresh(thresh);
    FunctionDefaults<2>::set_truncate_mode(1);
    FunctionDefaults<2>::set_cubic_cell(-L / 2, L / 2);
    SumOfGaussians<2> Rho(a, Q, charge_locations);
    Function<double, 2> f = FunctionFactory<double, 2>(world).special_level(20).special_points(Rho.charge_locations).
            functor(Rho);
    double norm = f.trace();
    f = Rho.Q / norm * f;
    std::vector<Function<double, 2> > f_vec = {f};
    auto COpPtr = std::shared_ptr<SeparatedConvolution<double, 2> >(CoulombOperatorNDPtr<2>(world, coulomb_lo, coulomb_eps));
    std::vector<Function<double, 2> > Cf = apply(world, *COpPtr, f_vec);
    std::string filename1="einsur"+std::to_string(2)+"D.dat";
    std::string filename2="delta"+std::to_string(2)+"D.dat";
    plot<2>(filename1.c_str(), Cf[0]);
    plot<2>(filename2.c_str(),f);
    plane_plot<2>(world,"einsur2D.dat",Cf[0],"xy",5);
    plane_plot<2>(world,"delta2D.dat",f,"xy",5);

}

void run3d(int argc, char **argv, bool wt) {
    World &world = initialize(argc, argv);
    startup(world, argc, argv);
    std::cout.precision(6);


    const double a = 1000000;
    const double Q = 1.0;
    double coulomb_lo = 0.001;
    double coulomb_eps = 1e-6;
    std::vector<Vector<double, 3> > charge_locations = {origin<3>};

    FunctionDefaults<3>::set_k(k);
    FunctionDefaults<3>::set_thresh(thresh);
    FunctionDefaults<3>::set_truncate_mode(1);
    FunctionDefaults<3>::set_cubic_cell(-L / 2, L / 2);
    SumOfGaussians<3> Rho(a, Q, charge_locations);
    Function<double, 3> f = FunctionFactory<double, 3>(world).special_level(10).special_points(Rho.charge_locations).
            functor(Rho);
    double norm = f.trace();
    f = Rho.Q / norm * f;
    std::vector<Function<double, 3> > f_vec = {f};
    std::vector<Function<double, 3> > Cf;
    if (wt) {
        auto COpPtr = std::shared_ptr<SeparatedConvolution<double, 3> >(CoulombOperatorNDPtr<3>(world, coulomb_lo, coulomb_eps));
        Cf = apply(world, *COpPtr, f_vec);
        std::string filename1="einsur"+std::to_string(3)+"D_wt.dat";
        std::string filename2="delta"+std::to_string(3)+"D_wt.dat";
        plot<3>(filename1.c_str(), Cf[0]);
        plot<3>(filename2.c_str(),f);
        plane_plot<3>(world,"einsur3D_wt.dat",Cf[0],"xy",5);
        plane_plot<3>(world,"delta3D_wt.dat",f,"xy",5);
    } else {
        auto COpPtr = std::shared_ptr<SeparatedConvolution<double, 3> >(CoulombOperatorPtr(world, coulomb_lo, coulomb_eps));
        Cf = apply(world, *COpPtr, f_vec);
        std::string filename1="einsur"+std::to_string(3)+"D.dat";
        std::string filename2="delta"+std::to_string(3)+"D.dat";
        plot<3>(filename1.c_str(), Cf[0]);
        plot<3>(filename2.c_str(),f);
        plane_plot<3>(world,"einsur3D.dat",Cf[0],"xy",5);
        plane_plot<3>(world,"delta3D.dat",f,"xy",5);
    }

}



void run1d(int argc, char **argv) {
    World &world = initialize(argc, argv);
    startup(world, argc, argv);
    std::cout.precision(6);


    const double a = 1000000000;
    const double Q = 1.0;
    double coulomb_lo = 0.001;
    double coulomb_eps = 1e-4;
    std::vector<Vector<double, 1> > charge_locations = {origin<1>};

    FunctionDefaults<1>::set_k(k);
    FunctionDefaults<1>::set_thresh(thresh);
    FunctionDefaults<1>::set_truncate_mode(1);
    FunctionDefaults<1>::set_cubic_cell(-L / 2, L / 2);
    SumOfGaussians<1> Rho(a, Q, charge_locations);
    Function<double, 1> f = FunctionFactory<double, 1>(world).special_level(20).special_points(Rho.charge_locations).
            functor(Rho);
    double norm = f.trace();
    f = Rho.Q / norm * f;
    std::vector<Function<double, 1> > f_vec = {f};
    auto COpPtr = std::shared_ptr<SeparatedConvolution<double, 1> >(CoulombOperatorNDPtr<1>(world, coulomb_lo, coulomb_eps));
    std::vector<Function<double, 1> > Cf = apply(world, *COpPtr, f_vec);
    std::string filename1="einsur"+std::to_string(1)+"D.dat";
    std::string filename2="delta"+std::to_string(1)+"D.dat";
    plot<1>(filename1.c_str(), Cf[0]);
    plot<1>(filename2.c_str(),f);

}



int main(int argc, char **argv) {
    run1d(argc,argv);
    finalize();

    return 0;
}
