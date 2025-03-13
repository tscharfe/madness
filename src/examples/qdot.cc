#include <fstream>
#include <iostream>
#include <madness/mra/mra.h>
#include <madness/mra/operator.h>

#include "SCF.h"

using namespace madness;

static const double L = 32.0;   // box size
static const long k = 7;        // wavelet order
static const double thresh = 1e-6; // precision
static const double F=0.0; // electric field


const std::string path_to_plots="/Users/timo/up_to_date_madness/madness/cmake-build-debug/plots/";
// Convenience routine for plotting
template<int NDIM>
void plot(const char* filename, const Function<double,NDIM>& f) {
    Vector<double,NDIM> lo(0.0), hi(0.0);
    lo[NDIM-1] = -L/2; hi[NDIM-1] = L/2;
    std::string full_path=path_to_plots+filename;
    plot_line(full_path.c_str(),401,lo,hi,f);
}


template <int NDIM>
double rho(const Vector<double,NDIM> &r) {
    double a=1000.0;
    return exp(-a*std::pow(r.normf(),2));
}

template <int NDIM>
double guess(const Vector<double,NDIM> &r) {
    return exp(-r.normf()/2.0)/sqrt(8.0*constants::pi);
}


template <int NDIM>
Function<double,NDIM> make_potential(World & world) {
    std::vector<Vector<double,NDIM> > points;
    Vector<double,NDIM> nullvector;
    points.push_back(nullvector);
    Function<double,NDIM> f=FunctionFactory<double,NDIM>(world).special_level(6).special_points(points).f(rho<NDIM>);
    auto norm=f.trace();
    f=1/norm*f;
    SeparatedConvolution<double,NDIM> op = BSHOperator<NDIM>(world, 0.0, 0.001,1e-6);
    auto V=op(f);
    V=V.truncate(thresh);
    return -4.0*constants::pi*V;
}



template <int NDIM>
void iterate(World& world, Function<double,NDIM>& V, Function<double,NDIM>& psi, double& eps) {
    Function<double,NDIM> Vpsi = (V*psi);
    Vpsi.scale(-2.0).truncate();
    SeparatedConvolution<double,NDIM> op = BSHOperator<NDIM>(world, sqrt(-2*eps), 0.001, 1e-6);
    Function<double,NDIM> tmp = apply(op,Vpsi).truncate();
    double norm = tmp.norm2();
    Function<double,NDIM> r = tmp-psi;
    double rnorm = r.norm2();
    double eps_new;
    if (rnorm > 0.2) {
        r *= 0.2/rnorm;
        print("step restriction");
        eps_new = eps;
    }
    else {
        // Only update energy once step restriction is lifted since this is only locally convergent
        eps_new = eps - 0.5*inner(Vpsi,r)/(norm*norm);
    }
    if (world.rank() == 0) {
        print("norm=",norm," eps=",eps," err(psi)=",rnorm," err(eps)=",eps_new-eps);
    }
    psi += r;
    psi.scale(1.0/psi.norm2());
    eps = eps_new;
}

template <int NDIM>
std::tuple<double,double,double> compute_energy(World& world, Function<double,NDIM>& Vnuc, Function<double,NDIM>& psi) {
    double kinetic_energy = 0.0;
    for (int axis=0; axis<NDIM; axis++) {
        Derivative<double,NDIM> D = free_space_derivative<double,NDIM>(world, axis);
        Function<double,NDIM> dpsi = D(psi);
        kinetic_energy += 0.5*inner(dpsi,dpsi);
    }
    Function<double,NDIM> rho = square(psi).truncate();
    double nuclear_attraction_energy = inner(Vnuc*psi,psi);
    double total_energy = kinetic_energy + nuclear_attraction_energy;
    return {kinetic_energy, nuclear_attraction_energy, total_energy};
}


template <int NDIM>
void run(World& world) {
    FunctionDefaults<NDIM>::set_k(k);
    FunctionDefaults<NDIM>::set_thresh(thresh);
    FunctionDefaults<NDIM>::set_truncate_mode(1);
    FunctionDefaults<NDIM>::set_cubic_cell(-L/2,L/2);

    Function<double,NDIM> Vnuc = make_potential<NDIM>(world);
    plot<NDIM>("Vnuc_plot",Vnuc);
    Function<double,NDIM> psi  = FunctionFactory<double,NDIM>(world).f(guess<NDIM>);
    print("initial", psi.norm2());
    psi.scale(1.0/psi.norm2());

    double eps = -0.5;
    for (int iter=0; iter<15; iter++) {
        Function<double,NDIM> rho = square(psi).truncate();
        iterate<NDIM>(world, Vnuc, psi, eps);
    }

    // Manually tabluate the orbital along a line along the z axis ... probably easier to use the lineplot routine
    /*
    coord_3d r(0.0);
    psi.reconstruct();
    for (int i=0; i<201; i++) {
        r[2] = -L/2 + L*i/200.0;
        print(r[2], psi(r));
    }
    */

    auto [kinetic_energy, nuclear_attraction_energy, total_energy] = compute_energy<NDIM>(world, Vnuc, psi);
    if (world.rank() == 0) {
        print("            Electric Field ", F);
        print("            Kinetic energy ", kinetic_energy);
        print(" Nuclear attraction energy ", nuclear_attraction_energy);
        print("              Total energy ", total_energy);
    }
}

int main(int argc, char** argv) {
    World& world = initialize(argc, argv);
    startup(world,argc,argv);
    std::cout.precision(6);



    run<3>(world);

    world.gop.fence();
    finalize();
    return 0;
}
