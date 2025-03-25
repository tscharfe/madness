#include <madness/mra/mra.h>
#include <madness/mra/funcplot.h>
#include <madness/mra/nonlinsol.h>

using namespace madness;

const double L = 7.0;
//const double DELTA = 3*L*L/2; // Use this to make fixed-point iteration converge
const double DELTA = 7.0;

// The initial guess wave function
double guess(const coord_1d& r) {
  return exp(-(r[0]*r[0])/1.0);
}

// The shifted potential
double potential(const coord_1d& r) {
  return 0.5*(r[0]*r[0]) - DELTA;
}

const std::string path_to_plots="/Users/truonthu/Documents/MRA/plots";
// Convenience routine for plotting
void plot(const char* filename, const real_function_1d& f) {
  coord_1d lo(0.0), hi(0.0);
  lo[0] = -L; hi[0] = L;
  std::string full_path=path_to_plots+filename;
  plot_line(full_path.c_str(),401,lo,hi,f);
}

double energy(World& world, const real_function_1d& phi, const real_function_1d& V) {
  double potential_energy = inner(phi,V*phi); // <phi|Vphi> = <phi|V|phi>
  double kinetic_energy = 0.0;
  for (int axis=0; axis<1; axis++) {
    real_derivative_1d D = Derivative<double,1>(world, axis);
    real_function_1d dphi = D(phi);
    kinetic_energy += 0.5*inner(dphi,dphi);  // (1/2) <dphi/dx | dphi/dx>
  }
  double energy = kinetic_energy + potential_energy;
  //print("kinetic",kinetic_energy,"potential", potential_energy, "total", energy);
  return energy;
}

int main(int argc, char** argv) {
    initialize(argc, argv);
    World world(SafeMPI::COMM_WORLD);
    startup(world,argc,argv);
    if (world.rank() == 0) printf("starting at time %.1f\n", wall_time());

    const double thresh = 1e-5;
    FunctionDefaults<1>::set_k(6);
    FunctionDefaults<1>::set_thresh(thresh);
    FunctionDefaults<1>::set_cubic_cell(-L,L);
    NonlinearSolverND<1> solver;

    real_function_1d phi = real_factory_1d(world).f(guess);
    real_function_1d V = real_factory_1d(world).f(potential);
    plot("potential.dat", V);

    phi.scale(1.0/phi.norm2());  // phi *= 1.0/norm

    double E = energy(world,phi,V);


    for (int iter=0; iter<100; iter++) {
      char filename[256];
      snprintf(filename, 256, "phi-%3.3d.dat", iter);
      plot(filename,phi);

      real_function_1d Vphi = V*phi;
      Vphi.truncate();
      real_convolution_1d op = BSHOperator<1>(world, sqrt(-2*E), 0.01, thresh);

      real_function_1d r = phi + 2.0 * op(Vphi); // the residual
      double err = r.norm2();

      phi = solver.update(phi, r);
      //phi = phi-r;

      double norm = phi.norm2();
      phi.scale(1.0/norm);  // phi *= 1.0/norm
      E = energy(world,phi,V);

      if (world.rank() == 0)
          print("iteration", iter, "energy", E, "norm", norm, "error",err);

      if (err < 5e-4) break;
    }

    print("Final energy without shift", E+DELTA);

    if (world.rank() == 0) printf("finished at time %.1f\n", wall_time());
    finalize();
    return 0;
}
