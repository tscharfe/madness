//
// Created by Timo Scharfe on 07.03.25.
//
#include<madness/mra/mra.h>
#include<vector>
#include<functional>

using namespace madness;

const double L=5.0;

double sinusf(const coord_1d& r){
  return std::sin(r[0]);
}
double guess(const coord_1d& r) {
    return exp(-r[0]*r[0]);
}



double potential_with_lambda(const double lambda, const coord_1d& r) {
     return 0.5*std::tanh((r[0]-1)/lambda)-0.5*std::tanh((r[0]+1)/lambda);
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

    double lambda=0.001;
    double potential = std::bind(potential_with_lambda,lambda,std::placeholders::_1);


    real_function_1d phi = real_factory_1d(world).f(guess);
    real_function_1d V = real_factory_1d(world).f(potential);
    real_function_1d Vphi = V*phi;


    double integral=Vphi.trace();

    print("integral = ",integral);

    if (world.rank() == 0) printf("finished at time %.1f\n", wall_time());
    finalize();
    return 0;
}