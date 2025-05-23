//
// Created by adrianhurtado on 2/3/22.
//

#include "FrequencyResponse.hpp"

#include "property.h"


void FrequencyResponse::initialize(World &world) {
    if (world.rank() == 0) { print("FrequencyResponse::initialize()"); }
    Chi = PQ.copy();
}


void FrequencyResponse::iterate(World &world) {
    size_t iter;
    // Variables needed to iterate
    madness::QProjector<double, 3> projector(ground_orbitals);
    size_t n = r_params.num_orbitals();// Number of ground state orbitals
    size_t m = r_params.num_states();  // Number of excited states

    real_function_3d v_xc;
    const double dconv = std::max(FunctionDefaults<3>::get_thresh() * 10,
                                  r_params.dconv());//.01 .0001 .1e-5
    auto thresh = FunctionDefaults<3>::get_thresh();
    auto density_target = dconv * std::max(size_t(5.0), molecule.natom());
    const double a_pow{0.50209};
    const double b_pow{-0.99162};

    const double x_relative_target =
            pow(thresh, a_pow) * pow(10, b_pow);//thresh^a*10^b
    Tensor<double> x_relative_residuals((int(m)));
    Tensor<double> density_residuals((int(m)));

    bool static_res = (omega == 0.0);
    bool compute_y = not static_res;
    int r_vector_size;
    all_done = false;
    r_vector_size = (compute_y) ? 2 * n : n;
    Tensor<double> v_polar(m, m);
    Tensor<double> polar;
    Tensor<double> res_polar;

    vecfuncT rho_omega_old(m);
    // initialize DFT XC functional operator
    XCOperator<double, 3> xc = make_xc_operator(world);
    // create X space residuals
    X_space residuals = X_space::zero_functions(world, m, n);
    // create a std vector of XNONLinearsolvers
    response_solver kain_x_space;
    for (size_t b = 0; b < m; b++) {
        kain_x_space.emplace_back(
                response_matrix_allocator(world, r_vector_size), false);
    }
    if (r_params.kain()) {
        for (auto &kain_space_b: kain_x_space) {
            kain_space_b.set_maxsub(r_params.maxsub());
        }
    }
    // We compute with positive frequencies
    if (world.rank() == 0) {
        print("Warning input frequency is assumed to be positive");
        print("Computing at positive frequency omega = ", omega);
    }
    double x_shifts = 0.0;
    double y_shifts = 0.0;
    // if less negative orbital energy + frequency is positive or greater than 0
    if ((ground_energies[long(n) - 1] + omega) >= 0.0) {
        // Calculate minimum shift needed such that \eps + \omega + shift < 0
        print("*** we are shifting just so you know!!!");
        x_shifts = -.05 - (omega + ground_energies[long(n) - 1]);
    }
    auto bsh_x_ops = make_bsh_operators_response(world, x_shifts, omega);
    std::vector<poperatorT> bsh_y_ops;
    bsh_y_ops = (compute_y)
                        ? make_bsh_operators_response(world, y_shifts, -omega)
                        : bsh_x_ops;
    auto max_rotation = .5;
    if (thresh >= 1e-2) {
        max_rotation = 2;
    } else if (thresh >= 1e-4) {
        max_rotation = 2 * x_relative_target;
    } else if (thresh >= 1e-6) {
        max_rotation = 2 * x_relative_target;
    } else if (thresh >= 1e-7) {
        max_rotation = .01;
    }
    PQ = generator(world, *this);

    vector<bool> converged(Chi.num_states(), false);
    Chi.reset_active();
    // make density for the first time
    auto rho_omega = response_context.compute_density(
            world, Chi, ground_orbitals,
            vector_real_function_3d(Chi.num_states()), false);

    for (iter = 0; iter < r_params.maxiter(); ++iter) {
        //if (world.rank() == 0) { print("At the start of iterate x", checkx); }
        iter_timing.clear();
        iter_function_data.clear();

        if (r_params.print_level() >= 1) {
            molresponse::start_timer(world);
            if (world.rank() == 0)
                printf("\n   Iteration %d at time %.1fs\n",
                       static_cast<int>(iter), wall_time());
            if (world.rank() == 0)
                print("-------------------------------------------");
        }
        if (iter < 2 || (iter % 5) == 0) { load_balance_chi(world); }
        if (iter > 0) {
            if (density_residuals.max() > 20 && iter > 5) {
                if (world.rank() == 0) { print("d-residual > 20...break"); }
                break;
            }

            auto chi_norms = (compute_y) ? Chi.norm2s() : Chi.x.norm2();
            auto rho_norms = madness::norm2s_T(world, rho_omega);

            // Todo add chi norm and chi_x
            if (world.rank() == 0) {
                function_data_to_json(j_molresponse, iter, chi_norms,
                                      x_relative_residuals, rho_norms,
                                      density_residuals);
                frequency_to_json(j_molresponse, iter, polar, res_polar);
            }
            if (r_params.print_level() >= 1) {
                if (world.rank() == 0) {
                    print("r_params.dconv(): ", r_params.dconv());
                    print("thresh: ", FunctionDefaults<3>::get_thresh());
                    print("k: ", FunctionDefaults<3>::get_k());
                    print("Chi Norms at start of iteration: ", iter);
                    print("||X||: ", chi_norms);
                    print("<< XI | XJ >>(omega): \n", polar);
                    print("targets : ||x||", x_relative_target,
                          "    ||delta_rho||", density_target);
                }
            }
            auto check_convergence = [&](auto &ri, auto &di) {
                if (world.rank() == 0) {
                    print("              ", ri, "    ", di);
                }
                return ((ri < x_relative_target) && (di < density_target));
            };

            for (const auto &b: Chi.active) {
                converged[b] = check_convergence(x_relative_residuals[b],
                                                 density_residuals[b]);
            }
            int b = 0;
            auto remove_converged = [&]() {
                Chi.reset_active();
                Chi.active.remove_if([&](auto x) { return converged[b++]; });
            };
            remove_converged();

            if (world.rank() == 0) {
                print("converged", converged);
                print("active", Chi.active);
            }
            b = 0;
            all_done = std::all_of(converged.begin(), converged.end(),
                                   [](const auto &ci) { return ci; });
            if (all_done || iter == r_params.maxiter()) {
                // if converged print converged
                if (world.rank() == 0 && all_done and
                    (r_params.print_level() > 1)) {
                    print("\nConverged!\n");
                }
                if (r_params.save()) {
                    molresponse::start_timer(world);
                    save(world, r_params.save_file());
                    if (r_params.print_level() >= 1)
                        molresponse::end_timer(world, "Save:");
                }
                if (r_params.plot_all_orbitals()) {
                    //plotResponseOrbitals(world, iter, Chi.x, Chi.y, r_params,
                     //                    ground_calc);
                }
                break;
            }
        }
        auto x_inner = ((compute_y) ? 2 : 1) * response_context.inner(Chi, Chi);
        inner_to_json(world, "x", x_inner, iter_function_data);

        auto rho_omega_norm = norm2s_T(world, rho_omega);
        inner_to_json(world, "density_norms", rho_omega_norm,
                      iter_function_data);
        auto [new_chi, new_res, new_rho] = update_response(
                world, Chi, xc, bsh_x_ops, bsh_y_ops, projector, x_shifts,
                omega, kain_x_space, iter, max_rotation, rho_omega,
                x_relative_residuals, residuals);

        auto old_rho = copy(world, rho_omega);
        rho_omega = copy(world, new_rho);
        // first thing we should do is update the density residuals
        // drho = rho(x)-rho(g(x))
        // new_rho= rho(g(x))

        for (const auto &b: Chi.active) {
            auto drho_b = rho_omega[b] - old_rho[b];
            auto drho_b_norm = drho_b.norm2();
            world.gop.fence();
            density_residuals[b] = drho_b_norm;
        }
        world.gop.fence();

        auto old_density_residual = copy(density_residuals);
        iter_function_data["r_d"] = old_density_residual;

        // Now we should update the orbitals and density
        // x= x+deltax
        // rho = rho(x+delta x)

        if (compute_y) {
            Chi = new_chi.copy();
        } else {
            Chi.x = new_chi.x.copy();
        }

        if (r_params.print_level() >= 1) { molresponse::start_timer(world); }
        x_relative_residuals = copy(new_res.residual_norms);
        residuals = new_res.residual.copy();
        if (r_params.print_level() >= 1) {
            molresponse::end_timer(world, "copy_response_data",
                                   "copy_response_data", iter_timing);
        }
        inner_to_json(world, "x_relative_residuals", x_relative_residuals,
                      iter_function_data);

        inner_to_json(world, "density_residuals", old_density_residual,
                      iter_function_data);

        auto dnorm = norm2s_T(world, rho_omega);
        iter_function_data["d"] = dnorm;
        polar = ((compute_y) ? -2 : -4) * response_context.inner(Chi, PQ);
        res_polar = ((compute_y) ? -2 : -4) *
                    response_context.inner(new_res.residual, PQ);
        inner_to_json(world, "alpha", polar, iter_function_data);
        inner_to_json(world, "r_alpha", res_polar, iter_function_data);
        if (r_params.print_level() >= 20) {
            if (world.rank() == 0) {
                printf("\n--------Response Properties after %d-------------\n",
                       static_cast<int>(iter));
                print("<<X,P>> at omega =", omega);
                print(polar);
                print("res r<<X,P>> at omega =", omega);
                print(res_polar);
            }
        }
        if (r_params.print_level() >= 1) {
            molresponse::end_timer(world, "Iteration Timing", "iter_total",
                                   iter_timing);
        }
        time_data.add_data(iter_timing);
        function_data.add_data(iter_function_data);
    }
    function_data.add_convergence_targets(FunctionDefaults<3>::get_thresh(),
                                          density_target, x_relative_target);
    Chi.reset_active();
    if (world.rank() == 0) print("\n");
    if (world.rank() == 0) print("   Finished Response Calculation ");
    if (world.rank() == 0) print("   ------------------------");
    if (world.rank() == 0) print("\n");

    // Did we converge?
    if (iter == r_params.maxiter() && not all_done) {
        if (world.rank() == 0) print("   Failed to converge. Reason:");
        if (world.rank() == 0) print("\n  ***  Ran out of iterations  ***\n");
    }
    if (world.rank() == 0) {
        print(" Final energy residuals X:");
        print(x_relative_residuals);
        print(" Final density residuals:");
        print(density_residuals);
    }
    //compute_and_print_polarizability(world, Chi, PQ, "Converged");
}

auto FrequencyResponse::update_response(
        World &world, X_space &chi, XCOperator<double, 3> &xc,
        std::vector<poperatorT> &bsh_x_ops, std::vector<poperatorT> &bsh_y_ops,
        QProjector<double, 3> &projector, double &x_shifts, double &omega_n,
        response_solver &kain_x_space, size_t iteration,
        const double &max_rotation, const vector_real_function_3d &rho_old,
        const Tensor<double> &old_residuals, const X_space &xres_old)
        -> std::tuple<X_space, residuals, vector_real_function_3d> {

    if (r_params.print_level() >= 1) { molresponse::start_timer(world); }

    auto x = chi.copy();
    X_space theta_X =
            compute_theta_X(world, x, rho_old, xc, r_params.calc_type());
    X_space new_chi = bsh_update_response(world, theta_X, bsh_x_ops, bsh_y_ops,
                                          projector, x_shifts);


    inner_to_json(world, "x_new", response_context.inner(new_chi, new_chi),
                  iter_function_data);

    auto [new_res, bsh] = update_residual(
            world, chi, new_chi, r_params.calc_type(), old_residuals, xres_old);
    inner_to_json(world, "r_x", response_context.inner(new_res, new_res),
                  iter_function_data);
    if (iteration >= 0) {// & (iteration % 3 == 0)) {
        new_chi = kain_x_space_update(world, chi, new_res, kain_x_space);
    }
    inner_to_json(world, "x_update", response_context.inner(new_chi, new_chi),
                  iter_function_data);
    // if (false) { x_space_step_restriction(world, chi, new_chi, compute_y, max_rotation); }
    if (r_params.print_level() >= 1) {
        molresponse::end_timer(world, "update response", "update", iter_timing);
    }

    auto new_rho = response_context.compute_density(
            world, new_chi, ground_orbitals, rho_old, true);

    return {new_chi, {new_res, bsh}, new_rho};
}

auto FrequencyResponse::bsh_update_response(World &world, X_space &theta_X,
                                            std::vector<poperatorT> &bsh_x_ops,
                                            std::vector<poperatorT> &bsh_y_ops,
                                            QProjector<double, 3> &projector,
                                            double &x_shifts) -> X_space {
    if (r_params.print_level() >= 1) { molresponse::start_timer(world); }
    size_t m = theta_X.x.size();
    size_t n = theta_X.x.size_orbitals();
    bool compute_y = omega != 0.0;

    if (compute_y) {
        theta_X += theta_X * x_shifts;
        theta_X += PQ;
        theta_X = -2 * theta_X;
        theta_X.truncate();
    } else {
        theta_X.x += theta_X.x * x_shifts;
        theta_X.x += PQ.x;
        theta_X.x = theta_X.x * -2;
        theta_X.x.truncate_rf();
    }
    // apply bsh
    X_space bsh_X(world, m, n);
    bsh_X.active = theta_X.active;
    bsh_X.x = apply(world, bsh_x_ops, theta_X.x);
    if (compute_y) { bsh_X.y = apply(world, bsh_y_ops, theta_X.y); }

    if (compute_y) {
        bsh_X.truncate();
    } else {
        bsh_X.x.truncate_rf();
    }

    auto apply_projector = [&](auto &xi) { return projector(xi); };
    if (compute_y) {
        bsh_X = oop_apply(bsh_X, apply_projector);
    } else {
        for (const auto &i: bsh_X.active) bsh_X.x[i] = projector(bsh_X.x[i]);
    }
    if (r_params.print_level() >= 1) {
        molresponse::end_timer(world, "bsh_update", "bsh_update", iter_timing);
    }
    if (compute_y) {
        bsh_X.truncate();
    } else {
        bsh_X.x.truncate_rf();
    }
    return bsh_X;
}

void FrequencyResponse::frequency_to_json(json &j_mol_in, size_t iter,
                                          const Tensor<double> &polar_ij,
                                          const Tensor<double> &res_polar_ij) {
    json j = {};
    j["iter"] = iter;
    j["polar"] = tensor_to_json(polar_ij);
    j["res_polar"] = tensor_to_json(res_polar_ij);
    auto index = j_mol_in["protocol_data"].size() - 1;
    j_mol_in["protocol_data"][index]["property_data"].push_back(j);
}

void FrequencyResponse::compute_and_print_polarizability(World &world,
                                                         X_space &Chi,
                                                         X_space &pq,
                                                         std::string message) {
    Tensor<double> G = -2 * inner(Chi, pq);
    if (world.rank() == 0) {
        print("Polarizability", message);
        print(G);
    }
}

void FrequencyResponse::save(World &world, const std::string &name) {
    // Archive to write everything to
    archive::ParallelOutputArchive ar(world, name.c_str(), 1);

    ar & r_params.archive();
    ar & r_params.tda();
    ar & r_params.num_orbitals();
    ar & r_params.num_states();

    for (size_t i = 0; i < r_params.num_states(); i++)
        for (size_t j = 0; j < r_params.num_orbitals(); j++) ar & Chi.x[i][j];
    for (size_t i = 0; i < r_params.num_states(); i++)
        for (size_t j = 0; j < r_params.num_orbitals(); j++) ar & Chi.y[i][j];
}

// Load a response calculation
void FrequencyResponse::load(World &world, const std::string &name) {
    if (world.rank() == 0) { print("FrequencyResponse::load() -state"); }
    // The archive to read from
    archive::ParallelInputArchive ar(world, name.c_str());
    ar & r_params.archive();
    ar & r_params.tda();
    ar & r_params.num_orbitals();
    ar & r_params.num_states();
    Chi = X_space(world, r_params.num_states(), r_params.num_orbitals());
    for (size_t i = 0; i < r_params.num_states(); i++)
        for (size_t j = 0; j < r_params.num_orbitals(); j++) ar & Chi.x[i][j];
    world.gop.fence();
    for (size_t i = 0; i < r_params.num_states(); i++)
        for (size_t j = 0; j < r_params.num_orbitals(); j++) ar & Chi.y[i][j];
    world.gop.fence();
}

auto nuclear_generator(World &world, FrequencyResponse &calc) -> X_space {
    auto [gc, molecule, r_params] = calc.get_parameter();
    X_space PQ(world, r_params.num_states(), r_params.num_orbitals());
    auto num_operators = size_t(molecule.natom() * 3);
    auto nuclear_vector = vecfuncT(num_operators);

    for (long atom = 0; atom < molecule.natom(); ++atom) {
        for (long axis = 0; axis < 3; ++axis) {
            functorT func(new madchem::MolecularDerivativeFunctor(molecule,
                                                                  atom, axis));
            nuclear_vector.at(atom * 3 + axis) =
                    functionT(factoryT(world)
                                      .functor(func)
                                      .nofence()
                                      .truncate_on_project()
                                      .truncate_mode(0));
        }
    }
    PQ.x = vector_to_PQ(world, nuclear_vector, calc.get_orbitals());
    PQ.y = PQ.x;
    return PQ;
}

auto dipole_generator(World &world, FrequencyResponse &calc) -> X_space {
    auto [gc, molecule, r_params] = calc.get_parameter();
    X_space PQ(world, r_params.num_states(), r_params.num_orbitals());
    vector_real_function_3d dipole_vectors(3);
    size_t i = 0;
    for (auto &d: dipole_vectors) {
        std::vector<int> f(3, 0);
        f[i++] = 1;
        d = real_factory_3d(world).functor(
                real_functor_3d(new MomentFunctor(f)));
    }
    //truncate(world, dipole_vectors, true);
    world.gop.fence();
    PQ.x = vector_to_PQ(world, dipole_vectors, calc.get_orbitals());
    PQ.y = PQ.x.copy();
    if (world.rank() == 0) { print("Made new PQ"); }
    return PQ;
}

auto vector_to_PQ(World &world, const vector_real_function_3d &rhs_operators,
                  const vector_real_function_3d &ground_orbitals)
        -> response_space {
    response_space rhs(world, rhs_operators.size(), ground_orbitals.size());
    auto orbitals = copy(world, ground_orbitals);
    reconstruct(world, orbitals);
    truncate(world, orbitals);
    QProjector<double, 3> Qhat(orbitals);
    int b = 0;
    for (const functionT &pi: rhs_operators) {
        auto op_phi = mul(world, pi, ground_orbitals, true);
        rhs[b] = Qhat(op_phi);
        b++;
    }
    return rhs;
}
//
