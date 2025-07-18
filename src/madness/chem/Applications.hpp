#pragma once

#include <madness/chem/InputWriter.hpp>
#include <madness/chem/ParameterManager.hpp>
#include <madness/chem/PathManager.hpp>
#include <madness/chem/Results.h>

namespace madness {
  // Scoped CWD: changes the current directory to the given one, and restores when
  // the object goes out of scope
  struct ScopedCWD {
    std::filesystem::path old_cwd;
    explicit ScopedCWD(std::filesystem::path const& new_dir) {
      old_cwd = std::filesystem::current_path();
      std::filesystem::current_path(new_dir);
    }
    ~ScopedCWD() { std::filesystem::current_path(old_cwd); }
  };


  /// common interface for any underlying MADNESS app
  class Application {
  public:
    explicit Application(const Params& p) : params_(p) {}
    virtual ~Application() = default;

    // run: write all outputs under the given directory
    virtual void run(const std::filesystem::path& workdir) = 0;

    // optional hook to return a JSON fragment of this app's main results
    [[nodiscard]] virtual nlohmann::json results() const = 0;

    // get the parameters used for this application
    [[nodiscard]] virtual const QCCalculationParametersBase& get_parameters() const = 0;

    virtual void print_parameters(World& world) const {
      std::string tag= get_parameters().get_tag();
      if (world.rank() == 0) get_parameters().print(tag,"end");
    }

    // get the working directory for this application
    [[nodiscard]] path get_workdir() const {
      return workdir_;
    }

    /// check if this calculation has a json with results
    [[nodiscard]] virtual bool has_results(std::string filename) const {
      // check if the results file exists
      // return std::filesystem::exists(workdir_ / filename);
      return std::filesystem::exists( filename);
    }

    [[nodiscard]] virtual bool verify_results(const nlohmann::json& j) const {
      // check if some key parameters of the calculation match:
      // molecule, box size, nmo_alpha, nmo_beta
      Molecule mol1 = params_.get<Molecule>();
      Molecule mol2;
      mol2.from_json(j["molecule"]);
      if (not (mol1 == mol2)) {
        print("molecule mismatch");
        mol1.print();
        mol2.print();
        return false;
      }
      return true;
    }

    /// read the results from a json file
    [[nodiscard]] virtual nlohmann::json read_results(std::string filename) const {
      if (has_results(filename)) {
        std::cout << "Found checkpoint file: " << filename << std::endl;
        // std::ifstream ifs(workdir_ / filename);
        std::ifstream ifs( filename);
        nlohmann::json j;
        ifs >> j;
        ifs.close();
        if (not verify_results(j)) {
          std::string msg= "Results file " + filename + " does not match the parameters of the calculation";
          print(msg);
          return nlohmann::json(); // return empty json
        }
        return j;
      } else {
        std::string msg= "Results file " + filename + " does not exist in " + workdir_.string();
        MADNESS_EXCEPTION(msg.c_str(), 1);
      }
      return nlohmann::json();
    }

    /// check if the wavefunctions are already computed
    [[nodiscard]] virtual bool has_wavefunctions(std::string filename) const {
      return std::filesystem::exists(workdir_ / filename);
    }

    /// read the wavefunctions from a file
    [[nodiscard]] virtual std::vector<double> read_wavefunctions(std::string filename) const {
      if (has_wavefunctions(filename)) {
        std::ifstream ifs(workdir_ / filename);
        std::vector<double> wfs;
        double value;
        while (ifs >> value) {
          wfs.push_back(value);
        }
        ifs.close();
        return wfs;
      } else {
        std::string msg= "Wavefunction file " + filename + " does not exist in " + workdir_.string();
        MADNESS_EXCEPTION(msg.c_str(), 1);
      }
      return {};
    }

    /// check if this calculation needs to be redone
    [[nodiscard]] virtual bool needs_redo() const {
      // read json and check if the results are already there
      if (std::filesystem::exists(workdir_ / "results.json")) {
        std::ifstream ifs(workdir_ / "results.json");
        nlohmann::json j;
        ifs >> j;
        ifs.close();
        // check if the results are already there
        return !j.contains("energy") || !j["energy"].is_number();
      }
      // by default, we assume that the calculation needs to be redone
      return true;
    }

  protected:
    const Params params_;
    path workdir_;
    nlohmann::json results_;
  };


  template <typename Library, typename ScfT = SCF>
  class SCFApplication : public Application, public ScfT, public std::enable_shared_from_this<SCFApplication<Library,ScfT>> {
  public:

    /// SCF ctor
    template <typename T = ScfT, std::enable_if_t<std::is_same_v<T, Nemo>, int> = 0>
    explicit SCFApplication(World& w, const Params& p)
        : Application(p), ScfT(w,p.get<CalculationParameters>(), p.get<Nemo::NemoCalculationParameters>(), p.get<Molecule>()), world_(w) {}

    /// Nemo ctor
    template <typename T = ScfT, std::enable_if_t<std::is_same_v<T, SCF>, int> = 0>
    explicit SCFApplication(World& w, const Params& p)
        : Application(p), ScfT(w,p.get<CalculationParameters>(), p.get<Molecule>()), world_(w) {}

    std::shared_ptr<const SCF> get_scf() const {
      return std::dynamic_pointer_cast<const SCF>(this->shared_from_this());
    }

    bool constexpr is_nemo() const {
      return std::is_same_v<ScfT, Nemo>;
    }

    std::shared_ptr<const Nemo> get_nemo() const {
      auto return_ptr= std::dynamic_pointer_cast<const Nemo>(this->shared_from_this());
      if (!return_ptr) {
        MADNESS_EXCEPTION("Could not cast SCFApplication to Nemo", 1);
      }
      return return_ptr;
    }

    std::shared_ptr<Nemo> get_nemo() {
      auto return_ptr= std::dynamic_pointer_cast<Nemo>(this->shared_from_this());
      if (!return_ptr) {
        MADNESS_EXCEPTION("Could not cast SCFApplication to Nemo", 1);
      }
      return return_ptr;
    }

    void print_parameters(World& world) const override {
      std::string tag= get_parameters().get_tag();
      if (world.rank() == 0) {
        if constexpr (std::is_same_v<ScfT, SCF>) { // SCF
          get_parameters().print("dft","end");
        } else {  // Nemo
          get_parameters().print("dft");
          get_nemo()->get_nemo_param().print();
          print("end");
        }
      }
    }

    const QCCalculationParametersBase& get_parameters() const override {
      if (is_nemo()) {
        return get_nemo()->get_calc_param();
      } else {
        return get_scf()->param;
      }
    }

    void run(const std::filesystem::path& workdir) override {
      // 1) set up a namedspaced directory for this run
      std::string label= is_nemo() ? "nemo" : Library::label();
      PathManager pm(workdir, label.c_str());
      pm.create();
      workdir_=pm.dir();
      world_.gop.fence();
      {
        ScopedCWD scwd(pm.dir());
        if (world_.rank() == 0) {
          std::cout << "Running SCF in " << pm.dir() << std::endl;
        }

        auto scfParams = params_.get<CalculationParameters>();
        bool needEnergy = true;
        bool needDipole = scfParams.dipole();
        bool needGradient = scfParams.derivatives();
        bool needWavefunctions = true;

        // 2) define the "checkpoint" file
        auto ckpt = params_.get<CalculationParameters>().prefix()+".calc_info.json";
        nlohmann::json j;
        if (has_results(ckpt)) j=read_results(ckpt);

        bool ok = true;
        if (needEnergy && !j.contains("energy")) ok = false;
        if (needDipole && !j.contains("dipole")) ok = false;
        if (needGradient && !j.contains("gradient")) ok = false;
        if (needWavefunctions) {
          try {
            double thresh=scfParams.protocol().back();
            if constexpr (std::is_same_v<ScfT, Nemo>) this->set_protocol(scfParams.econv());
            if constexpr (std::is_same_v<ScfT, SCF>) SCF::set_protocol<3>(world_,thresh);
            this->load_mos(world_);
          } catch (...) {
            // if we cannot load MOs, we need to recompute them
            ok = false;
          }
        }

        if (ok) {
          energy_ = j["energy"];
          if (needDipole) dipole_ = tensor_from_json<double>(j["dipole"]);
          if (needGradient) gradient_ = tensor_from_json<double>(j["gradient"]);
          return;
        }

        // we could dump params_ to JSON and pass as argv if desired…
        MetaDataResults metadata(world_);
        if constexpr (std::is_same_v<ScfT, SCF>) {
          results_ = Library::run_scf(world_, params_, pm.dir());
        } else if constexpr (std::is_same_v<ScfT, Nemo>) {
          results_ = Library::run_nemo(this->get_nemo());
        } else {
          MADNESS_CHECK_THROW("unknown SCF type", 1);
        }
        results_["metadata"] = metadata.to_json();

        // legacy
        PropertyResults properties= results_["properties"];
        energy_ = properties.energy;
        dipole_ = properties.dipole;
        gradient_ = properties.gradient;

      }
    }

    nlohmann::json results() const override {
      return results_;
    }

  private:
    World& world_;
    double energy_;

    std::optional<Tensor<double>> dipole_;
    std::optional<Tensor<double>> gradient_;
    std::optional<real_function_3d> density_;
  };

  /**
   * @brief Wrapper application to run the molresponse workflow
   *        via the molresponse_lib::run_response function.
   */
  template <typename Library>
  class ResponseApplication : public Application {
  public:
    /**
     * @param world   MADNESS world communicator
     * @param params  Unified Params containing ResponseParameters & Molecule
     * @param indir   Directory of precomputed ground-state (SCF) outputs
     */
    ResponseApplication(World& world, Params params, std::filesystem::path indir)
        : world_(world),
          Application(std::move(params)),
          indir_(std::move(indir)) {}


    const QCCalculationParametersBase& get_parameters() const override {
      return params_.get<ResponseParameters>();
    }
    /**
     * @brief Execute response + property workflow, writing into workdir/response
     */
    void run(const std::filesystem::path& workdir) override {
      // create a namespaced subdirectory for response outputs
      PathManager pm(workdir, Library::label());
      pm.create();
      {
        ScopedCWD scwd(pm.dir());

        auto res = Library::run_response(world_, params_, indir_, pm.dir());

        metadata_ = std::move(res.metadata);
        properties_ = std::move(res.properties);
      }
    }

    /**
     * @brief Return a JSON fragment summarizing results
     */
    [[nodiscard]] nlohmann::json results() const override {
      return {{"type", "response"},
              {"metadata", metadata_},
              {"properties", properties_}};
    }

  private:
    World& world_;
    std::filesystem::path indir_;
    nlohmann::json metadata_;
    nlohmann::json properties_;
  };

  template <typename Library, typename SCFApplicationT>
  class CC2Application : public Application, public CC2 {
  public:
    explicit CC2Application(World& w, const Params& p, const SCFApplicationT& reference)
        : Application(p), world_(w), reference_(reference),
          CC2(w, p.get<CCParameters>(), p.get<TDHFParameters>(), reference.get_nemo()) {
    }

    const QCCalculationParametersBase& get_parameters() const override {
      return parameters; // CCParameters
    }

    void run(const std::filesystem::path& workdir) override {
      // 1) set up a namedspaced directory for this run
      PathManager pm(workdir, Library::label());
      pm.create();
      world_.gop.fence();
      {
        ScopedCWD scwd(pm.dir());
        if (world_.rank() == 0) {
          std::cout << "Running CC2 in " << pm.dir() << std::endl;
        }

        // 2) define the "checkpoint" file
        std::string label=Library::label();
        auto ckpt = label + "_results.json";
        print("cc checkpoint file",ckpt);
        if (std::filesystem::exists(ckpt)) {
          if (world_.rank() == 0) {
            std::cout << "Found checkpoint file: " << ckpt << std::endl;
          }
          // read the checkpoint file
          std::ifstream ifs(ckpt);
          ifs >> results_;
          ifs.close();

          bool ok = true;
          bool needEnergy = true;
          if (needEnergy && !results_.contains("energy")) ok = false;

        }

        auto rel = std::filesystem::relative(reference_.get_workdir(), pm.dir());
        if (world.rank() == 0) {
          std::cout << "Running cc2 calculation in: " << pm.dir() << std::endl;
          std::cout << "Ground state archive: " << reference_.get_workdir() << std::endl;
          std::cout << "Relative path: " << rel << std::endl;
        }

        results_=this->solve();

      }
    }

    nlohmann::json results() const override {
      return results_;
    }

  private:
    World& world_;
    const Application& reference_;
  };



  template<typename SCFApplicationT>
  class TDHFApplication : public Application, public TDHF {
  public:
    explicit TDHFApplication(World& w, const Params& p, const SCFApplicationT& reference)
        : Application(p), world_(w), reference_(reference),
          TDHF(w, p.get<TDHFParameters>(), reference.get_nemo()) {
    }


    const QCCalculationParametersBase& get_parameters() const override {
      return TDHF::get_parameters(); // TDHFParameters
    }

    void run(const std::filesystem::path& workdir) override {
      // 1) set up a namedspaced directory for this run
      PathManager pm(workdir, "tdhf");
      pm.create();
      world_.gop.fence();
      {
        ScopedCWD scwd(pm.dir());
        if (world_.rank() == 0) {
          std::cout << "Running CIS in " << pm.dir() << std::endl;
        }

        // we could dump params_ to JSON and pass as argv if desired…
        try {
          const double time_scf_start = wall_time();
          this->prepare_calculation();
          const double time_scf_end = wall_time();
          if (world_.rank() == 0) printf(" at time %.1f\n", wall_time());

          const double time_cis_start = wall_time();
          std::vector<CC_vecfunction> roots = this->solve_cis();
          const double time_cis_end = wall_time();
          if (world_.rank() == 0) printf(" at time %.1f\n", wall_time());

          if (world_.rank() == 0) {
            std::cout << std::setfill(' ');
            std::cout << "\n\n\n";
            std::cout << "--------------------------------------------------\n";
            std::cout << "MRA-CIS ended \n";
            std::cout << "--------------------------------------------------\n";
            std::cout << std::setw(25) << "time scf" << " = " << time_scf_end - time_scf_start << "\n";
            std::cout << std::setw(25) << "time cis" << " = " << time_cis_end - time_cis_start << "\n";
            std::cout << "--------------------------------------------------\n";
          }
          auto j=this->analyze(roots);
          // funnel through CISResults to make sure we have the right format
          CISResults results(j);
          results_= results.to_json();

        } catch (std::exception& e) {
          print("Caught exception: ", e.what());
        }
      }
    }

    nlohmann::json results() const override {
      return results_;
    }

  private:
    World& world_;
    const Application& reference_;

  };



  template<typename SCFApplicationT>
  class OEPApplication : public Application, public OEP {
  public:
    explicit OEPApplication(World& w, const Params& p, const SCFApplicationT& reference)
        : Application(p), world_(w), reference_(reference),
          OEP(w, p.get<OEP_Parameters>(), reference.get_nemo()) {
    }

    const QCCalculationParametersBase& get_parameters() const override {
      return oep_param; // OEP_Parameters
    }

    void run(const std::filesystem::path& workdir) override {
      // 1) set up a namedspaced directory for this run
      PathManager pm(workdir, "oep");
      pm.create();
      world_.gop.fence();
      {
        ScopedCWD scwd(pm.dir());
        if (world_.rank() == 0) {
          std::cout << "Running OEP in " << pm.dir() << std::endl;
        }

        // 2) define the "checkpoint" file
        std::string label="oep";
        auto ckpt = label + "_results.json";
        print("cc checkpoint file",ckpt);
        if (std::filesystem::exists(ckpt)) {
          if (world_.rank() == 0) {
            std::cout << "Found checkpoint file: " << ckpt << std::endl;
          }
          // read the checkpoint file
          std::ifstream ifs(ckpt);
          nlohmann::json j;
          ifs >> j;
          ifs.close();

        }

        // we could dump params_ to JSON and pass as argv if desired…
        try {
          const double time_scf_start = wall_time();
          this->value();
          const double time_scf_end = wall_time();
          if (world_.rank() == 0) printf(" at time %.1f\n", wall_time());

          if (world_.rank() == 0) {
            std::cout << std::setfill(' ');
            std::cout << "\n\n\n";
            std::cout << "--------------------------------------------------\n";
            std::cout << "MRA-OEP ended \n";
            std::cout << "--------------------------------------------------\n";
            std::cout << std::setw(25) << "time scf" << " = " << time_scf_end - time_scf_start << "\n";
            std::cout << "--------------------------------------------------\n";
          }
        } catch (std::exception& e) {
          print("Caught exception: ", e.what());
        }
        // nlohmann::json results;
        results_=this->analyze();
      }
    }

    nlohmann::json results() const override {
      return results_;
    }

  private:
    World& world_;
    const Application& reference_;

    double energy_;
    std::optional<Tensor<double>> dipole_;
    std::optional<Tensor<double>> gradient_;
    std::optional<real_function_3d> density_;
  };

}


