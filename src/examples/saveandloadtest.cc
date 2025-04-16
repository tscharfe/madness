#include <madness/mra/mra.h>
#include <madness/mra/funcplot.h>
#include <madness/mra/nonlinsol.h>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

using namespace madness;




/// Reads the contents of a binary file and returns them as a string
std::string* read_binary_file(const std::string& filename) {
    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Read the file contents into a string
    std::ostringstream buffer;
    buffer << file.rdbuf();

    // Close the file
    file.close();

    // Return the contents as a string
    return &buffer.str();
}






class BinaryStreamOutputArchive : public archive::BinaryFstreamOutputArchive {
    static const std::size_t IOBUFSIZE = 4 * 1024 * 1024; ///< Buffer size.
    std::shared_ptr<char> iobuf; ///< Buffer.
    mutable std::ostream os; ///< The filestream.
    BinaryStreamOutputArchive(const char *filename, std::ios_base::openmode mode)
        : iobuf(), os(NULL) {
        if (filename) open(filename, mode);
    }

    void open(const char *filename, std::ios_base::openmode mode) {
        iobuf.reset(new char[IOBUFSIZE], std::default_delete<char[]>());
    }
};

class NullTeeStreamBuf : public std::streambuf {
public:
    NullTeeStreamBuf(std::ostream& ostream) : m_ostream(ostream) {}

protected:
    virtual int_type overflow(int_type ch) override {
        if (ch != EOF) {
            m_ostream.put(ch);
            if (m_ostream.fail()) {
                return EOF;
            }
        }
        return ch;
    }

    virtual std::streamsize xsputn(const char* s, std::streamsize n) override {
        m_ostream.write(s, n);
        if (m_ostream.fail()) {
            return 0;
        }
        return n;
    }

private:
    std::ostream& m_ostream;
};

template<typename Archive>
class BaseParallelArchive2 : public archive::BaseParallelArchive<Archive> {
    World *world; ///< The world.
    mutable Archive ar; ///< The local archive.
    int nio; ///< Number of I/O nodes (always includes node zero).
    bool do_fence = true; ///< If true (default), a read/write of parallel objects fences before and after I/O.
    char fname[256]; ///< Name of the archive.
    int nclient; ///< Number of clients of this node, including self. Zero if not I/O node.
public:
    template<typename X=Archive>
    BaseParallelArchive2(
        typename std::enable_if_t<std::is_same<X, archive::BinaryFstreamInputArchive>::value || std::is_same<X,
                                      BinaryStreamOutputArchive>::value, int> nio = 0)
        : world(nullptr), ar(), nio(nio), do_fence(true) {
    }
    void open(World& world, const char* filename, int nwriter=1) {}
};
/*
template <class localarchiveT=archive::BinaryFstreamOutputArchive>
        class ParallelOutputArchive2 : public BaseParallelArchive2<localarchiveT>, public BaseOutputArchive {
public:
    using basear = BaseParallelArchive<localarchiveT>;

    /// Default constructor.
    //ParallelOutputArchive() {}

    ParallelOutputArchive(World& world, localarchiveT& ar, int nio=1) : basear(world, ar, nio) {}

    /// Creates a parallel archive for output with given base filename and number of I/O nodes.

    /// \param[in] world The world.
    /// \param[in] filename Base name of the file.
    /// \param[in] nio The number of I/O nodes.
    ParallelOutputArchive(World& world, const char* filename, int nio=1)  {
        basear::open(world, filename, nio);
    }

    /// Creates a parallel archive for output with given base filename and number of I/O nodes.

    /// \param[in] world The world.
    /// \param[in] filename Base name of the file.
    /// \param[in] nio The number of I/O nodes.
    ParallelOutputArchive(World& world, const std::string filename, int nio=1)  {
        basear::open(world, filename.c_str(), nio);
    }

    /// Flush any data in the archive.
    void flush() {
        if (basear::is_io_node()) basear::local_archive().flush();
    }
};


template<typename T, size_t NDIM>
void load_function(World &world, std::vector<Function<T, NDIM> > &f,
                   const std::string name) {
    if (world.rank() == 0) print("loading vector of functions", name);
    archive::ParallelInputArchive<archive::BinaryFstreamInputArchive> ar(world, name.c_str(), 1);
    std::size_t fsize = 0;
    ar & fsize;
    f.resize(fsize);
    for (std::size_t i = 0; i < fsize; ++i) ar & f[i];
}

/// save a vector of functions
template<typename T, size_t NDIM>
void save_function(const std::vector<Function<T, NDIM> > &f, const std::string name) {
    if (f.size() > 0) {
        World &world = f.front().world();
        if (world.rank() == 0) print("saving vector of functions", name);
        archive::ParallelOutputArchive<archive::BinaryFstreamOutputArchive> ar(world, name.c_str(), 1);
        std::size_t fsize = f.size();
        ar & fsize;
        for (std::size_t i = 0; i < fsize; ++i) ar & f[i];
    }
}
*/
template<class T, std::size_t NDIM>
void mod_save(const Function<T, NDIM> &f, const std::string name) {
    archive::ParallelOutputArchive<archive::BinaryFstreamOutputArchive> ar2(f.world(), name.c_str(), 1);
    ar2 & f;
}

template<class T, std::size_t NDIM>
void mod_load(Function<T, NDIM> &f, const std::string name) {
    archive::ParallelInputArchive<archive::BinaryFstreamInputArchive> ar2(f.world(), name.c_str(), 1);
    ar2 & f;
}


const double L = 7.0;

// The initial guess wave function
double guess(const coord_1d &r) {
    return exp(-(r[0] * r[0]) / 1.0);
}

const std::string path_to_plots = "/Users/timo/workspace/up_to_date_madness/madness/src/examples/testfiles/";
// Convenience routine for plotting
void plot(const char *filename, const real_function_1d &f) {
    coord_1d lo(0.0), hi(0.0);
    lo[0] = -L;
    hi[0] = L;
    std::string full_path = path_to_plots + filename;
    plot_line(full_path.c_str(), 401, lo, hi, f);
}


int main(int argc, char **argv) {
    initialize(argc, argv);
    World world(SafeMPI::COMM_WORLD);
    startup(world, argc, argv);
    if (world.rank() == 0) printf("starting at time %.1f\n", wall_time());

    const double thresh = 1e-5;
    FunctionDefaults<1>::set_k(6);
    FunctionDefaults<1>::set_thresh(thresh);
    FunctionDefaults<1>::set_cubic_cell(-L, L);
    NonlinearSolverND<1> solver;

    real_function_1d phi = real_factory_1d(world).f(guess);
    double norm = phi.norm2();
    phi.scale(1.0 / norm);
    plot("phi.dat", phi);

    auto f = std::is_same<BinaryStreamOutputArchive, archive::BinaryFstreamOutputArchive>::value;
    std::cout << f << std::endl;
    save<double,1>(phi,"/Users/timo/workspace/up_to_date_madness/madness/src/examples/testfiles/test");
    real_function_1d phi2 = real_factory_1d(world);
    /*mod_load<double,1>(phi2,"/Users/timo/workspace/up_to_date_madness/madness/src/examples/testfiles/test");
    plot("phi2.dat",phi2);

    double ip=phi.inner(phi2);
    std::cout << ip << std::endl;
    */
    std::string* load_str = read_binary_file("/Users/timo/workspace/up_to_date_madness/madness/src/examples/testfiles/test.00000");
    std::cout << load_str << std::endl;
    if (world.rank() == 0) printf("finished at time %.1f\n", wall_time());
    finalize();
    return 0;
}
