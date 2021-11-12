#include <fstream>
#include <typeinfo>
#include <complex>
#include <stdlib.h>
#include <sys/time.h>
#include <fftw3.h>

#include "aweights.hpp"
#include "tipsy.h"


using namespace std;

typedef double real_type;
typedef std::complex<real_type> complex_type;

typedef blitz::Array<real_type,2> array2D_r;
typedef blitz::Array<real_type,3> array3D_r;

typedef blitz::Array<complex_type,3> array3D_c;


// A simple timing function
double getTime() {
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return tv.tv_sec + 1e-6*(double)tv.tv_usec;
}

// Read the particle file,
// return a 2d blitz array containing particle positions
array2D_r read_particles(string fname){
    double t0, elapsed;
    TipsyIO io;

    io.open(fname);
    cout << "Found "<<io.count() << " particles."<<endl;

    if (io.fail()) {
        cerr << "Unable to open file" << endl;
        abort();
    }

    // Allocate the particle buffer
    array2D_r p(io.count(),3);
    
    t0 = getTime();
    // Read the particles
    io.load(p);
    elapsed = getTime() - t0;

    cout << "particle reading: " << elapsed << " s" << endl;
    return p;
}

// Write a blitz array in a csv file format
template<typename T>
void write_array(T A, const char* filename){
    cout << "Writing to " << filename << endl;
    ofstream ofs(filename);
    if (ofs.bad()){
        cerr << "Unable to write to file: " << filename << endl;
        abort();;
    }

    ofs << A << endl;

    ofs.close(); 
    return;
}

// Projection of a 3D grid into a 2D grid (max pooling)
array2D_r project(array3D_r grid){
    auto shape = grid.shape();
    array2D_r ret(shape[0], shape[1]);
    blitz::thirdIndex k;
    ret = blitz::max(grid, k);
    return ret;
}


// Mass assignment for a single particle with order given by o
template<int o>
void _assign_mass(real_type x, real_type y, real_type z, array3D_r grid){
    auto shape = grid.shape();
    int i, j, k;

    AssignmentWeights<o> wx((x + 0.5)*shape[0]);
    AssignmentWeights<o> wy((y + 0.5)*shape[1]);
    AssignmentWeights<o> wz((z + 0.5)*shape[2]);
    for(int ii=0; ii<o; ii++){
        for(int jj=0; jj<o; jj++){
            for(int kk=0; kk<o; kk++){
                i = (wx.i+ii+shape[0])%shape[0];
                j = (wy.i+jj+shape[1])%shape[1];
                k = (wz.i+kk+shape[2])%shape[2];
		#pragma omp atomic
                grid(i,j,k) += wx.H[ii]*wy.H[jj]*wz.H[kk];
            }
        }
    }
    return;
}

// Wrapper for templated mass assignment
void assign_mass(int o, real_type x, real_type y, real_type z, array3D_r grid){
    switch(o){
        case 1: _assign_mass<1>(x,y,z,grid); break;
        case 2: _assign_mass<2>(x,y,z,grid); break;
        case 3: _assign_mass<3>(x,y,z,grid); break;
        case 4: _assign_mass<4>(x,y,z,grid); break;
        default: 
            cerr << "Incorrect mass assignment order: " << o << endl;
            abort();
    }
}

// Mass assignment for a list of particles
void assign_masses(int o, array2D_r p, array3D_r grid){
    double t0, elapsed;
    auto N_part = p.shape()[0];
    auto shape = grid.shape();

    // Compute the average density per grid cell
    real_type avg = 1.0*N_part / (shape[0]*shape[1]*shape[2]);

    t0 = getTime();
    #pragma omg parallel for
    for(int i=0; i<N_part; ++i)
        assign_mass(o, p(i,0), p(i,1), p(i,2), grid);

    // Turn the density into the over-density
    grid = (grid - avg) / avg;
    elapsed = getTime()-t0;
    cout << "mass assignment: " << elapsed << " s" << endl;
}

// Call mass assignment at every order to test implementation
void test_assignment(array2D_r p, int N){
    array3D_r grid(N,N,N);
    cout << "Testing mass assignment:" <<endl;
    
    for(int o=1; o<5; o++){
        cout << "Computing order " << o << endl;
        assign_masses(o, p, grid);
        array2D_r proj = project(grid);
        string fname = "order"+to_string(o)+".data";
        write_array(proj, fname.c_str());
        // Set the grid value to zero and compute next order
        grid *= 0;
    }
    cout << "Done" <<endl<<endl;
}

void compute_fft(array3D_r grid, array3D_c fft_grid, int N){
    double t0, elapsed;

    // Create FFTW plan
    t0 = getTime();
    auto plan = fftw_plan_dft_r2c_3d(N,N,N,
            grid.dataFirst(),
            reinterpret_cast<fftw_complex*>(fft_grid.dataFirst()),
            FFTW_ESTIMATE);
    elapsed = getTime()-t0;
    cout << "fftw_plan creation: " << elapsed << " s" << endl;


    // Execute FFTW plan
    t0 = getTime();
    fftw_execute(plan);
    elapsed = getTime()-t0;
    cout << "fftw_plan execution: " << elapsed << " s" << endl;

    // Destroy FFTW plan
    fftw_destroy_plan(plan);
}

void compute_pk(array3D_c fft_grid, int N){
    double t0, elapsed;
    int iNyquist = N / 2;
    int nBins = iNyquist;

    double k_max = sqrt(3) * (iNyquist+1);
    double scale = nBins * 1.0 / log(k_max);

    // LIN SPACED BINS:
    // bin_idx = floor( k_norm / k_max * nBins )

    // LOG SPACED BINS:
    // bin_idx = floor ( log(k_norm) * scale )
    //         = floor ( nBins * log(k_norm) / log(k_max) )

    blitz::Array<double,1>  log_k(nBins);
    blitz::Array<double,1>  power(nBins);
    blitz::Array<int64_t,1> n_power(nBins);


    // Fill arrays with zeros
    log_k = 0;
    power = 0;
    n_power = 0;

    // Mode ordering by fftw:
    // 0, 1, 2, ..., N/2, -(N/2-1), ..., -2, -1
    // 0, 1, 2, ..., N/2, -(N/2-1), ..., -2, -1
    // 0, 1, 2, ..., N/2

    t0 = getTime();
    for( auto index=fft_grid.begin(); index!=fft_grid.end(); ++index ) {
        auto pos = index.position();
        int kx = pos[0]>iNyquist ? N - pos[0] : pos[0];
        int ky = pos[1]>iNyquist ? N - pos[1] : pos[1];
        int kz = pos[2];

        int mult = (kz == 0) || (kz == iNyquist) ? 1 : 2;

        complex_type cplx_amplitude = *index;

        double k_norm = sqrt(kx*kx + ky*ky + kz*kz);

        int bin_idx = k_norm>0 ? floor(log(k_norm) * scale) : 0;

        assert(bin_idx>=0 && bin_idx<nBins);

        log_k(bin_idx)   += mult*log(k_norm);
        power(bin_idx)   += mult*norm(cplx_amplitude);
        n_power(bin_idx) += mult;
    }
    elapsed = getTime() - t0;

    cout << "P(k) measurement: " << elapsed << " s" << endl;

    for(int i=0; i<nBins; ++i) {
        if (n_power(i)>0) {
            cout << exp(log_k(i)/n_power(i)) << " " << power(i)/n_power(i) << endl;
        }
    }
}

int main() {
    int N = 512;
    string fname = "/store/uzh/uzh8/ESC412/ic_512.std"; 
    array2D_r p = read_particles(fname);
    
    test_assignment(p, N); // Test the assignment schemes

    array3D_r grid(N,N,N);
    assign_masses(4, p, grid);

    // Allocate the output buffer for the fft
    array3D_c fft_grid(N,N,N/2+1);
    
    // Compute the fft of the over-density field
    // The results are stored in fft_grid (out-of-place method)
    compute_fft(grid, fft_grid, N);
    
    // Compute the power spectrum
    compute_pk(fft_grid, N);
    
}
