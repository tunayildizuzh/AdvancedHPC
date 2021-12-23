#include <fstream>
#include <typeinfo>
#include <complex>
#include <stdlib.h>
#include <sys/time.h>
#include <cufftw.h>
#include "cufft.h"

#include "aweights.hpp"
#include "tipsy.h"

using namespace std;
using namespace blitz;

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
    cerr << "Found "<<io.count() << " particles."<<endl;

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

    cerr << "particle reading: " << elapsed << " s" << endl;
    return p;
}

// Write a blitz array in a csv file format
template<typename T>
void write_array(T A, const char* filename){
    cerr << "Writing to " << filename << endl;
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
array2D_r project(array3D_r &grid){
    auto shape = grid.shape();
    array2D_r ret(shape[0], shape[1]);
    blitz::thirdIndex k;
    ret = blitz::max(grid, k);
    return ret;
}


// Mass assignment for a single particle with order given by o
template<int o>
void _assign_mass(real_type x, real_type y, real_type z, array3D_r &grid){
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
                grid(i,j,k) += wx.H[ii]*wy.H[jj]*wz.H[kk];
            }
        }
    }
    return;
}

// Wrapper for templated mass assignment
void assign_mass(int o, real_type x, real_type y, real_type z, array3D_r &grid){
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
void assign_masses(int o, array2D_r &p, array3D_r &grid){
    double t0, elapsed;
    auto N_part = p.shape()[0];
    auto shape = grid.shape();

    // Compute the average density per grid cell
    real_type avg = 1.0*N_part / (shape[0]*shape[1]*shape[2]);

    t0 = getTime();
    for(int i=0; i<N_part; ++i)
        assign_mass(o, p(i,0), p(i,1), p(i,2), grid);

    // Turn the density into the over-density
    grid = (grid - avg) / avg;
    elapsed = getTime()-t0;
    cerr << "mass assignment: " << elapsed << " s" << endl;
}

// Call mass assignment at every order to test implementation
void test_assignment(array2D_r &p, int N){
    array3D_r grid(N,N,N);
    cerr << "Testing mass assignment:" <<endl;
    for(int o=1; o<5; o++){
        cerr << "Computing order " << o << endl;
        assign_masses(o, p, grid);
        array2D_r proj = project(grid);
        string fname = "order"+to_string(o)+".data";
        write_array(proj, fname.c_str());
        // Set the grid value to zero and compute next order
        grid *= 0;
    }
    cerr << "Done" <<endl<<endl;
}

void execute_plan(fftw_plan plan) {
    auto t0 = getTime();
    fftw_execute(plan);
    auto elapsed = getTime()-t0;
    cerr << "fftw_plan execution: " << elapsed << " s" << endl;
    }

// The simple way to calculate the 3D FFT - let FFTW worry about it
void compute_fft_cuda(array3D_r &grid, array3D_c &fft_grid, int N){
    double t0, elapsed;

    // Create FFTW plan
    t0 = getTime();
    auto plan = fftw_plan_dft_r2c_3d(N,N,N,
            grid.dataFirst(),
            reinterpret_cast<fftw_complex*>(fft_grid.dataFirst()),
            FFTW_ESTIMATE);
    elapsed = getTime()-t0;
    cerr << "fftw_plan creation: " << elapsed << " s" << endl;

    // Execute FFTW plan
    execute_plan(plan);

    // Destroy FFTW plan
    fftw_destroy_plan(plan);
}




//**********************************************************************

void compute_fft_2D_R2C(array3D_r &grid, array3D_c &fft_grid, int N) {
    int n[] = {N,N};       // 2D FFT of length NxN
    int inembed[] = {N,2*(N/2+1)};
    int onembed[] = {N,(N/2+1)};
    int howmany = N;
    int odist = N*(N/2+1); // Output distance is in "complex"
    int idist = 2*odist;   // Input distance is in "real"
    int istride = 1,       // Elements of each FFT are adjacent
	ostride = 1;
    double t0 = getTime();
#if 1
    cufftHandle plan;
    cufftPlanMany(&plan,sizeof(n)/sizeof(n[0]), n,
		    inembed,istride,idist,
		    onembed,ostride,odist,
		    CUFFT_D2Z,howmany);
    cufftDoubleComplex *data;
    cudaMalloc((void**)&data, sizeof(cufftDoubleComplex)*N*N*(N/2+1));
    cudaMemcpy(data, grid.dataFirst(), sizeof(cufftDoubleComplex)*N*N*(N/2+1), cudaMemcpyHostToDevice);
    cufftExecD2Z(plan,reinterpret_cast<cufftDoubleReal*>(data),data);
    cudaMemcpy(grid.dataFirst(), data,sizeof(cufftDoubleComplex)*N*N*(N/2+1), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(data);
    cufftDestroy(plan);
#else
    auto plan = fftw_plan_many_dft_r2c(
    	sizeof(n)/sizeof(n[0]), n, howmany,
    	grid.dataFirst(),
    	inembed,istride, idist,
	reinterpret_cast<fftw_complex*>(fft_grid.dataFirst()),
	onembed,ostride,odist,
	FFTW_ESTIMATE);
    execute_plan(plan);
    fftw_destroy_plan(plan);
#endif
    double elapsed = getTime()-t0;
    cerr << "2D R2C FFT CUDA: " << elapsed << " s" << endl;
}

void compute_fft_1D_C2C(array3D_c &fft_grid, int N){
    // 2D FFT of the 1st dimensions: C2C
    int n[] = {N};
    int *inembed = n, *onembed = n;
    int howmany = N*(N/2+1);
    int idist = 1;
    int odist = 1;
    int istride = N*(N/2+1), ostride = N*(N/2+1);
#if 1
    cufftHandle plan;
    cufftPlanMany(&plan,sizeof(n)/sizeof(n[0]), n,
                    inembed,istride,idist,
                    onembed,ostride,odist,
                    CUFFT_Z2Z,howmany);
    cufftDoubleComplex *data;
    cudaMalloc((void**)&data, sizeof(cufftDoubleComplex)*N*N*(N/2+1));
    cudaMemcpy(data, fft_grid.dataFirst(), sizeof(cufftDoubleComplex)*N*N*(N/2+1), cudaMemcpyHostToDevice);
    cufftExecZ2Z(plan,data,data,CUFFT_FORWARD);
    cudaMemcpy(fft_grid.dataFirst(), data,sizeof(cufftDoubleComplex)*N*N*(N/2+1), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(data);
    cufftDestroy(plan);
#else
    auto plan = fftw_plan_many_dft(sizeof(n)/sizeof(n[0]), n, howmany,
    	     	    reinterpret_cast<fftw_complex*>(fft_grid.dataFirst()),
    	     	    inembed, istride, idist,
		    reinterpret_cast<fftw_complex*>(fft_grid.dataFirst()),
		    onembed,ostride,odist, FFTW_FORWARD, FFTW_ESTIMATE);
    execute_plan(plan);
    fftw_destroy_plan(plan);
#endif
}

//**********************************************************************

void compute_fft_2D_R2C_stream(array3D_r &grid, array3D_c &fft_grid, int N) {
    int n[] = {N,N};       // 2D FFT of length NxN
    int inembed[] = {N,2*(N/2+1)};
    int onembed[] = {N,(N/2+1)};
    const int howmany = 16;// Number of slabs to do at once
    int odist = N*(N/2+1); // Output distance is in "complex"
    int idist = 2*odist;   // Input distance is in "real"
    int istride = 1,       // Elements of each FFT are adjacent
	ostride = 1;
    const int nStreams = 4;

    double t0 = getTime();

    // Allocate the CUDA streams. Each stream can execute independently
    cudaStream_t stream[nStreams];
    for(auto i=0; i<nStreams; ++i) cudaStreamCreate(stream+i);

    // Allocate a single chunk on the GPU, but separate it into blocks
    cufftDoubleComplex *data[nStreams];
    int block_count = howmany*N*(N/2+1);
    cudaMalloc((void**)data, sizeof(cufftDoubleComplex)*nStreams*block_count);
    for(auto i=1; i<nStreams; ++i) data[i] = data[0] + i*block_count;

    // Create a plan to do "howmany" slabs at a time
    // This plan will be run simultaneously on multiple streams so that
    // means we have to create "work areas" for each. Instead of calling
    // cufftPlanMany we call cufftMakePlanMany so we can disable auto allocation
    // of the work area. Later we need to call cufftSetWorkArea.
    cufftHandle plan;
    size_t workSize;
    cufftCreate(&plan);
    cufftSetAutoAllocation(plan,0);
    cufftMakePlanMany(plan,sizeof(n)/sizeof(n[0]), n,
		    inembed,istride,idist,
		    onembed,ostride,odist,
		    CUFFT_D2Z,howmany,&workSize);
    void *workArea[nStreams];
    // We allocate "nStreams" work areas and set pointer to them for each stream
    cudaMalloc(&workArea[0],workSize*nStreams);
    for(auto i=1; i<nStreams; ++i) workArea[i] = reinterpret_cast<char*>(workArea[0]) + i*workSize;


    // Distribute the work on the streams
    int iStream = 0;
    for(auto i=grid.lbound(0); i<=grid.ubound(0); i+=howmany) {
	cudaMemcpyAsync(data[iStream], &grid(i,0,0), sizeof(cufftDoubleComplex)*block_count, cudaMemcpyHostToDevice,stream[iStream]);
	cufftSetStream(plan,stream[iStream]);
	cufftSetWorkArea(plan,workArea[iStream]);
	cufftExecD2Z(plan,reinterpret_cast<cufftDoubleReal*>(data[iStream]),data[iStream]);
	cudaMemcpyAsync(&grid(i,0,0),data[iStream],sizeof(cufftDoubleComplex)*block_count, cudaMemcpyDeviceToHost,stream[iStream]);
	if (++iStream == nStreams) iStream = 0;
    }
    cudaDeviceSynchronize(); // Wait for all streams to complete
    cudaFree(data[0]);
    cudaFree(workArea[0]);
    cufftDestroy(plan);
    for(auto i=0; i<nStreams; ++i) cudaStreamDestroy(stream[i]);

    double elapsed = getTime()-t0;
    cerr << "2D R2C FFT streaming: " << elapsed << " s" << endl;
}



//**********************************************************************

void compute_pk(array3D_c &fft_grid, int N){
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

    cerr << "P(k) measurement: " << elapsed << " s" << endl;

    for(int i=0; i<nBins; ++i) {
        if (n_power(i)>0) {
            cout << exp(log_k(i)/n_power(i)) << " " << power(i)/n_power(i) << endl;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc!=3) {
	cerr << "Usage: " << argv[0] << " <input> <grid" << endl;
	return 1;
	}
    string fname = argv[1];
    int N = atoi(argv[2]);

    cerr << "Reading " << fname << endl;
    array2D_r p = read_particles(fname);
    
    //test_assignment(p, N); // Test the assignment schemes

    // The grid has to be slighly larger for an in-place transformation
    array3D_r raw_grid(N,N,2*(N/2+1));
    array3D_r grid(raw_grid,Range::all(),Range::all(),Range(0,N-1));
    assign_masses(4, p, grid);

    // We use the same memory as the input grid (in-place)
    array3D_c fft_grid(reinterpret_cast<complex_type*>(grid.data()),shape(N,N,N/2+1),neverDeleteData);
    
    // Compute the fft of the over-density field
    compute_fft_2D_R2C(grid, fft_grid, N);
    //compute_fft_2D_R2C_stream(grid,fft_grid,N);
    compute_fft_1D_C2C(fft_grid,N);


    // Compute the power spectrum
    compute_pk(fft_grid, N);
    
}
