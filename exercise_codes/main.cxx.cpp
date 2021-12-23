#include <fstream>
#include <typeinfo>
#include <complex>
#include <stdlib.h>
#include <sys/time.h>
#include <fftw3.h>
#include <mpi.h>
#include <omp.h>
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
array2D_r read_particles(string fname,int size,int rank){
    double t0, elapsed;
    TipsyIO io;

    io.open(fname);
    cout << "Found "<<io.count() << " particles."<<endl;

    if (io.fail()) {
        cerr << "Unable to open file" << endl;
        abort();
    }
    int N = io.count();
    int n = (N + (size-1))/size;
    int istart = rank*n
    int iend;
    int num_rows = N/size;
    if(istart + n-1 < N){
	iend = istart + n-1;
    } else {
        iend = N - 1;
    }
    // Allocate the particle buffer
    array2D_r p(blitz::Range(istart,iend),blitz::Range(0,2));
    
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
void assign_masses(int o, array2D_r p, array3D_r &grid, int rank, int size){
    double t0, elapsed;
    auto N_part = p.shape()[0]; // Do i still need this?
    auto shape = grid.shape();
    
    array3D_r grid_nopad = grid(blitz::Range::all(), blitz::Range::all(),blitz::Range(0,shape[2]-3]));    
    t0 = getTime();
    #pragma omg parallel for
    for(auto i=p.lbound(0), i <= p.ubound(0); ++i){
        assign_mass(o, p(i,0), p(i,1), p(i,2), grid_nopad);
    }
    if(rank==0){
	    MPI_Reduce(MPI_IN_PLACE,grid.data(),grid.size(),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    }else{
	    MPI_Reduce(grid.data(),NULL,grid.size(),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    }
    
    // Compute the average density per grid cell
    real_type avg = blitz::sum(grid_nopad) / (grid_nopad.size());

    // Turn the density into the over-density
    grid = (grid - avg) / avg;
    elapsed = getTime()-t0;
    cout << "mass assignment: " << elapsed << " s" << endl;
}

// Call mass assignment at every order to test implementation
/*
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
*/

void compute_fft(array3D_r grid, array3D_c fft_grid, int N, MPI_Comm comm){
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

    blitz::Array<double,1>  log_k(nBins);
    blitz::Array<double,1>  power(nBins);
    blitz::Array<int64_t,1> n_power(nBins);


    // Fill arrays with zeros
    log_k = 0;
    power = 0;
    n_power = 0;

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
// Communicating with FFTW to determine how the data is organized.
// N is being the grid size and returns the number of slabs (local_n0)
// local_0_start is the starting slab.
alloc_local = fftw_mpi_local_size_3d(N,N,N/2+1, MPI_COMM_WORLD, &local_n0, &local_0_start);
//blitz::Array<int,1> starts(size); // Array to hold the start slab of each rank.
//int s = local_0_start;

void count_sort(vector<double> &arr, vector<int> idx, int max_idx){ 
    vector<int> count(max_idx + 1);
    vector<double> out(arr.size());

    for (int i = 0; i < arr.size(); i++)
        count[idx[i]]++;

    for (int i = 1; i < count.size(); i++)
        count[i] += count[i - 1];

    for (int i = arr.size() - 1; i >= 0; i--){
        out[count[idx[i]] - 1] = arr[i];
        count[idx[i]]--;
    }

    for (int i = 0; i < arr.size(); i++)
        arr[i] = out[i];
}

void printArray(vector<double>& arr){
    for (int i = 0; i < arr.size(); i++)
        cout << arr[i] << " ";
    cout << "\n";
}

vector<int> idx_vector = {1,0,0};

vector<double> reorder_p(array2D_r p, int N, int starts_at) {
	array2D_r reordered_p(blitz::Range(0,N),blitz::Range(0,2));
	blitz::Array<int,1> starts_at(size+1); // Index starts from 0, so starts_at needs to be size+1.
	int s = local_n0_start;
        MPI_Allgather(&s, 1 ,MPI_INT, starts_at.data(), 1, MPI_INT, MPI_COMM_WORLD);

	if (rank ==0) {
		blitz::Array<int,1> cnts(N);
		cnts = 0;
		for(int j=p.lbound(0); j <= p.ubound(0); ++j{
			       AssignmentWeights<4> wx(p(j,0)+0.5)*N);
			       ++cnts;
		}
	}		
	count_sort(&p,starts_at.size(),size+1);
        MPI_Alltoall(&p,p.shape(),MPI_DOUBLE,&reordered_p,reordered_p.shape(),MPI_DOUBLE,MPI_COMM_WORLD);
	int send_count[size];
	for(int i =0; i < sizeof(send_count)/sizeof(send_count[0]); ++i) {
		send_count[i] = io.count/size; // This is the number of particles that are sent and received per each slab and rank.
	}
	int disp[];
	disp[0] = 0;
	for(int i =1; i < sizeof(cnts)/sizeof(cnts[0]); ++i){
		disp[i] = disp[i-1] + cnt[i-1];
	}
	MPI_Alltoallv(&p,send_count,disp,MPI_DOUBLE,reordered_p,disp,MPI_DOUBLE,MPI_COMM_WORLD);
}

void PPS_Binning(int rank,int size) {	// Parallel Power Spectrum binning with MPI_Reduce.


	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	MPI_Reduce(log_k, sizeof(log_k)/sizeof(log_k[0], MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(power, sizeof(power)/sizeof(power[0], MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(n_power, sizeof(n_power)/sizeof(n_power[0]), MPI_INT64_t, MPI_SUM, 0, MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
    int thread_support;
    int rank;
    int size;

    MPI_Init_thread(&argc,&argv, MPI_THREAD_FUNNELED,&thread_support);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    fftw_mpi_init();
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    int N = 64;
    string fname = "/store/uzh/uzh8/ESC412/ic_512.std"; 
    array2D_r p = read_particles(fname,rank,size);

   // MPI_Allgather(&s, 1, MPI_INT, starts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    //MPI_Alltoall(&s,count_list.size(), MPI_INT, count_list,count_list.size(), MPI_INT, MPI_COMM_WORLD);
    //Dummy Communicator fot FFTW-MPI (Rank 0 performs FFT).
    // MPI_Comm dummy_comm;
    //MPI_Comm_split(MPI_COMM_WORLD,rank,rank,&dummy_comm);

    ptrdiff_t alloc_local, local_n0, local_0_start;

    alloc_local = fftw_mpi_local_size_3d(N,N,N/2+1, MPI_COMM_WORLD, &local_n0, &local_0_start);
    blitz::GeneralArrayStorage<3> storage;
    storage.base(local_n0_start,0,0);
    double *local_grid_space = fftw_alloc_real((local_n0_start+3)*N*(N+2));  // For the real part. 
    array3D_r grid(local_grid_space,blitz::shape(local_n0_start+3,N,N+2,neverDeleteData,storage));    
    // array3D_r grid(local_n0,N,N+2);
    double *local_grid_space_complex = fftw_alloc_complex((local_n0+3)*N*(N/2));

    assign_masses(4,p,grid,rank,size);
    int dest = (rank +1) % 2 * size;
    int src = (rank -1) % 2 * size;
    // To avoid deadlock, MPI_Irecv is called before MPI_Send.
    MPI_Request request;
    MPI_Irecv(&grid, local_grid_space, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &request); 
    MPI_Send(&grid, local_grid_space, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
    array3D_c grid_complex(local_grid_space_complex,blitz::shape(local_n0+3,N,N+2,neverDeleteData,storage)); // For the complex part.
    //array3D_c fft_grid(local_n0,N,N/2+1);
    compute_fft(grid,grid_complex,N,MPI_COMM_WORLD);
    compute_pl(grid_complex,N);
    
    // Binning with MPI_Reduce.
    PPS_Binning(rank,size); 
    //cout << p << endl; //
    /*  
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
    */
    fftw_mpi_cleanup();
    MPI_Finalize();
    fftw_free(grid);
    fftw_free(grid_complex);
}