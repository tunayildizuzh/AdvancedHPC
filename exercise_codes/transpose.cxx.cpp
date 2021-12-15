#include "mpi.h"
#include <omp.h>
#include <stdio.h>
#include <fstream>
#include <complex>
#include <fftw3.h>
#include <fftw3-mpi.h>
#include "blitz/array.h"
using namespace std;
using namespace blitz;
int main(int argc, char *argv[]) {
    const int nGrid = 9;
    const ptrdiff_t n[] = {nGrid,nGrid,nGrid/2 + 1};
    // MPI Support + FFTW MPI
    int thread_support;
    MPI_Init_thread(&argc, &argv,
        MPI_THREAD_FUNNELED,
        &thread_support);
    int nRank, iRank;
    MPI_Comm_size(MPI_COMM_WORLD, &nRank); // Number of ranks
    MPI_Comm_rank(MPI_COMM_WORLD, &iRank); // my rank
    //fftw_init_threads();
    fftw_mpi_init();
    // fftw_plan_with_nthreads(omp_get_max_threads());
    ptrdiff_t howmany = 2;
    ptrdiff_t n0, s0, n1, s1;
    ptrdiff_t v = fftw_mpi_local_size_many(sizeof(n)/sizeof(n[0]), n, howmany, FFTW_MPI_DEFAULT_BLOCK, MPI_COMM_WORLD, &n0, &s0);
    printf("r: %d: start %llu count %llu number %llu\n", iRank, s0, n0, v);
    // v = fftw_mpi_local_size_many_transposed(sizeof(n)/sizeof(n[0]), n, howmany, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, MPI_COMM_WORLD, &n0, &s0, &n1, &s1);
    // printf("k: %d: start %llu,%llu count %llu,%llu number %llu\n", iRank, s0, s1, n0, n1, v);
    Array<complex<double>,3> grid(Range(s0,s0+n0-1),Range(0,nGrid-1),Range(0,nGrid/2));
    for(auto i=grid.begin(); i!=grid.end(); ++i) {
    	auto pos = i.position();
    	*i = pos[0]*10 + pos[1]*1 + pos[2]*0;
    }
    if (iRank==2) cout << grid(Range::all(),Range::all(),0) << endl;
    auto plan = fftw_mpi_plan_many_transpose(
	nGrid, nGrid, 2*(nGrid/2+1),
	FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
	reinterpret_cast<double*>(grid.data()),
	reinterpret_cast<double*>(grid.data()),
	MPI_COMM_WORLD, FFTW_ESTIMATE);
    fftw_execute(plan);

    if (iRank==2) cout << grid(Range::all(),Range::all(),0) << endl;
    fftw_mpi_cleanup();
    MPI_Finalize();
}