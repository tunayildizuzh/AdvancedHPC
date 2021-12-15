#include "tipsy.h"
#include <iostream>
#include "aweights.hpp"
#include <fstream>
//#include "testa.cxx"

// This function takes the particle location x,y,z and convert it to i,j,k coordinates of the grid.
int axis_conversion(float particle_loc)
{
    return (particle_loc * 64) + 32;
}
/*
int write_to_csv(blitz::Array<float,2> array, const char* fname) {
    std::ofstream out;
    out.open(fname);
    if (!out.is_open()) {
        return 1;
    }   

    for (auto i=0; i<array.extent(blitz::firstDim); ++i) {
        for (auto j=0; j<array.extent(blitz::secondDim); ++j) {
            out << array(i, j) << ",";
        }   
        out << std::endl;
    }   
    out.close();
    return 0;
}
*/
int main()
{
    TipsyIO io;

    io.open("/store/uzh/uzh8/ESC412/ic_512.std");
    std::cout << io.count() << std::endl;

    if (io.fail())
    {
        std::cerr << "Unable to open file" << std::endl;
        abort();
    }

    blitz::Array<float, 2> r(io.count(), 3);
    io.load(r);
    //std::cout << r << std::endl;

    // This blitz array is constructed to get the first 10 particle location as Exercise 2 asked.
    blitz::Array<float, 2> first10(10, 3);
    first10 = r(blitz::Range(10), blitz::Range(3));
    //std::cout << first10 << std::endl;

    // x,y,z is the particle locations we extracted from the .std file.
    // i,j,k is the grid axis that we construct which is 64x64x64.
    double x, y, z;

    // This blitz array shows a 3D cell with the dimensions of 64x64x64.
    int N = 512;
    blitz::Array<float, 3> new_axis(N, N, N);
    new_axis = 0.0;

    // This for loop goes through the particle locations and get the locations in terms of x,y,z coordinates.
    //By using the function defined, it writes the new coordinates of i,j,k to the new blitz array new_axis.
    int nParticle = io.count();
    for (int a = 0; a < nParticle; ++a)
    { 
        x = r(a, 0);
        y = r(a, 1);
        z = r(a, 2);

        for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                int t,y,u;
                
                AssignmentWeights<4> w4a(axis_conversion(i));
                AssignmentWeights<4> w4b(axis_conversion(j));
                AssignmentWeights<4> w4c(axis_conversion(k));
                t = (w4a.i+i+N) % N;
                y = (w4b.i+j+N) % N;
                u = (w4c.i+k+N) % N;
                new_axis(t , y , u) += w4a.H[i] * w4b.H[j] * w4c.H[k];
            }
        }
    }

    }

    //std::cout << new_axis << std::endl;
    /*
    // Mass assignment for 2nd order.
    AssignmentWeights<2> w2a(axis_conversion(x)), w2b(axis_conversion(y)), w2c(axis_conversion(z));
    
    for(i = 0 ; i < nParticle ; ++i){
        for(j=0 ; j < nParticle ; ++j) {
            for(k =0 ; k < nParticle ; ++k){
                new_axis = new_axis(w2a.i+i) + new_axis(w2b.i+j) + new_axis(w2c.i+k); 
            }
        }
    }
    std::cout << new_axis << std::endl;
    std::cout << "\n---------------------------------------------------------------------" << std::endl;
   
    //Mass assignment for 3rd order.
    AssignmentWeights<3> w3a(axis_conversion(x)), w3b(axis_conversion(y)), w3c(axis_conversion(z));

    for(i = 0 ; i < nParticle ; ++i){
        for(j=0 ; j < nParticle ; ++j) {
            for(k =0 ; k < nParticle ; ++k){
                new_axis = new_axis(w3a.i+i) + new_axis(w3b.i+j) + new_axis(w3c.i+k); 
            }
        }
    }
    std::cout << new_axis << std::endl;
    std::cout << "\n---------------------------------------------------------------------" << std::endl;
  
*/

    //Mass assignment for 4th order.
    // for (int i = 0; i < 4; ++i)
    // {
    //     for (int j = 0; j < 4; ++j)
    //     {
    //         for (int k = 0; k < 4; ++k)
    //         {
              
    //             AssignmentWeights<4> w4a(new_axis(i));
    //             AssignmentWeights<4> w4b(new_axis(j));
    //             AssignmentWeights<4> w4c(new_axis(k));
    //             new_axis((w4a.i+i) + (w4b.i+j) + (w4c.i+k)) += w4a.H[i] * w4b.H[j] * w4c.H[k];
    //         }
    //     }
    // }


   // std::cout << new_axis << std::endl;
   // std::cout << "\n---------------------------------------------------------------------" << std::endl;
    
    // blitz::thirdIndex k;
    // blitz::Array<float,2> reduced_axis(64,64);
    // reduced_axis = blitz::max(new_axis); 
    // write_to_csv(reduced_axis,"reduced_axis");
    



    return 0;
}
