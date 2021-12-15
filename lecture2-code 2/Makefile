CXX=CC
NV=nvcc
CXXFLAGS= -lcudart -lcufft -lcufftw# -O2
INCLUDE=-I/users/yerdemtu/blitz/include
all:main
tipsy.o: tipsy.cxx
	${CXX} -c $(CXXFLAGS) $(INCLUDE) -o $@ $<
main.o: main.cu  aweights.hpp
	${NV} -c ${CXXFLAGS} $(INCLUDE) -o $@ $<  
	#${CXX} -c $(CXXFLAGS) $(INCLUDE) -o $@ $<
main: tipsy.o main.o
	${CXX} $(CXXFLAGS) $(INCLUDE) -o main tipsy.o main.o
clean:
	rm -f tipsy.o main.o main
