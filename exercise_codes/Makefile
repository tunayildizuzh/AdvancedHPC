CXX=CC
NV=nvcc
CXXFLAGS= -lcudart -lcufft -lcufftw -lfftw3# -O2
INCLUDE=-I/users/yerdemtu/blitz/include
all:main
tipsy.o: tipsy.cxx
	${CXX} -c $(CXXFLAGS) $(INCLUDE) -o $@ $<
main.o: main.cu  aweights.hpp
	${NV} -c ${CXXFLAGS} $(INCLUDE) -o $@ $<  
	#${CXX} -c $(CXXFLAGS) $(INCLUDE) -o $@ $<
exercise1: exercise1.o
	${CXX} $(CXXFLAGS) $(INCLUDE) -o exercise1  exercise1.o
exercise1.o: Exercise1.cu
	${CXX} $(CXXFLAGS) $(INCLUDE) -o exercise1  exercise1.o
exercise2: exercise2.o tipsy.o
	${CXX} $(CXXFLAGS) $(INCLUDE) -o exercise2  exercise2.o tipsy.o
exercise2.o: Exercise2.cu 
	${NV} -c ${CXXFLAGS} $(INCLUDE) -o $@ $<  
main: tipsy.o main.o
	${CXX} $(CXXFLAGS) $(INCLUDE) -o main tipsy.o main.o
clean:
	rm -f tipsy.o main.o main
