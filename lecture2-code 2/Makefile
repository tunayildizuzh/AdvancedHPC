CXX=g++
CXXFLAGS=-std=c++17# -O2
INCLUDE=-I/Users/tunayildiz/Desktop/CPPDocuments/blitz
all:testio
tipsy.o:tipsy.cxx
	${CXX} -c $(CXXFLAGS) $(INCLUDE) -o $@ $<
testio.o: testio.cxx aweights.hpp
	${CXX} -c $(CXXFLAGS) $(INCLUDE) -o $@ $<
testio: tipsy.o testio.o
	${CXX} $(CXXFLAGS) $(INCLUDE) -o testio tipsy.o testio.o
clean:
	rm -f tipsy.o testio.o testio
