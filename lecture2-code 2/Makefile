CXX=CC
CXXFLAGS=-std=c++17# -O2
INCLUDE=-I/users/yerdemtu/blitz/include
all:main
tipsy.o:tipsy.cxx
	${CXX} -c $(CXXFLAGS) $(INCLUDE) -o $@ $<
main.o: main.cxx.cpp aweights.hpp
	${CXX} -c $(CXXFLAGS) $(INCLUDE) -o $@ $<
main: tipsy.o main.o
	${CXX} $(CXXFLAGS) $(INCLUDE) -o main tipsy.o main.o
clean:
	rm -f tipsy.o main.o main
