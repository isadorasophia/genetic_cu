all: genetic_cu

CXX=g++-4.9
CXXFLAGS=-O3 `sdl-config --cflags` -I/usr/include/SDL
LDFLAGS=`sdl-config --libs`	
OBJECTS=genetic.o serializer.o streamer.o image.o
	
genetic_cu: $(OBJECTS) point.h
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LDFLAGS) -o genetic

image.o: image.cpp
	nvcc -x cu -c image.cpp
 
clean:
	rm genetic_omp $(OBJECTS)
	 
