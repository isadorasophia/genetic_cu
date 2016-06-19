all: genetic_serial genetic_omp

CXX=g++-4.9
CXXFLAGS=-O3 `sdl-config --cflags` -I/usr/include/SDL -fopenmp
LDFLAGS=`sdl-config --libs`	
OBJECTS=genetic.o serializer.o streamer.o image.o

genetic_serial: $(OBJECTS) point.h
	g++-4.9 -o $@ $(CXXFLAGS) $(OBJECTS) $(LDFLAGS)

genetic_omp: $(OBJECTS) point.h
	g++-4.9 -o $@ $(CXXFLAGS) $(OBJECTS) $(LDFLAGS)

clean:
	rm genetic_omp $(OBJECTS)
	 
