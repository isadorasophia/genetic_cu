all: genetic_cu

CXX=g++-4.9
CXXFLAGS=-O3 `sdl-config --cflags` -I/usr/include/SDL -g
LDFLAGS=`sdl-config --libs`	
OBJECTS=genetic.o serializer.o streamer.o image.o image_op.o

CUDA_LIB=/usr/local/cuda-7.5/lib64
CUDA_FLAGS=-lcuda -lcudart 
	
genetic_cu: $(OBJECTS) point.h
	$(CXX) -L$(CUDA_LIB) $(CXXFLAGS) $(OBJECTS) $(LDFLAGS) -o genetic $(CUDA_FLAGS)

image_op.o: image_op.cpp
	nvcc -x cu -c -g image_op.cpp
 
clean:
	rm genetic_omp $(OBJECTS)
	 
