all: genetic_cu

CUDA_LIB=/usr/local/cuda-7.5/lib64
CUDA_FLAGS=-lcuda -lcudart 

CXX=g++-4.9
OBJECTS=image_op.o main.o

genetic_cu: $(OBJECTS)
	$(CXX) -L$(CUDA_LIB) $(OBJECTS) -o ok $(CUDA_FLAGS)

image_op.o: image_op.cpp
	nvcc -x cu -c image_op.cpp

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -c 

clean:
	rm -f *.o
	 
