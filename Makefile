CXX 	   = g++-4.9
CXXFLAGS   = -O3 `sdl-config --cflags` -I/usr/include/SDL -g
LDFLAGS    = `sdl-config --libs`
OBJECTS    = genetic.o serializer.o streamer.o image.o image_op.o

CUDA_LIB   = /usr/local/cuda-7.5/lib64
CUDA_FLAGS = -lcuda -lcudart

INPUTDIR   = input
INPUT      = $(shell find $(INPUTDIR) -name '*.bmp')
OUTPUT     = genetic

all:
	make clean 
	make serial
	make clean
	make cuda

serial: OUTPUT  = genetic_serial
serial: genetic

cuda: CXXFLAGS += -DCUDA
cuda: OUTPUT    = genetic_cuda
cuda: genetic

run:
	@echo "------ Input 1 ------"
	@echo "> Serial code:"
	./genetic_serial ./$(INPUTDIR)/hair.bmp
	@echo "> Parallel code:"
	./genetic_cuda   ./$(INPUTDIR)/hair.bmp

	@echo "------ Input 2 ------"
	@echo "> Serial code:"
	./genetic_serial ./$(INPUTDIR)/lemon.bmp
	@echo "> Parallel code:"
	./genetic_cuda   ./$(INPUTDIR)/lemon.bmp

	@echo "------ Input 3 ------"
	@echo "> Serial code:"
	./genetic_serial ./$(INPUTDIR)/pic.bmp
	@echo "> Parallel code:"
	./genetic_cuda   ./$(INPUTDIR)/pic.bmp

	@echo "------ Input 4 ------"
	@echo "> Serial code:"
	./genetic_serial ./$(INPUTDIR)/robot.bmp
	@echo "> Parallel code:"
	./genetic_cuda   ./$(INPUTDIR)/robot.bmp

genetic: $(OBJECTS) point.h
	$(CXX) -L$(CUDA_LIB) $(CXXFLAGS) $(OBJECTS) $(LDFLAGS) -o $(OUTPUT) $(CUDA_FLAGS)

image_op.o: image_op.cpp
	nvcc -x cu -c -g image_op.cpp
 
clean:
	rm -f *.o
