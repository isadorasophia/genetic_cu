> Able to compile .cpp code as .cu by:
	`` nvcc -x cu file.cpp ``

---

Portable program that realize the algorithm for polygonal image generation proposed by Roger Alsin in his blog (www.rogeralsing.com).

The original code was a .net program, I realized a portable C++ version. This version seems 10/20 times faster than Roger .net code.

This version requires SDL.

Provided you have SDL development libraries installed you can compile it with "make" on a linux machine without additional efforts. You may need to hack a makefile to compile it on windows or OSX machines.

To execute it:

./genetic image.bmp [progressfile]

Image must be a 32bit BMP image, resolution doesn't matter, but the smaller the image is the faster the algorithm will be.

Progress file is an optional file created by a previous run of the program and saved pressing "s".

During the program execution you can use the following keys:
s - save a progress file named imagename.idx
b - save a bmp file named imagename.out.idx
q - quit the program

News of this version:
- FASTER
- Internal polygon drawing.
- Improved confidence function.
- More object oriented design.
- View original image on the right.
- Load/Save progress files and bitmaps.
- Doesn't depend anymore from SDL_gfx

original version made by:
	gabrielegreco@gmail.com
