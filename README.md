Portable program that realize the algorithm for polygonal image generation proposed by Roger Alsin in his blog (www.rogeralsing.com).

The original code was a .net program. This version seems 10/20 times faster than Roger .net code.

This version requires SDL.

To execute it:

	./genetic image.bmp [progressfile]

Image must be a 32bit BMP image, resolution doesn't matter, but the smaller the image is the faster the algorithm will be.

Progress file is an optional file created by a previous run of the program and saved pressing "s".

During the program execution you can use the following keys:
s - save a progress file named imagename.idx
b - save a bmp file named imagename.out.idx
q - quit the program

---
original version made by:
	gabrielegreco@gmail.com

---
cuda implementation by:
	Isadora Sophia e João Guilherme Fidélis

---
Para executar o programa, é necessário obter a library do SDL, em que o projeto é implementado. Em seguida, execute:

	make
	make run

E o programa já passará a executar todas as entradas e seu respectivo tempo.
