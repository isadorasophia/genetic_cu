#include <cstdio>
#include <algorithm>

#include "image_op.h"

#define SIZE 1000000

int comp( const void* a, const void* b ) {
    return ( *( int* )a - *( int* )b );
}

int main () {
	int vector[SIZE];

	for (int i = 0; i < SIZE; i++)
		vector[i] = SIZE - i;

	sort(vector, SIZE);

	// printf("done!\n");

	// qsort(vector, SIZE, sizeof(int), comp);

	// printf("ugh... done\n");

	return 0;
}