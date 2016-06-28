#include <inttypes.h>

/* make operations regarding comparation of images */
extern unsigned long long int compare_cu(const unsigned char *s, const unsigned char *d, uint16_t pitch, uint16_t bpp, int h, int w);

/* sort an array of a given size */
extern void sort(int* array, int size);
