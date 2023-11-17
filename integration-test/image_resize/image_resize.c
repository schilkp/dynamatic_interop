#include "image_resize.h"
#include "../integration_utils.h"
#include <stdlib.h>

void image_resize(inout_int_t a[N][N], in_int_t c) {
  for (unsigned i = 0; i < N; i++) {
    for (unsigned j = 0; j < N; j++) {
      a[i][j] = c - a[i][j];
    }
  }
}

int main(void) {
  inout_int_t a[N][N];
  for (int y = 0; y < N; ++y) {
    for (int x = 0; x < N; ++x)
      a[y][x] = rand() % 100;
  }

  CALL_KERNEL(image_resize, a, 1000);
  return 0;
}
