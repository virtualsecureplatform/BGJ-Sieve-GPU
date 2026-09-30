#include <cmath>
#include <cstdio>

#include "../include/lattice.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s BASIS\n", argv[0]);
        return 2;
    }
    Lattice_QP basis(argv[1]);
    if (!basis.NumRows()) {
        std::fprintf(stderr, "failed to load basis: %s\n", argv[1]);
        return 1;
    }
    basis.compute_gso_QP();
    VEC_QP gso = basis.get_B();
    for (long i = 0; i < basis.NumRows(); ++i) {
        std::printf("%.17g%c", 0.5 * std::log(gso.hi[i]),
                    i + 1 == basis.NumRows() ? '\n' : ' ');
    }
    return 0;
}
