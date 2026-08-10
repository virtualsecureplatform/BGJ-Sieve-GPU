#include <cstdlib>
#include <iostream>

#include <NTL/LLL.h>

NTL_CLIENT;

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " DIMENSION SEED\n";
        return 2;
    }
    const long dimension = std::strtol(argv[1], nullptr, 10);
    ZZ seed(INIT_VAL, argv[2]);
    if (dimension < 1) return 2;

    SetSeed(seed);
    ZZ determinant;
    GenPrime(determinant, 10 * dimension);
    mat_ZZ basis;
    basis.SetDims(dimension, dimension);
    clear(basis);
    basis(1, 1) = determinant;
    for (long i = 2; i <= dimension; ++i) {
        RandomBnd(basis(i, 1), determinant);
        basis(i, i) = 1;
    }
    std::cout << basis << '\n';
}
