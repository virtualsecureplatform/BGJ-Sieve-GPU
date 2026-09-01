#include <cmath>
#include <cstdio>

#include "../include/lattice.h"

static void report(const char *label, const char *path) {
    Lattice_QP basis(path);
    if (!basis.NumRows()) {
        std::fprintf(stderr, "basis_quality label=%s error=load_failed path=%s\n",
                     label, path);
        return;
    }
    basis.compute_gso_QP();
    const long n = basis.NumRows();
    VEC_QP B = basis.get_B();

    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (long i = 0; i < n; ++i) {
        const double y = 0.5 * std::log2(B.hi[i]);
        sx += i;
        sy += y;
        sxx += (double)i * i;
        sxy += (double)i * y;
    }
    const double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    const long tail60 = n > 60 ? n - 60 : 0;
    const long lift120 = n > 120 ? n - 120 : 0;

    std::printf("basis_quality label=%s rows=%ld cols=%ld pot_log2=%.12f "
                "det_root=%.12f gh_full=%.12f b1=%.12f "
                "gso_log2_slope=%.12f gso_at_tail60=%.12f "
                "gh_tail60=%.12f gh_tail120=%.12f pump_red_msd=%ld path=%s\n",
                label, n, basis.NumCols(), basis.Pot(), basis.detn(), basis.gh(),
                std::sqrt(B.hi[0]), slope, std::sqrt(B.hi[tail60]),
                basis.gh(tail60, n), basis.gh(lift120, n),
                basis.pump_red_msd(), path);
}

int main(int argc, char **argv) {
    // Preserve the original two-path interface for existing jobs, while also
    // accepting explicit LABEL PATH pairs for preprocessing comparisons.
    if (argc == 3) {
        report("fplll_bkz72", argv[1]);
        report("hd_bkz120", argv[2]);
        return 0;
    }
    if (argc < 3 || (argc % 2) == 0) {
        std::fprintf(stderr,
                     "usage: %s BASIS1 BASIS2 | LABEL1 BASIS1 [LABEL2 BASIS2 ...]\n",
                     argv[0]);
        return 2;
    }
    for (int i = 1; i < argc; i += 2) report(argv[i], argv[i + 1]);
    return 0;
}
