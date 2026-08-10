# SVP-142, lattice seed 42

This recipe targets exact squared norm `8949279` (displayed norm `2992`).
The challenge site's online generator currently returns an empty response for
this instance, so preparation reproduces it with the official algorithm and
the required NTL 9.3 pseudorandom generator. The NTL source archive is vendored
and hash-pinned; preparation performs no dependency download.

Preprocessing is LLL followed by pruned BKZ-60 with at most eight loops. The
four-A100 sieve defaults to RNG seed 0, MLD 118, and TSD 132.
