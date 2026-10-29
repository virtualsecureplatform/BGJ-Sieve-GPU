#include "../include/sieve_cache_budget.h"
#include <cassert>
#include <cstdio>
int main() {
    const long pool[]={94183,108754,125578,145005,167437,193339};
    for(long p:pool) {
        const long s=bounded_solution_slots(229217,p,320,p+1336);
        assert(p+s+320<=229217 && s>=512);
    }
    assert(bounded_solution_slots(229217,94183,320,95519)==95519);
    assert(bounded_solution_slots(229217,193339,320,194675)==35558);
    bool failed=false;
    try {bounded_solution_slots(1000,900,100,800);} catch(const std::runtime_error&){failed=true;}
    assert(failed);
    // Original retained PWC allocation pattern exhausts the shared arena.
    assert(122341+(94183-81567)+95519>229217);
    assert(94183+95519+320<229217);
    puts("PASS: shared arena budgets and CSD139 retained-slot regression");
}
