CUBIN = $(filter-out hd_sieve_mpi,$(patsubst bin_%.cu,%, $(wildcard *.cu)))
CXX = g++
CFLAGS = -O3 -g -fopenmp -march=native -pthread -rdynamic -Wno-unused-result -Wno-format-truncation
LIBS = -L../src -lllib -lntl -lgmp -lm
INC_DIR =

SURP = -diag-suppress 549,550,815,1650 -Wno-deprecated-gpu-targets -gencode arch=compute_80,code=sm_80 -static-global-template-stub=false

NVCC = nvcc

all: $(CUBIN)

MPI_CPPFLAGS ?= $(shell mpicxx --showme:compile)
MPI_LIBDIR ?= $(shell mpicxx --showme:link | tr ' ' '\n' | sed -n 's/^-L//p' | head -1)
MPI_LDFLAGS ?= -L$(MPI_LIBDIR) -lmpi -Xlinker -rpath -Xlinker $(MPI_LIBDIR)

hd_sieve_mpi: bin_hd_sieve_mpi.cu mpi_bkz_driver.inc mpi_sieve_runtime.cpp mpi_sieve_pipeline.inc ../src/libllib.a
	$(NVCC) -ccbin $(CXX) -Xcompiler "$(CFLAGS)" -O3 \
		bin_hd_sieve_mpi.cu mpi_sieve_runtime.cpp $(INC_DIR) $(CPPFLAGS) $(MPI_CPPFLAGS) \
		$(LIBS) $(MPI_LDFLAGS) $(SURP) -lcudart -lnuma -o $@

hd_phase_bench: bin_hd_phase_bench.cu bin_hd_sieve.cu ../src/libllib.a
	$(NVCC) -ccbin $(CXX) -Xcompiler "$(CFLAGS)" -O3 bin_hd_phase_bench.cu $(CPPFLAGS) $(LIBS) $(SURP) -lcudart -lnuma -o $@

hd_phase_bench_mpi: bin_hd_phase_bench.cu bin_hd_sieve.cu mpi_sieve_runtime.cpp mpi_sieve_pipeline.inc ../src/libllib.a
	$(NVCC) -ccbin $(CXX) -Xcompiler "$(CFLAGS)" -O3 -DHD_PHASE_BENCH_MPI=1 \
		bin_hd_phase_bench.cu mpi_sieve_runtime.cpp $(CPPFLAGS) $(MPI_CPPFLAGS) $(LIBS) $(MPI_LDFLAGS) $(SURP) -lcudart -lnuma -o $@

.PHONY: clean

clean:
	-rm -f $(CUBIN)

%: bin_%.cu ../src/libllib.a
	$(NVCC) -ccbin $(CXX) -Xcompiler "$(CFLAGS)" -O3 $< $(INC_DIR) $(CPPFLAGS) $(LIBS) $(SURP) -lcudart -lnuma -o $*
