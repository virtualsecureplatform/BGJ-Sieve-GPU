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

hd_sieve_mpi: bin_hd_sieve_mpi.cu mpi_sieve_runtime.cpp ../src/libllib.a
	$(NVCC) -ccbin $(CXX) -Xcompiler "$(CFLAGS)" -O3 \
		bin_hd_sieve_mpi.cu mpi_sieve_runtime.cpp $(INC_DIR) $(CPPFLAGS) $(MPI_CPPFLAGS) \
		$(LIBS) $(MPI_LDFLAGS) $(SURP) -lcudart -lnuma -o $@

.PHONY: clean

clean:
	-rm -f $(CUBIN)

%: bin_%.cu ../src/libllib.a
	$(NVCC) -ccbin $(CXX) -Xcompiler "$(CFLAGS)" -O3 $< $(INC_DIR) $(CPPFLAGS) $(LIBS) $(SURP) -lcudart -lnuma -o $*
