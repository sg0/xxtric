CXX = CC
DBGFLAGS = -g -fomit-frame-pointer
#OPTFLAGS = $(DBGFLAGS) -O3 -DPRINT_EXTRA_NEDGES #-DDEBUG_PRINTF   
OPTFLAGS = -O3 -DPRINT_EXTRA_NEDGES #-DFAST_CHECK_EDGELIST -DUSE_SHMEM_ORDER #-DUSE_SHMEM_ORDER -DUSE_SHMEM_SIGNAL -DUSE_MPI_RMA_PASSIVE#-DDEBUG_PRINTF   
# -DPRINT_EXTRA_NEDGES prints extra edges when -p <> is passed to 
#  add extra edges randomly on a generated graph
# use export ASAN_OPTIONS=verbosity=1 to check ASAN output
SNTFLAGS = -std=c++2a -fsanitize=address -O1 -fno-omit-frame-pointer
CXXFLAGS = --std=c++20 $(OPTFLAGS) -I.

OBJ = main.o
TARGET = tric

all: $(TARGET)

ENABLE_OPENMP=0
ifeq ($(ENABLE_OPENMP),1)
CXXFLAGS += -fopenmp -DUSE_OPENMP
endif

ENABLE_SHMEM=0
ifeq ($(ENABLE_SHMEM), 1)
CXX = CC
LDFLAGS += -L/opt/cray/pe/pmi/6.1.11/lib -lpmi -lpmi2 
endif

ENABLE_CUDA=0
ifeq ($(ENABLE_CUDA),1)
NVCXX = nvcc
MPI_PATH = /opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3
CUDA_PATH = /opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2
CXXFLAGS += -DUSE_CUDA
CUDA_CXXFLAGS = -gencode arch=compute_80,code=sm_80 
INCLUDE += -I$(MPI_PATH)/include -I$(CUDA_PATH)/include 
LDFLAGS += -lm -lrt -ldl -L$(MPI_PATH)/lib -lmpi -L$(CUDA_PATH)/lib64 -lcudart
OBJ += counting.o
endif

ENABLE_TAUPROF=0
ifeq ($(ENABLE_TAUPROF),1)
TAU=/people/ghos167/builds/tau-2.32/x86_64/lib
CXX = tau_cxx.sh -tau_makefile=$(TAU)/Makefile.tau-mpi-openmp-mpit
endif

%.o: %.cpp
ifeq ($(ENABLE_CUDA),1)
	$(NVCXX) $(INCLUDE) $(CXXFLAGS) $(CUDA_CXXFLAGS) -c counting.cu
	$(CXX) $(INCLUDE) $(CXXFLAGS) -c -o $@ $^
else
	$(CXX) $(INCLUDE) $(CXXFLAGS) -c -o $@ $^
endif

$(TARGET):  $(OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ 


.PHONY: clean

clean:
	rm -rf *~ $(OBJ) $(TARGET) *.o *.dSYM
