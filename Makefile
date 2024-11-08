CXX = mpicxx
DBGFLAGS = -g -fomit-frame-pointer
#OPTFLAGS = $(DBGFLAGS) -O3 -DPRINT_EXTRA_NEDGES #-DDEBUG_PRINTF   
OPTFLAGS = -O3 -DPRINT_EXTRA_NEDGES #-DFAST_CHECK_EDGELIST -DUSE_SHMEM_ORDER #-DUSE_SHMEM_ORDER -DUSE_SHMEM_SIGNAL -DUSE_MPI_RMA_PASSIVE#-DDEBUG_PRINTF   
# -DPRINT_EXTRA_NEDGES prints extra edges when -p <> is passed to 
#  add extra edges randomly on a generated graph
# use export ASAN_OPTIONS=verbosity=1 to check ASAN output
SNTFLAGS = -std=c++2a -fsanitize=address -O1 -fno-omit-frame-pointer
CXXFLAGS = -std=c++2a $(OPTFLAGS) -I.

OBJ = main.o
TARGET = tric

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
MPI_PATH = /share/apps/openmpi/5.0.5/gcc/14.1.0
CUDA_PATH = /share/apps/cuda/11.1
CUDA_CXXFLAGS = -DUSE_CUDA -gencode arch=compute_80,code=sm_80 
CUDA_INCLUDE += -I$(CUDA_PATH)/include #-I$(MPI_PATH)/include
LDFLAGS += -ldl -L$(MPI_PATH)/lib -lmpi -L$(CUDA_PATH)/lib64 -lcudart
endif

ENABLE_TAUPROF=0
ifeq ($(ENABLE_TAUPROF),1)
TAU=/people/ghos167/builds/tau-2.32/x86_64/lib
CXX = tau_cxx.sh -tau_makefile=$(TAU)/Makefile.tau-mpi-openmp-mpit
endif


all: $(TARGET)

%.o: %.cpp
	$(CXX) $(INCLUDE) $(CXXFLAGS) -c -o $@ $^

ifeq ($(ENABLE_CUDA),1)
%.o: %.cu
	$(NVCXX) $(CUDA_INCLUDE) $(CUDA_CXXFLAGS) -c -o $@ $^
endif

$(TARGET):  $(OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ 

.PHONY: clean

clean:
	rm -rf *~ $(OBJ) $(TARGET) *.dSYM
