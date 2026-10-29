CXX = mpicxx
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

ENABLE_SHMEM=0
ifeq ($(ENABLE_SHMEM), 1)
CXX = CC
LDFLAGS += -L/opt/cray/pe/pmi/6.1.11/lib -lpmi -lpmi2 
endif

ENABLE_COMPRESSION=0
ifeq ($(ENABLE_COMPRESSION),1)
CMP_PATH = /opt/homebrew/opt/lz4
INCLUDE += -I$(CMP_PATH)/include -DCOMPRESS_COMM_BUFFERS 
LDFLAGS += -L$(CMP_PATH)/lib -llz4
endif

ENABLE_CUDA=0
ifeq ($(ENABLE_CUDA),1)
NVCXX = nvcc
MPI_PATH = /share/apps/openmpi/5.0.5/gcc/12.2.0
CUDA_PATH = /share/apps/cuda/12.3
CXXFLAGS += -DUSE_CUDA #-DUSE_CUDA_IPC #-DUSE_CG_CLUSTER_GROUP #-DUSE_SINGLE_DEVICE
CUDA_CXXFLAGS = -gencode arch=compute_90,code=sm_90 
INCLUDE += -I$(MPI_PATH)/include -I$(CUDA_PATH)/include 
LDFLAGS += -lm -lrt -ldl -L$(MPI_PATH)/lib -lmpi -L$(CUDA_PATH)/lib64 -lcudart
OBJ += counting.o
endif

ENABLE_OPENMP=0
ifeq ($(ENABLE_OPENMP)$(ENABLE_CUDA),1)
CXXFLAGS += -Xcompiler -fopenmp -DUSE_OPENMP
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
