CXX = mpicxx
DBGFLAGS = -g -fomit-frame-pointer
#OPTFLAGS = $(DBGFLAGS) -O3 -DPRINT_EXTRA_NEDGES #-DDEBUG_PRINTF   
OPTFLAGS = -Ofast -DFAST_CHECK_EDGELIST -DPRINT_EXTRA_NEDGES #-DDEBUG_PRINTF   
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

ENABLE_TAUPROF=0
ifeq ($(ENABLE_TAUPROF),1)
TAU=/people/ghos167/builds/tau-2.32/x86_64/lib
CXX = tau_cxx.sh -tau_makefile=$(TAU)/Makefile.tau-mpi-openmp-mpit
endif


all: $(TARGET)

%.o: %.cpp
	$(CXX) $(INCLUDE) $(CXXFLAGS) -c -o $@ $^

$(TARGET):  $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ 

.PHONY: clean

clean:
	rm -rf *~ $(OBJ) $(TARGET) *.dSYM
