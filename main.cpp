// ***********************************************************************
//
//                              TRIC
//
// ***********************************************************************
//
//       Copyright (2019) Battelle Memorial Institute
//                      All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************ 



#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#include <cassert>
#include <cstdlib>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "deftypes.hpp"
#include "utils.hpp"
#include "graph.hpp"

#if defined(USE_OPENMP)
#include <omp.h>
#endif

#if defined(USE_CUDA)
#include "xxtric-gpu.hpp"
#elif defined(USE_CUDA_IPC)
#include "xxtric-gpu-ipc.hpp"
#elif defined(USE_CUDA_IPC_MPI)
#include "xxtric-gpu-ipc-mpi.hpp"
#else
#include "xxtric.hpp"
#endif

static std::string inputFileName;
static int me, nprocs;
static int ranksPerNode = 1;
static GraphElem nvRGG = 0;
static int generateGraph = 0;
static bool readBalanced = false;
static bool readBalancedDynamic = false;
static GraphWeight randomEdgePercent = 0.0;
static bool randomNumberLCG = false;
static long bufferSize = -1;

// parse command line parameters
static void parseCommandLine(const int argc, char * const argv[]);

int main(int argc, char *argv[])
{
  double t0, t1, td, td0, td1;

#if defined(DISABLE_THREAD_MULTIPLE_CHECK) || !defined(USE_OPENMP)
  MPI_Init(&argc, &argv);
#else  
  int max_threads;

  max_threads = omp_get_max_threads();

  if (max_threads > 1) {
      int provided;
      MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
      if (provided < MPI_THREAD_MULTIPLE) {
          std::cerr << "MPI library does not support MPI_THREAD_MULTIPLE." << std::endl;
          MPI_Abort(MPI_COMM_WORLD, -99);
      }
  } else {
      MPI_Init(&argc, &argv);
  }
#endif

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  // command line options
  parseCommandLine(argc, argv);

  Graph* g = nullptr;

  td0 = MPI_Wtime();

  // generate graph only supports RGG as of now
  if (generateGraph) 
  {
    if (!is_pwr2(nprocs)) 
    {
      std::cout << "Error: random geometric graph generation require power-of-2 #processes." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
    }
    GenerateRGG gr(nvRGG);
    g = gr.generate(randomNumberLCG, true /*isUnitEdgeWeight*/, randomEdgePercent);
  }
  else 
  {   // read input graph
    BinaryEdgeList rm;
    if (readBalanced == true || readBalancedDynamic == true)
    {
      if (me == 0)
      {
        std::cout << std::endl;
        std::cout << "Trying to balance the edge distribution while reading: " << std::endl;
        std::cout << inputFileName << std::endl;
      }
      if (readBalanced)
        g = rm.read_balanced(me, nprocs, ranksPerNode, inputFileName, 0);
      else
        g = rm.read_balanced(me, nprocs, ranksPerNode, inputFileName, 1); // use dynamic programming
    }
    else
      g = rm.read(me, nprocs, ranksPerNode, inputFileName);
  }

#if defined(PRINT_GRAPH_EDGES)        
  g->print();
#endif
  g->print_dist_stats();
  assert(g != nullptr);

  MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG_PRINTF  
  assert(g);
#endif
  td1 = MPI_Wtime();
  td = td1 - td0;

  double tdt = 0.0;
  MPI_Reduce(&td, &tdt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (me == 0)  
  {
    if (!generateGraph)
      std::cout << "Time to read input file and create distributed graph (secs.): " 
        << tdt << std::endl;
    else
      std::cout << "Time to generate distributed graph of " 
        << nvRGG << " vertices (secs.): " << tdt << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

#if defined(USE_CUDA_IPC)
  if (bufferSize != -1)
  {
    if (me == 0)
      std::cout << "*Warning*: Buffersize is ignored for selected CUDA IPC version." << std::endl;
  }
  TriangleCounting tr(g);
#elif defined(USE_SINGLE_DEVICE)
  TriangleCountingSingle tr(g);
#else
  /*
  if (bufferSize < 100)
    bufferSize = DEFAULT_BUF_SIZE;
  */
  TriangleCounting tr(g, bufferSize);
#endif

  MPI_Barrier(MPI_COMM_WORLD);

  t0 = MPI_Wtime();
  GraphElem ntris = tr.count();
  t1 = MPI_Wtime();
  double p_tot = t1 - t0, t_tot = 0.0, t_max = 0.0, t_min = 0.0;

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(&p_tot, &t_tot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&p_tot, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&p_tot, &t_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

  if (me == 0) 
  {   
    double avg_t = (double)(t_tot/(double)nprocs);
#if defined(USE_CUDA_IPC)
      std::cout << "Performed counting (using CUDA IPC) on " << nprocs << " GPUs" << std::endl;
#elif defined(USE_CUDA_IPC_MPI)
      std::cout << "Using CUDA IPC and MPI: Avg/Min/Max execution time (secs.) for triangle counting on " << nprocs << " processes: " 
        << avg_t << ", " << t_min << ", " << t_max << std::endl;
#else
      std::cout << "Avg/Min/Max execution time (secs.) for triangle counting on " << nprocs << " processes: " 
        << avg_t << ", " << t_min << ", " << t_max << std::endl;
#endif

#if defined(USE_CUDA_IPC) || defined(USE_SINGLE_DEVICE)
#else
    std::cout << "User initialized per-PE buffer count: " << bufferSize << std::endl;
#endif    
    std::cout << "Number of triangles: " << ntris << std::endl;

    std::cout << "TEPS: " << g->get_ne()/avg_t << std::endl;
    std::cout << "Resolution of MPI_Wtime: " << MPI_Wtick() << std::endl;
  }

  tr.clear(); 
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();

  return 0;
}

void parseCommandLine(const int argc, char * const argv[])
{
  int ret;

  while ((ret = getopt(argc, argv, "f:r:n:p:olbds:")) != -1) {
    switch (ret) {
      case 'f':
        inputFileName.assign(optarg);
        break;
      case 'b':
        readBalanced = true;
        break;
      case 'd':
        readBalancedDynamic = true;
        break;
      case 'r':
        ranksPerNode = atoi(optarg);
        break;
      case 'n':
        nvRGG = atol(optarg);
        if (nvRGG > 0)
          generateGraph = true; 
        break;
      case 'l':
        randomNumberLCG = true;
        break;
      case 'p':
        randomEdgePercent = atof(optarg);
        break;
      case 's':
        bufferSize = atol(optarg);
        break;
      default:
        assert(0 && "Should not reach here!!");
        break;
    }
  }

  // warnings/info

  if (me == 0 && generateGraph && (readBalanced || readBalancedDynamic)) 
  {
    std::cout << "Balanced read (option -b) is only applicable for real-world graphs. "
      << "This option does nothing for generated (synthetic) graphs." << std::endl;
  } 

  // errors
  if (me == 0 && (argc == 1)) 
  {
    std::cerr << "Must specify some options." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -99);
  }

  if (me == 0 && !generateGraph && inputFileName.empty()) 
  {
    std::cerr << "Must specify a binary file name with -f or provide parameters for generating a graph." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -99);
  }

  if (me == 0 && !generateGraph && randomNumberLCG) 
  {
    std::cerr << "Must specify -g for graph generation using LCG." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -99);
  }

  if (me == 0 && !generateGraph && (randomEdgePercent > 0.0)) 
  {
    std::cerr << "Must specify -g for graph generation first to add random edges to it." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -99);
  } 

  if (me == 0 && generateGraph && ((randomEdgePercent < 0.0) || (randomEdgePercent >= 100.0))) 
  {
    std::cerr << "Invalid random edge percentage for generated graph!" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -99);
  }
} // parseCommandLine
