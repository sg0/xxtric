## About

`xxtric` is a distributed-memory exact triangle counting code that assumes a 1-D vertex-based 
distribution of a graph. There are several GPU variants as well, tested on contemporary scale-out 
and scale-up based NVIDIA&reg GPU platforms (considering NVIDIA&reg "Ampere" and "Hopper" architectures). 
`xxtric` adopts standard vertex-numbering based precedence rules to facilitate
single counting across GPUs and CPUs. 

![Distributed counting precedence rules](image/distributed-counting-precedence.pdf)

Papers:

[1] Ghosh S. Enumeration-based Triangle Counting on GPUs. arXiv preprint arXiv:2610.29988. 2026 Oct 21.

Please `*` this repository if you find it useful.

## Compile

`xxtric` is a C++ header-only interface and requires MPI and CUDA. It uses MPI nonblocking Send/Recv and
collectives for communication, in addition to CUDA memory copy interface. Please update the Makefile 
with compiler flags and use a C++17 compliant compiler of your choice. Invoke `make clean; make` 
after setting paths to MPI and CUDA for generating the binary. Use `mpirun` or `mpiexec` or `srun` to 
execute the code with specific runtime arguments mentioned in the next section.

Pass `-DPRINT_DIST_STATS` while building for printing distributed graph characteristics. Relevant macros:

1. `-DUSE_CUDA`: standard multi-node CUDA + MPI based counting ($PPN == \#devices$), use `-DUSE_SINGLE_DEVICE` in addition for single-device counting.

2. `-DUSE_CUDA_CPU`: hybrid multi-node CPU and GPU counting ($PPN > \#devices$).

3. `-DUSE_CUDA_IPC`: single-node CUDA IPC variant ($\#processes == \#devices$); `-DUSE_CG_CLUSTER_GROUP` to enable cluster-based reduction within a device (on Hopper).

4. `-DUSE_CUDA_IPC_MPI`: multi-node MPI and CUDA IPC (peer copy between devices/node, and MPI communication for inter-node transfers)

5. Default: multi-node MPI CPU based counting; use `-DCOMPRESS_COMM_BUFFERS` to enable outgoing/incoming buffer compression via LZ4.

There are other macros which can be enabled to study performance variations
(see respective header or CUDA implementation). 

## Run

`xxtric` assumes input graph in a binary format, see: https://github.com/ECP-ExaGraph/vite/blob/master/README#L96
 
```bash
mpiexec -n 2 bin/./tric -f karate.bin 
mpiexec -n 2 bin/./tric -n 100 -t 0
mpiexec -n 2 bin/./tric -p 2 -n 100
```

Possible options (can be combined):

1.  -f <bin-file>   : Specify input binary file after this argument. 
2.  -b              : Only valid for real-world inputs. Attempts to distribute approximately 
                      equal number of edges among processes. Irregular number of vertices
                      owned by a particular process. Increases the distributed graph creation
                      time due to serial overheads, but may improve overall execution time.
3.  -d              : Only valid for real-world inputs. Attempts to distribute approximately 
                      equal number of edges among processes using dynamic programming, 
                      linear partitioning (from Prof. Skiena's book).
4.  -n <vertices>   : Only valid for synthetically generated inputs. Pass total number of 
                      vertices of the generated graph.
5.  -l              : Use distributed LCG for randomly choosing edges. If this option 
                      is not used, we will use C++ random number generator (using 
                      std::default_random_engine).
6.  -p <percent>    : Only valid for synthetically generated inputs. Specify percent of overall 
                      edges to be randomly generated between processes.
7.  -r <nranks>     : This is used to control the number of aggregators in MPI I/O and is
                      meaningful when an input binary graph file is passed with option "-f".
                      naggr := (nranks > 1) ? (nprocs/nranks) : nranks;
8.  -s <size>       : Buffer size, only relevant when -DAGGR_<> is passed for aggregate-buffered 
                      triangle counting. Default buffer size can be set by updating DEFAULT_BUF_SIZE. 
