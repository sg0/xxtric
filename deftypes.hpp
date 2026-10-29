#pragma once
#ifndef DEFTYPES_HPP
#define DEFTYPES_HPP

#include <mpi.h>

#define PI                          (3.14159)
#define MAX_PRINT_NEDGE             (100000)

// Read https://en.wikipedia.org/wiki/Linear_congruential_generator#Period_length
// about choice of LCG parameters
// From numerical recipes
// TODO FIXME investigate larger periods
#define MLCG                        (2147483647)    // 2^31 - 1
#define ALCG                        (16807)         // 7^5
#define BLCG                        (0)

#define SR_UP_TAG                   100
#define SR_DOWN_TAG                 101
#define SR_SIZES_UP_TAG             102
#define SR_SIZES_DOWN_TAG           103
#define SR_X_UP_TAG                 104
#define SR_X_DOWN_TAG               105
#define SR_Y_UP_TAG                 106
#define SR_Y_DOWN_TAG               107
#define SR_LCG_TAG                  108

#define MAX(x,y)                    ((x > y) ? (x) : (y))
#define MIN(x,y)                    ((x < y) ? (x) : (y))
#define MID(x,y)                    (x + (y - x) / 2)

#define ONE                         (1)
#define TWO                         (2)
#define FIVE                        (5)
#define PERCOF(x,y)                 std::ceil((y) * ((x) / 100))
#define ONEPERCOF(y)                PERCOF(ONE, y)
#define TWOPERCOF(y)                PERCOF(TWO, y)
#define FIVEPERCOF(y)               PERCOF(FIVE, y)

#ifdef USE_32_BIT_GRAPH
using GraphElem = int32_t;
using GraphWeight = float;
const MPI_Datatype MPI_GRAPH_TYPE = MPI_INT32_T;
const MPI_Datatype MPI_WEIGHT_TYPE = MPI_FLOAT;
#else
using GraphElem = int64_t;
using GraphWeight = double;
const MPI_Datatype MPI_GRAPH_TYPE = MPI_INT64_T;
const MPI_Datatype MPI_WEIGHT_TYPE = MPI_DOUBLE;
#endif

#ifndef DEFAULT_BUF_SIZE
#define DEFAULT_BUF_SIZE   (8000)
#endif

#endif 
