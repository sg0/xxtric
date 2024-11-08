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
