/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#define restrict __restrict__

#define THREADS1 1024  /* must be a power of 2 */
#define THREADS2 1024
#define THREADS3 384   /* shared-memory limited on some devices */
#define THREADS4 1024
#define THREADS5 1024
#define THREADS6 1024

#define FACTOR1 2
#define FACTOR2 2
#define FACTOR3 1  /* must all be resident at the same time */
#define FACTOR4 1  /* must all be resident at the same time */
#define FACTOR5 2
#define FACTOR6 2

#include <float.h>
#include <raft/cudart_utils.h>
#include <raft/device_atomics.cuh>
#include <stdio.h>

namespace ML {
namespace TSNE {
namespace BH {

/**
 * Intializes the states of objects. This speeds the overall kernel up.
 */
template <typename value_t>
__global__ void InitializationKernel(unsigned int* restrict limiter,
                                     value_t* restrict radiusd)
{
  limiter[0]   = 0;
  radiusd[0]   = 0.0f;
}

/**
 * Reset normalization back to 0.
 */
template <typename value_idx, typename value_t>
__global__ void Reset_Normalization(value_t* restrict Z_norm,
                                    value_t* restrict radiusd_squared,
                                    value_idx* restrict bottomd,
                                    const value_idx NNODES,
                                    const value_t* restrict radiusd)
{
  Z_norm[0]          = 0.0f;
  radiusd_squared[0] = radiusd[0] * radiusd[0];
  // create root node
  bottomd[0] = NNODES;
}

/**
 * Find 1/Z
 */
template <typename value_idx, typename value_t>
__global__ void Find_Normalization(value_t* restrict Z_norm, const value_idx N)
{
  Z_norm[0] = 1.0f / (Z_norm[0] - N);
}

/**
 * Figures the bounding boxes for every point in the embedding.
 */
template <int dim, typename value_idx, typename value_t>
__global__ __launch_bounds__(THREADS1) void BoundingBoxKernel(value_idx* restrict startd,
                                                              value_idx* restrict childd,
                                                              value_t* restrict massd,
                                                              value_t* restrict posxd,
                                                              value_t* restrict posyd,
                                                              value_t* restrict poszd,
                                                              value_t* restrict maxxd,
                                                              value_t* restrict maxyd,
                                                              value_t* restrict maxzd,
                                                              value_t* restrict minxd,
                                                              value_t* restrict minyd,
                                                              value_t* restrict minzd,
                                                              const value_idx EIGHT_NNODES,
                                                              const value_idx NNODES,
                                                              const value_idx N,
                                                              unsigned* restrict limiter,
                                                              value_t* restrict radiusd)
{
  value_t val, minx, maxx, miny, maxy, minz, maxz;
  __shared__ value_t sminx[THREADS1], smaxx[THREADS1], sminy[THREADS1], smaxy[THREADS1];
  __shared__ value_t  sminz[(dim == 2) ? THREADS1 : 1], smaxz[(dim == 2) ? THREADS1 : 1];

  // initialize with valid data (in case #bodies < #threads)
  minx = maxx = posxd[0];
  miny = maxy = posyd[0];
  if constexpr (dim == 3) minz = maxz = poszd[0];

  // scan all bodies
  const auto i   = threadIdx.x;
  const auto inc = THREADS1 * gridDim.x;
  for (auto j = i + blockIdx.x * THREADS1; j < N; j += inc) {
    val = posxd[j];
    if (val < minx)
      minx = val;
    else if (val > maxx)
      maxx = val;

    val = posyd[j];
    if (val < miny)
      miny = val;
    else if (val > maxy)
      maxy = val;

    if constexpr (dim == 3) {
      val = poszd[j];
      if (val < minz)
        minz = val;
      else if (val > maxz)
        maxz = val;
    }
  }

  // reduction in shared memory
  sminx[i] = minx;
  smaxx[i] = maxx;
  sminy[i] = miny;
  smaxy[i] = maxy;
  if constexpr (dim == 3) {
    sminz[i] = minz;
    smaxz[i] = maxz;
  }

  for (auto j = THREADS1 / 2; j > i; j /= 2) {
    __syncthreads();
    const auto k = i + j;
    sminx[i] = minx = fminf(minx, sminx[k]);
    smaxx[i] = maxx = fmaxf(maxx, smaxx[k]);
    sminy[i] = miny = fminf(miny, sminy[k]);
    smaxy[i] = maxy = fmaxf(maxy, smaxy[k]);
    if constexpr (dim == 3) {
      sminz[i] = minz = fminf(minz, sminz[k]);
      smaxz[i] = maxz = fmaxf(maxz, smaxz[k]);
    }
  }

  if (i == 0) {
    // write block result to global memory
    const auto k = blockIdx.x;
    minxd[k]     = minx;
    maxxd[k]     = maxx;
    minyd[k]     = miny;
    maxyd[k]     = maxy;
    if constexpr (dim == 3) {
      minzd[k]     = minz;
      maxzd[k]     = maxz;
    }
    __threadfence();

    const auto inc = gridDim.x - 1;
    if (inc != atomicInc(limiter, inc)) return;

    // I'm the last block, so combine all block results
    for (auto j = 0; j <= inc; j++) {
      minx = fminf(minx, minxd[j]);
      maxx = fmaxf(maxx, maxxd[j]);
      miny = fminf(miny, minyd[j]);
      maxy = fmaxf(maxy, maxyd[j]);
      if constexpr (dim == 3) {
        minz = fminf(minz, minzd[j]);
        maxz = fmaxf(maxz, maxzd[j]);
      }
    }

    // compute 'radius'
    val = fmaxf(maxx - minx, maxy - miny);
    if constexpr (dim == 3) {
      val = fmaxf(val, maxz - minz);
    }
    atomicExch(radiusd, val * 0.5f);

    massd[NNODES]  = -1.0f;
    startd[NNODES] = 0;
    posxd[NNODES]  = (minx + maxx) * 0.5f;
    posyd[NNODES]  = (miny + maxy) * 0.5f;
    if constexpr (dim == 3) {
    poszd[NNODES]  = (minz + maxz) * 0.5f;
    }

    constexpr auto dim2 = (dim == 2) ? 4 : 8;
#pragma unroll
    for (auto a = 0; a < dim2; a++)
      childd[EIGHT_NNODES + a] = -1;
  }
}

/**
 * Clear some of the state vectors up.
 */
template <typename value_idx>
__global__ __launch_bounds__(1024, 1) void ClearKernel1(value_idx* const restrict childd,
                                                        const value_idx EIGHT_NNODES,
                                                        const value_idx EIGHT_N)
{
  const auto inc = blockDim.x * gridDim.x;
  value_idx k    = (EIGHT_N & -32) + threadIdx.x + blockIdx.x * blockDim.x;
  if (k < EIGHT_N) k += inc;

// iterate over all cells assigned to thread
#pragma unroll
  for (; k < EIGHT_NNODES; k += inc)
    childd[k] = -1;
}

/**
 * Build the actual QuadTree.
 * See: https://iss.oden.utexas.edu/Publications/Papers/burtscher11.pdf
 */
template <int dim, typename value_idx, typename value_t>
__global__ __launch_bounds__(THREADS2) void TreeBuildingKernel(value_idx* const restrict childd,
                                                               const value_t* const restrict posxd,
                                                               const value_t* const restrict posyd,
                                                               const value_t* const restrict poszd,
                                                               const value_idx NNODES,
                                                               const value_idx N,
                                                               value_idx* const restrict bottomd,
                                                               const value_t* const restrict radiusd)
{
  const value_idx maxdepth = 32;
  value_idx j, depth;
  value_t x, y, z, r;
  value_t px, py, pz;
  value_idx ch, n, locked, patch;
  constexpr int dim2 = (dim == 2) ? 4 : 8;

  // cache root data
  const value_t radius = radiusd[0] * 0.5f;
  const value_t rootx  = posxd[NNODES];
  const value_t rooty  = posyd[NNODES];
  const value_t rootz  = (dim == 2) ? 0 : poszd[NNODES];

  value_idx skip          = 1;

  const auto inc = blockDim.x * gridDim.x;
  value_idx i    = threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all bodies assigned to thread
  while (i < N) {
    px = posxd[i];
    py = posyd[i];
    if constexpr (dim == 3) pz = poszd[i];
    if (skip != 0) {
      // new body, so start traversing at root
      skip  = 0;
      n     = NNODES;
      depth = 1;
      r     = radius;

      /* Select child node 'j'
                    rootx < px  rootx > px
       * rooty < py   1 -> 3    0 -> 2
       * rooty > py   1 -> 1    0 -> 0
       */
      x = rootx + ((rootx < px) ? (j = 1, r) : (j = 0, -r));
      y = rooty + ((rooty < py) ? (j |= 2, r) : (-r));
      if constexpr (dim == 3) z = rootz + ((rootz < pz) ? (j |= 4, r) : (-r));
    }

    // follow path to leaf cell
    while ((ch = childd[n * dim2 + j]) >= N) {
      depth++;
      n = ch;
      r *= 0.5f;

      x += ((x < px) ? (j = 1, r) : (j = 0, -r));
      y += ((y < py) ? (j |= 2, r) : (-r));
      if constexpr (dim == 3) z += ((z < pz) ? (j |= 4, r) : (-r));
    }

    // (ch)ild will be '-1' (nullptr), '-2' (locked), or an Integer corresponding to a body offset
    // in the lower [0, N) blocks of childd
    if (ch != -2) {
      // skip if child pointer was locked when we examined it, and try again later.
      locked = n * dim2 + j;
      // store the locked position in case we need to patch in a cell later.

      if (ch == -1) {
        // Child is a nullptr ('-1'), so we write our body index to the leaf, and move on to the
        // next body.
        if (atomicCAS(&childd[locked], (value_idx)-1, i) == -1) {
          i += inc;  // move on to next body
          skip = 1;
        }
      } else {
        // Child node isn't empty, so we store the current value of the child, lock the leaf, and
        // patch in a new cell
        if (ch == atomicCAS(&childd[locked], ch, (value_idx)-2)) {
          patch = -1;

          while (ch >= 0) {
            depth++;
            if (depth > maxdepth) {
              // Maximum depth exceeded (bodies are too close together)
              printf("Maximum depth\n");
              break;
              //asm("trap;");
            }

            const value_idx cell = atomicAdd(bottomd, (value_idx)-1) - 1;
            if (cell <= N) {
              // Out of cell memory
              printf("Out of cell memory\n");
              break;
              //asm("trap;");
            }

            if (patch != -1) childd[n * dim2 + j] = cell;

            if (cell > patch) patch = cell;

            // Insert migrated child node
            j = (x < posxd[ch]) ? 1 : 0;
            if (y < posyd[ch]) j |= 2;
            if constexpr (dim == 3) {
              if (z < poszd[ch]) j |= 4;
            }

            childd[cell * dim2 + j] = ch;
            n                       = cell;
            r                       *= 0.5f;

            x += ((x < px) ? (j = 1, r) : (j = 0, -r));
            y += ((y < py) ? (j |= 2, r) : (-r));
            if constexpr (dim == 3) z += ((z < pz) ? (j |= 4, r) : (-r));

            ch = childd[n * dim2 + j];
            //if (r <= 1e-10) { printf("radius\n"); break;}
          }

          childd[n * dim2 + j] = i;

          i += inc;  // move on to next body
          skip = 2;
        }
      }
    }

    __syncthreads(); // optional barrier for performance
    __threadfence();

    if (skip == 2) childd[locked] = patch;
  }
}

/**
 * Clean more state vectors.
 */
template <typename value_idx, typename value_t>
__global__ __launch_bounds__(1024, 1) void ClearKernel2(value_idx* const restrict startd,
                                                        value_t* const restrict massd,
                                                        const value_idx NNODES,
                                                        const value_idx* const restrict bottomd)
{
  const auto bottom = bottomd[0];
  const auto inc    = blockDim.x * gridDim.x;
  auto k            = (bottom & -32) + threadIdx.x + blockIdx.x * blockDim.x;
  if (k < bottom) k += inc;

// iterate over all cells assigned to thread
#pragma unroll
  for (; k < NNODES; k += inc) {
    massd[k]  = -1.0f;
    startd[k] = -1;
  }
}

/**
 * Summarize the KD Tree via cell gathering
 */
template <int dim, typename value_idx, typename value_t>
__global__ __launch_bounds__(THREADS3,
                             FACTOR3) void SummarizationKernel(value_idx* const restrict countd,
                                                               const value_idx* const restrict childd,
                                                               volatile value_t* const restrict massd,
                                                               value_t* const restrict posxd,
                                                               value_t* const restrict posyd,
                                                               value_t* const restrict poszd,
                                                               const value_idx NNODES,
                                                               const value_idx N,
                                                               const value_idx* const restrict bottomd)
{
  value_t cm, px, py, pz;
  constexpr int dim2 = (dim == 2) ? 4 : 8;
  __shared__ value_idx child[THREADS3 * dim2];
  __shared__ value_t mass[THREADS3 * dim2];

  const auto bottom = bottomd[0];
  const auto inc    = blockDim.x * gridDim.x;
  auto k            = (bottom & -32) + threadIdx.x + blockIdx.x * blockDim.x;
  if (k < bottom) k += inc;

  const auto restart = k;

  for (int j = 0; j < 3; j++)  // wait-free pre-passes
  {
    // iterate over all cells assigned to thread
    while (k <= NNODES) {
      if (massd[k] < 0.0f) {
        for (int i = 0; i < dim2; i++) {
          const auto ch                     = childd[k * dim2 + i];
          child[i * THREADS3 + threadIdx.x] = ch;

          if ((ch >= N) and ((mass[i * THREADS3 + threadIdx.x] = massd[ch]) < 0))
            goto CONTINUE_LOOP;
        }

        // all children are ready
        cm       = 0.0f;
        px       = 0.0f;
        py       = 0.0f;
        pz       = 0.0f;
        auto cnt = 0;

#pragma unroll
        for (int i = 0; i < dim2; i++) {
          const int ch = child[i * THREADS3 + threadIdx.x];
          if (ch >= 0) {
            const value_t m = (ch >= N) ? (cnt += countd[ch], mass[i * THREADS3 + threadIdx.x])
                                        : (cnt++, massd[ch]);
            // add child's contribution
            cm += m;
            px += posxd[ch] * m;
            py += posyd[ch] * m;
            if constexpr (dim == 3) pz += poszd[ch] * m;
          }
        }

        countd[k]       = cnt;
        const value_t m = 1.0f / cm;
        posxd[k]        = px * m;
        posyd[k]        = py * m;
        if constexpr (dim == 3) poszd[k] = pz * m;
        __threadfence();  // make sure data are visible before setting mass
        massd[k] = cm;
      }

    CONTINUE_LOOP:
      k += inc;  // move on to next cell
    }
    k = restart;
  }

  int j = 0;
  // iterate over all cells assigned to thread
  while (k <= NNODES) {
    if (massd[k] >= 0) {
      k += inc;
      continue;
    }

    if (j == 0) {
      j = dim2;
      for (int i = 0; i < dim2; i++) {
        const auto ch = childd[k * dim2 + i];

        child[i * THREADS3 + threadIdx.x] = ch;
        if ((ch < N) or ((mass[i * THREADS3 + threadIdx.x] = massd[ch]) >= 0)) j--;
      }
    } else {
      j = dim2;
      for (int i = 0; i < dim2; i++) {
        const auto ch = child[i * THREADS3 + threadIdx.x];

        if ((ch < N) or (mass[i * THREADS3 + threadIdx.x] >= 0) or
            ((mass[i * THREADS3 + threadIdx.x] = massd[ch]) >= 0))
          j--;
      }
    }

    if (j == 0) {
      // all children are ready
      cm       = 0.0f;
      px       = 0.0f;
      py       = 0.0f;
      pz       = 0.0f;
      auto cnt = 0;

#pragma unroll
      for (int i = 0; i < dim2; i++) {
        const auto ch = child[i * THREADS3 + threadIdx.x];
        if (ch >= 0) {
          const auto m =
            (ch >= N) ? (cnt += countd[ch], mass[i * THREADS3 + threadIdx.x]) : (cnt++, massd[ch]);
          // add child's contribution
          cm += m;
          px += posxd[ch] * m;
          py += posyd[ch] * m;
          if constexpr (dim == 3) pz += poszd[ch] * m;
        }
      }

      countd[k]       = cnt;
      const value_t m = 1.0f / cm;
      posxd[k]        = px * m;
      posyd[k]        = py * m;
      if constexpr (dim == 3) poszd[k] = pz * m;
      __threadfence();
      massd[k]        = cm;
    k += inc;
    }
  }
}

/**
 * Sort the cells
 */
template <int dim, typename value_idx>
__global__ __launch_bounds__(THREADS4, FACTOR4) void SortKernel(value_idx* restrict sortd,
                                                                const value_idx* restrict countd,
                                                                volatile value_idx* restrict startd,
                                                                value_idx* restrict childd,
                                                                const value_idx NNODES,
                                                                const value_idx N,
                                                                const value_idx* restrict bottomd)
{
  const value_idx bottom = bottomd[0];
  const value_idx dec    = blockDim.x * gridDim.x;
  value_idx k            = NNODES + 1 - dec + threadIdx.x + blockIdx.x * blockDim.x;
  value_idx start;
  value_idx limiter = 0;
  constexpr int dim2 = (dim == 2) ? 4 : 8;

  // iterate over all cells assigned to thread
  while (k >= bottom) {
    // To control possible infinite loops
    if (++limiter > NNODES) break;

    // Not a child so skip
    if ((start = startd[k]) < 0) continue;

    int j = 0;
    for (int i = 0; i < dim2; i++) {
      const auto ch = childd[k * dim2 + i];
      if (ch >= 0) {
        if (i != j) {
          // move children to front (needed later for speed)
          childd[k * dim2 + i] = -1;
          childd[k * dim2 + j] = ch;
        }
        if (ch >= N) {
          // child is a cell
          startd[ch] = start;
          start += countd[ch];  // add #bodies in subtree
        } else if (start <= NNODES and start >= 0) {
          // child is a body
          sortd[start++] = ch;
        }
        j++;
      }
    }
    k -= dec;  // move on to next cell
  }
}

/**
 * Calculate the repulsive forces using the KD Tree
 */
template <int dim, typename value_idx, typename value_t>
__global__ __launch_bounds__(
  THREADS5, 1) void RepulsionKernel(const float epssqd,  // correction for zero distance
                                    const value_idx* restrict sortd,
                                    const value_idx* restrict childd,
                                    const value_t* restrict massd,
                                    const value_t* restrict posxd,
                                    const value_t* restrict posyd,
                                    const value_t* restrict poszd,
                                    value_t* restrict velxd,
                                    value_t* restrict velyd,
                                    value_t* restrict velzd,
                                    value_t* restrict Z_norm,
                                    const value_t itolsq,
                                    const value_idx NNODES,
                                    const value_idx EIGHT_NNODES,
                                    const value_idx N,
                                    const value_t* restrict radiusd_squared)
{
  // Return if max depth is too deep
  // Not possible since I limited it to 32
  // if (maxdepthd[0] > 32)
  // {
  //   atomicExch(errd, max_depth);
  //   return;
  // }
  const value_t EPS_PLUS_1 = epssqd + 1.0f;
  const int warpsize = 32;

  __shared__ value_idx pos[THREADS5], node[THREADS5];
  __shared__ value_t dq[THREADS5];

  if (threadIdx.x == 0) {
    const auto max_depth = 32;
    dq[0]                = 4 * radiusd_squared[0] * itolsq;

    for (auto i = 1; i < max_depth; i++) {
      dq[i] = dq[i - 1] * 0.25f;
      dq[i - 1] += epssqd;
    }
    dq[max_depth - 1] += epssqd;

    // Add one so EPS_PLUS_1 can be compared
    for (auto i = 0; i < max_depth; i++)
      dq[i] += 1.0f;
  }

  __syncthreads();
  // figure out first thread in each warp (lane 0)
  // const int base = threadIdx.x / 32;
  // const int sbase = base * 32;
  const int sbase            = (threadIdx.x / warpsize) * warpsize;
  const bool SBASE_EQ_THREAD = (sbase == threadIdx.x);

  const int diff = threadIdx.x - sbase;
  // make multiple copies to avoid index calculations later
  // Always true
  // if (diff < 32)
  dq[diff + sbase] = dq[diff];

  //__syncthreads();
  __threadfence_block();

  // iterate over all bodies assigned to thread
  constexpr int dim2 = (dim == 2) ? 4 : 8;
  const auto MAX_SIZE = EIGHT_NNODES + dim2;

  for (auto k = threadIdx.x + blockIdx.x * blockDim.x; k < N; k += blockDim.x * gridDim.x) {
    const auto i = sortd[k];  // get permuted/sorted index
    // cache position info
    if (i < 0 or i >= MAX_SIZE) continue;

    const value_t px = posxd[i];
    const value_t py = posyd[i];
    const value_t pz = (dim == 2) ? 0 : poszd[i];

    value_t ax      = 0.0f;
    value_t ay      = 0.0f;
    value_t az      = 0.0f;
    value_t normsum = 0.0f;
    // initialize iteration stack, i.e., push root node onto stack
    int depth = sbase;

    if (SBASE_EQ_THREAD == true) {
      pos[sbase]  = 0;
      node[sbase] = EIGHT_NNODES;
    }

    do {
      // stack is not empty
      auto pd = pos[depth];
      auto nd = node[depth];

      while (pd < dim2) {
        const auto index = nd + pd++;
        if (index < 0 or index >= MAX_SIZE) break;

        const auto n = childd[index];  // load child pointer

        // Non child
        if (n < 0 or n > NNODES) break;

        const value_t dx   = px - posxd[n];
        const value_t dy   = py - posyd[n];
        const value_t dz   = (dim == 2) ? 0 : pz - poszd[n];
        const value_t dxy1 = dx * dx + dy * dy + dz * dz + EPS_PLUS_1;

        if ((n < N) or __all_sync(0xffffffff, dxy1 >= dq[depth])) {
          const value_t tdist_2 = rsqrtf(dxy1);
          const value_t tdist_3 = massd[n] * tdist_2 * tdist_2 * tdist_2;
          normsum += tdist_3 * dxy1;
          ax += dx * tdist_3;
          ay += dy * tdist_3;
          if constexpr (dim == 3) az += dz * tdist_3;
        } else {
          // push cell onto stack
          if (SBASE_EQ_THREAD == true) {
            pos[depth]  = pd;
            node[depth] = nd;
          }
          depth++;
          pd = 0;
          nd = n * dim2;
        }
      }

    } while (--depth >= sbase);  // done with this level

    /*
    TODO: accVeld
    */

    // update velocity
    velxd[i] += ax;
    velyd[i] += ay;
    if constexpr (dim == 2) velzd[i] += az;
    atomicAdd(Z_norm, normsum);
  }
}

/**
 * Fast attractive kernel. Uses COO matrix.
 */
template <int dim, typename value_idx, typename value_t>
__global__ void attractive_kernel_bh(const value_t* restrict VAL,
                                     const value_idx* restrict COL,
                                     const value_idx* restrict ROW,
                                     const value_t* restrict Y1,
                                     const value_t* restrict Y2,
                                     const value_t* restrict Y3,
                                     value_t* restrict attract1,
                                     value_t* restrict attract2,
                                     value_t* restrict attract3,
                                     value_t* restrict Qs,
                                     const value_idx NNZ,
                                     const value_t dof)
{
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= NNZ) return;
  const auto i = ROW[index];
  const auto j = COL[index];

  const value_t y1d = Y1[i] - Y1[j];
  const value_t y2d = Y2[i] - Y2[j];
  const value_t y3d = (dim == 2) ? 0 : Y3[i] - Y3[j];
  value_t dist      = y1d * y1d + y2d * y2d + y3d * y3d;
  // As a sum of squares, SED is mathematically >= 0. There might be a source of
  // NaNs upstream though, so until we find and fix them, enforce that trait.
  if (!(dist >= 0)) dist = 0.0f;

  const value_t P  = VAL[index];
  const value_t Q  = compute_q(dist, dof);
  const value_t PQ = P * Q;

  // Apply forces
  atomicAdd(&attract1[i], PQ * y1d);
  atomicAdd(&attract2[i], PQ * y2d);
  if constexpr (dim == 3) atomicAdd(&attract3[i], PQ * y3d);

  if (Qs) {  // when computing KL div
    Qs[index] = Q;
  }

  // TODO: Convert attractive forces to CSR format
}

/**
 * Apply gradient updates.
 */
template <int dim, typename value_idx, typename value_t>
__global__ __launch_bounds__(THREADS6, 1) void IntegrationKernel(const float eta,
                                                                 const float momentum,
                                                                 const float exaggeration,
                                                                 value_t* restrict Y1,
                                                                 value_t* restrict Y2,
                                                                 value_t* restrict Y3,
                                                                 const value_t* restrict attract1,
                                                                 const value_t* restrict attract2,
                                                                 const value_t* restrict attract3,
                                                                 const value_t* restrict repel1,
                                                                 const value_t* restrict repel2,
                                                                 const value_t* restrict repel3,
                                                                 value_t* restrict gains1,
                                                                 value_t* restrict gains2,
                                                                 value_t* restrict gains3,
                                                                 value_t* restrict old_forces1,
                                                                 value_t* restrict old_forces2,
                                                                 value_t* restrict old_forces3,
                                                                 const value_t* restrict Z,
                                                                 const value_idx N)
{
  value_t ux, uy, uz, gx, gy, gz;

  // iterate over all bodies assigned to thread
  const auto inc       = blockDim.x * gridDim.x;
  const value_t Z_norm = Z[0];

  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += inc) {
    const value_t dx = exaggeration * attract1[i] - Z_norm * repel1[i];
    const value_t dy = exaggeration * attract2[i] - Z_norm * repel2[i];
    const value_t dz = (dim == 2) ? 0 : exaggeration * attract3[i] - Z_norm * repel3[i];

    if (signbit(dx) != signbit(ux = old_forces1[i]))
      gx = gains1[i] + 0.2f;
    else
      gx = gains1[i] * 0.8f;
    if (gx < 0.01f) gx = 0.01f;

    if (signbit(dy) != signbit(uy = old_forces2[i]))
      gy = gains2[i] + 0.2f;
    else
      gy = gains2[i] * 0.8f;
    if (gy < 0.01f) gy = 0.01f;
  
    if constexpr (dim == 3) {
      if (signbit(dz) != signbit(uz = old_forces3[i]))
        gz = gains3[i] + 0.2f;
      else
        gz = gains3[i] * 0.8f;
      if (gz < 0.01f) gz = 0.01f;
    }

    gains1[i] = gx;
    gains2[i] = gy;
    if constexpr (dim == 3) gains3[i] = gz;

    old_forces1[i] = ux = momentum * ux - eta * gx * dx;
    old_forces2[i] = uy = momentum * uy - eta * gy * dy;
    if constexpr (dim == 3) old_forces3[i] = uz = momentum * uz - eta * gz * dz;

    Y1[i] += ux;
    Y2[i] += uy;
    if constexpr (dim == 3) Y3[i] += uz;
  }
}

}  // namespace BH
}  // namespace TSNE
}  // namespace ML
