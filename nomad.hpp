/*
 * Copyright (c) 2013 Hyokun Yun
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 */
#ifndef NOMAD_NOMAD_HPP_
#define NOMAD_NOMAD_HPP_

#define MAT_FILE_CLASSID 1211216    /* used to indicate matrices in binary files */
#define LONG_FILE_CLASSID 1015    /* used to indicate matrices in binary files with large number of nonzeroes */

// BUGBUG: there should be better way than this
#undef TBB_IMPLEMENT_CPP0X
#define TBB_IMPLEMENT_CPP0X 1


#include <random>
typedef std::mt19937_64 rng_type;

#include "tbb/scalable_allocator.h"
#include "tbb/cache_aligned_allocator.h"

template <typename T>
using sallocator = tbb::scalable_allocator<T>;

template <typename T>
using callocator = tbb::cache_aligned_allocator<T>;

using real=double;

// void SYByteSwapInt(int *buff,int n)
// {
//   int  i,j,tmp;
//   char *ptr1,*ptr2 = (char*)&tmp;
//   for (j=0; j<n; j++) {
//     ptr1 = (char*)(buff + j);
//     for (i=0; i<(int)sizeof(int); i++) {
//       ptr2[i] = ptr1[sizeof(int)-1-i];
//     }
//     buff[j] = tmp;
//   }
// }

// void SYByteSwapScalar(scalar *buff,int n)
// {
//   int    i,j;
//   double tmp,*buff1 = (double*)buff;
//   char   *ptr1,*ptr2 = (char*)&tmp;
//   for (j=0; j<n; j++) {
//     ptr1 = (char*)(buff1 + j);
//     for (i=0; i<(int)sizeof(double); i++) {
//       ptr2[i] = ptr1[sizeof(double)-1-i];
//     }
//     buff1[j] = tmp;
//   }
// }


#endif
