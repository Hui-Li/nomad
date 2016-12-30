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

#ifndef NOMAD_POOL_HPP_
#define NOMAD_POOL_HPP_

#include "nomad.hpp"
#include "tbb/tbb.h"

namespace nomad { 

  struct ColumnData {

  public:
    int col_index_;
    long flag_;
    int *perm_;
    int pos_;
    scalar *values_;

  };

  class Pool {

  public:
    Pool(int dim, int num_threads, int init_size) : 
    queue_(),
    dim_(dim),
    num_threads_(num_threads),
    alloc_(),
    int_alloc_(),
    scalar_alloc_()
    {
      
    }

    ~Pool() {

      while (true) {

	ColumnData *p_col=nullptr;
	bool succeed = queue_.try_pop(p_col);
	if (succeed) {	 
	  int_alloc_.deallocate(p_col->perm_, num_threads_);
	  scalar_alloc_.deallocate(p_col->values_, dim_);
	  alloc_.destroy(p_col);
	  alloc_.deallocate(p_col, 1);
	}		
	else {
	  break;
	}

      }

    }

    ColumnData *allocate() {

      ColumnData *ret = alloc_.allocate(1);
      ret->perm_ = int_alloc_.allocate(num_threads_);
      ret->values_ = scalar_alloc_.allocate(dim_);
      return ret;

    }

    void push(ColumnData *p_col) {
      queue_.push(p_col);
    }

    ColumnData *pop() {
      
      ColumnData *ret=nullptr;
      bool succeed = queue_.try_pop(ret);
      
      if (succeed) {
	return ret;
      }
      else {
	return allocate();
      }

    }

  private:
    tbb::concurrent_queue<ColumnData *, callocator<ColumnData *> > queue_;
    int dim_;
    int num_threads_;
    callocator< ColumnData > alloc_;
    callocator< int > int_alloc_;
    callocator< scalar > scalar_alloc_;
  };

}

#endif
