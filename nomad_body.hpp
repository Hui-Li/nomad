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

#ifndef NOMAD_NOMAD_BODY_HPP_
#define NOMAD_NOMAD_BODY_HPP_

#include "nomad.hpp"
#include "nomad_option.hpp"
#include "pool.hpp"

#include "tbb/tbb.h"
#include <tbb/compat/thread>

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <condition_variable>


#include "mpi.h"

const int UNITS_PER_MSG = 100;

using std::cout; using std::cerr; using std::endl; 
using std::ifstream; using std::ofstream;
using std::endl;
using std::ios;

using std::vector;
using std::pair;
using std::string;

using tbb::atomic;
using tbb::tick_count;

using nomad::ColumnData;

typedef tbb::concurrent_queue<ColumnData *, callocator<ColumnData *> > colque;

using nomad::NomadOption;

class NomadBody {

protected:
  virtual NomadOption *create_option() = 0;

  virtual bool load_train(NomadOption& option, 
			  int part_index, int num_parts, 
			  int &min_row_index,
			  int &local_num_rows,
			  vector<int, sallocator<int> > &col_offset,
			  vector<int, sallocator<int> > &row_idx,
			  vector<scalar, sallocator<scalar> > &row_val
			  ) = 0;

  virtual bool load_test(NomadOption& option, 
			 int part_index, int num_parts, 
			 int &min_row_index,
			 int &local_num_rows,
			 vector<int, sallocator<int> > &col_offset,
			 vector<int, sallocator<int> > &row_idx,
			 vector<scalar, sallocator<scalar> > &row_val
			 ) = 0;

  
public:
  int run(int argc, char **argv) {

    int numtasks, rank, hostname_len;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    // check whether MPI provides multiple threading
    int mpi_thread_provided;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &mpi_thread_provided);
    if (mpi_thread_provided != MPI_THREAD_MULTIPLE) {
      cerr << "MPI multiple thread not provided!!! ("
	   << mpi_thread_provided << " != " << MPI_THREAD_MULTIPLE << ")" << endl;
      exit(1);
    }


    // retrieve MPI task info
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Get_processor_name(hostname, &hostname_len);
  

    nomad::NomadOption *p_option = create_option();
    nomad::NomadOption &option = *p_option;

    if (option.parse_command(argc, argv) == false) {
      return 1;
    }

    cout << boost::format("processor name: %s, number of tasks: %d, rank: %d")
      % hostname % numtasks % rank << endl;

    const int num_parts = numtasks * option.num_threads_;

    cout << "number of threads: " << option.num_threads_
	 << ", number of parts: " << num_parts << endl;

    // read number of columns 
    int global_num_cols = option.get_num_cols();


    // create a column pool with big enough size
    // this serves as a memory pool. global_num_cols * 3 / num_parts is arbitrary big enough number.
    // when the capacity is exceeded, it automatically assigns additional memory. 
    // therefore no need to worry too much
    nomad::Pool column_pool(option.latent_dimension_, option.num_threads_,
			    std::min(global_num_cols * 3 / num_parts, global_num_cols));

    // setup initial queues of columns
    // each thread owns each queue with corresponding access
    colque *job_queues = callocator<colque>().allocate(option.num_threads_);  
    for (int i=0; i < option.num_threads_; i++) {
      callocator<colque>().construct(job_queues + i);
    }

    // a queue of columns to be sent to other machines via network
    colque send_queue;

    // count the number of threads in the machine which initial setup for training is done
    atomic<int> count_setup_threads;
    count_setup_threads = 0;

    // this flag will be turned on when all threads are ready for training
    atomic<bool> flag_train_ready;
    flag_train_ready = false;

    // this flag will be used to send signals to all threads that it has to stop training
    atomic<bool> flag_train_stop;
    flag_train_stop = false;
  
    // this flag will be turned on when all threads are ready for testing
    atomic<bool> flag_test_ready;
    flag_test_ready = false;

    // this flag will be used to send signals to all threads that it has to stop testing
    atomic<bool> flag_test_stop;
    flag_test_stop = false;


    // distribution used to initialize parameters
    // distribution is taken from Hsiang-Fu's implementation of DSGD
    std::uniform_real_distribution<> init_dist(0, 1.0/sqrt(option.latent_dimension_));

    // maintain the number of updates for each thread
    atomic<long long> *num_updates = callocator< atomic<long long> >().allocate(option.num_threads_);
    for (int i=0; i < option.num_threads_; i++) {
      callocator< atomic<long long> >().construct(num_updates + i);
      num_updates[i] = 0;
    }

    // also maintain a number of pop failures from job queue
    atomic<long long> *num_failures = callocator< atomic<long long> >().allocate(option.num_threads_);
    for (int i=0; i < option.num_threads_; i++) {
      callocator< atomic<long long> >().construct(num_failures + i);
      num_failures[i] = 0;
    }

    // used to compute the number of empty columns inside a machine
    // BUGBUG: right now it does not have any purpose other than computing statistics
    // we may enhance the efficiency of communication by taking advantage of this information
    atomic<bool> *is_column_empty = callocator< atomic<bool> >().allocate(global_num_cols);
    for (int i=0; i < global_num_cols; i++) {
      is_column_empty[i] = true;
    }

    // these arrays will be used to calculate test error
    // each thread will calculate test error on its own, and the results will be aggregated
    int *train_count_errors = callocator<int>().allocate(option.num_threads_);
    real *train_sum_errors = callocator<real>().allocate(option.num_threads_);
    int *test_count_errors = callocator<int>().allocate(option.num_threads_);
    real *test_sum_errors = callocator<real>().allocate(option.num_threads_);

    std::fill_n(train_count_errors, option.num_threads_, 0);
    std::fill_n(train_sum_errors, option.num_threads_, 0.0);
    std::fill_n(test_count_errors, option.num_threads_, 0);
    std::fill_n(test_sum_errors, option.num_threads_, 0.0);

    // array used to remember the sizes of send_queue in each machine
    atomic<int> *queue_current_sizes = callocator< atomic<int> >().allocate( numtasks );
    for (int i=0; i < numtasks; i++) {
      queue_current_sizes[i] = 0;
    }
    // we try to bound the size of send_queue's by this number
    const int queue_upperbound = global_num_cols * 4 / numtasks;


    std::mutex print_mutex;
    std::condition_variable print_waiter;

    /////////////////////////////////////////////////////////
    // Define Updater Thread
    /////////////////////////////////////////////////////////

    tbb::atomic<int> wait_number;
    wait_number = 0;

    std::function<void(int)> updater_func = [&](int thread_index)->void {


      int part_index = rank * option.num_threads_ + thread_index;
      cout << boost::format("rank: %d, thread_index: %d, part_index: %d") % rank % thread_index % part_index << endl;


      /////////////////////////////////////////////////////////
      // Read Data
      /////////////////////////////////////////////////////////

      // each thread reads its own portion of data and stores in CSC format
      vector<int, sallocator<int> > train_col_offset, test_col_offset;
      vector<int, sallocator<int> > train_row_idx, test_row_idx;
      vector<scalar, sallocator<scalar> > train_row_val, test_row_val;

      int local_num_rows;
      int min_row_index;

      bool succeed = load_train(option, part_index, num_parts,
				min_row_index,
				local_num_rows,
				train_col_offset, train_row_idx, train_row_val);
      if (succeed == false) {
	cerr << "error in reading training file" << endl;
	exit(11);
      }

      succeed = load_test(option, part_index, num_parts,
			  min_row_index,
			  local_num_rows,
			  test_col_offset, test_row_idx, test_row_val);
      if (succeed == false) {
	cerr << "error in reading testing file" << endl;
	exit(11);
      }

      for (int i=0; i < global_num_cols; i++) {
	if (train_col_offset[i+1] > train_col_offset[i]) {
	  is_column_empty[i].compare_and_swap(false, true);
	}
      }


      /////////////////////////////////////////////////////////
      // Initialize Data Structure
      /////////////////////////////////////////////////////////

      // now assign parameters for rows
      scalar *latent_rows = sallocator<scalar>().allocate(local_num_rows * option.latent_dimension_);

      // initialize random number generator
      rng_type rng(option.seed_ + rank * 131 + thread_index + 1);
      for (int i=0; i < local_num_rows * option.latent_dimension_; i++) {
	latent_rows[i] = init_dist(rng);
      }
    
      int *col_update_counts = sallocator<int>().allocate(global_num_cols);
      std::fill_n(col_update_counts, global_num_cols, 0);
    

      // copy some essential parameters explicitly

      const int dim = option.latent_dimension_;
      const scalar learn_rate = option.learn_rate_;
      const scalar decay_rate = option.decay_rate_;
      const scalar par_lambda = option.par_lambda_;
      const int num_threads = option.num_threads_;
      const int num_reuse = option.num_reuse_;
    
      long long local_num_updates = 0;
      long long local_num_failures = 0;


      // notify that the thread is ready to run
      count_setup_threads++;

      for (unsigned int timeout_iter = 0; timeout_iter < option.timeouts_.size(); timeout_iter++) {

	cout << "thread: " << thread_index << " ready to train!" << endl;

	// wait until all threads are ready
	while (flag_train_ready == false) {
	  std::this_thread::yield();
	}

	/////////////////////////////////////////////////////////
	// Training
	/////////////////////////////////////////////////////////

	while (flag_train_stop == false) {

	  ColumnData *p_col;
	  bool pop_succeed = job_queues[thread_index].try_pop(p_col);      

	  if (pop_succeed) { // there was an available column in job queue to process
	
	    const int col_index = p_col->col_index_;

	    const scalar step_size = learn_rate * 1.5 / 
	      (1.0 + decay_rate * pow(col_update_counts[col_index] + 1, 1.5));

	    scalar *col = p_col->values_;

	    // for each data point
	    for (int offset = train_col_offset[col_index];
		 offset < train_col_offset[col_index + 1]; offset++) {

	      // retrieve the point
	      int row_index = train_row_idx[offset];
	      scalar *row = latent_rows + dim * row_index;
	    
	      scalar cur_error = std::inner_product(col, col + dim, row, -train_row_val[offset]);

	      // calculate error	  
	      // scalar cur_error = -train_row_val[offset];
	      // for (int i=0; i < dim; i++) {
	      // 	cur_error += col[i] * row[i];
	      // }

	      // update both row and column
	      for (int i=0; i < dim; i++) {
		scalar tmp = row[i];
	      
		row[i] -= step_size * (cur_error * col[i] + par_lambda * tmp);
		col[i] -= step_size * (cur_error * tmp + par_lambda * col[i]);
	      }
	    
	      local_num_updates++;
	    
	    }
	  
	    col_update_counts[col_index]++;
	  
	    // send to the next thread
	    p_col->pos_++;
	    // if the column was circulated in every thread inside the machine, send to another machine
	    if (p_col->pos_ >= num_threads * num_reuse) {
	      // BUGBUG: now treating one machine case as special.. should I continue doing this?
	      if (numtasks == 1) {
		p_col->pos_ = 0;
		job_queues[p_col->perm_[p_col->pos_ % num_threads]].push(p_col);
	      }
	      else {
		send_queue.push(p_col);
	      }
	    }
	    else {
	      job_queues[p_col->perm_[p_col->pos_ % num_threads]].push(p_col);
	    }

	  }
	  else {
	    local_num_failures++;
	    std::this_thread::yield();
	  }
	
	}

	num_updates[thread_index] = local_num_updates;
	num_failures[thread_index] = local_num_failures;

	while (flag_test_ready == false) {
	  std::this_thread::yield();
	}

	/////////////////////////////////////////////////////////
	// Testing
	/////////////////////////////////////////////////////////

	int num_col_processed = 0;

	real train_sum_squared_error = 0.0;
	int train_count_error = 0;

	real test_sum_squared_error = 0.0;
	int test_count_error = 0;

	int monitor_num = 0;
	tbb::tick_count start_time = tbb::tick_count::now();

	// test until every column is processed
	while (num_col_processed < global_num_cols) {
      
	  double elapsed_seconds = (tbb::tick_count::now() - start_time).seconds();
	  if (monitor_num < elapsed_seconds) {
	    cout << "test updater alive," << rank << "," 
		 << monitor_num << "," << num_col_processed << "/" << global_num_cols << "" << endl;
	    monitor_num++;
	  }

	  ColumnData *p_col;
      
	  if (job_queues[thread_index].try_pop(p_col)) {
	
	    scalar *col = p_col->values_;
	    const int col_index = p_col->col_index_;

	    // for each training data point
	    for (int offset = train_col_offset[col_index];
		 offset < train_col_offset[col_index + 1]; offset++) {

	      // retrieve the point
	      int row_index = train_row_idx[offset];
	      scalar *row = latent_rows + dim * row_index;

	      // calculate error	  
	      scalar cur_error = -train_row_val[offset];
	      for (int i=0; i < dim; i++) {
		cur_error += col[i] * row[i];
	      }

	      train_sum_squared_error += cur_error * cur_error;
	      train_count_error++;

	    }

	    // for each test data point
	    for (int offset = test_col_offset[col_index];
		 offset < test_col_offset[col_index + 1]; offset++) {

	      // retrieve the point
	      int row_index = test_row_idx[offset];
	      scalar *row = latent_rows + dim * row_index;

	      // calculate error	  
	      scalar cur_error = -test_row_val[offset];
	      for (int i=0; i < dim; i++) {
		cur_error += col[i] * row[i];
	      }

	      test_sum_squared_error += cur_error * cur_error;
	      test_count_error++;

	    }

	    if (thread_index < num_threads - 1) {
	      job_queues[thread_index + 1].push(p_col);
	    }
	    else {
	      send_queue.push(p_col);
	    }

	    num_col_processed++;

	  }
	  else {
	    std::this_thread::yield();
	  }

	}

	train_count_errors[thread_index] = train_count_error;
	train_sum_errors[thread_index] = train_sum_squared_error;

	test_count_errors[thread_index] = test_count_error;
	test_sum_errors[thread_index] = test_sum_squared_error;

	// notify that this thread has finished testing
	count_setup_threads++;

      }

      // print to the file
      if (option.output_path_.length() > 0) { 

	while (wait_number < part_index % option.num_threads_) {
	  std::this_thread::yield();
	}

	ofstream::openmode mode = (part_index % option.num_threads_ == 0) ? ofstream::out : (ofstream::out | ofstream::app);
	ofstream ofs(option.output_path_ + boost::lexical_cast<string>(rank), mode);
	
	cout << "min_row_index: " << min_row_index << endl;
	for (int i=0; i < local_num_rows; i++) {
	  scalar *row = latent_rows + dim * i;
	  ofs << "row," << (min_row_index + i);
	  for (int t=0; t < dim; t++) {
	    ofs << "," << row[t];
	  }
	  ofs << endl;
	}
	ofs.close();
	
	wait_number++;

      }

      sallocator<int>().deallocate(col_update_counts, global_num_cols);

      sallocator<scalar>().deallocate(latent_rows, local_num_rows * option.latent_dimension_);

      return;

    };



    // create and run updater threads
    std::thread* updater_threads = callocator<std::thread>().allocate(option.num_threads_);
    for (int i=0; i < option.num_threads_; i++) {
      callocator<std::thread>().construct(updater_threads + i, updater_func, i);
    }
    while (count_setup_threads < option.num_threads_) {
      // wait until data loading and initializaiton of rows are done in every updater thread
      std::this_thread::yield();
    }


    /////////////////////////////////////////////////////////
    // Initialize Columns
    /////////////////////////////////////////////////////////

    rng_type rng(option.seed_ + rank * 131 + 139);

    int columns_per_machine = global_num_cols/numtasks + ((global_num_cols % numtasks > 0) ? 1 : 0);
    int col_start = columns_per_machine * rank;
    int col_end = std::min(columns_per_machine * (rank + 1), global_num_cols);

    // generate columns
    for (int i=col_start; i < col_end; i++) {
    
      // create additional RNG, to make it identical to other programs
      rng_type rng_temp(option.seed_ + rank + 137);

      // create a column
      ColumnData *p_col = column_pool.pop();
      p_col->col_index_ = i;
      p_col->flag_ = 0;
      p_col->pos_ = 0;
      // create initial permutation for the column
      for (int j=0; j < option.num_threads_; j++) {
	p_col->perm_[j] = j;
      }
      std::shuffle(p_col->perm_, p_col->perm_ + option.num_threads_, rng_temp);

      // initialize parameter
      for (int j=0; j < option.latent_dimension_; j++) {
	p_col->values_[j] = init_dist(rng);
      }

      // push to the job queue
      job_queues[p_col->perm_[p_col->pos_]].push(p_col);
    }

    // define constants needed for network communication
    // col_index + vector
    const int unit_bytenum = sizeof(int) + sizeof(long) + sizeof(scalar) * option.latent_dimension_;
    // current queue size + number of columns + columns
    const int msg_bytenum = sizeof(int) + sizeof(int) + unit_bytenum * UNITS_PER_MSG;


    long long local_send_count = 0;

    for (double ttt : option.timeouts_) {
      cout << "timeout: " << ttt << endl;
    }

    // save columns here, and push to job_queues again before next train starts
    vector< ColumnData *, sallocator<ColumnData *> > saved_columns;

    for (unsigned int main_timeout_iter=0; main_timeout_iter < option.timeouts_.size(); main_timeout_iter++) {

      const double timeout = (main_timeout_iter == 0) ? option.timeouts_[0] : 
	option.timeouts_[main_timeout_iter] - option.timeouts_[main_timeout_iter-1];

      /////////////////////////////////////////////////////////
      // Define Training Sender Thread
      /////////////////////////////////////////////////////////

      std::thread train_send_thread([&]() {

	  rng_type send_rng(rank * 17 + option.seed_ + option.num_threads_ + 2);
	  std::uniform_int_distribution<> target_dist(0, numtasks - 1);
 
	  const int dim = option.latent_dimension_;

	  while (flag_train_ready == false) {
	    std::this_thread::yield();
	  }

	  const tick_count start_time = tick_count::now();
	  int monitor_num = 0;

	  char *send_message = sallocator<char>().allocate(msg_bytenum);
	  char *cur_pos = send_message + sizeof(int) + sizeof(int);
	  int cur_num = 0;

	  while (true) {

	    double elapsed_seconds = (tbb::tick_count::now() - start_time).seconds();
	    if (elapsed_seconds > timeout) {
	      break;
	    }

	    if (monitor_num < elapsed_seconds) {
	      cout << "sender thread alive," << rank << "," 
		   << monitor_num << "," << send_queue.unsafe_size() << ",endline" << endl;
	      monitor_num++;
	    }

	    ColumnData *p_col = nullptr;

	    if (send_queue.try_pop(p_col)) {
	  
	      *(reinterpret_cast<int *>(cur_pos)) = p_col->col_index_;
	      *(reinterpret_cast<long *>(cur_pos + sizeof(int))) = p_col->flag_;
	      scalar *dest = reinterpret_cast<scalar *>(cur_pos + sizeof(long) + sizeof(int));
	      std::copy(p_col->values_, p_col->values_ + dim, dest);

	      column_pool.push(p_col);

	      cur_pos += unit_bytenum;
	      cur_num++;

	      if (cur_num >= UNITS_PER_MSG) {

		*(reinterpret_cast<int *>(send_message)) = send_queue.unsafe_size();
		*(reinterpret_cast<int *>(send_message) + 1) = cur_num;

		local_send_count += cur_num;

		// choose destination
	    
		while (true) {

		  int target_rank = target_dist(send_rng);
	      
		  if (queue_current_sizes[target_rank] < queue_upperbound) {

		    int rc = MPI_Ssend(send_message, msg_bytenum, MPI_CHAR,
				       target_rank, 1, MPI_COMM_WORLD);
	      
		    // BUGBUG: in rank 0, arbitrary delay is added
		    if (rank == 0 && option.rank0_delay_ > 0) {
		      std::this_thread::sleep_for( tbb::tick_count::interval_t(option.rank0_delay_) );
		    }

		    if (rc != MPI_SUCCESS) {
		      std::cerr << "SendTask MPI Error" << std::endl;
		      exit(64);
		    }
		
		    cur_pos = send_message + sizeof(int) + sizeof(int);
		    cur_num = 0;

		    break;
		
		  }

		}

	      }

	    }
	    else {
	      std::this_thread::yield();
	    }

	  }

	  {
	    // send remaining columns to random machine
	    *(reinterpret_cast<int *>(send_message)) = send_queue.unsafe_size();
	    *(reinterpret_cast<int *>(send_message) + 1) = cur_num;
	    int target_rank = target_dist(send_rng);
	    int rc = MPI_Ssend(send_message, msg_bytenum, MPI_CHAR,
			       target_rank, 1, MPI_COMM_WORLD);
	  
	    local_send_count += cur_num;

	    if (rc != MPI_SUCCESS) {
	      std::cerr << "SendTask MPI Error" << std::endl;
	      exit(64);
	    }

	  }

	  // send dying message to every machine
	  *(reinterpret_cast<int *>(send_message) + 1) = -(rank + 1);
	  for (int i=0; i < numtasks; i++) {
	    int rc = MPI_Ssend(send_message, msg_bytenum, MPI_CHAR,
			       i, 1, MPI_COMM_WORLD);
	  
	    if (rc != MPI_SUCCESS) {
	      std::cerr << "SendTask MPI Error" << std::endl;
	      exit(64);
	    }
	  }
      
	  sallocator<char>().deallocate(send_message, msg_bytenum);

	  cout << "send thread finishing" << endl;

	  return;

	});


      // wait until every machine is ready
      MPI_Barrier(MPI_COMM_WORLD);

      /////////////////////////////////////////////////////////
      // Start Training
      /////////////////////////////////////////////////////////

      // now we are ready to train
      flag_train_ready = true;

      // do receiving
      {
	const int dim = option.latent_dimension_;
	char *recv_message = sallocator<char>().allocate(msg_bytenum);

	const tick_count start_time = tick_count::now();
	int monitor_num = 0;

	int num_dead = 0;

	MPI_Status status;

	while (num_dead < numtasks) {
      
	  double elapsed_seconds = (tbb::tick_count::now() - start_time).seconds();

	  if (monitor_num < elapsed_seconds) {
	    cout << "receiver thread alive," << rank << ","
		 << monitor_num << endl;
	    monitor_num++;
	  }

	  int rc = MPI_Recv(recv_message, msg_bytenum, MPI_CHAR,
			    MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);

	  if (rc != MPI_SUCCESS) {
	    std::cerr << "ReceiveTask MPI Error" << std::endl;
	    exit(64);
	  }
      
	  int queue_size = *(reinterpret_cast<int *>(recv_message));
	  int num_received = *(reinterpret_cast<int *>(recv_message) + 1);

	  queue_current_sizes[status.MPI_SOURCE] = queue_size;

	  // negative numbers are dying messages
	  if (num_received < 0) {
	    num_dead++;
	  }
	  else {
	    char *cur_pos = recv_message + sizeof(int) + sizeof(int);
	    for (int i=0; i < num_received; i++) {
	    
	      ColumnData *p_col = column_pool.pop();
	    
	      p_col->col_index_ = *(reinterpret_cast<int *>(cur_pos));
	      p_col->flag_ = *(reinterpret_cast<long *>(cur_pos + sizeof(int)));
	  
	      scalar *dest = reinterpret_cast<scalar *>(cur_pos + sizeof(int) + sizeof(long));
	      std::copy(dest, dest+dim, p_col->values_);	  

	      p_col->pos_ = 0;
	  
	      // generate permutation
	      for (int j=0; j < option.num_threads_; j++) {
		p_col->perm_[j] = j;
	      }
	      std::shuffle(p_col->perm_, p_col->perm_ + option.num_threads_, rng);
	    
	      job_queues[p_col->perm_[p_col->pos_]].push(p_col);	  

	      cur_pos += unit_bytenum;

	    }
	  }

	}

	sallocator<char>().deallocate(recv_message, msg_bytenum);    

      }

      train_send_thread.join();

      flag_train_stop = true;
      flag_train_ready = false;
      count_setup_threads = 0;

      // prepare for test
      {
	// gather everything that is within the machine
	vector< ColumnData *, sallocator<ColumnData *> > local_columns;

	int num_columns_prepared = 0;
	int global_num_columns_prepared = 0;

	while (global_num_columns_prepared < global_num_cols) {
    
	  for (int i=0; i < option.num_threads_; i++) {
	    ColumnData *p_col;
	    while (job_queues[i].try_pop(p_col)) {
	      local_columns.push_back(p_col);
	      num_columns_prepared++;
	    }
	  }
	
	  {
	    ColumnData *p_col;
	    while (send_queue.try_pop(p_col)) {
	      local_columns.push_back(p_col);
	      num_columns_prepared++;
	    }
	  }
	
	
	  MPI_Allreduce(&num_columns_prepared, &global_num_columns_prepared, 1, MPI_INT,
			MPI_SUM, MPI_COMM_WORLD);

	  if (rank == 0) {
	    cout << "num columns prepared: " << global_num_columns_prepared
		 << " / " << global_num_cols << endl;
	  }

	}

	for (ColumnData *p_col : local_columns) {
	  p_col->flag_ = 0;
	  job_queues[0].push(p_col);
	}
    
      }

      // wait until every machine is ready
      MPI_Barrier(MPI_COMM_WORLD);

      /////////////////////////////////////////////////////////
      // Start Testing
      /////////////////////////////////////////////////////////

      // now start actual computation
      flag_test_ready = true;

      std::thread test_send_thread([&]() {

	  const long mask = (1L << rank);

	  const int dim = option.latent_dimension_;

	  const tick_count start_time = tick_count::now();
	  int monitor_num = 0;

	  char *send_message = sallocator<char>().allocate(msg_bytenum);
	  char *cur_pos = send_message + sizeof(int);
	  int cur_num = 0;

	  int send_count = 0;

	  int target_rank = rank + 1;
	  target_rank %= numtasks;

	  while (send_count < global_num_cols) {

	    double elapsed_seconds = (tbb::tick_count::now() - start_time).seconds();
	    if (monitor_num < elapsed_seconds) {
	      cout << "test sender thread alive: "
		   << monitor_num << endl;
	      monitor_num++;
	    }

	    ColumnData *p_col;
	
	    if (send_queue.try_pop(p_col)) {

	      // if the column was not already processed
	      if ((p_col->flag_ & mask) == 0) {

		p_col->flag_ |= mask;

		*(reinterpret_cast<int *>(cur_pos)) = p_col->col_index_;
		*(reinterpret_cast<long *>(cur_pos + sizeof(int))) = p_col->flag_;
		scalar *dest = reinterpret_cast<scalar *>(cur_pos + sizeof(long) + sizeof(int));
		std::copy(p_col->values_, p_col->values_ + dim, dest);

		cur_pos += unit_bytenum;
		cur_num++;

		send_count++;
	      
		if (cur_num >= UNITS_PER_MSG) {
		
		  *(reinterpret_cast<int *>(send_message)) = cur_num;
		
		  // choose destination
		  int rc = MPI_Ssend(send_message, msg_bytenum, MPI_CHAR,
				     target_rank, 1, MPI_COMM_WORLD);
		
		  if (rc != MPI_SUCCESS) {
		    std::cerr << "SendTask MPI Error" << std::endl;
		    exit(64);
		  }
	      
		  cur_pos = send_message + sizeof(int);
		  cur_num = 0;
	      
		}
	    
	      }
	      else {
		cout << "!!! should not happen! flag:" << p_col->flag_ << "???" << endl;	    
	      }

	      column_pool.push(p_col);

	    }
	    else {

	      // even if pop was failed, if there is remaining message send it to another machine
	      if (cur_num > 0) {

		*(reinterpret_cast<int *>(send_message)) = cur_num;

		// choose destination
		int rc = MPI_Ssend(send_message, msg_bytenum, MPI_CHAR,
				   target_rank, 1, MPI_COMM_WORLD);
	    
		if (rc != MPI_SUCCESS) {
		  std::cerr << "SendTask MPI Error" << std::endl;
		  exit(64);
		}
	      
		cur_pos = send_message + sizeof(int);
		cur_num = 0;
	    
	      }
	      else {
		std::this_thread::yield();
	      }
	    }

	  }

	  if (cur_num > 0) {
	    // send remaining columns to designated machine
	    *(reinterpret_cast<int *>(send_message)) = cur_num;
	    int rc = MPI_Ssend(send_message, msg_bytenum, MPI_CHAR,
			       target_rank, 1, MPI_COMM_WORLD);
	  
	    if (rc != MPI_SUCCESS) {
	      std::cerr << "SendTask MPI Error" << std::endl;
	      exit(64);
	    }
	  }

	  sallocator<char>().deallocate(send_message, msg_bytenum);

	  cout << "test send thread finishing," << rank << endl;
	
	  return;

	});


      // receive columns for testing
      {
	const int dim = option.latent_dimension_;
	char *recv_message = sallocator<char>().allocate(msg_bytenum);

	const tick_count start_time = tick_count::now();
	int monitor_num = 0;
      
	int recv_count = 0;

	MPI_Status status;

	const long mask = (1L << rank);

	while (recv_count < global_num_cols) {
      
	  double elapsed_seconds = (tbb::tick_count::now() - start_time).seconds();

	  if (monitor_num < elapsed_seconds) {
	    cout << "receiver thread alive: "
		 << monitor_num << endl;
	    monitor_num++;
	  }

	  int rc = MPI_Recv(recv_message, msg_bytenum, MPI_CHAR,
			    MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);

	  if (rc != MPI_SUCCESS) {
	    std::cerr << "ReceiveTask MPI Error" << std::endl;
	    exit(64);
	  }

	  int num_received = *(reinterpret_cast<int *>(recv_message));
	  // negative numbers are dying messages
	  char *cur_pos = recv_message + sizeof(int);
	  for (int i=0; i < num_received; i++) {
	  
	    ColumnData *p_col = column_pool.pop();

	    p_col->col_index_ = *(reinterpret_cast<int *>(cur_pos));
	    p_col->flag_ = *(reinterpret_cast<long *>(cur_pos + sizeof(int)));
	
	    scalar *dest = reinterpret_cast<scalar *>(cur_pos + sizeof(int) + sizeof(long));
	    std::copy(dest, dest+dim, p_col->values_);	  

	    if ((mask & p_col->flag_) == 0) {
	      job_queues[0].push(p_col);	 
	    }
	    else {
	      // discard the column
	      saved_columns.push_back(p_col);
	    }
	  
	    cur_pos += unit_bytenum;

	    recv_count++;

	  }

	}

	sallocator<char>().deallocate(recv_message, msg_bytenum);    

	cout << "test receive done," << rank << endl;

      }

      test_send_thread.join();

      // test done
      flag_test_stop = true;

      cout << "waiting to join with updaters," << rank << endl;

      while (count_setup_threads < option.num_threads_) {
	std::this_thread::yield();
      }

  
      /////////////////////////////////////////////////////////
      // Compute Statistics
      /////////////////////////////////////////////////////////

      long long machine_num_updates = 0; // std::accumulate(num_updates, num_updates + option.num_threads_, 0);
      for (int i=0; i < option.num_threads_; i++) {
	machine_num_updates += num_updates[i];
      }
      cout << "machine_num_updates: " << machine_num_updates << endl;
    
      long long machine_num_failures = 0; // std::accumulate(num_updates, num_updates + option.num_threads_, 0);
      for (int i=0; i < option.num_threads_; i++) {
	machine_num_failures += num_failures[i];
      }
      cout << "machine_num_failures: " << machine_num_failures << endl;


      int machine_train_count_error = std::accumulate(train_count_errors, train_count_errors + option.num_threads_, 0);
      int machine_test_count_error = std::accumulate(test_count_errors, test_count_errors + option.num_threads_, 0);
      real machine_train_sum_error = std::accumulate(train_sum_errors, train_sum_errors + option.num_threads_, 0.0);
      real machine_test_sum_error = std::accumulate(test_sum_errors, test_sum_errors + option.num_threads_, 0.0);

      int global_train_count_error = 0;
      MPI_Allreduce(&machine_train_count_error, &global_train_count_error, 1, MPI_INT,
		    MPI_SUM, MPI_COMM_WORLD);
      int global_test_count_error = 0;
      MPI_Allreduce(&machine_test_count_error, &global_test_count_error, 1, MPI_INT,
		    MPI_SUM, MPI_COMM_WORLD);

      real global_train_sum_error = 0.0;
      MPI_Allreduce(&machine_train_sum_error, &global_train_sum_error, 1, MPI_DOUBLE,
		    MPI_SUM, MPI_COMM_WORLD);
      real global_test_sum_error = 0.0;
      MPI_Allreduce(&machine_test_sum_error, &global_test_sum_error, 1, MPI_DOUBLE,
		    MPI_SUM, MPI_COMM_WORLD);

      long long global_num_updates = 0;
      MPI_Allreduce(&machine_num_updates, &global_num_updates, 1, MPI_LONG_LONG,
		    MPI_SUM, MPI_COMM_WORLD);
    
      long long global_num_failures = 0;
      MPI_Allreduce(&machine_num_failures, &global_num_failures, 1, MPI_LONG_LONG,
		    MPI_SUM, MPI_COMM_WORLD);

      long long global_send_count = 0;
      MPI_Allreduce(&local_send_count, &global_send_count, 1, MPI_LONG_LONG,
		    MPI_SUM, MPI_COMM_WORLD);

      int machine_col_empty = 0;
      for (int i=0; i < global_num_cols; i++) {
	if (is_column_empty[i]) {
	  machine_col_empty++;
	}
      }
  
      int global_col_empty = 0;
      MPI_Allreduce(&machine_col_empty, &global_col_empty, 1, MPI_INT,
		    MPI_SUM, MPI_COMM_WORLD);

      MPI_Barrier(MPI_COMM_WORLD);


      if (option.flag_pause_) {
	std::this_thread::sleep_for( tbb::tick_count::interval_t(5.0) );
      }
      if (rank == 0) {
	cout << "=====================================================" << endl;
	cout << "elapsed time: " << option.timeouts_[main_timeout_iter] << endl;
	cout << "current training RMSE: " << sqrt(global_train_sum_error/global_train_count_error) << endl;
	cout << "current test RMSE: " << sqrt(global_test_sum_error/global_test_count_error)  << endl;

	cout << "testgrep," << numtasks << "," << option.num_threads_ << ","
	     << option.timeouts_[main_timeout_iter] << "," << global_num_updates << ","
	     << sqrt(global_test_sum_error/global_test_count_error) 
	     << "," << global_test_sum_error << "," << global_test_count_error 
	     << "," << global_num_failures << "," << global_col_empty 
	     << "," << global_send_count << ","
	     << sqrt(global_train_sum_error/global_train_count_error) 
	     << "," << global_train_sum_error << "," << global_train_count_error 
	     << endl;
	cout << "=====================================================" << endl;
      }
      if (option.flag_pause_) {
	std::this_thread::sleep_for( tbb::tick_count::interval_t(5.0) );
      }


      // initialize state variables
      flag_train_ready = false;
      flag_train_stop = false;
      flag_test_ready = false;
      flag_test_stop = false;


      // BUGBUG: saved_columns: do initialization and push to job queue again
      for (ColumnData *p_col : saved_columns) {

	p_col->flag_ = 0;
	p_col->pos_ = 0;
	// create initial permutation for the column
	for (int j=0; j < option.num_threads_; j++) {
	  p_col->perm_[j] = j;
	}
	std::shuffle(p_col->perm_, p_col->perm_ + option.num_threads_, rng);

	// push to the job queue
	job_queues[p_col->perm_[p_col->pos_]].push(p_col);

      }
      
      // if at the last iteration, do not clear this thing to print out to file
      if (main_timeout_iter < option.timeouts_.size() - 1) {
	saved_columns.clear();	
      }

    }  // end of timeout loop
  
    cout << "Waiting for updater threads to join" << endl;
    for (int i=0; i < option.num_threads_; i++) {
      updater_threads[i].join();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (option.output_path_.length() > 0) {
      for (int task_iter=0; task_iter < numtasks; task_iter++) {
	if (task_iter == rank) {
	  ofstream ofs(option.output_path_ + boost::lexical_cast<string>(rank), 
		       ofstream::out | ofstream::app);
	  for (ColumnData *p_col : saved_columns) {	
	    ofs << "column," << (p_col->col_index_);
	    for (int t=0; t < option.latent_dimension_; t++) {
	      ofs << "," << p_col->values_[t];
	    }
	    ofs << endl;
	  }
	  ofs.close();
	}
	MPI_Barrier(MPI_COMM_WORLD);
      }
    }

    

    cout << "All done, now free memory" << endl;

    callocator<colque>().deallocate(job_queues, option.num_threads_);  
  

    for (int i=0; i < option.num_threads_; i++) {
      callocator<std::thread>().destroy(updater_threads + i);
      callocator< atomic<long long> >().destroy(num_updates + i);
      callocator< atomic<long long> >().destroy(num_failures + i);
    }
    callocator< atomic<long long> >().deallocate(num_updates, option.num_threads_);
    callocator< atomic<long long> >().deallocate(num_failures, option.num_threads_);
  
    callocator<std::thread>().deallocate(updater_threads, option.num_threads_);

    callocator<int>().deallocate(train_count_errors, option.num_threads_);
    callocator<real>().deallocate(train_sum_errors, option.num_threads_);
    callocator<int>().deallocate(test_count_errors, option.num_threads_);
    callocator<real>().deallocate(test_sum_errors, option.num_threads_);


    callocator< atomic<bool> >().deallocate(is_column_empty, global_num_cols);

    callocator< atomic<int> >().deallocate( queue_current_sizes, numtasks );

    MPI_Finalize();

    return 0;

    
  }

};

#endif
