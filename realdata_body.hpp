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
#ifndef NOMAD_REALDATA_BODY_HPP_
#define NOMAD_REALDATA_BODY_HPP_

#include "nomad.hpp"

#include <iostream>
#include <vector>
#include <fstream>

#include "nomad_option.hpp"
#include "nomad_body.hpp"

using std::cout; using std::cerr; using std::endl;

using std::string;
using std::vector;
using std::ifstream;
using std::ios;


bool read_data(const string filename, int part_index, int num_parts, 
	       int &min_row_index,
	       int &local_num_rows,
	       vector<int, sallocator<int> > &col_offset,
	       vector<int, sallocator<int> > &row_idx,
	       vector<scalar, sallocator<scalar> > &row_val
	       ) {

  ifstream data_file (filename, ios::in | ios::binary);

  int file_id;
  int nrows;
  int ncols;
  long long total_nnz;

  // read the ID of the file to figure out whether it is normal file format
  // or long file format
  if (!data_file.read(reinterpret_cast<char*>(&file_id), sizeof(int))) {
    cout << "Error in reading ID from file" << endl;
    return false;
  }

  // in this case, the file is in regular PETSc foramt
  if (file_id == MAT_FILE_CLASSID) {
    int header[3];

    if (!data_file.read(reinterpret_cast<char*>(header), 3*sizeof(int))) {
      cout << "Error in reading header from file" << endl;
      return false;
    }

    nrows = header[0];
    ncols = header[1];
    total_nnz = header[2];
  }
  // in this case, it is in PETSc format as well, but the nnz is in long long.
  else if (file_id == LONG_FILE_CLASSID) {
    int header[2];

    if (!data_file.read(reinterpret_cast<char*>(header), 2*sizeof(int))) {
      cout << "Error in reading header from file" << endl;
      return false;
    }

    nrows = header[0];
    ncols = header[1];
    
    if (!data_file.read(reinterpret_cast<char*>(&total_nnz), sizeof(long long))) {
      cout << "Error in reading nnz from file" << endl;
      return false;
    }

  }
  else {
    cout << file_id << " does not identify a valid binary matrix file!" << endl;
    exit(1);
  }

  if (part_index == 0) {
    cout << "nrows: " << nrows << ", ncols: " << ncols << ", total_nnz: " << total_nnz << endl;
  }


  // calculate how many number of rows is to be stored locally
  const int num_rows_per_part = nrows/num_parts + ((nrows%num_parts > 0) ? 1 : 0);
  const int min_row = num_rows_per_part * part_index;
  min_row_index = min_row;
  const int max_row = std::min(num_rows_per_part * (part_index + 1), nrows);
  
  // return the number of rows stored in the machine, by reference
  local_num_rows = max_row - min_row;

  int* total_nnz_rows = sallocator<int>().allocate(nrows);
  if (!data_file.read(reinterpret_cast<char*>(total_nnz_rows), nrows*sizeof(int))) {
    cout << "Error in reading nnz values from file!" << endl;
    return false;
  }

  // calculate how many number of entries we'd have to skip to get to the 
  // region of file that is interesting locally
  long long begin_skip=std::accumulate(total_nnz_rows,total_nnz_rows+min_row,0LL);
  long long nnz=std::accumulate(total_nnz_rows+min_row,total_nnz_rows+max_row,0LL);
  long long end_skip=total_nnz-nnz-begin_skip;

  // BUGBUG: this is just for debugging purpose
  if (part_index == 3) {
    cout << "nrows: " << nrows << ", ncols: " << ncols << ", total_nnz: " << total_nnz << endl;
    cout << "min_row: " << min_row << ", max_row: " << max_row << endl;
    cout << "begin_skip: " << begin_skip << ", nnz: " << nnz << endl;
  }

  // Skip over the begin_nnz number of column indices in the file
  data_file.seekg(begin_skip*sizeof(int), std::ios_base::cur);

  cout << "read column indices" << endl;

  int* col_idx = sallocator<int>().allocate(nnz);  
  if (!data_file.read(reinterpret_cast<char*>(col_idx), nnz*sizeof(int))) {
    cout << "Error in reading column indices from file!" << endl;
    return false;
  }
  
  // Skip over remaining nnz and the beginning of data as well
  data_file.seekg(end_skip*sizeof(int)+begin_skip*sizeof(double), std::ios_base::cur);

  cout << "read values" << endl;

  double* col_val = sallocator<double>().allocate(nnz);  
  if (!data_file.read(reinterpret_cast<char*>(col_val), nnz*sizeof(double))) {
    cout << "Error in reading matrix values from file" << endl;
    exit(1);
  }

  data_file.close();
  
  // Now convert everything to column major format
  cout << "form column-wise data structure" << endl;

  // First create vector of vectors 
  vector<vector<int>, sallocator<int> > row_idx_vec(ncols);
  vector<vector<scalar>, sallocator<int> > row_val_vec(ncols);
  int* col_idx_ptr=col_idx;
  double* val_ptr=col_val;

  for(int i=min_row; i<max_row; i++){
    for(int j=0; j<total_nnz_rows[i]; j++){
      int my_col_idx=*col_idx_ptr;
      double my_val=*val_ptr;
      // use relative indices for rows
      row_idx_vec[my_col_idx].push_back(i - min_row);
      row_val_vec[my_col_idx].push_back(static_cast<scalar>(my_val));
      col_idx_ptr++;
      val_ptr++;
    }
  }

  // Free up some space
  sallocator<int>().deallocate(col_idx, nnz);
  sallocator<double>().deallocate(col_val, nnz);
  sallocator<int>().deallocate(total_nnz_rows, nrows);

  cout << "form CSC" << endl;

  // Now convert everything into CSC format
  // vector<int, sallocator<int> > col_offset(ncols+1);
  // vector<int, sallocator<int> > row_idx(nnz);
  // vector<scalar, sallocator<scalar> > row_val(nnz);

  col_offset.resize(ncols+1);
  row_idx.resize(nnz);
  row_val.resize(nnz);

  int offset=0;  
  col_offset[0]=0;

  for(int i=0; i < ncols; i++){
    copy(row_idx_vec[i].begin(), row_idx_vec[i].end(), row_idx.begin()+offset);
    copy(row_val_vec[i].begin(), row_val_vec[i].end(), row_val.begin()+offset);
    offset+=row_idx_vec[i].size();
    col_offset[i+1]=offset;
  }

  row_idx_vec.clear();
  row_val_vec.clear();
  
  return true;
  
}


namespace nomad {

  using std::ifstream;

  struct RealDataOption : public NomadOption {
    
    string path_ = "";
    
    RealDataOption() :
      NomadOption("nomad") 
    {
      option_desc_.add_options()
	("path", 
	 boost::program_options::value<string>(&path_),
	 "path of data")	
	;
    }

    virtual bool is_option_OK() {
      if (path_.length() <= 0) {
	cerr << "--path has to be specified." << endl;
	return false;
      }
      return NomadOption::is_option_OK();
    }


  protected:


    virtual int get_num_cols() {
      
      const string train_filename = path_ + "/train.dat";

      // read number of columns from the data file
      int global_num_cols;
      {
	ifstream data_file (train_filename, ios::in | ios::binary);
	int header[4];
  
	if (!data_file.read(reinterpret_cast<char*>(header), 4*sizeof(int))) {
	  cout << "Error in reading ID from file" << endl;
	  exit(11);
	}
	
	global_num_cols = header[2];

	data_file.close();
      }

      return global_num_cols;

    }



  };

}

using nomad::NomadOption;
using nomad::RealDataOption;

class RealDataBody : public NomadBody {

public:
  virtual NomadOption *create_option() {
    return new nomad::RealDataOption();
  }

protected:
  virtual bool load_train(NomadOption& option, 
			  int part_index, int num_parts, 
			  int &min_row_index,
			  int &local_num_rows,
			  vector<int, sallocator<int> > &col_offset,
			  vector<int, sallocator<int> > &row_idx,
			  vector<scalar, sallocator<scalar> > &row_val
			  ) {
    
    RealDataOption& real_option = dynamic_cast<RealDataOption &>(option);
    const string train_filename = real_option.path_ + "/train.dat";
    return read_data(train_filename, part_index, num_parts,
		     min_row_index,
		     local_num_rows,
		     col_offset, row_idx, row_val);
    
  }

  virtual bool load_test(NomadOption& option, 
			 int part_index, int num_parts, 
			 int &min_row_index,
			 int &local_num_rows,
			 vector<int, sallocator<int> > &col_offset,
			 vector<int, sallocator<int> > &row_idx,
			 vector<scalar, sallocator<scalar> > &row_val
			 ) {
    

    RealDataOption& real_option = dynamic_cast<RealDataOption &>(option);
    const string test_filename = real_option.path_ + "/test.dat";
    return read_data(test_filename, part_index, num_parts,
		     min_row_index,
		     local_num_rows,
		     col_offset, row_idx, row_val);
    
  }


};

#endif
