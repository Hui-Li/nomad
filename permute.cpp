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
#include "nomad.hpp"

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <set>

#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>


using std::cout; using std::cerr; using std::endl;
using std::string;
using std::ifstream; using std::ofstream;
using std::ios;
using std::vector;
using std::set;
using std::pair;

using boost::lexical_cast;
using boost::tokenizer;
using boost::char_separator;

int main(int argc, char **argv) {

  vector<string> default_path;
  default_path.push_back("/Users/bikestra/Research/data/mc/ml-1m-new/train.dat");
  default_path.push_back("/Users/bikestra/Research/data/mc/ml-1m-new/test.dat");
  vector<string> paths;

  int seed;

  boost::program_options::options_description option_desc("synthgen options");
  option_desc.add_options()
    ("help,h", "produce help message")
    ("output", boost::program_options::value<vector<string>>(&paths)->multitoken()->
     default_value(default_path, 
		   "/Users/bikestra/Research/data/mc/ml-1m-new/train.dat"),
     "list of files to be permuted")
    ("seed", boost::program_options::value<int>(&seed)->
     default_value(12345),
     "random seed")
    ;
  
  bool flag_help = false;

  try {

    boost::program_options::variables_map var_map;
    boost::program_options::store(boost::program_options::
				  parse_command_line(argc, argv, option_desc), 
				  var_map);
    boost::program_options::notify(var_map);

    if (var_map.count("help")) {
      flag_help = true;
    }
    
  }
  catch (std::exception& excep) {
    cerr << "error: " << excep.what() << "\n";
    flag_help = true;
  }
  catch(...) {
    cerr << "Exception of unknown type!\n";
    flag_help = true;
  }
  
  if (true == flag_help) {
    cerr << option_desc << endl;
    return 1;
  }

  rng_type rng(seed);

  ifstream data_files[paths.size()];
  long long nnzs[paths.size()];

  int nrows=-1, ncols=-1;

  for (unsigned int i=0; i < paths.size(); i++) {
    string path = paths[i];
    cout << "path: " << path << endl;

    int file_id, this_nrows, this_ncols;
    long long this_nnz;

    data_files[i].open(path, ios::in | ios::binary);
    
    ifstream &data_file = data_files[i];
    if (!data_file.read(reinterpret_cast<char*>(&file_id), sizeof(int))) {
      cout << "Error in reading ID from file" << endl;
      return 1;
    }

    if (!data_file.read(reinterpret_cast<char*>(&this_nrows), sizeof(int))) {
      cout << "Error in reading number of rows from file" << endl;
      return 1;
    }

    if (!data_file.read(reinterpret_cast<char*>(&this_ncols), sizeof(int))) {
      cout << "Error in reading number of columns from file" << endl;
      return 1;
    }

    if (i == 0) {
      nrows = this_nrows;
      ncols = this_ncols;
    }
    else {
      if (nrows != this_nrows) {
	cerr << "number of rows is different from that of previous file: " << nrows << " vs. " << this_nrows << endl;
	return 1;
      }
      if (ncols != this_ncols) {
	cerr << "number of columns is different from that of previous file: " << nrows << " vs. " << this_nrows << endl;
	return 1;
      }
    }
    
    if (file_id == MAT_FILE_CLASSID) {
      int tmp;
      if (!data_file.read(reinterpret_cast<char*>(&tmp), sizeof(int))) {
	cerr << "Error in reading nnz from file" << endl;
	return 1;
      }
      this_nnz = static_cast<int>(tmp);
    }
    else if (file_id == LONG_FILE_CLASSID) {
      if (!data_file.read(reinterpret_cast<char*>(&this_nnz), sizeof(long long))) {
	cerr << "Error in reading nnz from file" << endl;
	return 1;
      }
    }

    cout << "nnz: " << this_nnz << endl;
    nnzs[i] = this_nnz;

  }

  cout << "creating permutations" << endl;
  
  vector<int> row_new2old, col_new2old;
  row_new2old.reserve(nrows); col_new2old.reserve(ncols);

  for (int i=0; i < nrows; i++) {
    row_new2old.push_back(i);
  }
  for (int i=0; i < ncols; i++) {
    col_new2old.push_back(i);
  }

  std::shuffle(row_new2old.begin(), row_new2old.end(), rng);
  std::shuffle(col_new2old.begin(), col_new2old.end(), rng);

  int long_id = LONG_FILE_CLASSID;

  for (unsigned int i=0; i < paths.size(); i++) {

    string outfile_name = paths[i] + ".permute";
    cout << "converting file: " << paths[i] << " to " << outfile_name << endl;
    ofstream outfile(outfile_name, ios::out | ios::binary);

    outfile.write(reinterpret_cast<char*>(&long_id), sizeof(long_id));
    outfile.write(reinterpret_cast<char*>(&nrows), sizeof(nrows));
    outfile.write(reinterpret_cast<char*>(&ncols), sizeof(ncols));
    outfile.write(reinterpret_cast<char*>(nnzs + i), sizeof(long long));

    cout << "assign memory" << endl;

    ifstream &data_file = data_files[i];
    int *row_nnzs = new int[nrows];
    int *col_inds = new int[nnzs[i]];
    double *values = new double[nnzs[i]];

    cout << "reading row nnzs" << endl;

    if (!data_file.read(reinterpret_cast<char*>(row_nnzs), nrows*sizeof(int))) {
      cout << "Error in reading nnz values from file!" << endl;
      return 1;
    }

    if (!data_file.read(reinterpret_cast<char*>(col_inds), nnzs[i]*sizeof(int))) {
      cout << "Error in reading column indices values from file!" << endl;
      return 1;
    }

    if (!data_file.read(reinterpret_cast<char*>(values), nnzs[i]*sizeof(double))) {
      cout << "Error in reading values from file!" << endl;
      return 1;
    }

    long long *row_ptrs = new long long[nrows + 1];
    row_ptrs[0] = 0;
    for (int j=0; j < nrows; j++) {
      row_ptrs[j+1] = row_ptrs[j] + row_nnzs[j];
    }

    // write row_nnzs
    for (int row_index = 0; row_index < nrows; row_index++) {
      int old_row_index = row_new2old[row_index];
      outfile.write(reinterpret_cast<char*>(row_nnzs + old_row_index), sizeof(int));
    }

    // write col_inds
    for (int row_index = 0; row_index < nrows; row_index++) {
      if (row_index % 100000 == 0) {
	cout << "col_ind progress: " << row_index << " / " << nrows << " (" << (row_index*100.0/nrows) << "%)" << endl;
      }
      int old_row_index = row_new2old[row_index];
      set< pair<int, double> > coords;
      for (long long j=row_ptrs[old_row_index]; j < row_ptrs[old_row_index + 1]; j++) {
	coords.insert( pair<int,double>(col_new2old[col_inds[j]], values[j]) );
      }
      for (pair<int, double> coord : coords) {
	outfile.write(reinterpret_cast<char*>(&(coord.first)), sizeof(int));
      }
    }

    // write values
    for (int row_index = 0; row_index < nrows; row_index++) {
      if (row_index % 100000 == 0) {
	cout << "values progress: " << row_index << " / " << nrows << " (" << (row_index*100.0/nrows) << "%)" << endl;
      }
      int old_row_index = row_new2old[row_index];
      set< pair<int, double> > coords;
      for (long long j=row_ptrs[old_row_index]; j < row_ptrs[old_row_index + 1]; j++) {
	coords.insert( pair<int,double>(col_new2old[col_inds[j]], values[j]) );
      }
      for (pair<int, double> coord : coords) {
	outfile.write(reinterpret_cast<char*>(&(coord.second)), sizeof(double));
      }
    }


    delete [] row_ptrs;
    delete [] row_nnzs;
    delete [] col_inds;
    delete [] values;
  }


  return 0;

}
