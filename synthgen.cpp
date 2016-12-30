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

using boost::lexical_cast;
using boost::tokenizer;
using boost::char_separator;

int main(int argc, char **argv) {

  int num_rows, num_cols;
  string row_filename, col_filename, output_prefix;
  int seed, dim;

  boost::program_options::options_description option_desc("synthgen options");
  option_desc.add_options()
    ("help,h", "produce help message")
    ("seed", boost::program_options::value<int>(&seed)->default_value(12345),
     "RNG seed")
    ("dim", boost::program_options::value<int>(&dim)->default_value(100),
     "dimension")
    ("nrow", boost::program_options::value<int>(&num_rows)->default_value(480189),
     "number of rows")
    ("ncol", boost::program_options::value<int>(&num_cols)->default_value(17770),
     "number of columns")
    ("rowfile", boost::program_options::value<string>(&row_filename)->default_value("../Results/synth/netflix_rowdegree.txt"),
     "location of row degree frequencies file")
    ("colfile", boost::program_options::value<string>(&col_filename)->default_value("../Results/synth/netflix_coldegree.txt"),
     "location of column degree frequencies file")
    ("output", boost::program_options::value<string>(&output_prefix)->default_value("../Results/temp/first/"),
     "location of column degree frequencies file")
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

  // read row and column degree frequencies
  vector<int> row_degrees;
  vector<int> col_degrees;
  vector<double> row_freqs;
  vector<int> col_freqs;
  
  // row frequencies
  {
    ifstream row_file(row_filename.c_str());
    char_separator<char> sep(",");
    string line;
    cout << "reading file: " << row_filename << endl;

    if (!row_file.is_open()) {
      cerr << "could not open: " << row_filename << endl;
      return 1;
    }

    while (row_file.good()) {
      getline(row_file, line);

      tokenizer< char_separator<char> > tokens(line, sep);      
      tokenizer< char_separator<char> >::iterator iter = tokens.begin();
      
      if (iter == tokens.end()) {
	break;
      }

      // all indicies are subtracted by 0 to make it 0-based index
      int row_degree = lexical_cast<int>(*iter);
      ++iter;

      int freq = lexical_cast<int>(*iter);
      ++iter;
      
      row_degrees.push_back(row_degree);
      row_freqs.push_back(static_cast<double>(freq));
    }    
  }

  // col frequencies
  {
    ifstream col_file(col_filename.c_str());
    char_separator<char> sep(",");
    string line;
    cout << "reading file: " << col_filename << endl;

    if (!col_file.is_open()) {
      cerr << "could not open: " << col_filename << endl;
      return 1;
    }

    while (col_file.good()) {
      getline(col_file, line);

      tokenizer< char_separator<char> > tokens(line, sep);      
      tokenizer< char_separator<char> >::iterator iter = tokens.begin();
      
      if (iter == tokens.end()) {
	break;
      }

      // all indicies are subtracted by 0 to make it 0-based index
      int col_degree = lexical_cast<int>(*iter);
      ++iter;

      int freq = lexical_cast<int>(*iter);
      ++iter;
      
      col_degrees.push_back(col_degree);
      col_freqs.push_back(freq);
    }    
  }

  std::mt19937 rng(seed);
  std::discrete_distribution<> col_degree_dist(col_freqs.begin(), col_freqs.end());
  std::discrete_distribution<> row_degree_dist(row_freqs.begin(), row_freqs.end());
  std::normal_distribution<> normal_dist(0,1);

  auto sample_rowdegree = [&]()->int{
    return row_degrees[row_degree_dist(rng)];
  };

  int temp = 0;
  for (unsigned int i=0; i < row_degrees.size(); i++) {
    temp += row_degrees[i] * row_freqs[i];
  }
  cout << "temp: " << temp << endl;

  temp = 0;
  for (unsigned int i=0; i < col_degrees.size(); i++) {
    temp += col_degrees[i] * col_freqs[i];
  }    
  cout << "temp2: " << temp << endl;

  vector<int> column_weights;
  column_weights.reserve(num_cols);

  for (int i=0; i < num_cols; i++) {
    column_weights.push_back(col_degrees[col_degree_dist(rng)]);
  }

  std::discrete_distribution<> col_index_dist(column_weights.begin(), column_weights.end());

  double *matrix_H = new double[dim * num_cols];
  for (int i=0; i < dim * num_cols; i++) {
    matrix_H[i] = normal_dist(rng);
  }

  ofstream train_header_file(output_prefix + "train_header.dat", ios::out | ios::binary);
  ofstream train_size_file(output_prefix + "train_size.dat", ios::out | ios::binary);
  ofstream train_ind_file(output_prefix + "train_ind.dat", ios::out | ios::binary);
  ofstream train_val_file(output_prefix + "train_val.dat", ios::out | ios::binary);

  ofstream test_header_file(output_prefix + "test_header.dat", ios::out | ios::binary);
  ofstream test_size_file(output_prefix + "test_size.dat", ios::out | ios::binary);
  ofstream test_ind_file(output_prefix + "test_ind.dat", ios::out | ios::binary);
  ofstream test_val_file(output_prefix + "test_val.dat", ios::out | ios::binary);

  long long train_nnz = 0, test_nnz = 0;
  
  int latent_row[dim];
  for (int row_index=0; row_index < num_rows; row_index++) {
    
    if (row_index % 10000 == 0) {
      cout << "progress: " << row_index << " / " << num_rows << " (" << 
	static_cast<double>(row_index) / num_rows * 100.0 << "%)" << endl;
    }

    for (int i=0; i < dim; i++) {
      latent_row[i] = normal_dist(rng);
    }

    int row_degree = std::min(sample_rowdegree() * 5 / 4, num_cols);
    unsigned int train_degree = static_cast<int>(row_degree * 0.8);
    unsigned int test_degree = row_degree - train_degree;
    set<int> train_sampled_cols, test_sampled_cols;
    for (unsigned int j=0; j < static_cast<unsigned int>(row_degree); j++) {
      while(true) {
	int col_index = col_index_dist(rng);
	if (j < train_degree) {
	  if (train_sampled_cols.find(col_index) == train_sampled_cols.end()) {
	    train_sampled_cols.insert(col_index);
	    break;
	  }
	}
	else {
	  if (test_sampled_cols.find(col_index) == test_sampled_cols.end()) {
	    test_sampled_cols.insert(col_index);
	    break;
	  }
	}
      }
    }

    if (train_degree != train_sampled_cols.size()) {
      cout << "!!!" << train_degree << "," << train_sampled_cols.size() << endl;
    }
    if (test_degree != test_sampled_cols.size()) {
      cout << "???" << test_degree << "," << test_sampled_cols.size() << endl;
    }


    for (int col_index : train_sampled_cols) {
      double true_val = 0;
      for (int t=0; t < dim; t++) {
	true_val += latent_row[t] * matrix_H[dim * col_index + t];
      }
      true_val += normal_dist(rng) * 0.1;
      // write 
      train_ind_file.write((char *)&col_index, sizeof(col_index));
      train_val_file.write((char *)&true_val, sizeof(true_val));
    }
    for (int col_index : test_sampled_cols) {
      double true_val = 0;
      for (int t=0; t < dim; t++) {
	true_val += latent_row[t] * matrix_H[dim * col_index + t];
      }
      true_val += normal_dist(rng) * 0.1;
      // write 
      test_ind_file.write((char *)&col_index, sizeof(col_index));
      test_val_file.write((char *)&true_val, sizeof(true_val));
    }

    train_size_file.write((char *)&train_degree, sizeof(train_degree));
    test_size_file.write((char *)&test_degree, sizeof(test_degree));

    train_nnz += train_degree;
    test_nnz += test_degree;
  }

  cout << "train_nnz: " << train_nnz << endl;
  cout << "test_nnz: " << test_nnz << endl;
  int file_id = 1015;

  train_header_file.write((char *)&file_id, sizeof(file_id));
  train_header_file.write((char *)&num_rows, sizeof(num_rows));
  train_header_file.write((char *)&num_cols, sizeof(num_cols));
  train_header_file.write((char *)&train_nnz, sizeof(train_nnz));

  test_header_file.write((char *)&file_id, sizeof(file_id));
  test_header_file.write((char *)&num_rows, sizeof(num_rows));
  test_header_file.write((char *)&num_cols, sizeof(num_cols));
  test_header_file.write((char *)&test_nnz, sizeof(test_nnz));

  train_header_file.close();
  train_size_file.close();
  train_ind_file.close();
  train_val_file.close();

  test_header_file.close();
  test_size_file.close();
  test_ind_file.close();
  test_val_file.close();
    

  delete matrix_H;

  return 0;

}
