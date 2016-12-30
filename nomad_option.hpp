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
#ifndef NOMAD_NOMAD_OPTION_HPP_
#define NOMAD_NOMAD_OPTION_HPP_

#include "nomad.hpp"

#include <random>
#include <vector>

#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

namespace nomad {

  using std::vector;
  using std::string;

  struct NomadOption {

    boost::program_options::options_description option_desc_;
    int num_threads_;
    scalar learn_rate_;
    scalar decay_rate_;
    scalar par_lambda_;
    rng_type::result_type seed_;
    int latent_dimension_;
    vector<double> timeouts_;
    int pipeline_token_num_;
    int num_reuse_;
    bool flag_pause_;
    double rank0_delay_;
    string output_path_;

    /** 
     * Create NomadOption instance
     * 
     * @param exec_name 
     * 
     */
    NomadOption(const char *program_name) :
    option_desc_((boost::format("%s options") % program_name).str().c_str())
    {
      option_desc_.add_options()
        ("help,h", "produce help message")
        ("nthreads", 
         boost::program_options::value<int>(&num_threads_)->default_value(4),
         "number of threads to use (0: automatic)")
        ("lrate,l", 
         boost::program_options::value<scalar>(&learn_rate_)->default_value(0.001),
         "learning rate")
        ("drate,d", 
         boost::program_options::value<scalar>(&decay_rate_)->default_value(0.1),
         "decay rate")
        ("reg,r", 
         boost::program_options::value<scalar>(&par_lambda_)->default_value(1.0),
         "regularization parameter lambda")
        ("seed,s", 
         boost::program_options::value<rng_type::result_type>(&seed_)->default_value(12345),
         "seed value of random number generator")
        ("timeout,t", 
         boost::program_options::value<vector<double> >(&timeouts_)->multitoken()->default_value( vector<double>(1, 10.0) , "10.0" ),
         "timeout seconds until completion")
        ("ptoken,p", 
         boost::program_options::value<int>(&pipeline_token_num_)->default_value(1024),
         "number of tokens in the pipeline")
	("dim,d",
	 boost::program_options::value<int>(&latent_dimension_)->default_value(100),
	 "dimension of latent space")
	("reuse",
	 boost::program_options::value<int>(&num_reuse_)->default_value(1),
	 "number of column reuse")
	("pause",
	 boost::program_options::value<bool>(&flag_pause_)->default_value(true),
	 "number of column reuse")
	("r0delay",
	 boost::program_options::value<double>(&rank0_delay_)->default_value(0),
	 "arbitrary network delay added to communication of rank 0 machine")
	("output",
	 boost::program_options::value<string>(&output_path_)->default_value(""),
	 "path of the file the result will be printed into")
        ;
    }

    virtual int get_num_cols() = 0;

    virtual bool is_option_OK() {
      return true;
    }

    /** 
     * 
     * 
     * @param argc 
     * @param argv 
     * 
     * @return 
     */
    bool parse_command(int &argc, char **& argv) {

      using std::cerr;
      using std::endl;
     
      bool flag_help = false;

      try {

        boost::program_options::variables_map var_map;
        boost::program_options::store(boost::program_options::
                                      parse_command_line(argc, argv, option_desc_), 
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

      if (true == flag_help || false == is_option_OK()) {
        cerr << option_desc_ << endl;
        return false;
      }

      return true;
 
    }

  };

}

#endif
