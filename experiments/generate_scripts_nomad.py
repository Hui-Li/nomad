#-*-coding:utf-8-*-
import os

def generate_scripts_pnomad(script_template):

    dimensions=[50, 100]
    thread_nums=[1, 4, 8, 16]
    parameters={
        "netflix":{
            "data_folder": "/data/huilee/mf_data/netflix",
            "lambda": 0.05,
            "l": 0.002,
            "timeouts": "10 100 200 500 1000 1500 2000 4000 6000 8000"
        },
        "yahoo":{
            "data_folder": "/data/huilee/mf_data/yahoo",
            "lambda": 1,
            "l": 0.0001,
            "timeouts": "10 50 100 150 200 300 500 1000"
        }
    }

    queue="special"
    node_num=1

    if not os.path.exists("./PNOMAD"):
        os.makedirs("./PNOMAD")

    for dimension in dimensions:
        folder_name = "./PNOMAD/"+str(dimension)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for thread_num in thread_nums:
            for (dataset, parameter) in parameters.iteritems():
                running_template = 'cd ../../../build/\n\n' \
                                   'data_folder="{0}"\n\n' \
                                   'default_timeouts="{1}"\n' \
                                   'dimension="{2}"\n' \
                                   'lambda="{3}"\n' \
                                   'l="{4}"\n' \
                                   'thread="{5}"\n\n' \
                                   'OUTFILE="log-{6}-{7}-1-{8}.txt"\n\n' \
                                   'for timout in $default_timeouts; do\n' \
                                   '	time ./nomad_double --nthreads $thread --lrate $l --reg $lambda --dim $dimension --path $data_folder --timeout $timout >> ${{OUTFILE}}\n' \
                                   'done\n\n' \
                                   'mv $OUTFILE $log_file $PBS_O_WORKDIR'

                running_command = running_template.format(parameter["data_folder"], parameter["timeouts"], dimension,
                                                          parameter["lambda"], parameter["l"], thread_num, dataset, dimension,
                                                          thread_num)

                job_name = "PNOMAD-{0}-{1}-{2}-{3}".format(dataset[0], dimension, node_num, thread_num)
                script_content = script_template.format(job_name, queue, node_num, thread_num, running_command)

                script_name = dataset + "-" + str(dimension) + "-" + str(node_num) + "-" + str(thread_num) + ".cmd"
                with open(folder_name + "/" + script_name, "w") as output:
                    output.write(script_content)

if __name__=="__main__":
    script_template='#!/bin/sh\n' \
    '###############################################################################\n' \
    '###  This is a sample PBS job script for Serial C/F90 program                 #\n' \
    '###  1. To use GNU Compilers (Default)					      #\n' \
    '###     gcc hello.c -o hello-gcc					      #\n' \
    '###     gfortran hello.f90 -o hello-gfortran				      #\n' \
    '###  2. To use PGI Compilers						      #\n' \
    '###     module load pgi							      #\n' \
    '###     pgcc hello.c -o hello-pgcc					      #\n' \
    '###     pgf90 hello.f90 -o hello-pgf90					      #\n' \
    '###  3. To use Intel Compilers						      #\n' \
    '###     module load intel					              #\n' \
    '###     icc hello.c -o hello-icc        				      #\n' \
    '###     ifort hello.f90 -o hello-ifort					      #\n' \
    '###############################################################################\n\n' \
    '### Job name\n' \
    '#PBS -N {0}\n' \
    '### Declare job non-rerunable\n' \
    '#PBS -r n\n' \
    '#PBS -k oe\n\n' \
    '###  Queue name (debug, parallel or fourday)   ################################\n' \
    '###    Queue debug   : Walltime can be  00:00:01 to 00:30:00                  #\n' \
    '###    Queue parallel: Walltime can be  00:00:01 to 24:00:00                  #\n' \
    '###    Queue fourday : Walltime can be  24:00:01 to 96:00:00                  #\n' \
    '###  #PBS -q parallel                                                         #\n' \
    '###############################################################################\n' \
    '#PBS -q {1}\n\n' \
    '###  Wall time required. This example is 30 min  ##############################\n' \
    '###  #PBS -l walltime=00:30:00                   			      #\n' \
    '###############################################################################\n' \
    '#PBS -l walltime=24:00:00\n\n' \
    '###  Number of node and cpu core  #############################################\n' \
    '###  For serial program, 1 core is used					      #\n' \
    '###  #PBS -l nodes=1:ppn=1						      #\n' \
    '###############################################################################\n' \
    '#PBS -l nodes={2}:ppn={3}\n\n' \
    '###############################################################################\n' \
    '#The following stuff will be executed in the first allocated node.            #\n' \
    '#Please don\'t modify it                                                       #\n' \
    '#                                                                             #\n' \
    'echo $PBS_JOBID : `wc -l < $PBS_NODEFILE` CPUs allocated: `cat $PBS_NODEFILE`\n' \
    'PATH=$PBS_O_PATH\n' \
    'JID=`echo ${{PBS_JOBID}}| sed "s/.hpc2015-mgt.hku.hk//"`\n' \
    '###############################################################################\n\n' \
    'echo ===========================================================\n' \
    'echo "Job Start  Time is `date "+%Y/%m/%d -- %H:%M:%S"`"\n\n' \
    'cd $PBS_O_WORKDIR\n\n' \
    '######################################\n' \
    '{4}\n' \
    '######################################\n\n' \
    'echo "Job Finish Time is `date "+%Y/%m/%d -- %H:%M:%S"`"\n' \
    'mv $HOME/${{PBS_JOBNAME}}.e${{JID}} $HOME/${{PBS_JOBNAME}}.o${{JID}} $PBS_O_WORKDIR\n' \
    'exit 0'

    generate_scripts_pnomad(script_template)