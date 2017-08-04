#!/bin/sh
###############################################################################
###  This is a sample PBS job script for Serial C/F90 program                 #
###  1. To use GNU Compilers (Default)					      #
###     gcc hello.c -o hello-gcc					      #
###     gfortran hello.f90 -o hello-gfortran				      #
###  2. To use PGI Compilers						      #
###     module load pgi							      #
###     pgcc hello.c -o hello-pgcc					      #
###     pgf90 hello.f90 -o hello-pgf90					      #
###  3. To use Intel Compilers						      #
###     module load intel					              #
###     icc hello.c -o hello-icc        				      #
###     ifort hello.f90 -o hello-ifort					      #
###############################################################################

### Job name
#PBS -N PNOMAD-n-100-1-1
### Declare job non-rerunable
#PBS -r n
#PBS -k oe

###  Queue name (debug, parallel or fourday)   ################################
###    Queue debug   : Walltime can be  00:00:01 to 00:30:00                  #
###    Queue parallel: Walltime can be  00:00:01 to 24:00:00                  #
###    Queue fourday : Walltime can be  24:00:01 to 96:00:00                  #
###  #PBS -q parallel                                                         #
###############################################################################
#PBS -q special

###  Wall time required. This example is 30 min  ##############################
###  #PBS -l walltime=00:30:00                   			      #
###############################################################################
#PBS -l walltime=24:00:00

###  Number of node and cpu core  #############################################
###  For serial program, 1 core is used					      #
###  #PBS -l nodes=1:ppn=1						      #
###############################################################################
#PBS -l nodes=1:ppn=1

###############################################################################
#The following stuff will be executed in the first allocated node.            #
#Please don't modify it                                                       #
#                                                                             #
echo $PBS_JOBID : `wc -l < $PBS_NODEFILE` CPUs allocated: `cat $PBS_NODEFILE`
PATH=$PBS_O_PATH
JID=`echo ${PBS_JOBID}| sed "s/.hpc2015-mgt.hku.hk//"`
###############################################################################

echo ===========================================================
echo "Job Start  Time is `date "+%Y/%m/%d -- %H:%M:%S"`"

cd $PBS_O_WORKDIR

######################################
cd ../../../build/

data_folder="/data/huilee/mf_data/netflix"

default_timeouts="10 100 200 500 1000 1500 2000 4000 6000 8000"
dimension="100"
lambda="0.05"
l="0.002"
thread="1"

OUTFILE="log-netflix-100-1-1.txt"

for timout in $default_timeouts; do
	time ./nomad_double --nthreads $thread --lrate $l --reg $lambda --dim $dimension --path $data_folder --timeout $timout >> ${OUTFILE}
done

mv $OUTFILE $log_file $PBS_O_WORKDIR
######################################

echo "Job Finish Time is `date "+%Y/%m/%d -- %H:%M:%S"`"
mv $HOME/${PBS_JOBNAME}.e${JID} $HOME/${PBS_JOBNAME}.o${JID} $PBS_O_WORKDIR
exit 0