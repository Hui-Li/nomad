# Nomad

This is a modified version of [author's implementation](http://bigdata.ices.utexas.edu/software/nomad/).  Input file format has been changed to  [CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29) and we use CMake to help compile.
## Environment

- Ubuntu 16.04
- CMake 2.8
- Intel TBB 2018
- GCC 5.4
- Boost 1.63 
- MPICH 3.1.4

## Data
The input data format is changed to [CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29). Additionally, there should be a meta file with the following format:
```
69878 10677
7972661 train.dat
2027393 test.dat
```
where 69878 is the number of users, 10677 is the number of items, 7972661 is the number of training ratings, train.dat is the path to training file (in CSR format), 2027393 is the number of testing ratings and test.dat is the path to testing file (in CSR format).

You can use our tool [MFDataPreparation](https://github.com/Hui-Li/MFDataPreparation) to transform public datasets to CSR format.


## Examples

See shell script `runNomad.sh` for example. 

## Reference
NOMAD: Non-locking, stOchastic Multi-machine algorithm for Asynchronous and Decentralized matrix completion (pdf, software)
H. Yun, H. Yu, C. Hsieh, S. Vishwanathan, I. Dhillon.
In International Conference on Very Large Data Bases (VLDB), pp. 975-986, July 2014.
