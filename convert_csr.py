from scipy import *
from scipy.sparse import *
import sys
import random
import numpy as np
import struct
import numpy.random

# script for the data format used in nomad (CSR)

random.seed(12345)

# if len(sys.argv) < 3:
#     print "usage: %s [train_filename] [test_filename] [output_path]" % sys.argv[0]
#     exit(1)

train_filename = "./netflix_mm"
test_filename = "./netflix_mme"
output_path = "/home/huilee/Hui_Workspace"

user_ids = set()
item_ids = set()

# index users and items

for index, line in enumerate(open(train_filename)):

    if index < 3:
        continue

    if index % 1000000 == 0:
        print "1st pass training:", index

    tokens = line.strip().split()

    user_ids.add(tokens[0])
    item_ids.add(tokens[1])

    #if index > 200:
    #    break

for index, line in enumerate(open(test_filename)):
    if index < 3:
        continue

    if index % 1000000 == 0:
        print "1st pass test:", index

    tokens = line.strip().split()

    user_ids.add(tokens[0])
    item_ids.add(tokens[1])

    #if index > 200:
    #    break



user_id_list = list(user_ids)
item_id_list = list(item_ids)
random.shuffle(user_id_list)
random.shuffle(item_id_list)

user_indexer = {key:value for value, key in enumerate(user_id_list)}
item_indexer = {key:value for value, key in enumerate(item_id_list)}


# now parse the data
train_user_indices = list()
train_item_indices = list()
train_values = list()

for index, line in enumerate(open(train_filename)):

    if index < 3:
        continue
    if index % 1000000 == 0:
        print "2nd pass training:", index

    tokens = line.strip().split()

    train_user_indices.append(user_indexer[tokens[0]])
    train_item_indices.append(item_indexer[tokens[1]])
    train_values.append(float(tokens[2]))

    #if index > 200:
    #    break


#print user_indices
#print item_indices
#print values

print "form training csr matrix"
train_mat = csr_matrix( (train_values,(train_user_indices,train_item_indices)), shape=(len(user_indexer),len(item_indexer)) )

print "calculate size of rows"
train_row_sizes = train_mat.indptr[1:] - train_mat.indptr[:-1]

#print user_indexer
#print len(user_indexer)
#print train_row_sizes

#print train_mat
#print mat.indices
#print mat.data

print "write train binary file"
ofile = open(output_path + "/train.dat", "wb")
# ofile.write(struct.pack("=iiii", 1211216, len(user_indexer), len(item_indexer), train_mat.getnnz()))
ofile.write(struct.pack("=%sd" % len(train_mat.data), *train_mat.data))
ofile.write(struct.pack("=%si" % len(train_mat.indptr), *train_mat.indptr))
ofile.write(struct.pack("=%si" % len(train_mat.indices), *train_mat.indices))

ofile.close()



test_user_indices = list()
test_item_indices = list()
test_values = list()

for index, line in enumerate(open(test_filename)):

    if index < 3:
        continue

    if index % 1000000 == 0:
        print "2nd pass test:", index

    tokens = line.strip().split()

    test_user_indices.append(user_indexer[tokens[0]])
    test_item_indices.append(item_indexer[tokens[1]])
    test_values.append(float(tokens[2]))

    #if index > 200:
    #    break


#print user_indices
#print item_indices
#print values

print "form test csr matrix"
test_mat = csr_matrix( (test_values,(test_user_indices,test_item_indices)), shape=(len(user_indexer),len(item_indexer)) )

print "calculate size of rows"
test_row_sizes = test_mat.indptr[1:] - test_mat.indptr[:-1]

#print row_sizes
#print mat.indices
#print mat.data

print "write test binary file"
ofile = open(output_path + "/test.dat", "wb")
# ofile.write(struct.pack("=iiii", 1211216, len(user_indexer), len(item_indexer), test_mat.getnnz()))
ofile.write(struct.pack("=%sd" % len(test_mat.data), *test_mat.data))
ofile.write(struct.pack("=%si" % len(test_mat.indptr), *test_mat.indptr))
ofile.write(struct.pack("=%si" % len(test_mat.indices), *test_mat.indices))

ofile.close()

print "write user index mappings"
ofile = open(output_path + "/user_ids.txt", "w")
for user_id in user_id_list:
    ofile.write("%s\n" % user_id)
ofile.close()

print "write item index mappings"
ofile = open(output_path + "/item_ids.txt", "w")
for item_id in item_id_list:
    ofile.write("%s\n" % item_id)
ofile.close()