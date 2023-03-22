
import os

my_n_tree = [20,50,100,150,200]
my_mtree = [10,20,30,40,50]
my_min_leaf = [10,20,30,40]

ram = 35
which_HPC = 'sharc'

count = 0

path_to_save = './'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)


for n_tree in my_n_tree:
    for mtree in my_mtree:
        for min_leaf in my_min_leaf:
            f = open(path_to_save + "bash" + str(count) + ".sge", "w+")
            if which_HPC == 'sharc' and count%1 == 0:
                f.write("#!/bin/bash\n#$ -P rse\n#$ -l rmem=%dG #Ram memory\n#$ -cwd\n\n"  
                        "module load apps/matlab/2021a/binary\nmatlab -nodesktop -nosplash -r \"All_Drugs_LungCancer_FRF_job(%d,%d,%d,%d)\"" % (
                        ram, n_tree, mtree, min_leaf,count))

            elif which_HPC == 'bessemer':
                "ToDo"
                pass
            f.close()
            count = count + 1

