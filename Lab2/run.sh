
#!/bin/bash
#PBS -N ptest
#PBS -l nodes=n008:ppn=4:gpus=1

cd $PBS_O_WORKDIR

./a.out >> out.out
