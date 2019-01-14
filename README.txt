cmake3 -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_PREFIX_PATH=/matieres/5MMPMP6/pnl ..

make

mpirun --mca pml ob1 --mca btl self,tcp -np 4 ./pricer fichier

