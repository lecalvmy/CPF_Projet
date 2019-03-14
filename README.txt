Projet Calcul Parallèle en Finance 
Daniella Teukeng Mobou, Mylène Le Calvez

Pour utiliser le pricer vous devez : 

- Se placer dans le dossier build/

- Pour compiler le projet : 
cmake3 -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_PREFIX_PATH=/matieres/5MMPMP6/pnl ..
make

- Pour lancer : 
mpirun --mca pml ob1 --mca btl self,tcp -np nbThreads ./pricer fichier 
mpirun --mca pml ob1 --mca btl self,tcp -np nbThreads ./pricer fichier precision


- Fichiers disponibles : 
../data/asian.dat
../data/basket.dat
../data/basket_1.dat
../data/basket_2.dat
../data/call.dat
../data/perf.dat
