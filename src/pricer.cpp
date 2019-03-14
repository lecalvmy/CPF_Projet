#include <iostream>
#include <string>
#include "jlparser/parser.hpp"
#include "Model.hpp"
#include "BlackScholesModel.hpp"
#include "MonteCarlo.hpp"
#include "AsianOption.hpp"
#include "BasketOption.hpp"
#include "CallOption.hpp"
#include "PerformanceOption.hpp"
#include <time.h>
#include "mpi.h"

#include "pnl/pnl_finance.h"



using namespace std;

int main(int argc, char **argv)
{
    clock_t t1, t2;
    double t3, t4;
    double fdStep = 0.1;
    double T, r, strike, correlation;
    PnlVect *spot, *sigma, *divid, *payoff_coef;
    string type;
    int size, timestep;
    //int hedging_dates_number;
    int optionc = 0;
    double precision = 0;

    size_t n_samples;
    char *infile, *market_file;
    if (argc >= 5){
        throw std::invalid_argument( "Invalid number of arguments for function" );
    } else if (argc == 2){
        infile = argv[1];
    } else if (argc == 3) {
        infile = argv[1];
        precision = atof(argv[2]);
    }else if (argc == 4){
        if (strcmp(argv[1], "-c") != 0){
            throw std::invalid_argument( "Option not implemented for function" );
        } else {
            optionc = 1;
            infile = argv[3];
            market_file = argv[2];
        }

    }

    Param *P = new Parser(infile);
    P->extract("option type", type);
    P->extract("maturity", T);
    P->extract("option size", size);
    P->extract("spot", spot, size);
    P->extract("correlation", correlation);
    P->extract("volatility", sigma, size);
    P->extract("interest rate", r);
    if (P->extract("dividend rate", divid, size, true) == false)
    {
        divid = pnl_vect_create_from_zero(size);
    }
    if (type != "performance"){
        P->extract("strike", strike);
    }
    P->extract("timestep number", timestep);
    P->extract("sample number", n_samples);
    //P->extract("hedging dates number", hedging_dates_number);
    P->extract("payoff coefficients", payoff_coef, size);
    PnlVect* trend = pnl_vect_create_from_zero(size);


    Option* opt;

    Model *bsmodel = new BlackScholesModel(size, r, correlation, sigma, spot, trend);
    if (type == "asian"){
        opt = new AsianOption(T, timestep, size, payoff_coef, strike);
    } else if ( type == "basket"){
        opt = new BasketOption(T, timestep, size, payoff_coef, strike);
    } else if ( type == "performance"){
        opt = new PerformanceOption(T, timestep, size, payoff_coef);
    }else if ( type == "call"){
        opt = new CallOption(T, timestep, size, payoff_coef, strike);
    }

    PnlRng *rng= pnl_rng_create(PNL_RNG_MERSENNE);


    MonteCarlo *mCarlo = new MonteCarlo(bsmodel, opt, rng, fdStep, n_samples);
/*
    double prix = 0.0;
    double ic = 0.0;
    t1 = clock();
    mCarlo->price(prix , ic);
    t2 = clock();
    printf("============== \nPrix: %f \nIc: %f \n", prix, ic);
    float diff ((float)t2-(float)t1);
    float seconds = diff / CLOCKS_PER_SEC;
    printf("%f sec\n==============\n", seconds);

*/

    if (argc == 2) {
        int size_th, rank;
        MPI_Init (&argc, &argv);
        MPI_Comm_size (MPI_COMM_WORLD, &size_th);
        MPI_Comm_rank (MPI_COMM_WORLD, &rank);
        /*
        printf("rank : %i\n", rank);
        printf("size : %i\n", size_th);
        */
        double prix_para = 0.0;
        double ic_para = 0.0;
        t3 = MPI_Wtime();
        mCarlo->price(prix_para , ic_para, size_th, rank);
        t4 = MPI_Wtime();
        if (rank == 0) {
            printf("============== \nPrix mpi: %f \nIc mpi: %f \n ",  prix_para, ic_para);
            printf("%f sec\n==============\n", t4-t3);
        }
        MPI_Finalize ();
    }

    else if (argc == 3) {
        int size_th, rank;
        MPI_Init (&argc, &argv);
        MPI_Comm_size (MPI_COMM_WORLD, &size_th);
        MPI_Comm_rank (MPI_COMM_WORLD, &rank);
/*
        printf("rank : %i\n", rank);
        printf("size : %i\n", size_th);
*/
        double prix_para = 0.0;
        double nbSamplesNeeded = 0.0;
        double ic = 0.0;
        t3 = MPI_Wtime();
        mCarlo->price(prix_para, size_th, rank, precision, nbSamplesNeeded, ic);
        t4 = MPI_Wtime();
        if (rank==0) {
            printf("============== \nPrix mpi: %f \nNombre samples: %f \n  ic trouve : %f\n", prix_para, nbSamplesNeeded, ic);
            printf("%f sec\n==============\n", t4-t3);
        }
        MPI_Finalize ();
    }



    //PnlMat *past = pnl_mat_create_from_scalar(1, size, 100);
    //PnlVect *delta = pnl_vect_create(size);
    //PnlVect *conf_delta = pnl_vect_create(size);


    /*mCarlo->delta(past, 0, delta, conf_delta);
    for (int i =0; i < size; i ++){
        printf("Delta actif %u: %f\n", i+1, pnl_vect_get(delta, i));
        printf("Standard Deviation actif %u: %f\n", i+1, pnl_vect_get(conf_delta, i));
    }
    t3 = clock();
    float diff2 ((float)t3-(float)t1);
    float seconds2 = diff2 / CLOCKS_PER_SEC;
    printf("%f sec\n==============\n", seconds2);*/

    /*if (optionc == 1){
        PnlMat *market = pnl_mat_create_from_file(market_file);
        double pnl = 0;
        mCarlo->pnl(pnl, market, hedging_dates_number);
        printf("P&L: %f\n", pnl);
        t4 = clock();
        float diff3 ((float)t4-(float)t1);
        float seconds3 = diff3 / CLOCKS_PER_SEC;
        printf("%f sec\n", seconds3);
    }*/

    //pnl_mat_free(&past);
    //pnl_vect_free(&delta);
    //pnl_vect_free(&conf_delta);

    //pnl_vect_free(&spot);
    //pnl_vect_free(&sigma);
    //pnl_vect_free(&divid);
    //pnl_rng_free(&rng);
    delete P;
    //delete bsmodel;
    //delete opt;
    delete mCarlo;
}
