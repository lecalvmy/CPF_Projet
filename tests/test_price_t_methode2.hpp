//
// Created by lecalvmy on 9/20/18.
//

#ifndef MC_PRICER_TEST_PRICE_T_METHODE_2_HPP
#define MC_PRICER_TEST_PRICE_T_METHODE_2_HPP

#include <gtest/gtest.h>
#include "jlparser/parser.hpp"
#include "pnl/pnl_random.h"
#include "pnl/pnl_vector.h"
#include "pnl/pnl_matrix.h"

/**
* Programme de test pour le prix en t des différentes options
* Méthode 2
*/

TEST(MonteCarlo, Call_price_t_2){

    const char *infile = "../data/simul_call.dat";
    const PnlMat *callPath = pnl_mat_create_from_file(infile);

    double fdStep = 1; // valeur quelconque car non utilisee pour ce test

    int size = 1;
    double r = 0.02;
    double rho = 0;
    PnlVect *sigma = pnl_vect_create_from_scalar(size,0.200000);
    PnlVect *spot = pnl_vect_create_from_scalar(size,100.000000);
    PnlVect *trend = pnl_vect_create_from_zero(size);

    double T = 2.000000;
    int nbTimeSteps = 365;
    double strike = 100;

    PnlRng *rng= pnl_rng_create(PNL_RNG_MERSENNE);

    size_t n_samples = 50000;

  PnlVect *weights = pnl_vect_create_from_scalar(1, 1.0);

    BlackScholesModel *bsmodel = new BlackScholesModel(size, r, rho, sigma, spot, trend);
    Option *call = new CallOption(T, nbTimeSteps, size, weights, strike);

    pnl_rng_init(rng, PNL_RNG_MERSENNE);
    pnl_rng_sseed(rng, time(NULL));

    MonteCarlo *mCarlo = new MonteCarlo(bsmodel, call, rng, fdStep, n_samples);

    double prix = 0.0;
    double ic = 0.0;
    double t = 0;
    double step = floor(t * nbTimeSteps / T) + 1;

    PnlMat *past = pnl_mat_create_from_scalar(1, size, 100.0000);
    pnl_mat_extract_subblock(past, callPath, 0, step,  0, size);

    mCarlo->price(past, t, prix, ic);

    PnlVect *spot2 = pnl_vect_create(size);
    pnl_mat_get_row(spot2, past, (int) (past->m - 1));

    double prix2 = 0;
    double ic2 = 0;

    BlackScholesModel *bsmodel2 = new BlackScholesModel(size, r, rho, sigma, spot2, trend);
    Option *call2  = new CallOption(T - t, nbTimeSteps - step, size, weights, strike);
    MonteCarlo *mCarlo2 = new MonteCarlo(bsmodel2, call2, rng, fdStep, n_samples);

    mCarlo2->price(prix2, ic2);

    ASSERT_TRUE(abs(prix-prix2)/prix2 <= 0.05); // ecart relatif inf a 5%

    pnl_mat_free(&past);
    pnl_vect_free(&sigma);
    pnl_vect_free(&spot);
    pnl_vect_free(&spot2);
    pnl_rng_free(&rng);
    delete call;
    delete bsmodel;
    delete mCarlo;
    delete call2;
    delete bsmodel2;
    delete mCarlo2;
}

TEST(MonteCarlo, Asian_price_t_2){

    const char *infile = "../data/simul_asian.dat";
    const PnlMat *asianPath = pnl_mat_create_from_file(infile);

    double fdStep = 1; // valeur quelconque car non utilisee pour ce test

    int size = 2;
    double r = 0.02;
    double rho = 0;
    PnlVect *sigma = pnl_vect_create_from_scalar(size,0.200000);
    PnlVect *spot = pnl_vect_create_from_scalar(size,100.000000);
    PnlVect *trend = pnl_vect_create_from_zero(size);

    double T = 1.500000;
    int nbTimeSteps = 450;
    double strike = 90;

    PnlRng *rng= pnl_rng_create(PNL_RNG_MERSENNE);

    size_t n_samples = 50000;

    PnlVect *weights = pnl_vect_create_from_scalar(2, 0.5);

    BlackScholesModel *bsmodel = new BlackScholesModel(size, r, rho, sigma, spot, trend);
    Option *asian = new AsianOption(T, nbTimeSteps, size, weights, strike);

    pnl_rng_init(rng, PNL_RNG_MERSENNE);
    pnl_rng_sseed(rng, time(NULL));

    MonteCarlo *mCarlo = new MonteCarlo(bsmodel, asian, rng, fdStep, n_samples);

    double prix = 0.0;
    double ic = 0.0;
    double t = 0;
    double step = floor(t * nbTimeSteps / T) + 1;

    PnlMat *past = pnl_mat_create_from_scalar(1, size, 100.0000 );
    pnl_mat_extract_subblock(past, asianPath, 0, step,  0, size);
    //pnl_mat_print(past);
    mCarlo->price(past, t, prix, ic);

    PnlVect *spot2 = pnl_vect_create(size);
    pnl_mat_get_row(spot2, past, (int) (past->m - 1));

    double prix2 = 0;
    double ic2 = 0;

    BlackScholesModel *bsmodel2 = new BlackScholesModel(size, r, rho, sigma, spot2, trend);
    Option *asian2  = new AsianOption(T - t, nbTimeSteps - step, size, weights, strike);
    MonteCarlo *mCarlo2 = new MonteCarlo(bsmodel2, asian2, rng, fdStep, n_samples);
    mCarlo2->price(prix2, ic2);

    ASSERT_TRUE(abs(prix-prix2)/prix2 <= 0.05); // ecart relatif inf a 5%

    pnl_mat_free(&past);
    pnl_vect_free(&sigma);
    pnl_vect_free(&spot);
    pnl_vect_free(&spot2);
    pnl_rng_free(&rng);
    delete asian;
    delete bsmodel;
    delete mCarlo;
    delete asian2;
    delete bsmodel2;
    delete mCarlo2;

}



TEST(MonteCarlo, Basket_price_t_2){

    const char *infile = "../data/simul_basket.dat";
    const PnlMat *basketPath = pnl_mat_create_from_file(infile);

    double fdStep = 1; // valeur quelconque car non utilisee pour ce test

    int size = 5;
    double r = 0.04879;
    double rho = 0;
    PnlVect *sigma = pnl_vect_create_from_scalar(size,0.200000);
    PnlVect *spot = pnl_vect_create_from_scalar(size,100.000000);
    PnlVect *trend = pnl_vect_create_from_zero(size);

    double T = 1.000000;
    int nbTimeSteps = 365;
    double strike = 100;

    PnlRng *rng= pnl_rng_create(PNL_RNG_MERSENNE);

    size_t n_samples = 50000;

    PnlVect *weights = pnl_vect_create_from_scalar(5, 0.2);


    BlackScholesModel *bsmodel = new BlackScholesModel(size, r, rho, sigma, spot, trend);
    Option *basket = new BasketOption(T, nbTimeSteps, size, weights, strike);

    pnl_rng_init(rng, PNL_RNG_MERSENNE);
    pnl_rng_sseed(rng, time(NULL));

    MonteCarlo *mCarlo = new MonteCarlo(bsmodel, basket, rng, fdStep, n_samples);

    double prix = 0.0;
    double ic = 0.0;
    double t = 0;
    double step = floor(t * nbTimeSteps / T) + 1;

    PnlMat *past = pnl_mat_create_from_scalar(1, size, 100.0000 );
    pnl_mat_extract_subblock(past, basketPath, 0, step,  0, size);
    //pnl_mat_print(past);
    mCarlo->price(past, t, prix, ic);

    PnlVect *spot2 = pnl_vect_create(size);
    pnl_mat_get_row(spot2, past, (int) (past->m - 1));

    double prix2 = 0;
    double ic2 = 0;

    BlackScholesModel *bsmodel2 = new BlackScholesModel(size, r, rho, sigma, spot2, trend);
    Option *basket_2  = new BasketOption(T - t, nbTimeSteps - step, size, weights, strike);
    MonteCarlo *mCarlo2 = new MonteCarlo(bsmodel2, basket_2, rng, fdStep, n_samples);
    mCarlo2->price(prix2, ic2);

    ASSERT_TRUE(abs(prix-prix2)/prix2 <= 0.05); // ecart relatif inf a 5%

    pnl_mat_free(&past);
    pnl_vect_free(&sigma);
    pnl_vect_free(&spot);
    pnl_vect_free(&spot2);
    pnl_rng_free(&rng);
    delete basket;
    delete bsmodel;
    delete mCarlo;
    delete basket_2;
    delete bsmodel2;
    delete mCarlo2;

}

TEST(MonteCarlo, Basket_1_price_t_2){

    const char *infile = "../data/simul_basket_1.dat";
    const PnlMat *basket1Path = pnl_mat_create_from_file(infile);

    double fdStep = 1; // valeur quelconque car non utilisee pour ce test

    int size = 40;
    double r = 0.04879;
    double rho = 0;
    PnlVect *sigma = pnl_vect_create_from_scalar(size,0.200000);
    PnlVect *spot = pnl_vect_create_from_scalar(size,100.000000);
    PnlVect *trend = pnl_vect_create_from_zero(size);

    double T = 3.000000;
    int nbTimeSteps = 1000;
    double strike = 100;

    PnlRng *rng= pnl_rng_create(PNL_RNG_MERSENNE);

    size_t n_samples = 5000;

    PnlVect *weights = pnl_vect_create_from_scalar(40, 0.025);

    BlackScholesModel *bsmodel = new BlackScholesModel(size, r, rho, sigma, spot, trend);
    Option *basket1 = new BasketOption(T, nbTimeSteps, size, weights, strike);

    pnl_rng_init(rng, PNL_RNG_MERSENNE);
    pnl_rng_sseed(rng, time(NULL));

    MonteCarlo *mCarlo = new MonteCarlo(bsmodel, basket1, rng, fdStep, n_samples);

    double prix = 0.0;
    double ic = 0.0;
    double t = 0;
    double step = floor(t * nbTimeSteps / T) + 1;

    PnlMat *past = pnl_mat_create_from_scalar(1, size, 100.0000 );
    pnl_mat_extract_subblock(past, basket1Path, 0, step,  0, size);
    //pnl_mat_print(past);
    mCarlo->price(past, t, prix, ic);

    PnlVect *spot2 = pnl_vect_create(size);
    pnl_mat_get_row(spot2, past, (int) (past->m - 1));

    double prix2 = 0;
    double ic2 = 0;

    BlackScholesModel *bsmodel2 = new BlackScholesModel(size, r, rho, sigma, spot2, trend);
    Option *basket1_2  = new BasketOption(T - t, nbTimeSteps - step, size, weights, strike);
    MonteCarlo *mCarlo2 = new MonteCarlo(bsmodel2, basket1_2, rng, fdStep, n_samples);
    mCarlo2->price(prix2, ic2);

    ASSERT_TRUE(abs(prix-prix2)/prix2 <= 0.05); // ecart relatif inf a 5%

    pnl_mat_free(&past);
    pnl_vect_free(&sigma);
    pnl_vect_free(&spot);
    pnl_vect_free(&spot2);
    pnl_rng_free(&rng);
    delete basket1;
    delete bsmodel;
    delete mCarlo;
    delete basket1_2;
    delete bsmodel2;
    delete mCarlo2;

}

TEST(MonteCarlo, Basket_2_price_t_2){

    const char *infile = "../data/simul_basket_2.dat";
    const PnlMat *basket2Path = pnl_mat_create_from_file(infile);

    double fdStep = 1; // valeur quelconque car non utilisee pour ce test

    int size = 40;
    double r = 0.04879;
    double rho = 0.7;
    PnlVect *sigma = pnl_vect_create_from_scalar(size,0.200000);
    PnlVect *spot = pnl_vect_create_from_scalar(size,100.000000);
    PnlVect *trend = pnl_vect_create_from_zero(size);

    double T = 1.000000;
    int nbTimeSteps = 365;
    double strike = 100;

    PnlRng *rng= pnl_rng_create(PNL_RNG_MERSENNE);

    size_t n_samples = 5000;

    PnlVect *weights = pnl_vect_create_from_scalar(40, 0.025);

    BlackScholesModel *bsmodel = new BlackScholesModel(size, r, rho, sigma, spot, trend);
    Option *basket2 = new BasketOption(T, nbTimeSteps, size, weights, strike);

    pnl_rng_init(rng, PNL_RNG_MERSENNE);
    pnl_rng_sseed(rng, time(NULL));

    MonteCarlo *mCarlo = new MonteCarlo(bsmodel, basket2, rng, fdStep, n_samples);

    double prix = 0.0;
    double ic = 0.0;
    double t = 0;
    double step = floor(t * nbTimeSteps / T) + 1;

    PnlMat *past = pnl_mat_create_from_scalar(1, size, 100.0000 );
    pnl_mat_extract_subblock(past, basket2Path, 0, step,  0, size);
    //pnl_mat_print(past);
    mCarlo->price(past, t, prix, ic);

    PnlVect *spot2 = pnl_vect_create(size);
    pnl_mat_get_row(spot2, past, (int) (past->m - 1));

    double prix2 = 0;
    double ic2 = 0;

    BlackScholesModel *bsmodel2 = new BlackScholesModel(size, r, rho, sigma, spot2, trend);
    Option *basket2_2  = new BasketOption(T - t, nbTimeSteps - step, size, weights, strike);
    MonteCarlo *mCarlo2 = new MonteCarlo(bsmodel2, basket2_2, rng, fdStep, n_samples);
    mCarlo2->price(prix2, ic2);

    ASSERT_TRUE(abs(prix-prix2)/prix2 <= 0.05); // ecart relatif inf a 5%

    pnl_mat_free(&past);
    pnl_vect_free(&sigma);
    pnl_vect_free(&spot);
    pnl_vect_free(&spot2);
    pnl_rng_free(&rng);
    delete basket2;
    delete bsmodel;
    delete mCarlo;
    delete basket2_2;
    delete bsmodel2;
    delete mCarlo2;

}

TEST(MonteCarlo, Basket_2d_price_t_2){

    const char *infile = "../data/simul_basket_2d.dat";
    const PnlMat *basket2dPath = pnl_mat_create_from_file(infile);

    double fdStep = 1; // valeur quelconque car non utilisee pour ce test

    int size = 2;
    double r = 0.04879;
    double rho = 0;
    PnlVect *sigma = pnl_vect_create_from_scalar(size,0.200000);
    PnlVect *spot = pnl_vect_create_from_scalar(size,100.000000);
    PnlVect *trend = pnl_vect_create_from_zero(size);

    double T = 1.000000;
    int nbTimeSteps = 365;
    double strike = 100;

    PnlRng *rng= pnl_rng_create(PNL_RNG_MERSENNE);

    size_t n_samples = 50000;

    PnlVect *weights = pnl_vect_create_from_scalar(2, 0.5);

    BlackScholesModel *bsmodel = new BlackScholesModel(size, r, rho, sigma, spot, trend);
    Option *basket2d = new BasketOption(T, nbTimeSteps, size, weights, strike);

    pnl_rng_init(rng, PNL_RNG_MERSENNE);
    pnl_rng_sseed(rng, time(NULL));

    MonteCarlo *mCarlo = new MonteCarlo(bsmodel, basket2d, rng, fdStep, n_samples);

    double prix = 0.0;
    double ic = 0.0;
    double t = 0;
    double step = floor(t * nbTimeSteps / T) + 1;

    PnlMat *past = pnl_mat_create_from_scalar(1, size, 100.0000 );
    pnl_mat_extract_subblock(past, basket2dPath, 0, step,  0, size);
    //pnl_mat_print(past);
    mCarlo->price(past, t, prix, ic);

    PnlVect *spot2 = pnl_vect_create(size);
    pnl_mat_get_row(spot2, past, (int) (past->m - 1));

    double prix2 = 0;
    double ic2 = 0;

    BlackScholesModel *bsmodel2 = new BlackScholesModel(size, r, rho, sigma, spot2, trend);
    Option *basket2d_2  = new BasketOption(T - t, nbTimeSteps - step, size, weights, strike);
    MonteCarlo *mCarlo2 = new MonteCarlo(bsmodel2, basket2d_2, rng, fdStep, n_samples);
    mCarlo2->price(prix2, ic2);

    ASSERT_TRUE(abs(prix-prix2)/prix2 <= 0.05); // ecart relatif inf a 5%

    pnl_mat_free(&past);
    pnl_vect_free(&sigma);
    pnl_vect_free(&spot);
    pnl_vect_free(&spot2);
    pnl_rng_free(&rng);
    delete basket2d;
    delete bsmodel;
    delete mCarlo;
    delete basket2d_2;
    delete bsmodel2;
    delete mCarlo2;

}

TEST(MonteCarlo, Perf_price_t_2){

    const char *infile = "../data/simul_perf.dat";
    const PnlMat *perfPath = pnl_mat_create_from_file(infile);

    double fdStep = 1; // valeur quelconque car non utilisee pour ce test

    int size = 5;
    double r = 0.03;
    double rho = 0.5;
    PnlVect *sigma = pnl_vect_create_from_scalar(size,0.200000);
    PnlVect *spot = pnl_vect_create_from_scalar(size,100.000000);
      PnlVect *trend = pnl_vect_create_from_zero(size);

    double T = 2.000000;
    int nbTimeSteps = 96;

    PnlRng *rng= pnl_rng_create(PNL_RNG_MERSENNE);

    size_t n_samples = 50000;

    PnlVect *weights = pnl_vect_create_from_scalar(5, 0.2);

    BlackScholesModel *bsmodel = new BlackScholesModel(size, r, rho, sigma, spot, trend);
    Option *perf = new PerformanceOption(T, nbTimeSteps, size, weights);

    pnl_rng_init(rng, PNL_RNG_MERSENNE);
    pnl_rng_sseed(rng, time(NULL));

    MonteCarlo *mCarlo = new MonteCarlo(bsmodel, perf, rng, fdStep, n_samples);

    double prix = 0.0;
    double ic = 0.0;
    double t = 0;
    double step = floor(t * nbTimeSteps / T) + 1;

    PnlMat *past = pnl_mat_create_from_scalar(1, size, 100.0000 );
    pnl_mat_extract_subblock(past, perfPath, 0, step,  0, size);
    //pnl_mat_print(past);
    mCarlo->price(past, t, prix, ic);

    PnlVect *spot2 = pnl_vect_create(size);
    pnl_mat_get_row(spot2, past, (int) (past->m - 1));

    double prix2 = 0;
    double ic2 = 0;

    BlackScholesModel *bsmodel2 = new BlackScholesModel(size, r, rho, sigma, spot2, trend);
    Option *perf2  = new PerformanceOption(T - t, nbTimeSteps - step, size, weights);
    MonteCarlo *mCarlo2 = new MonteCarlo(bsmodel2, perf2, rng, fdStep, n_samples);
    mCarlo2->price(prix2, ic2);

    ASSERT_TRUE(abs(prix-prix2)/prix2 <= 0.05); // ecart relatif inf a 5%

    pnl_mat_free(&past);
    pnl_vect_free(&sigma);
    pnl_vect_free(&spot);
    pnl_vect_free(&spot2);
    pnl_rng_free(&rng);
    delete perf;
    delete bsmodel;
    delete mCarlo;
    delete perf2;
    delete bsmodel2;
    delete mCarlo2;

}

#endif //MC_PRICER_TEST_PRICE_T_METHODE_2_HPP
