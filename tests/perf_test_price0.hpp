#ifndef CHAMILO_TEST_PERF_PRICE0
#define CHAMILO_TEST_PERF_PRICE0
#include <gtest/gtest.h>
#include "jlparser/parser.hpp"
#include "pnl/pnl_random.h"
#include "pnl/pnl_vector.h"

/**
* Programme de test pour le prix en 0 d'une option performance
*/

TEST(MonteCarlo, Performance_price_0){
    double fdStep = 1;
    double T, r, strike, correlation;
    PnlVect *spot, *sigma, *divid, *payoff_coef;
    string type;
    int size, timestep;
    size_t n_samples;

    const char *infile = "../data/perf.dat";
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
    P->extract("timestep number", timestep);
    P->extract("sample number", n_samples);
    P->extract("payoff coefficients", payoff_coef, size);
    PnlVect *trend = pnl_vect_create_from_zero(size);

    BlackScholesModel *bsmodel = new BlackScholesModel(size, r, correlation, sigma, spot, trend);
    Option *pOption = new PerformanceOption(T, timestep, size, payoff_coef);
    PnlRng *rng= pnl_rng_create(PNL_RNG_MERSENNE);
    //

    pnl_rng_init(rng, PNL_RNG_MERSENNE);
    pnl_rng_sseed(rng, time(NULL));
    MonteCarlo *mCarlo = new MonteCarlo(bsmodel, pOption, rng, fdStep, n_samples);
    double prix = 0.0;
    double ic = 0.0;
    mCarlo->price(prix , ic);
    ASSERT_LE(1.257353 - ic, prix) << "Error, price at t=0 not in confidence interval, too low";
    ASSERT_GE(1.257353 + ic, prix) << "Error, price at t=0 not in confidence interval, too high";
    ASSERT_TRUE(abs(ic / 1.96 - 0.000587)/0.000587 <= 0.05); // erreur relative inf a 5%

    pnl_vect_free(&spot);
    pnl_vect_free(&sigma);
    pnl_vect_free(&divid);
    pnl_rng_free(&rng);
    delete P;
    delete bsmodel;
    delete pOption;
    delete mCarlo;
}


#endif //CHAMILO_TEST_ASIAN_PRICE0
