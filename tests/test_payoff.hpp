//
// Created by teukengd on 9/19/18.
//
#pragma once


#ifndef CHAMILO_TEST_ASSET_H
#define CHAMILO_TEST_ASSET_H
#include <gtest/gtest.h>
#include "pnl/pnl_matrix.h"



TEST(CallOption, callOptionPos){
    //PnlMat *callPath = pnl_mat_create_from_file("../../market-data/simul_call.dat");
    const char *file = "../../market-data/simul_call.dat";
    PnlMat *callPath = pnl_mat_create_from_file(file);
    PnlVect *weights = pnl_vect_create_from_scalar(1, 1.0);
    CallOption *call  = new CallOption(1, 365, 1, weights, 98);
    double result = call->payoff(callPath);
    ASSERT_TRUE(result >= 16.088379 - 0.0001 && result <= 16.088379 + 0.0001);

    pnl_mat_free(&callPath);
    delete call;
}

TEST(CallOption, callOptionNul){
    const char *file = "../../market-data/simul_call.dat";
    PnlMat *callPath = pnl_mat_create_from_file(file);
    PnlVect *weights = pnl_vect_create_from_scalar(1, 1.0);    
    CallOption *call  = new CallOption(1, 365, 1, weights, 118);
    double result = call->payoff(callPath);
    ASSERT_EQ(result, 0);

    pnl_mat_free(&callPath);
    delete call;
}

TEST(BasketOption, BasketOption2dPos){
    const char *file = "../../market-data/simul_basket_2d.dat";
    PnlMat *basketPath = pnl_mat_create_from_file(file);
    PnlVect *weights = pnl_vect_create_from_scalar(2, 0.5);
    BasketOption *basket  = new BasketOption(1, 365, 2, weights, 100);
    double result = basket->payoff(basketPath);
    ASSERT_TRUE(result >= 15.614674 - 0.0001 && result <= 15.614674 + 0.0001);

    pnl_mat_free(&basketPath);
    delete basket;
}

TEST(BasketOption, BasketOption2dNul){
    const char *file = "../../market-data/simul_basket_2d.dat";
    PnlMat *basketPath = pnl_mat_create_from_file(file);
    PnlVect *weights = pnl_vect_create_from_scalar(2, 0.5);    
    BasketOption *basket  = new BasketOption(1, 365, 2, weights, 120);
    double result = basket->payoff(basketPath);
    ASSERT_EQ(result,0);

    pnl_mat_free(&basketPath);
    delete basket;
}

TEST(BasketOption, BasketOptionPos){
    const char *file = "../../market-data/simul_basket.dat";
    PnlMat *basketPath = pnl_mat_create_from_file(file);
    PnlVect *weights = pnl_vect_create_from_scalar(5, 0.2);    
    BasketOption *basket  = new BasketOption(1, 365, 5, weights, 100);
    double result = basket->payoff(basketPath);
    ASSERT_TRUE(result >= 3.4349628 - 0.0001 && result <= 3.4349628 + 0.0001);

    pnl_mat_free(&basketPath);
    delete basket;
}

TEST(BasketOption, BasketOptionNul){
    const char *file = "../../market-data/simul_basket.dat";
    PnlMat *basketPath = pnl_mat_create_from_file(file);
    PnlVect *weights = pnl_vect_create_from_scalar(5, 0.2);        
    BasketOption *basket  = new BasketOption(1, 365, 5, weights, 110);
    double result = basket->payoff(basketPath);
    ASSERT_EQ(result,0);

    pnl_mat_free(&basketPath);
    delete basket;
}

TEST(BasketOption, BasketOption1Pos){
    const char *file = "../../market-data/simul_basket_1.dat";
    PnlMat *basketPath = pnl_mat_create_from_file(file);
    PnlVect *weights = pnl_vect_create_from_scalar(40, 0.025);        
    BasketOption *basket  = new BasketOption(3, 1000, 40, weights, 100);
    double result = basket->payoff(basketPath);
    ASSERT_TRUE(result >= 3.79563912499998 - 0.0001 && result <= 3.79563912499998 + 0.0001);

    pnl_mat_free(&basketPath);
    delete basket;
}

TEST(BasketOption, BasketOption1Nul){
    const char *file = "../../market-data/simul_basket_1.dat";
    PnlMat *basketPath = pnl_mat_create_from_file(file);
    PnlVect *weights = pnl_vect_create_from_scalar(40, 0.025);            
    BasketOption *basket  = new BasketOption(3, 1000, 40, weights, 110);
    double result = basket->payoff(basketPath);
    ASSERT_EQ(result,0);

    pnl_mat_free(&basketPath);
    delete basket;
}

TEST(BasketOption, BasketOption2Pos){
    const char *file = "../../market-data/simul_basket_2.dat";
    PnlMat *basketPath = pnl_mat_create_from_file(file);
    PnlVect *weights = pnl_vect_create_from_scalar(40, 0.025);            
    BasketOption *basket  = new BasketOption(1, 365, 40, weights, 100);
    double result = basket->payoff(basketPath);
    ASSERT_TRUE(result >= 35.6169229500001 - 0.0001 && result <= 35.6169229500001 + 0.0001);

    pnl_mat_free(&basketPath);
    delete basket;
}

TEST(BasketOption, BasketOption2Nul){
    const char *file = "../../market-data/simul_basket_2.dat";
    PnlMat *basketPath = pnl_mat_create_from_file(file);
    PnlVect *weights = pnl_vect_create_from_scalar(40, 0.025);            
    BasketOption *basket  = new BasketOption(1, 365, 40, weights, 140);
    double result = basket->payoff(basketPath);
    ASSERT_EQ(result,0);

    pnl_mat_free(&basketPath);
    delete basket;
}

TEST(AsianOption, AsianOptionPos){
    const char *file = "../../market-data/simul_asian.dat";
    PnlMat *asianPath = pnl_mat_create_from_file(file);
    PnlVect *weights = pnl_vect_create_from_scalar(2, 0.5);            
    AsianOption *asian  = new AsianOption(1.5, 450, 2, weights, 90);
    double result = asian->payoff(asianPath);
    ASSERT_TRUE(result >= 2.1552890886918 - 0.0001 && result <= 2.1552890886918 + 0.0001);

    pnl_mat_free(&asianPath);
    delete asian;
}

TEST(AsianOption, AsianOptionNul){
    const char *file = "../../market-data/simul_asian.dat";
    PnlMat *asianPath = pnl_mat_create_from_file(file);
    PnlVect *weights = pnl_vect_create_from_scalar(2, 0.5);                
    AsianOption *asian  = new AsianOption(1.5, 450, 2, weights, 95);
    double result = asian->payoff(asianPath);
    ASSERT_EQ(result,0);

    pnl_mat_free(&asianPath);
    delete asian;
}

TEST(PerformanceOption, PerformanceOptionPos){
    const char *file = "../../market-data/simul_perf.dat";
    PnlMat *perfPath = pnl_mat_create_from_file(file);
    PnlVect *weights = pnl_vect_create_from_scalar(5, 0.2);                    
    PerformanceOption *perf  = new PerformanceOption(2, 96, 5, weights);
    double result = perf->payoff(perfPath);
    ASSERT_TRUE(result >= 1.88337442199339 - 0.0001 && result <= 1.88337442199339 + 0.0001);

    pnl_mat_free(&perfPath);
    delete perf;
}



#endif //CHAMILO_TEST_ASSET_H
