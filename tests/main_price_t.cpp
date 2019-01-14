//
// Created by lecalvmy on 9/21/18.
//

#include <gtest/gtest.h>
#include "../src/BlackScholesModel.cpp"
#include "../src/MonteCarlo.cpp"
#include "../src/AsianOption.cpp"
#include "../src/BasketOption.cpp"
#include "../src/CallOption.cpp"
#include "../src/PerformanceOption.cpp"
#include "test_price_t_methode1.hpp"
#include "test_price_t_methode2.hpp"

int main(int argc,char *argv[]) {
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
