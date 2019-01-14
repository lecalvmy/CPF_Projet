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
#include "asian_test_price0.hpp"
#include "basket_test_price0.hpp"
#include "basket1_test_price0.hpp"
#include "basket2_test_price0.hpp"
#include "perf_test_price0.hpp"

int main(int argc,char *argv[]) {
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}