
add_executable(test1 main.cpp ${SOURCESTEST} ${HEADERSTEST})
target_link_libraries(test1 gtest gtest_main ${LIBS})
target_link_libraries(test1 jl_parser ${LIBS})

add_executable(payoff main_payoff.cpp ${SOURCESTEST} ${HEADERSTEST})
target_link_libraries(payoff gtest gtest_main ${LIBS})
target_link_libraries(payoff jl_parser ${LIBS})

add_executable(price_0 main_price_0.cpp ${SOURCESTEST} ${HEADERSTEST})
target_link_libraries(price_0 gtest gtest_main ${LIBS})
target_link_libraries(price_0 jl_parser ${LIBS})

add_executable(price_t main_price_t.cpp ${SOURCESTEST} ${HEADERSTEST})
target_link_libraries(price_t gtest gtest_main ${LIBS})
target_link_libraries(price_t jl_parser ${LIBS})
