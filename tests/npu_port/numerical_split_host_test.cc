// Copyright 2026, XGBoost Contributors
#include "numerical_split_cases.h"
int main() {
  try {numerical_test::HostTests();std::cout<<"NUMERICAL_SPLIT_HOST_PASS checks=4 cases="<<numerical_test::Cases().size()<<'\n';return 0;}
  catch (std::exception const& e) {std::cerr<<"NUMERICAL_SPLIT_HOST_FAIL: "<<e.what()<<'\n';return 1;}
}