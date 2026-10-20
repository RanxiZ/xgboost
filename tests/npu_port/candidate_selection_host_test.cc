// Copyright 2026, XGBoost Contributors
#include "candidate_selection_cases.h"
int main() {
  try {selection_test::HostTests();std::cout<<"CANDIDATE_SELECTION_HOST_PASS checks=4 cases="<<selection_test::Cases().size()<<'\n';return 0;}
  catch (std::exception const& e) {std::cerr<<"CANDIDATE_SELECTION_HOST_FAIL: "<<e.what()<<'\n';return 1;}
}