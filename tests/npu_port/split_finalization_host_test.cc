// Copyright 2026, XGBoost Contributors
#include "split_finalization_cases.h"
namespace p=xgboost::common::npu::precise;
namespace t=finalization_test;
int main() {
  try {
    quantizer_test::Require(std::fegetround()==FE_TONEAREST,"host round mode");
    auto nodes=t::NodeCases();
    for (auto const& c:nodes) t::Compare(p::NodeQuantities(c.input.gradient,c.input.hessian,c.input.parameters),c);
    quantizer_test::Passed("host_node_quantities_bitwise");
    auto directions=t::DirectionCases();
    for (auto v:directions) t::Compare(p::ChooseDirection(v.missing_left,v.missing_right),v);
    quantizer_test::Passed("host_missing_direction_bitwise");
    auto gains=t::GainCases();
    for (auto v:gains) t::Compare(p::SubtractParent(v.selected_child_gain,v.parent_gain),v);
    quantizer_test::Passed("host_parent_subtraction_bitwise");
    t::OrderingWitness();quantizer_test::Passed("host_subtract_after_selection_witness");
    std::cout<<"SPLIT_FINALIZATION_HOST_PASS checks=4 nodes="<<nodes.size()<<" directions="<<directions.size()
             <<" gains="<<gains.size()<<" candidate_reduction=not_implemented cuda_execution=not_run\n";
    return 0;
  } catch (std::exception const& e) {std::cerr<<"SPLIT_FINALIZATION_HOST_FAIL: "<<e.what()<<'\n';return 1;}
}
