// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_TESTS_NPU_PORT_HISTOGRAM_CASES_H_
#define XGBOOST_TESTS_NPU_PORT_HISTOGRAM_CASES_H_
#include "quantizer_cases.h"
#include "../../src/common/npu/histogram_contract.h"
#include <algorithm>
#include <numeric>
#include <set>

namespace histogram_test {
namespace qt=quantizer_test;
struct Case {
  std::string name;
  std::vector<int64_t> gradient,hessian,bin_counts;
  std::vector<std::vector<int64_t>> columns;
  std::vector<int64_t> selected;
};
struct Expected {
  std::vector<std::vector<int64_t>> gradient,hessian;
  int64_t root_g{0},root_h{0};
};
inline int64_t Add(int64_t a,int64_t b) {
  if ((b>0 && a>INT64_MAX-b) || (b<0 && a<INT64_MIN-b)) throw std::overflow_error("test sum overflow");
  return a+b;
}
inline Expected Oracle(Case const& c,std::vector<int64_t> const& rows) {
  xgboost::common::npu::ValidateHistogramLayout(c.gradient.size(),c.bin_counts);
  qt::Require(c.hessian.size()==c.gradient.size() && c.columns.size()==c.bin_counts.size(),"bad fixture shape");
  Expected out;
  for (std::size_t f=0;f<c.columns.size();++f) {
    qt::Require(c.columns[f].size()==c.gradient.size(),"bad column fixture shape");
    out.gradient.emplace_back(std::max<int64_t>(1,c.bin_counts[f]),0);
    out.hessian.emplace_back(std::max<int64_t>(1,c.bin_counts[f]),0);
    for (auto bin:c.columns[f]) qt::Require(bin>=-1 && bin<c.bin_counts[f],"bad fixture bin");
  }
  std::set<int64_t> unique;
  for (auto row:rows) {
    qt::Require(row>=0 && static_cast<std::size_t>(row)<c.gradient.size() && unique.insert(row).second,"bad fixture row");
    out.root_g=Add(out.root_g,c.gradient[row]); out.root_h=Add(out.root_h,c.hessian[row]);
    for (std::size_t f=0;f<c.columns.size();++f) {
      auto bin=c.columns[f][row]; if (bin==-1) continue;
      out.gradient[f][bin]=Add(out.gradient[f][bin],c.gradient[row]);
      out.hessian[f][bin]=Add(out.hessian[f][bin],c.hessian[row]);
    }
  }
  return out;
}
inline std::vector<Case> Cases() {
  constexpr int64_t s=INT64_C(1)<<58, big=(INT64_C(1)<<62)-1;
  std::vector<Case> cases{
    {"cuda_fixed",{-2*s,s,3*s,0,-s,2*s,4*s,0},{s,s,2*s,0,s,3*s,2*s,0},{4},
     {{0,1,1,2,-1,0,2,-1}},{0,1,2,3,4,5,6,7}},
    {"feature_local_bins_and_shuffled_rows",{7,-9,11,0,13,-15},{1,2,3,0,4,5},{3,2,0},
     {{0,1,2,-1,0,2},{1,0,-1,1,0,0},{-1,-1,-1,-1,-1,-1}},{5,0,3,2}},
    {"all_missing",{1,-2,3},{1,0,2},{2,0},{{-1,-1,-1},{-1,-1,-1}},{0,2,1}},
    {"constant_and_zero_weight",{0,0,17,-17},{0,0,1,1},{1},{{0,0,0,0}},{3,1,0,2}},
    {"empty_node",{3,-4},{1,1},{2},{{0,1}},{}},
    {"single_row",{-big},{big},{1},{{0}},{0}},
    {"near_both_int64_limits",{big,big,1,-big,-big,-1},{big,big,1,big,big,1},{1,2},
     {{0,0,0,0,0,0},{0,1,0,0,1,1}},{0,3,2,5,1,4}},
    {"integer_low_word_carry",{(INT64_C(1)<<32)-1,2,-3,(INT64_C(1)<<54)+1},
     {1,(INT64_C(1)<<32)+1,3,7},{2},{{0,0,1,1}},{3,0,2,1}}
  };
  // Keep both channel sign totals <=INT64_MAX, including the near-limit case.
  cases[6].hessian={big,big,1,0,0,0};
  Case collision{"high_collision",std::vector<int64_t>(65537),std::vector<int64_t>(65537,1),{1},
                 {std::vector<int64_t>(65537,0)},std::vector<int64_t>(65537)};
  for (std::size_t i=0;i<collision.gradient.size();++i) collision.gradient[i]=(i%2 ? -1 : 1)*((INT64_C(1)<<40)+1);
  std::iota(collision.selected.begin(),collision.selected.end(),0);
  cases.push_back(std::move(collision));
  std::mt19937 random{20261018};
  Case r{"random_sparse",std::vector<int64_t>(4096),std::vector<int64_t>(4096),{17,3,1},
         std::vector<std::vector<int64_t>>(3,std::vector<int64_t>(4096)),{}};
  for (std::size_t i=0;i<4096;++i) {
    r.gradient[i]=static_cast<int64_t>(random()%2000001)-1000000;
    r.hessian[i]=random()%10000;
    for (std::size_t f=0;f<3;++f) r.columns[f][i]=static_cast<int64_t>(random()%(r.bin_counts[f]+1))-1;
    if (random()%3) r.selected.push_back(i);
  }
  std::shuffle(r.selected.begin(),r.selected.end(),random); cases.push_back(std::move(r));
  return cases;
}
inline std::pair<std::vector<int64_t>,std::vector<int64_t>> Split(std::vector<int64_t> const& rows) {
  std::pair<std::vector<int64_t>,std::vector<int64_t>> result;
  for (std::size_t i=0;i<rows.size();++i) (i%2?result.second:result.first).push_back(rows[i]);
  return result;
}
}  // namespace histogram_test
#endif  // XGBOOST_TESTS_NPU_PORT_HISTOGRAM_CASES_H_
