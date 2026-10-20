// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_TESTS_NPU_PORT_SPLIT_STATISTICS_CASES_H_
#define XGBOOST_TESTS_NPU_PORT_SPLIT_STATISTICS_CASES_H_
#include "histogram_cases.h"

namespace split_statistics_test {
namespace ht = histogram_test;
namespace qt = quantizer_test;
// Independent oracle classifies ORIGINAL rows for each candidate. It does not
// construct a histogram or scan, and is test scaffolding, not a host backend.
struct Channel {
  int64_t missing{0};
  std::vector<int64_t> left_missing_right, right_missing_right;
  std::vector<int64_t> left_missing_left, right_missing_left;
};
inline Channel Oracle(ht::Case const& c, std::vector<int64_t> const& rows,
                      std::size_t f, std::vector<int64_t> const& values) {
  Channel out;
  for (auto row : rows) if (c.columns[f][row] == -1) out.missing = ht::Add(out.missing, values[row]);
  for (int64_t candidate = 0; candidate < c.bin_counts[f]; ++candidate) {
    int64_t lr=0, rr=0, ll=0, rl=0;
    for (auto row : rows) {
      auto bin = c.columns[f][row]; auto value = values[row];
      if (bin >= 0 && bin < candidate) lr=ht::Add(lr,value); else rr=ht::Add(rr,value);
      if (bin == -1 || bin < candidate) ll=ht::Add(ll,value); else rl=ht::Add(rl,value);
    }
    out.left_missing_right.push_back(lr); out.right_missing_right.push_back(rr);
    out.left_missing_left.push_back(ll); out.right_missing_left.push_back(rl);
  }
  return out;
}
inline std::vector<ht::Case> Cases() {
  auto cases = ht::Cases();
  // Cross CUDA's warp/chunk boundaries, with empty bins and missing rows.
  for (int64_t bins : {31,32,33,65,257}) {
    ht::Case c{"chunks_"+std::to_string(bins),{},{},{bins},{{}}, {}};
    for (int64_t row=0; row<4*bins; ++row) {
      c.gradient.push_back((row%2 ? -1 : 1)*((INT64_C(1)<<53)+row));
      c.hessian.push_back((INT64_C(1)<<40)+row);
      c.columns[0].push_back(row%5 == 0 ? -1 : (row*7)%bins);
      c.selected.push_back(row);
    }
    std::reverse(c.selected.begin(),c.selected.end()); cases.push_back(std::move(c));
  }
  return cases;
}
inline void CheckOracle(ht::Case const& c, std::vector<int64_t> const& rows) {
  auto hist = ht::Oracle(c, rows);
  for (std::size_t f=0; f<c.bin_counts.size(); ++f) {
    for (bool hessian : {false,true}) {
      auto const& values = hessian ? c.hessian : c.gradient;
      auto root = hessian ? hist.root_h : hist.root_g;
      auto out=Oracle(c,rows,f,values);
      for (int64_t i=0; i<c.bin_counts[f]; ++i) {
        qt::Require(ht::Add(out.left_missing_right[i],out.right_missing_right[i])==root &&
                    ht::Add(out.left_missing_left[i],out.right_missing_left[i])==root,
                    "candidate child sums do not conserve root");
        qt::Require(ht::Add(out.left_missing_right[i],out.missing)==out.left_missing_left[i],
                    "candidate missing reassignment mismatch");
      }
      if (c.bin_counts[f]) qt::Require(out.left_missing_right[0]==0,"first candidate must precede first bin");
    }
  }
}
}  // namespace split_statistics_test
#endif  // XGBOOST_TESTS_NPU_PORT_SPLIT_STATISTICS_CASES_H_
