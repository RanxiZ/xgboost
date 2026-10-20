// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_PRECISE_SCORING_H_
#define XGBOOST_COMMON_NPU_PRECISE_SCORING_H_
#include <cstdint>
// Portable integer arithmetic shared by host tests and an Ascend SIMT kernel.
// No host numerical fallback: production use is exclusively the device kernel.
#ifndef XGB_NPU_PRECISE_INLINE
#define XGB_NPU_PRECISE_INLINE inline
#endif
namespace xgboost::common::npu::precise {
using U = uint64_t;
constexpr U kSign = U{1} << 63, kHidden = U{1} << 52, kFraction = kHidden - 1;
// Supported binary64 domain: signed zero and normal finite numbers. Rounded
// nonzero subnormal results and overflow are rejected. Correctly rounded signed
// zero underflow is supported; no flush-to-zero approximation is performed.
// Float32 subnormals ARE supported at the explicit float rounding boundaries.
struct Arithmetic {
  uint32_t error{0};
  XGB_NPU_PRECISE_INLINE U Fail() { error |= 1; return 0; }
  XGB_NPU_PRECISE_INLINE bool Valid(U a) {
    auto e = (a >> 52) & 2047;
    if (e == 2047 || (e == 0 && (a & kFraction))) { Fail(); return false; }
    return true;
  }
  XGB_NPU_PRECISE_INLINE U Jam(U a, unsigned shift) {
    if (!shift) return a;
    if (shift >= 64) return a != 0;
    return (a >> shift) | ((a << (64-shift)) != 0);
  }
  XGB_NPU_PRECISE_INLINE U Round(U a, unsigned shift) {
    if (!shift) return a;
    if (shift > 64) return 0;
    if (shift == 64) return a > (U{1} << 63);
    U q=a>>shift, r=a&((U{1}<<shift)-1), half=U{1}<<(shift-1);
    return q + (r>half || (r==half && (q&1)));
  }
  // z has its leading bit at 55; low three bits are guard/round/sticky.
  XGB_NPU_PRECISE_INLINE U Pack(U sign, int exponent, U z) {
    if (!z) return sign;
    if (exponent>=2047) return Fail();
    if (exponent < -64) return sign;
    if (exponent<=0) { z=Jam(z,static_cast<unsigned>(1-exponent)); exponent=1; }
    U m=z>>3, r=z&7;
    m += r>4 || (r==4 && (m&1));
    if (!m) return sign;
    if (m<kHidden) return Fail();
    if (m >= (U{1}<<53)) { m>>=1; ++exponent; }
    if (exponent>=2047) return Fail();
    return sign | (static_cast<U>(exponent)<<52) | (m&kFraction);
  }
  XGB_NPU_PRECISE_INLINE U Neg(U a) { return a ^ kSign; }
  XGB_NPU_PRECISE_INLINE U Abs(U a) { return a & ~kSign; }
  XGB_NPU_PRECISE_INLINE bool Less(U a,U b) {
    if (!Abs(a) && !Abs(b)) return false;
    if ((a^b)&kSign) return (a&kSign)!=0;
    return (a&kSign) ? a>b : a<b;
  }
  XGB_NPU_PRECISE_INLINE U Add(U a,U b) {
    if (!Valid(a) || !Valid(b)) return 0;
    if (Abs(a)<Abs(b)) { U tmp=a; a=b; b=tmp; }
    if (!Abs(b)) return !Abs(a) ? (a&b&kSign) : a;
    int ea=static_cast<int>((a>>52)&2047), eb=static_cast<int>((b>>52)&2047);
    U ma=((a&kFraction)|kHidden)<<3, mb=Jam(((b&kFraction)|kHidden)<<3,ea-eb), z;
    if (!((a^b)&kSign)) {
      z=ma+mb;
      if (z&(U{1}<<56)) { z=Jam(z,1); ++ea; }
    } else {
      z=ma-mb;
      if (!z) return 0;
      while (z<(U{1}<<55)) { z<<=1; --ea; }
    }
    return Pack(a&kSign,ea,z);
  }
  XGB_NPU_PRECISE_INLINE U Mul(U a,U b) {
    if (!Valid(a) || !Valid(b)) return 0;
    U sign=(a^b)&kSign;
    if (!Abs(a) || !Abs(b)) return sign;
    U ma=(a&kFraction)|kHidden, mb=(b&kFraction)|kHidden;
    U al=ma&UINT32_MAX, ah=ma>>32, bl=mb&UINT32_MAX, bh=mb>>32;
    U t=al*bl, low=t&UINT32_MAX;
    t=ah*bl+(t>>32); U middle=t&UINT32_MAX, high=t>>32;
    t=al*bh+middle;
    U lo=(t<<32)|low, hi=ah*bh+high+(t>>32);
    bool carry=(hi&(U{1}<<41))!=0; unsigned shift=carry?50:49;
    U z=(hi<<(64-shift))|(lo>>shift)|((lo&((U{1}<<shift)-1))!=0);
    int e=static_cast<int>((a>>52)&2047)+static_cast<int>((b>>52)&2047)-1023+carry;
    return Pack(sign,e,z);
  }
  XGB_NPU_PRECISE_INLINE U Div(U a,U b) {
    if (!Valid(a) || !Valid(b) || !Abs(b)) return Fail();
    U sign=(a^b)&kSign;
    if (!Abs(a)) return sign;
    U ma=(a&kFraction)|kHidden, mb=(b&kFraction)|kHidden;
    int e=static_cast<int>((a>>52)&2047)-static_cast<int>((b>>52)&2047)+1023;
    if (ma<mb) { ma<<=1; --e; }
    U remainder=ma-mb, q=1;
    // Restoring division generates 53 retained + 3 rounding bits exactly.
    for (unsigned i=0;i<55;++i) {
      remainder<<=1; q<<=1;
      if (remainder>=mb) { remainder-=mb; ++q; }
    }
    q|=remainder!=0;
    return Pack(sign,e,q);
  }
  XGB_NPU_PRECISE_INLINE U FromInt(int64_t value,int scale_exponent) {
    if (scale_exponent < -1022 || scale_exponent > 1022) return Fail();
    U sign=value<0?kSign:0, magnitude=value<0?U{0}-static_cast<U>(value):static_cast<U>(value);
    if (!magnitude) return 0;
    unsigned top=0; for (U n=magnitude;n>>1;n>>=1) ++top;
    U z=top>55?Jam(magnitude,top-55):magnitude<<(55-top);
    U rounded=Pack(sign,1023+static_cast<int>(top),z);
    int e=static_cast<int>((rounded>>52)&2047)-scale_exponent;
    if (e<=0 || e>=2047) return Fail();
    return (rounded&(kSign|kFraction)) | (static_cast<U>(e)<<52);
  }
  XGB_NPU_PRECISE_INLINE U FromFloat(uint32_t a) {
    U sign=static_cast<U>(a>>31)<<63; unsigned e=(a>>23)&255; U m=a&0x7fffff;
    if (e==255) return Fail();
    if (!e) {
      if (!m) return sign;
      int exponent=-126;
      while (m<(U{1}<<23)) { m<<=1; --exponent; }
      return sign | (static_cast<U>(exponent+1023)<<52) | ((m&0x7fffff)<<29);
    }
    return sign | (static_cast<U>(e+896)<<52) | (m<<29);
  }
  XGB_NPU_PRECISE_INLINE uint32_t ToFloat(U a) {
    if (!Valid(a)) return 0;
    uint32_t sign=static_cast<uint32_t>(a>>63)<<31;
    if (!Abs(a)) return sign;
    int e=static_cast<int>((a>>52)&2047)-896;
    U m=(a&kFraction)|kHidden;
    if (e<=0) return sign | static_cast<uint32_t>(Round(m,static_cast<unsigned>(30-e)));
    m=Round(m,29);
    if (m>=(U{1}<<24)) { m>>=1; ++e; }
    if (e>=255) { Fail(); return 0; }
    return sign | (static_cast<uint32_t>(e)<<23) | static_cast<uint32_t>(m&0x7fffff);
  }
};
struct Parameters {
  uint32_t min_child_weight, reg_lambda, reg_alpha, max_delta_step;
  int32_t gradient_scale, hessian_scale;
};
struct Score { uint32_t left_weight, right_weight, gain, error; };
struct Child { uint32_t weight, gain; };
XGB_NPU_PRECISE_INLINE Child WeightGain(Arithmetic& a,U g,U h,U lambda,U alpha,U delta) {
  if (!a.Less(0,h)) return {0,0};
  U threshold=0;
  if (a.Less(alpha,g)) threshold=a.Add(g,a.Neg(alpha));
  else if (a.Less(g,a.Neg(alpha))) threshold=a.Add(g,alpha);
  U weight=a.Div(a.Neg(threshold),a.Add(h,lambda));
  if (a.Abs(delta) && a.Less(delta,a.Abs(weight))) weight=delta|(weight&kSign);
  auto w32=a.ToFloat(weight); U w=a.FromFloat(w32);
  // The official expression squares its float weight IN FLOAT before promotion.
  U squared=a.FromFloat(a.ToFloat(a.Mul(w,w)));
  U two=U{1024}<<52;
  U t1=a.Mul(a.Mul(two,g),w);
  U t2=a.Mul(a.Add(h,lambda),squared);
  U t3=a.Mul(a.Mul(two,alpha),a.Abs(w));
  auto gain=a.ToFloat(a.Neg(a.Add(a.Add(t1,t2),t3)));
  return {w32,gain};
}
// Mirrors scalar TreeEvaluator::SplitEvaluator, WITHOUT monotone constraints.
// No FMA contraction; actual CUDA compiler contraction remains a separate test.
XGB_NPU_PRECISE_INLINE Score Evaluate(int64_t lg,int64_t lh,int64_t rg,int64_t rh,Parameters p) {
  Arithmetic a;
  U minimum=a.FromFloat(p.min_child_weight),lambda=a.FromFloat(p.reg_lambda);
  U alpha=a.FromFloat(p.reg_alpha),delta=a.FromFloat(p.max_delta_step);
  if ((p.min_child_weight|p.reg_lambda|p.reg_alpha|p.max_delta_step)&0x80000000U ||
      p.gradient_scale < -1022 || p.gradient_scale > 1022 ||
      p.hessian_scale < -1022 || p.hessian_scale > 1022) a.error|=2;
  if (a.error) return {0,0,0,a.error};
  U lgrad=a.FromInt(lg,p.gradient_scale),lhess=a.FromInt(lh,p.hessian_scale);
  U rgrad=a.FromInt(rg,p.gradient_scale),rhess=a.FromInt(rh,p.hessian_scale);
  if (a.error) return {0,0,0,a.error};
  if (!a.Less(0,lhess) || !a.Less(0,rhess) || a.Less(lhess,minimum) || a.Less(rhess,minimum)) {
    return {0,0,0xff800000U,0};
  }
  auto left=WeightGain(a,lgrad,lhess,lambda,alpha,delta);
  auto right=WeightGain(a,rgrad,rhess,lambda,alpha,delta);
  auto gain=a.ToFloat(a.Add(a.FromFloat(left.gain),a.FromFloat(right.gain)));
  return {left.weight,right.weight,gain,a.error};
}
}  // namespace xgboost::common::npu::precise
#endif  // XGBOOST_COMMON_NPU_PRECISE_SCORING_H_
