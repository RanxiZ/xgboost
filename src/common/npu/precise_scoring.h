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
  // Exact doubling, bit-identical to Mul(U{1024}<<52, a) (that constant is the
  // binary64 pattern of 2.0). Proof: with mb == kHidden == 2^52 the 53x53
  // product is exactly ma<<52, so the split yields hi = ma>>12 < 2^41 (carry 0
  // -> shift 49), lo = (ma&0xfff)<<52, and z = (hi<<15)|(lo>>49)|sticky =
  // ma<<3 with sticky 0 (bits below 49 of the product are all zero). Pack then
  // sees r == 0, m == ma in [2^52,2^53) so it neither rounds nor renormalises
  // and returns sign | (ea+1)<<52 | (ma&kFraction). Zero, invalid and
  // overflow (ea+1 == 2047) take exactly the branches Mul takes.
  XGB_NPU_PRECISE_INLINE U Double(U a) {
    if (!Valid(a)) return Fail();
    U sign=a&kSign;
    if (!(a&~kSign)) return sign;
    int e=static_cast<int>((a>>52)&2047)+1;
    if (e>=2047) return Fail();
    return sign | (static_cast<U>(e)<<52) | (a&kFraction);
  }
  // Less(0,b) for every representable b. Less keeps the "both zero -> false"
  // guard, and since a is +0 the sign-mismatch branch is taken exactly when b
  // is negative, where Less also returns false; otherwise it returns 0<b,
  // i.e. "positive and nonzero". Verified for +-0, subnormals, inf and NaN
  // bit patterns, which is the whole input space of this predicate.
  XGB_NPU_PRECISE_INLINE bool Positive(U b) {
    return !(b&kSign) && (b&~kSign)!=0;
  }
  XGB_NPU_PRECISE_INLINE bool Less(U a,U b) {
    if (!Abs(a) && !Abs(b)) return false;
    if ((a^b)&kSign) return (a&kSign)!=0;
    return (a&kSign) ? a>b : a<b;
  }
  // Closed-domain fast paths. Every value this header produces (FromInt,
  // FromFloat, Pack inside Mul/Add/Div, ToFloat/FromFloat round trips) is either
  // a normal binary64 or +-0: Pack returns the bare sign on its zero/underflow
  // paths and Fail() == 0 elsewhere, and both are Valid. Every call site in
  // WeightGainCore/EvaluatePrepared feeds only such values back in, so the
  // validity scan of Add/Mul/Div/ToFloat can never take its failure branch
  // there: skipping it changes neither a result bit nor an error bit. The
  // checked names remain the public entry points for the other stages.
  XGB_NPU_PRECISE_INLINE U Add(U a,U b) {
    if (!Valid(a) || !Valid(b)) return 0;
    return AddRaw(a,b);
  }
  XGB_NPU_PRECISE_INLINE U AddRaw(U a,U b) {
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
    return MulRaw(a,b);
  }
  XGB_NPU_PRECISE_INLINE U MulRaw(U a,U b) {
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
  // MulRaw specialised to a second operand produced by FromFloat, whose low 29
  // fraction bits are zero: mb == B<<29 with B < 2^24. Then ma*mb == W<<29 with
  // W = ma*B (< 2^77) and the 128-bit split of the product gives
  //   hi = W>>35, lo = (W & (2^35-1))<<29 (product = W<<29), carry = bit 76 of W (bit 44 of Whi),
  //   z = carry ? (W>>21)|sticky(W,21) : (W>>20)|sticky(W,20),
  //   e = ea+eb-1023+carry,
  // so Pack receives exactly the z, e and sign that MulRaw passes it. Only two
  // 32x32 multiplies are needed (32x24 and 21x24) instead of four 32x32 ones.
  XGB_NPU_PRECISE_INLINE U MulRawF32(U a,U b) {
    U sign=(a^b)&kSign;
    if (!Abs(a) || !Abs(b)) return sign;
    U ma=(a&kFraction)|kHidden;
    U B=((b&kFraction)|kHidden)>>29;
    U P=(ma&UINT32_MAX)*B;
    U Whi=(ma>>32)*B+(P>>32), Wlo=P&UINT32_MAX;
    bool carry=(Whi&(U{1}<<44))!=0;
    unsigned k=carry?21U:20U;
    U z=((Whi<<(32-k))|(Wlo>>k))|((Wlo&((U{1}<<k)-1))!=0);
    int e=static_cast<int>((a>>52)&2047)+static_cast<int>((b>>52)&2047)-1023+carry;
    return Pack(sign,e,z);
  }
  XGB_NPU_PRECISE_INLINE U Div(U a,U b) {
    if (!Valid(a) || !Valid(b) || !Abs(b)) return Fail();
    return DivRaw(a,b);
  }
  XGB_NPU_PRECISE_INLINE U DivRaw(U a,U b) {
    if (!Abs(b)) return Fail();
    U sign=(a^b)&kSign;
    if (!Abs(a)) return sign;
    U ma=(a&kFraction)|kHidden, mb=(b&kFraction)|kHidden;
    int e=static_cast<int>((a>>52)&2047)-static_cast<int>((b>>52)&2047)+1023;
    if (ma<mb) { ma<<=1; --e; }
    U remainder=ma-mb, q=1;
    // Restoring division generates 53 retained + 3 rounding bits exactly. Each
    // radix-2^k step is *exactly* k radix-2 steps of the portable loop
    // (`remainder<<=1; q<<=1; if (remainder>=mb) { remainder-=mb; ++q; }`):
    // before a step the remainder is below mb, so after shifting by k it is
    // below 2^k*mb (below 2^57 for k=4, no overflow) and the greedy tests
    // against 8mb, 4mb, 2mb, mb recover the same k quotient bits in
    // descending order. 13*4 + 3 = 55 appended bits, as before, so the
    // quotient, its three guard bits and the final sticky bit are unchanged.
    U twice=mb<<1, four=mb<<2, eight=mb<<3;
    for (unsigned i=0;i<13;++i) {
      remainder<<=4; q<<=4;
      if (remainder>=eight) { remainder-=eight; q|=8; }
      if (remainder>=four) { remainder-=four; q|=4; }
      if (remainder>=twice) { remainder-=twice; q|=2; }
      if (remainder>=mb) { remainder-=mb; q|=1; }
    }
    remainder<<=2; q<<=2;
    if (remainder>=twice) { remainder-=twice; q|=2; }
    if (remainder>=mb) { remainder-=mb; q|=1; }
    remainder<<=1; q<<=1;
    if (remainder>=mb) { remainder-=mb; q|=1; }
    q|=remainder!=0;
    return Pack(sign,e,q);
  }
  XGB_NPU_PRECISE_INLINE U FromInt(int64_t value,int scale_exponent) {
    if (scale_exponent < -1022 || scale_exponent > 1022) return Fail();
    U sign=value<0?kSign:0, magnitude=value<0?U{0}-static_cast<U>(value):static_cast<U>(value);
    if (!magnitude) return 0;
    // floor(log2(magnitude)) by width halving. This is exactly the value the
    // portable `for (n=magnitude; n>>1; n>>=1) ++top;` loop produces, but it
    // needs 6 independent tests instead of up to 63 dependent shift steps - a
    // per-candidate cost that dominates this translation on integer data.
    unsigned top=0; U n=magnitude;
    if (n>>32) { top+=32; n>>=32; }
    if (n>>16) { top+=16; n>>=16; }
    if (n>> 8) { top+= 8; n>>= 8; }
    if (n>> 4) { top+= 4; n>>= 4; }
    if (n>> 2) { top+= 2; n>>= 2; }
    if (n>> 1) { top+= 1; }
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
    return ToFloatRaw(a);
  }
  // squared == FromFloat(ToFloatRaw(MulRaw(w,w))) where w == FromFloat(w32).
  // w is exact and w*w is exact in binary64 (a 24-bit significand square needs
  // 48 bits), so the only rounding is ToFloat's round-to-nearest-even of A*A to
  // 24 bits. For binary32 exponent fields 64..189 that rounding has neither a
  // subnormal result (biased exponent stays >= 1) nor an overflow (stays
  // <= 253 including the mantissa-overflow renormalisation), so it can be
  // evaluated directly; every other pattern takes the unchanged composition.
  // Both branches return the same bits and neither raises an error bit in the
  // fast window, because ToFloatRaw there cannot fail.
  XGB_NPU_PRECISE_INLINE U SquaredFromF32(U w,uint32_t w32) {
    unsigned e=(w32>>23)&255U;
    if (e-64U<126U) {                      // 64 <= e <= 189
      U A=(static_cast<U>(w32)&0x7fffffU)|(U{1}<<23);
      U P=A*A;                             // 2^46 .. 2^48
      unsigned s=(P>>47)?24U:23U;
      U q=P>>s, r=P&((U{1}<<s)-1), half=U{1}<<(s-1);
      q+=r>half||(r==half&&(q&1));
      if (q==(U{1}<<24)) { q=U{1}<<23; ++s; }
      return (static_cast<U>(2U*e-150U+s+896U)<<52)|((q&0x7fffffU)<<29);
    }
    return FromFloat(ToFloatRaw(MulRaw(w,w)));
  }
  XGB_NPU_PRECISE_INLINE uint32_t ToFloatRaw(U a) {
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
// two_alpha == Double(alpha) and two_alpha_error == the error bit that
// Mul(U{1024}<<52, alpha) would raise; both are loop invariant for the batched
// callers, so they are hoisted into Prepared. The error bit is applied at the
// exact program point where that multiply used to happen (just before t3),
// because the early return above must not publish it.
XGB_NPU_PRECISE_INLINE Child WeightGainCore(Arithmetic& a,U g,U h,U lambda,U alpha,U delta,
                                            U two_alpha,uint32_t two_alpha_error) {
  if (!a.Positive(h)) return {0,0};
  U threshold=0;
  if (a.Less(alpha,g)) threshold=a.AddRaw(g,a.Neg(alpha));
  else if (a.Less(g,a.Neg(alpha))) threshold=a.AddRaw(g,alpha);
  // One Add per child instead of two: the Div denominator and t2's first factor
  // are the same call on the same operands, and Add's value is a function of
  // its operands alone (it only ORs error bit 0, which is idempotent), so the
  // shared value is bit-identical to evaluating the expression twice.
  U denom=a.AddRaw(h,lambda);
  U weight=a.DivRaw(a.Neg(threshold),denom);
  if (a.Abs(delta) && a.Less(delta,a.Abs(weight))) weight=delta|(weight&kSign);
  auto w32=a.ToFloatRaw(weight); U w=a.FromFloat(w32);
  // The official expression squares its float weight IN FLOAT before promotion.
  U squared=a.SquaredFromF32(w,w32);
  // `two` in the portable code is the binary64 pattern of 2.0; Double() is
  // bit-identical and needs no 52x52 multiply. two_alpha is Double(alpha), loop
  // invariant, and therefore hoisted to Prepared by every batched caller.
  U t1=a.MulRawF32(a.Double(g),w);
  U t2=a.MulRawF32(denom,squared);
  a.error|=two_alpha_error;
  U t3=a.MulRawF32(two_alpha,a.Abs(w));
  auto gain=a.ToFloatRaw(a.Neg(a.AddRaw(a.AddRaw(t1,t2),t3)));
  return {w32,gain};
}
XGB_NPU_PRECISE_INLINE Child WeightGain(Arithmetic& a,U g,U h,U lambda,U alpha,U delta) {
  if (!a.Positive(h)) return {0,0};
  Arithmetic b;
  U two_alpha=b.Double(alpha);
  return WeightGainCore(a,g,h,lambda,alpha,delta,two_alpha,b.error);
}
// The Parameters -> binary64 decoding of Evaluate is loop invariant: it reads
// only p. Hoisting it out of a candidate loop is bit-exact because the decode
// writes nothing into the per-candidate error state that a later step reads.
struct Prepared {
  U minimum, lambda, alpha, delta;
  U two_alpha;   // Double(alpha): exact 2.0*alpha, loop invariant per launch.
  int32_t gradient_scale, hessian_scale;
  uint32_t error, two_alpha_error;
};
XGB_NPU_PRECISE_INLINE Prepared Prepare(Parameters p) {
  Arithmetic a;
  Prepared q;
  q.minimum=a.FromFloat(p.min_child_weight);q.lambda=a.FromFloat(p.reg_lambda);
  q.alpha=a.FromFloat(p.reg_alpha);q.delta=a.FromFloat(p.max_delta_step);
  // Separate arithmetic so the hoisted multiply cannot publish an error bit
  // that the original only raised once a child was actually evaluated.
  Arithmetic b;
  q.two_alpha=b.Double(q.alpha);
  q.two_alpha_error=b.error;
  if ((p.min_child_weight|p.reg_lambda|p.reg_alpha|p.max_delta_step)&0x80000000U ||
      p.gradient_scale < -1022 || p.gradient_scale > 1022 ||
      p.hessian_scale < -1022 || p.hessian_scale > 1022) a.error|=2;
  q.gradient_scale=p.gradient_scale;q.hessian_scale=p.hessian_scale;
  q.error=a.error;
  return q;
}
// Mirrors scalar TreeEvaluator::SplitEvaluator, WITHOUT monotone constraints.
// No FMA contraction; actual CUDA compiler contraction remains a separate test.
XGB_NPU_PRECISE_INLINE Score EvaluatePrepared(int64_t lg,int64_t lh,int64_t rg,int64_t rh,Prepared q) {
  Arithmetic a;
  a.error=q.error;
  if (a.error) return {0,0,0,a.error};
  U lgrad=a.FromInt(lg,q.gradient_scale),lhess=a.FromInt(lh,q.hessian_scale);
  U rgrad=a.FromInt(rg,q.gradient_scale),rhess=a.FromInt(rh,q.hessian_scale);
  if (a.error) return {0,0,0,a.error};
  if (!a.Positive(lhess) || !a.Positive(rhess) || a.Less(lhess,q.minimum) || a.Less(rhess,q.minimum)) {
    return {0,0,0xff800000U,0};
  }
  auto left=WeightGainCore(a,lgrad,lhess,q.lambda,q.alpha,q.delta,q.two_alpha,q.two_alpha_error);
  auto right=WeightGainCore(a,rgrad,rhess,q.lambda,q.alpha,q.delta,q.two_alpha,q.two_alpha_error);
  auto gain=a.ToFloatRaw(a.AddRaw(a.FromFloat(left.gain),a.FromFloat(right.gain)));
  return {left.weight,right.weight,gain,a.error};
}
XGB_NPU_PRECISE_INLINE Score Evaluate(int64_t lg,int64_t lh,int64_t rg,int64_t rh,Parameters p) {
  return EvaluatePrepared(lg,lh,rg,rh,Prepare(p));
}
}  // namespace xgboost::common::npu::precise
#endif  // XGBOOST_COMMON_NPU_PRECISE_SCORING_H_
