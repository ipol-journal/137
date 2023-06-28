#ifndef _MATOPS_HPP_
#define  _MATOPS_HPP_

void mmprc(float *, size_t, size_t, const float *, const float *, size_t, const float *);
void mmT(float *, size_t, size_t, const float *, const float *, size_t);
void mTm(float *, size_t, size_t, const float *, const float *, size_t);
void mTma(float *, size_t, size_t, const float *, const float *, size_t, const float *);
void omsq(float *, const float *, size_t);
void rsum(float *, const float *, size_t, size_t);
void mtanh(float *, size_t);
void axpb(float *, size_t, float, float);

#endif  /* !_MATOPS_HPP_ */
