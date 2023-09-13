//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "BigInt.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "Util.hpp"

namespace {
struct DoubleRepr {
  constexpr static unsigned exponentSize = 11;
  constexpr static int bias = std::numeric_limits<double>::max_exponent - 1;
  constexpr static int mantissaSize = std::numeric_limits<double>::digits - 1;

  std::uint64_t mantissa : mantissaSize;
  std::uint64_t biasedExponent : exponentSize;
  std::uint64_t sign : 1;

  explicit DoubleRepr(double value) : mantissa{}, biasedExponent{}, sign{} {
    std::memcpy(this, &value, sizeof(DoubleRepr));
  }

  [[nodiscard]] double toDouble() const {
    double result;
    std::memcpy(&result, this, sizeof(double));
    return result;
  }

  [[nodiscard]] int getExponent() const {
    return static_cast<int>(biasedExponent) - bias;
  }

  void setExponent(int exponent) {
    biasedExponent = exponent + bias;
  }
};

static_assert(sizeof(DoubleRepr) == sizeof(double));

} // namespace

std::pair<pylir::BigInt, pylir::BigInt> pylir::toRatio(double value) {
  static_assert(std::numeric_limits<double>::radix == 2 &&
                std::numeric_limits<double>::is_iec559);

  PYLIR_ASSERT(std::isfinite(value));

  // Special case below does not handle 0, as it is a special representation in
  // the floating point representation, where mantissa and exponent are all
  // zero.
  if (value == 0.0)
    return {BigInt(0), BigInt(1)};

  DoubleRepr bitCopy(value);

  // Calculate how many bits we have to shift up the fractional part, for it to
  // become an integer. This can easily be calculated by counting the trailing
  // zeros and subtracting the size of the mantissa. In the case the fraction
  // part is 0, we insert the implicit 1, to get a shift amount of 0.
  std::uint64_t fractionPart =
      bitCopy.mantissa | (1uLL << DoubleRepr::mantissaSize);
  unsigned shiftRequired =
      DoubleRepr::mantissaSize - countTrailingZeros(fractionPart);

  DoubleRepr temp = bitCopy;
  // Shifting a floating point is simply adding to the exponent.
  temp.setExponent(shiftRequired);
  double dest = temp.toDouble();

  BigInt numerator(dest);
  BigInt denominator(1);
  std::int64_t exponent =
      bitCopy.getExponent() - static_cast<int>(shiftRequired);
  // At this point we are in the form '(nom / denom) * 2^exponent'. Based on
  // whether the exponent is negative or not, we have to multiply either
  // numerator or denominator with '2^abs(exponent)', which are just binary
  // shifts.
  if (exponent > 0)
    numerator <<= exponent;
  else
    denominator <<= -exponent;

  return {std::move(numerator), std::move(denominator)};
}
