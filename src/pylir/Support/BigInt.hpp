//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>

#include <tommath.h>

#include "Macros.hpp"

namespace pylir {
class BigInt;

BigInt pow(const BigInt& base, int expo);
BigInt powmod(const BigInt& base, const BigInt& expo, const BigInt& mod);

// TODO think about how to handle allocation failure
class BigInt {
  // Abuse array to pointer decay
  mp_int m_int[1]{};

  static void cantFail(mp_err err) {
    PYLIR_ASSERT(err == MP_OKAY);
  }

  friend BigInt pow(const BigInt& base, int expo);
  friend BigInt powmod(const BigInt& base, const BigInt& expo,
                       const BigInt& mod);

public:
  BigInt() {
    cantFail(mp_init(m_int));
  }

  template <class T, std::enable_if_t<std::is_arithmetic_v<T>>* = nullptr>
  explicit BigInt(T value) {
    if constexpr (std::is_floating_point_v<T>) {
      cantFail(mp_init(m_int));
      cantFail(mp_set_double(m_int, value));
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
      cantFail(mp_init_u32(m_int, value));
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
      cantFail(mp_init_i32(m_int, value));
    } else if constexpr (std::is_same_v<T, long>) {
      cantFail(mp_init_l(m_int, value));
    } else if constexpr (std::is_same_v<T, unsigned long>) {
      cantFail(mp_init_ul(m_int, value));
    } else if constexpr (std::is_signed_v<T>) {
      cantFail(mp_init_i64(m_int, value));
    } else {
      cantFail(mp_init_u64(m_int, value));
    }
  }

  explicit BigInt(std::string ascii, std::uint8_t radix = 10) : BigInt() {
    cantFail(mp_read_radix(m_int, ascii.data(), radix));
  }

  ~BigInt() {
    mp_clear(m_int);
  }

  BigInt(const BigInt& rhs) {
    cantFail(mp_init_copy(m_int, rhs.m_int));
  }

  BigInt& operator=(const BigInt& rhs) {
    if (&rhs == this)
      return *this;

    cantFail(mp_copy(rhs.m_int, m_int));
    return *this;
  }

  BigInt(BigInt&& rhs) noexcept : BigInt() {
    mp_exch(m_int, rhs.m_int);
  }

  BigInt& operator=(BigInt&& rhs) noexcept {
    mp_exch(m_int, rhs.m_int);
    return *this;
  }

  BigInt& operator++() {
    cantFail(mp_incr(m_int));
    return *this;
  }

  BigInt operator++(int) {
    auto copy = *this;
    ++(*this);
    return copy;
  }

  BigInt& operator--() {
    cantFail(mp_decr(m_int));
    return *this;
  }

  BigInt operator--(int) {
    auto copy = *this;
    --(*this);
    return copy;
  }

  BigInt operator-() const {
    BigInt result;
    cantFail(mp_neg(m_int, result.m_int));
    return result;
  }

  const BigInt& operator+() const {
    return *this;
  }

  BigInt operator~() const {
    BigInt result;
    cantFail(mp_complement(m_int, result.m_int));
    return result;
  }

  BigInt& operator+=(const BigInt& rhs) {
    cantFail(mp_add(m_int, rhs.m_int, m_int));
    return *this;
  }

  BigInt& operator-=(const BigInt& rhs) {
    cantFail(mp_sub(m_int, rhs.m_int, m_int));
    return *this;
  }

  BigInt& operator*=(const BigInt& rhs) {
    cantFail(mp_mul(m_int, rhs.m_int, m_int));
    return *this;
  }

  BigInt& operator/=(const BigInt& rhs) {
    cantFail(mp_div(m_int, rhs.m_int, m_int, nullptr));
    return *this;
  }

  BigInt& operator%=(const BigInt& rhs) {
    cantFail(mp_mod(m_int, rhs.m_int, m_int));
    return *this;
  }

  [[nodiscard]] double roundToDouble() const {
    return mp_get_double(m_int);
  }

  template <class T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
  [[nodiscard]] std::optional<T> tryGetInteger() const {
    constexpr auto max = std::numeric_limits<T>::max();
    constexpr auto min = std::numeric_limits<T>::lowest();
    if (mp_cmp(m_int, BigInt(max).m_int) == 1)
      return std::nullopt;
    if constexpr (!std::is_signed_v<T>) {
      return mp_get_u64(m_int);
    } else {
      if (mp_cmp(m_int, BigInt(min).m_int) == -1)
        return std::nullopt;
      return mp_get_i64(m_int);
    }
  }

  template <class T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
  [[nodiscard]] T getInteger() const {
    auto optional = tryGetInteger<T>();
    PYLIR_ASSERT(optional);
    return *optional;
  }

  [[nodiscard]] std::string toString(std::uint8_t radix = 10) const {
    std::size_t size;
    cantFail(mp_radix_size_overestimate(m_int, radix, &size));
    std::string result(size, '\0');
    cantFail(mp_to_radix(m_int, result.data(), result.size(), &size, radix));
    // mp_to_radix includes the null terminator in size. We don't do that around
    // here
    result.resize(size - 1);
    return result;
  }

  [[nodiscard]] bool isZero() const {
    return mp_iszero(m_int);
  }

  [[nodiscard]] bool isNegative() const {
    return mp_isneg(m_int);
  }

  std::pair<BigInt, BigInt> divmod(const BigInt& rhs) {
    BigInt div;
    BigInt mod;
    cantFail(mp_div(m_int, rhs.m_int, div.m_int, mod.m_int));
    return {std::move(div), std::move(mod)};
  }

  friend BigInt operator+(const BigInt& lhs, const BigInt& rhs) {
    BigInt result;
    cantFail(mp_add(lhs.m_int, rhs.m_int, result.m_int));
    return result;
  }

  friend BigInt operator-(const BigInt& lhs, const BigInt& rhs) {
    BigInt result;
    cantFail(mp_sub(lhs.m_int, rhs.m_int, result.m_int));
    return result;
  }

  friend BigInt operator*(const BigInt& lhs, const BigInt& rhs) {
    BigInt result;
    cantFail(mp_mul(lhs.m_int, rhs.m_int, result.m_int));
    return result;
  }

  friend BigInt operator/(const BigInt& lhs, const BigInt& rhs) {
    BigInt result;
    cantFail(mp_div(lhs.m_int, rhs.m_int, result.m_int, nullptr));
    return result;
  }

  friend BigInt operator%(const BigInt& lhs, const BigInt& rhs) {
    BigInt result;
    cantFail(mp_mod(lhs.m_int, rhs.m_int, result.m_int));
    return result;
  }

  friend BigInt operator<<(const BigInt& lhs, int rhs) {
    BigInt result;
    cantFail(mp_mul_2d(lhs.m_int, rhs, result.m_int));
    return result;
  }

  friend BigInt& operator<<=(BigInt& lhs, int rhs) {
    cantFail(mp_mul_2d(lhs.m_int, rhs, lhs.m_int));
    return lhs;
  }

  friend BigInt operator>>(const BigInt& lhs, int rhs) {
    BigInt result;
    cantFail(mp_div_2d(lhs.m_int, rhs, result.m_int, nullptr));
    return result;
  }

  friend BigInt operator&(const BigInt& lhs, const BigInt& rhs) {
    BigInt result;
    cantFail(mp_and(lhs.m_int, rhs.m_int, result.m_int));
    return result;
  }

  friend BigInt operator|(const BigInt& lhs, const BigInt& rhs) {
    BigInt result;
    cantFail(mp_or(lhs.m_int, rhs.m_int, result.m_int));
    return result;
  }

  friend BigInt operator^(const BigInt& lhs, const BigInt& rhs) {
    BigInt result;
    cantFail(mp_xor(lhs.m_int, rhs.m_int, result.m_int));
    return result;
  }

  friend bool operator<(const BigInt& lhs, const BigInt& rhs) {
    return mp_cmp(lhs.m_int, rhs.m_int) == MP_LT;
  }

  friend bool operator>(const BigInt& lhs, const BigInt& rhs) {
    return mp_cmp(lhs.m_int, rhs.m_int) == MP_GT;
  }

  friend bool operator<=(const BigInt& lhs, const BigInt& rhs) {
    return mp_cmp(lhs.m_int, rhs.m_int) != MP_GT;
  }

  friend bool operator>=(const BigInt& lhs, const BigInt& rhs) {
    return mp_cmp(lhs.m_int, rhs.m_int) != MP_LT;
  }

  friend bool operator==(const BigInt& lhs, const BigInt& rhs) {
    return mp_cmp(lhs.m_int, rhs.m_int) == MP_EQ;
  }

  friend bool operator!=(const BigInt& lhs, const BigInt& rhs) {
    return mp_cmp(lhs.m_int, rhs.m_int) != MP_EQ;
  }

  friend void swap(BigInt& lhs, BigInt& rhs) {
    mp_exch(lhs.m_int, rhs.m_int);
  }

  mp_int& getHandle() {
    return m_int[0];
  }

  [[nodiscard]] const mp_int& getHandle() const {
    return m_int[0];
  }
};

static_assert(std::is_standard_layout_v<BigInt>);

inline BigInt pow(const BigInt& base, int expo) {
  BigInt result;
  BigInt::cantFail(mp_expt_n(base.m_int, expo, result.m_int));
  return result;
}

inline BigInt powmod(const BigInt& base, const BigInt& exp, const BigInt& mod) {
  BigInt result;
  BigInt::cantFail(mp_exptmod(base.m_int, exp.m_int, mod.m_int, result.m_int));
  return result;
}

/// Returns a fraction as nominator and denominator which is EXACTLY equal to
/// 'value'. If 'value' is negative, then the nominator is negative. Denominator
/// is always positive. The denominator and nominator are always coprime.
///
/// If the value passed is not finite, the behaviour is undefined.
std::pair<BigInt, BigInt> toRatio(double value);

} // namespace pylir
