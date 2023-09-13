//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdio>
#include <cstdlib>

#ifdef NDEBUG

#ifndef PYLIR_USE_ASSERTS

#ifdef __clang__

#define PYLIR_UNREACHABLE    \
  do                         \
    __builtin_unreachable(); \
  while (0)

#define PYLIR_ASSERT(...) __builtin_assume(bool(__VA_ARGS__))

#elif defined(__GNUC__)

#define PYLIR_UNREACHABLE    \
  do                         \
    __builtin_unreachable(); \
  while (0)

#define PYLIR_ASSERT(...) (void)0

#elif defined(_MSC_VER)

#define PYLIR_UNREACHABLE \
  do                      \
    __assume(false);      \
  while (0)

#define PYLIR_ASSERT(...) __assume((bool)(__VA_ARGS__))

#else

#define PYLIR_UNREACHABLE (void)0

#define PYLIR_ASSERT(...) (void)0

#endif

#else

#define PYLIR_UNREACHABLE \
  do                      \
    std::abort();         \
  while (0)

#define PYLIR_ASSERT(...)                                                 \
  do {                                                                    \
    if (!(__VA_ARGS__)) {                                                 \
      std::fprintf(stderr, __FILE__ ":%d: " #__VA_ARGS__ "\n", __LINE__); \
      std::abort();                                                       \
    }                                                                     \
  } while (0)

#endif

#else

#define PYLIR_ASSERT(...)                                                 \
  do {                                                                    \
    if (!(__VA_ARGS__)) {                                                 \
      std::fprintf(stderr, __FILE__ ":%d: %s\n", __LINE__, #__VA_ARGS__); \
      std::abort();                                                       \
    }                                                                     \
  } while (0)

#define PYLIR_UNREACHABLE                                                     \
  do {                                                                        \
    std::abort(); /* So that the compiler sees code afterwards is unreachable \
                   */                                                         \
  } while (0)

#endif

#ifdef __clang__
#define PYLIR_NON_NULL _Nonnull
#define PYLIR_NULLABLE _Nullable
#else
#define PYLIR_NON_NULL
#define PYLIR_NULLABLE
#endif
