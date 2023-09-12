//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <pylir/Support/Macros.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <utility>

namespace pylir::rt {
class PyObject;

struct PyObjectHasher {
  std::size_t operator()(PyObject* object) const noexcept;
};

struct PyObjectEqual {
  bool operator()(PyObject* lhs, PyObject* rhs) const noexcept;
};

template <class T>
struct MallocAllocator {
  using pointer = T*;
  using value_type = T;
  using const_void_pointer = const void*;
  using size_type = std::size_t;
  using is_always_equal = std::true_type;

  pointer allocate(std::size_t n) noexcept {
    return (pointer)std::malloc(sizeof(T) * n);
  }

  void deallocate(pointer p, std::size_t) noexcept {
    free(p);
  }

  constexpr bool operator==(const MallocAllocator&) {
    return true;
  }

  constexpr bool operator!=(const MallocAllocator&) {
    return false;
  }
};

template <class Fn>
class function_ref;

template <class Ret, class... Params>
class function_ref<Ret(Params...)> {
  Ret (*m_callback)(void*, Params... params){};
  void* m_callable{};

public:
  function_ref() = default;

  template <class Callable,
            std::enable_if_t<!std::is_same_v<std::decay_t<Callable>,
                                             function_ref>>* = nullptr>
  function_ref(Callable&& callable)
      : m_callback(+[](void* callable, Params... params) -> Ret {
          return (*reinterpret_cast<std::remove_reference_t<Callable>*>(
              callable))(std::forward<Params>(params)...);
        }),
        m_callable(&callable) {}

  template <class... Args>
  Ret operator()(Args&&... args) const {
    PYLIR_ASSERT(m_callable);
    return m_callback(m_callable, std::forward<Args>(args)...);
  }

  explicit operator bool() const {
    return m_callable;
  }
};

std::uintptr_t readULEB128(const std::uint8_t** data);

std::intptr_t readSLEB128(const std::uint8_t** data);

} // namespace pylir::rt
