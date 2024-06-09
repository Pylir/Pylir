//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunused-private-field"

#include <pylir/Runtime/GC/GC.hpp>
#include <pylir/Support/BigInt.hpp>
#include <pylir/Support/HashTable.hpp>

#include <array>
#include <string_view>
#include <type_traits>

#include <unwind.h>

#include "Builtins.hpp"
#include "Support.hpp"

namespace pylir::rt {

struct KeywordArg {
  std::string_view name;
  PyObject& arg;
};

struct Keyword {
  std::string_view name;

  ~Keyword() = default;
  Keyword(const Keyword&) = delete;
  Keyword& operator=(const Keyword&) = delete;
  Keyword(Keyword&&) = delete;
  Keyword& operator=(Keyword&&) = delete;

  // Blatant abuse
  KeywordArg operator=(PyObject& object) {
    return {name, object};
  }
};

inline Keyword operator""_kw(const char* s, std::size_t n) {
  return Keyword{std::string_view{s, n}};
}

struct PyObjectStorage {
  PyTypeObject* type;
};

class PyObject {
  PyObjectStorage& getStorage() {
    return *reinterpret_cast<PyObjectStorage*>(this);
  }

public:
  constexpr PyObject() = default;

  ~PyObject() = default;
  PyObject(const PyObject&) = delete;
  PyObject(PyObject&&) noexcept = delete;
  PyObject& operator=(const PyObject&) = delete;
  PyObject& operator=(PyObject&&) noexcept = delete;

  friend PyTypeObject& type(PyObject& obj) {
    return *reinterpret_cast<PyTypeObject*>(
        reinterpret_cast<std::uintptr_t>(obj.getStorage().type) &
        ~std::uintptr_t{0b11});
  }

  PyObject* getSlot(int index);

  PyObject* getSlot(std::string_view name);

  void setSlot(int index, PyObject& object);

  template <class... Args>
  PyObject& operator()(Args&&... args);

  bool operator==(PyObject& other);

  bool is(PyObject& other) {
    return this == &other;
  }

  template <class T>
  bool isa();

  template <class T>
  T& cast() {
    return *reinterpret_cast<T*>(this);
  }

  template <class T>
  // Name taken from LLVMs RTTI system.
  // NOLINTNEXTLINE(readability-identifier-naming)
  T* dyn_cast() {
    return isa<T>() ? &cast<T>() : nullptr;
  }

  void clearMarking() {
    getStorage().type = reinterpret_cast<PyTypeObject*>(
        reinterpret_cast<std::uintptr_t>(getStorage().type) &
        ~std::uintptr_t(0b11));
  }

  template <class T>
  void setMark(T value) {
    getStorage().type = reinterpret_cast<PyTypeObject*>(
        (reinterpret_cast<std::uintptr_t>(getStorage().type) &
         ~std::uintptr_t(0b11)) |
        static_cast<std::uintptr_t>(value));
  }

  template <class T>
  T getMark() {
    return static_cast<T>(reinterpret_cast<std::uintptr_t>(getStorage().type) &
                          0b11);
  }
};

void destroyPyObject(PyObject& object);

bool isinstance(PyObject& obj, PyTypeObject& type);

class PyTypeObject : public PyObject {
  friend class PyObject;

  PyObjectStorage m_base;
  std::size_t m_offset;
  PyTypeObject* m_layoutType;
  PyTuple* m_mroTuple;
  PyTuple* m_instanceSlots;

public:
  constexpr static auto& layoutTypeObject = Builtins::Type;

  enum Slots {
#define TYPE_SLOT(x, y) y,
#include <pylir/Interfaces/Slots.def>
  };

  [[nodiscard]] std::size_t getOffset() const noexcept {
    return m_offset;
  }

  [[nodiscard]] PyTuple& getMROTuple() const noexcept {
    return *m_mroTuple;
  }

  [[nodiscard]] PyTuple& getInstanceSlots() const noexcept {
    return *m_instanceSlots;
  }

  [[nodiscard]] PyTypeObject& getLayoutType() const noexcept {
    return *m_layoutType;
  }
};

using PyUniversalCC = PyObject& (*)(PyFunction&, PyTuple&, PyDict&);

class PyFunction : public PyObject {
  friend class PyObject;

  PyObjectStorage m_base;
  PyUniversalCC m_function;
  std::uint32_t m_closureSizeBytes{};

  // Memory layout after:
  // * PyObject* m_slots[];
  // * Closure parameters, which may be of any type.
  // * std::uint8_t refInClosureBitSet[ceil(m_closureSizeBytes / (8 *
  // sizeof(void*))];
  //
  // where bit 'i' of 'refInClosureBitSet[j]' being set means that
  // '(std::byte*)this + sizeof(PyFunction) + (8 * j + i) * sizeof(PyObject*)'
  // contains a 'PyObject*'.

  /// Retuns a pointer to the beginning of the closure arguments.
  std::byte* getClosureStart() {
    return reinterpret_cast<std::byte*>(
        reinterpret_cast<PyObject**>(std::next(this)) +
        // Using the fact that functions cannot be subclasses and have
        // known slot count.
        PyTypeTraits<Builtins::Function>::slotCount);
  }

  /// Returns a pointer to the beginning of the mask array for references in
  /// the closure arguments.
  std::int8_t* mask_begin() {
    return reinterpret_cast<std::int8_t*>(getClosureStart()) +
           m_closureSizeBytes;
  }

  /// Returns a pointer to the end of the mask array for references in
  /// the closure arguments.
  std::int8_t* mask_end() {
    std::uint32_t endMaskOffset = m_closureSizeBytes / (8 * sizeof(void*));
    if (m_closureSizeBytes % (8 * sizeof(void*)))
      endMaskOffset++;
    return mask_begin() + endMaskOffset;
  }

public:
  constexpr explicit PyFunction(PyUniversalCC function)
      : m_base{&Builtins::Function}, m_function(function) {}

  constexpr static auto& layoutTypeObject = Builtins::Function;

  enum Slots {
#define FUNCTION_SLOT(x, y) y,
#include <pylir/Interfaces/Slots.def>
  };

  class ClosureRefIterator {
    PyFunction* m_function{};
    std::uint32_t m_currMask{};
    std::uint8_t m_index{};

    void advanceToNext() {
      std::int8_t* mask = m_function->mask_begin();
      std::int8_t* maskEnd = m_function->mask_end();
      for (; mask + m_currMask != maskEnd; m_currMask++) {
        for (; m_index < 8; m_index++)
          if (mask[m_currMask] & (1 << m_index))
            return;

        m_index = 0;
      }
    }

  public:
    ClosureRefIterator() = default;

    ClosureRefIterator(PyFunction* function, uint32_t currMask)
        : m_function(function), m_currMask(currMask) {
      advanceToNext();
    }

    using difference_type = std::ptrdiff_t;
    using value_type = PyObject*;
    using reference = value_type&;
    using pointer = value_type*;

    reference operator*() const {
      return *(reinterpret_cast<PyObject**>(m_function->getClosureStart()) +
               8 * m_currMask + m_index);
    }

    pointer operator->() const {
      return &**this;
    }

    ClosureRefIterator& operator++() {
      m_index++;
      advanceToNext();
      return *this;
    }

    ClosureRefIterator operator++(int) {
      ClosureRefIterator temp = *this;
      ++*this;
      return temp;
    }

    bool operator==(const ClosureRefIterator& rhs) const {
      return std::tie(m_function, m_currMask) ==
             std::tie(rhs.m_function, rhs.m_currMask);
    }

    bool operator!=(const ClosureRefIterator& rhs) const {
      return !(rhs == *this);
    }
  };

  /// Returns an iterator iterating over all 'PyObject*'s within the closure
  /// arguments of this function object.
  ClosureRefIterator closure_ref_begin() {
    return ClosureRefIterator(this, 0);
  }

  ClosureRefIterator closure_ref_end() {
    return ClosureRefIterator(this, mask_end() - mask_begin());
  }
};

class PyTuple : public PyObject {
  PyObjectStorage m_base;
  std::size_t m_size;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
  PyObject* m_trailing[];
#pragma GCC diagnostic pop

public:
  explicit PyTuple(std::size_t size, PyTypeObject& type = Builtins::Tuple)
      : m_base{&type}, m_size(size) {}

  constexpr static auto& layoutTypeObject = Builtins::Tuple;

  PyObject** begin() {
    return m_trailing;
  }

  PyObject** end() {
    return m_trailing + m_size;
  }

  std::size_t len() {
    return m_size;
  }

  PyObject& getItem(std::size_t index) {
    return *m_trailing[index];
  }
};

class PyList : public PyObject {
  PyObjectStorage m_base;
  std::size_t m_size{};
  PyTuple* m_tuple{};

public:
  explicit PyList(PyTypeObject& type = Builtins::List) : m_base{&type} {}

  constexpr static auto& layoutTypeObject = Builtins::List;

  PyTuple* getTuple() {
    return m_tuple;
  }

  PyObject** begin() {
    return m_tuple->begin();
  }

  PyObject** end() {
    return m_tuple->begin() + m_size;
  }

  std::size_t len() {
    return m_size;
  }

  PyObject& getItem(std::size_t index) {
    return m_tuple->getItem(index);
  }
};

class PyString : public PyObject {
  PyObjectStorage m_base;
  BufferComponent<char, MallocAllocator> m_buffer;

public:
  explicit PyString(std::string_view string, PyTypeObject& type = Builtins::Str)
      : m_base{&type}, m_buffer(string.begin(), string.end()) {}

  constexpr static auto& layoutTypeObject = Builtins::Str;

  friend bool operator==(PyString& lhs, std::string_view sv) {
    return lhs.view() == sv;
  }

  friend bool operator==(const std::string_view sv, PyString& rhs) {
    return rhs.view() == sv;
  }

  [[nodiscard]] std::string_view view() const {
    return std::string_view{m_buffer.data(), m_buffer.size()};
  }

  [[nodiscard]] std::size_t len() const {
    return m_buffer.size();
  }
};

class PyDict : public PyObject {
  PyObjectStorage m_base;
  HashTable<PyObject*, PyObject*, PyObjectHasher, PyObjectEqual,
            MallocAllocator>
      m_table;

public:
  explicit PyDict(PyTypeObject& type = Builtins::Dict) : m_base{&type} {}

  constexpr static auto& layoutTypeObject = Builtins::Dict;

  PyObject* tryGetItem(PyObject& key) {
    auto* result = m_table.find(&key);
    if (result == m_table.end())
      return nullptr;

    return result->value;
  }

  PyObject* tryGetItem(PyObject& key, std::size_t hash) {
    auto* result = m_table.find_hash(hash, &key);
    if (result == m_table.end())
      return nullptr;

    return result->value;
  }

  void setItem(PyObject& key, PyObject& value) {
    m_table.insert_or_assign(&key, &value);
  }

  void setItem(PyObject& key, std::size_t hash, PyObject& value) {
    m_table.insert_or_assign_hash(hash, &key, &value);
  }

  void setItemUnique(PyObject& key, std::size_t hash, PyObject& value) {
    m_table.insert_or_assign_hash<PyObject*, true>(hash, &key, &value);
  }

  void delItem(PyObject& key) {
    m_table.erase(&key);
  }

  void delItem(PyObject& key, std::size_t hash) {
    m_table.erase_hash(hash, &key);
  }

  auto begin() {
    return m_table.begin();
  }

  auto end() {
    return m_table.end();
  }
};

class PyInt : public PyObject {
  PyObjectStorage m_base;
  BigInt m_integer;

public:
  constexpr static auto& layoutTypeObject = Builtins::Int;

  bool boolean() {
    return !m_integer.isZero();
  }

  template <class T>
  T to() {
    return m_integer.getInteger<T>();
  }
};

class PyBaseException : public PyObject {
  PyObjectStorage m_base;
  std::uintptr_t m_landingPad;
  _Unwind_Exception m_unwindHeader;
  std::uint32_t m_typeIndex;

public:
  constexpr static std::uint64_t EXCEPTION_CLASS =
      0x50594C5250590000; // PYLRPY\0\0

  constexpr static auto& layoutTypeObject = Builtins::BaseException;

  enum Slots {
#define BASEEXCEPTION_SLOT(x, y) y,
#include <pylir/Interfaces/Slots.def>
  };

  _Unwind_Exception& getUnwindHeader() {
    static_assert(offsetof(PyBaseException, m_unwindHeader) ==
                  alignof(_Unwind_Exception));
    return m_unwindHeader;
  }

  static PyBaseException* fromUnwindHeader(_Unwind_Exception* header) {
    PYLIR_ASSERT(header->exception_class == EXCEPTION_CLASS);
    return reinterpret_cast<PyBaseException*>(
        reinterpret_cast<char*>(header) -
        offsetof(PyBaseException, m_unwindHeader));
  }

  [[nodiscard]] std::uintptr_t getLandingPad() const {
    return m_landingPad;
  }

  void setLandingPad(std::uintptr_t landingPad) {
    m_landingPad = landingPad;
  }

  [[nodiscard]] std::uint32_t getTypeIndex() const {
    return m_typeIndex;
  }

  void setTypeIndex(std::uint32_t typeIndex) {
    m_typeIndex = typeIndex;
  }
};

template <PyTypeObject& typeObject,
          std::size_t slotCount = PyTypeTraits<typeObject>::slotCount>
class StaticInstance {
  using InstanceType = typename PyTypeTraits<typeObject>::instanceType;
  static_assert(alignof(InstanceType) >= alignof(PyObject*));
  // NOLINTNEXTLINE(bugprone-sizeof-expression)
  alignas(InstanceType)
      std::array<std::byte, sizeof(InstanceType) +
                                slotCount * sizeof(PyObject*)> m_buffer{};

public:
  template <class... Args>
  StaticInstance(
      std::initializer_list<std::pair<typename InstanceType::Slots, PyObject&>>
          slotsInit,
      Args&&... args) {
    static_assert(
        std::is_standard_layout_v<std::remove_reference_t<decltype(*this)>>);
    new (m_buffer.data()) InstanceType(std::forward<Args>(args)...);
    std::array<PyObject*, slotCount> slots{};
    for (auto& [index, object] : slotsInit)
      slots[index] = &object;

    std::memcpy(m_buffer.data() + sizeof(InstanceType), slots.data(),
                slots.size() * sizeof(PyObject*));
  }

  /*implicit*/ operator PyObject&() {
    return get();
  }

  /*implicit*/ operator InstanceType&() {
    return *reinterpret_cast<InstanceType*>(m_buffer);
  }

  InstanceType& get() {
    return *this;
  }
};

template <PyTypeObject& typeObject>
class StaticInstance<typeObject, 0> {
  using InstanceType = typename PyTypeTraits<typeObject>::instanceType;
  InstanceType m_object;

public:
  template <class... Args>
  explicit StaticInstance(Args&&... args)
      : m_object(std::forward<Args>(args)...) {}

  /*implicit*/ operator PyObject&() {
    return m_object;
  }

  /*implicit*/ operator InstanceType&() {
    return m_object;
  }

  InstanceType& get() {
    return m_object;
  }
};

namespace details {

template <PyTypeObject& type>
struct AllocType {
  template <class... Args>
  decltype(auto) operator()(Args&&... args) const noexcept {
    using InstanceType = typename PyTypeTraits<type>::instanceType;
    void* memory = gc.alloc(sizeof(InstanceType) +
                            sizeof(PyObject*) * PyTypeTraits<type>::slotCount);
    return *new (memory) InstanceType(std::forward<Args>(args)...);
  }
};

template <>
struct AllocType<Builtins::Tuple> {
  template <class... Args>
  decltype(auto) operator()(std::size_t count, Args&&... args) const noexcept {
    using InstanceType = typename PyTypeTraits<Builtins::Tuple>::instanceType;
    auto* memory = reinterpret_cast<std::byte*>(
        gc.alloc(sizeof(InstanceType) + sizeof(PyObject*) * count));
    return *new (memory) InstanceType(count, std::forward<Args>(args)...);
  }
};

} // namespace details

template <PyTypeObject& type>
constexpr details::AllocType<type> alloc;

template <class... Args>
PyObject& PyObject::operator()(Args&&... args) {
  constexpr std::size_t tupleCount =
      (... + std::is_base_of_v<PyObject, std::remove_reference_t<Args>>);
  auto& tuple = alloc<Builtins::Tuple>(tupleCount);
  auto& dict = alloc<Builtins::Dict>();
  auto* iter = tuple.begin();
  (
      [&](auto&& arg) {
        static_assert(
            std::is_base_of_v<PyObject,
                              std::remove_reference_t<decltype(arg)>> ||
            std::is_same_v<KeywordArg&&, decltype(arg)>);
        if constexpr (std::is_same_v<KeywordArg&&, decltype(arg)>)
          dict.setItem(alloc<Builtins::Str>(arg.name), arg.arg);
        else
          *iter++ = &arg;
      }(std::forward<Args>(args)),
      ...);
  return Builtins::pylir__call__(*this, tuple, dict);
}

template <>
inline bool PyObject::isa<PyObject>() {
  return true;
}

template <class T>
inline bool PyObject::isa() {
  return type(*this).m_layoutType == &T::layoutTypeObject;
}

} // namespace pylir::rt

#pragma GCC diagnostic pop
