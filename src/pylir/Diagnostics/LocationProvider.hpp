//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>

#include <pylir/Support/Macros.hpp>
#include <pylir/Support/Variant.hpp>

#include <iterator>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace pylir::Diag {

class LazyLocation;

/// Class representing a range in source code. More specifically, it consists of
/// a half-open range with the first index referring to the beginning of the
/// source range in UTF32 code units and the second value consisting of the
/// first UTF32 code unit no longer part of the location. The location may also
/// be null referring to an unknown source location.
class Location {
  std::optional<std::pair<std::size_t, std::size_t>> m_value;

public:
  /// Constructs an unknown location.
  Location() = default;

  /// Constructs a location ranging from 'from' to 'to'.
  Location(std::size_t from, std::size_t to)
      : m_value(std::in_place, from, to) {
    PYLIR_ASSERT(from <= to);
  }

  /// Constructs a location ranging from 'value.first' to 'value.second'.
  /*implicit*/ Location(std::pair<std::size_t, std::size_t> value)
      : Location(value.first, value.second) {}

  /// Constructs an unknown location.
  /*implicit*/ Location(std::nullopt_t) : m_value(std::nullopt) {}

  /// Constructs a location from all elements in 'providers'. The start of the
  /// source location is determined from the very first element with a known
  /// source location and the end of the range from the very first element from
  /// the back with a known source location.
  /*implicit*/ Location(std::initializer_list<LazyLocation> providers);

  /// Returns true if this location is not unknown.
  explicit operator bool() const {
    return m_value.has_value();
  }

  std::pair<std::size_t, std::size_t>* operator->() {
    return m_value.operator->();
  }

  const std::pair<std::size_t, std::size_t>* operator->() const {
    return m_value.operator->();
  }
};

/// This class templates allows for arbitrary types to provide the location of
/// where a diagnostic should go within the diagnostics subsystem. Types opt
/// into this behaviour by providing a template specialization for
/// 'LocationProvider' and the implementing specific methods. These are:
///
/// `static Location getRange` callable with `T`:
///     This method HAS to be implemented and returns the source of range of T
///     via two indices. These refer to the offset of the UTF32 text within the
///     `Document` instance of the file `T` has been parsed in. The first index
///     refers to the inclusive start of 'T', while the second index refers to
///     the exclusive end of 'T' (in other words, the first index, that is not
///     part of T. This may be one past the very last element in the document).
///     If it's impossible to gather a source location from a given value then
///     'std::nullopt' should be returned.
///
///     Additionally, one may optionally add a second 'context' parameter which
///     is the optional argument passed as the second argument to 'rangeLoc'.
///     It's purpose is for 'T's that are not capable of getting source
///     locations themselves to get some kind of context.
///
/// 'static std::size_t getPoint' callable with 'T':
///     This method maybe OPTIONALLY implemented at which point it may be called
///     'pointLoc'. It's purpose is for 'T' which are very large (in terms of
///     source range) or otherwise hard to look at, to provide a specific
///     location by which they are most recognizable. For a function or class in
///     the Python AST eg. this may be their name. If this function is not
///     implemented, then 'pointLoc' will instead use the start position
///     returned by 'getRange'.
template <class T, class = void>
struct LocationProvider;

/// Type erased lazy getter of 'Location'. This class is assignable and
/// constructible from any value that can be passed to 'rangeLoc'. The actual
/// call to 'rangeLoc' only happens if the lazy location is passed to 'rangeLoc'
/// or its call operator is used. The value it was constructed with must be
/// alive while the 'LazyLocation' is in use.
///
/// A type may opt-into having its location be computed eagerly as an
/// optimization by adding 'using non_lazy = void' to its 'LocationProvider'.
class LazyLocation {
  struct Lazy {
    const void* arg;
    Location (*call)(const void*);
  };
  std::variant<Location, Lazy> m_variant;

  template <class T>
  using has_non_lazy_typedef = typename T::non_lazy;

  /// True for any type 'T' that should be computed eagerly as an optimization.
  template <class T>
  constexpr static bool nonLazy =
      llvm::is_detected<has_non_lazy_typedef, LocationProvider<T>>::value;

public:
  /// Lazy location returning an unknown location by default.
  LazyLocation() = default;

  /// Constructs a 'LazyLocation' from a type whose location should be
  /// calculated lazily.
  template <class T,
            std::enable_if_t<!std::is_same_v<std::decay_t<T>, LazyLocation> &&
                             !LazyLocation::nonLazy<T>>* = nullptr>
  LazyLocation(const T& argument);

  /// Constructs a 'LazyLocation' from a type whose location should be
  /// calculated eagerly.
  template <class T,
            std::enable_if_t<!std::is_same_v<std::decay_t<T>, LazyLocation> &&
                             LazyLocation::nonLazy<T>>* = nullptr>
  LazyLocation(const T& argument);

  /// Reassigns the lazy location to now return the location of 'argument'.
  template <class T,
            std::enable_if_t<!std::is_same_v<std::decay_t<T>, LazyLocation>>* =
                nullptr>
  LazyLocation& operator=(const T& argument) noexcept {
    this->~LazyLocation();
    new (this) LazyLocation(argument);
    return *this;
  }

  /// Retrieves the location stored in the 'LazyLocation'.
  Location operator()() const {
    if (const auto* lazy = std::get_if<Lazy>(&m_variant))
      return lazy->call(lazy->arg);
    return pylir::get<Location>(m_variant);
  }
};

template <class T>
struct LocationProvider<T, std::enable_if_t<std::is_integral_v<T>>> {
  static Location getRange(T value) noexcept {
    return Location(value, value + 1);
  }

  using non_lazy = void;
};

template <>
struct LocationProvider<Location> {
  static Location getRange(Location value) noexcept {
    return value;
  }

  using non_lazy = void;
};

template <>
struct LocationProvider<LazyLocation> {
  static Location getRange(LazyLocation value) noexcept {
    return value();
  }
};

namespace detail {
template <class T, class = void>
struct HasLocationProviderRangeOneArg : std::false_type {};

template <class T>
struct HasLocationProviderRangeOneArg<
    T, std::void_t<decltype(LocationProvider<std::decay_t<T>>::getRange(
           std::declval<std::decay_t<T>>()))>> : std::true_type {};

template <class T, class Context, class = void>
struct HasLocationProviderRangeTwoArg : std::false_type {};

template <class T, class Context>
struct HasLocationProviderRangeTwoArg<
    T, Context,
    std::void_t<decltype(LocationProvider<std::decay_t<T>>::getRange(
        std::declval<std::decay_t<T>>(), std::declval<const Context&>()))>>
    : std::true_type {};
} // namespace detail

/// Convenient function for calling the 'LocationProvider::getRange'
/// specialization of 'value'. If the type of 'value' does not have a
/// specialization of 'LocationProvider::getRange' it triggers a static
/// assertion. The second argument serves as context for the 'getRange'
/// implementation of 'value'.
template <class T, class Context>
Location rangeLoc(const T& value, const Context& context) {
  if constexpr (detail::HasLocationProviderRangeTwoArg<T, Context>{})
    return LocationProvider<T>::getRange(value, context);
  else
    return LocationProvider<T>::getRange(value);
}

template <class T>
Location rangeLoc(const T& value) {
  return LocationProvider<T>::getRange(value);
}

namespace detail {
template <class T, class = void>
struct HasLocationProviderPointer : std::false_type {};

template <class T>
struct HasLocationProviderPointer<
    T, std::void_t<decltype(LocationProvider<std::decay_t<T>>::getPoint(
           std::declval<std::decay_t<T>>()))>> : std::true_type {};
} // namespace detail

/// Convenient function for calling the 'LocationProvider::getPoint'
/// specialization of 'value' if it exists. If it does not exist, a default
/// implementation is called, which returns the starting index of 'value's
/// 'getRange' implementation.
template <class T>
std::optional<std::size_t> pointLoc(const T& value) {
  if constexpr (detail::HasLocationProviderPointer<T>{}) {
    return LocationProvider<T>::getPoint(value);
  } else {
    Location loc = rangeLoc(value);
    return loc ? std::optional{loc->first} : std::nullopt;
  }
}

template <class T, class Context = void>
constexpr bool hasLocationProvider_v =
    std::disjunction_v<detail::HasLocationProviderRangeOneArg<T>,
                       detail::HasLocationProviderRangeTwoArg<T, Context>>;

/// Location provider specialization for any range where the value type has a
/// location provider and the range's iterators are bidirectional iterators.
template <class Range>
struct LocationProvider<
    Range, std::enable_if_t<std::is_base_of_v<
               std::bidirectional_iterator_tag,
               typename std::iterator_traits<
                   typename Range::iterator>::iterator_category>>> {
  template <class... Context>
  static Location getRange(const Range& range,
                           const Context&... context) noexcept {
    auto begin = range.begin();
    auto end = range.end();
    auto rbegin = std::make_reverse_iterator(range.end());
    auto rend = std::make_reverse_iterator(range.begin());

    Location firstLocation;
    for (; !firstLocation && begin != end; begin++)
      firstLocation = rangeLoc(*begin, context...);

    if (!firstLocation)
      return std::nullopt;

    Location lastLocation;
    for (; !lastLocation && rbegin != rend; rbegin++)
      lastLocation = rangeLoc(*rbegin, context...);

    PYLIR_ASSERT(lastLocation &&
                 "must have found last location if first was found");
    return Location(firstLocation->first, lastLocation->second);
  }
};

/// Location provider specialization for variants where every alternative
/// has a location provider specialization.
template <class... Args>
struct LocationProvider<std::variant<Args...>> {
  template <class... Context>
  static Location getRange(const std::variant<Args...>& variant,
                           const Context&... context) noexcept {
    return match(variant, [&](const auto& alternative) {
      return rangeLoc(alternative, context...);
    });
  }
};

template <>
struct LocationProvider<std::monostate> {
  template <class... Context>
  static Location getRange(std::monostate, const Context&...) noexcept {
    return std::nullopt;
  }

  using non_lazy = void;
};

template <class T>
struct LocationProvider<std::optional<T>> {
  template <class... Context>
  static Location getRange(const std::optional<T>& optional,
                           const Context&... context) noexcept {
    if (!optional)
      return std::nullopt;
    return rangeLoc(*optional, context...);
  }
};

template <>
struct LocationProvider<std::nullopt_t> {
  template <class... Context>
  static Location getRange(std::nullopt_t, const Context&...) noexcept {
    return std::nullopt;
  }

  using non_lazy = void;
};

template <class T, class Deleter>
struct LocationProvider<std::unique_ptr<T, Deleter>> {
  template <class... Context>
  static Location getRange(const std::unique_ptr<T, Deleter>& pointer,
                           const Context&... context) noexcept {
    if (!pointer)
      return std::nullopt;
    return rangeLoc(*pointer, context...);
  }
};

inline Location::Location(std::initializer_list<LazyLocation> providers)
    : Location(rangeLoc(providers)) {}

template <class U, class V>
struct LocationProvider<std::pair<U, V>> {
  static Location getRange(const std::pair<U, V>& value) noexcept {
    return {value.first, value.second};
  }
};

template <class T,
          std::enable_if_t<!std::is_same_v<std::decay_t<T>, LazyLocation> &&
                           !LazyLocation::nonLazy<T>>*>
inline LazyLocation::LazyLocation(const T& argument)
    : m_variant(Lazy{&argument, +[](const void* arg) {
                       return Diag::rangeLoc(*reinterpret_cast<const T*>(arg));
                     }}) {}

template <class T,
          std::enable_if_t<!std::is_same_v<std::decay_t<T>, LazyLocation> &&
                           LazyLocation::nonLazy<T>>*>
inline LazyLocation::LazyLocation(const T& argument)
    : m_variant(Diag::rangeLoc(argument)) {}

} // namespace pylir::Diag
