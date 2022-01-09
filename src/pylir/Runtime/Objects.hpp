
#pragma once

#include <pylir/Support/BigInt.hpp>
#include <pylir/Support/HashTable.hpp>

#include <array>
#include <string_view>
#include <type_traits>

#include <unwind.h>

#include "PylirGC.hpp"
#include "Support.hpp"

namespace pylir::rt
{

class PyTypeObject;
class PySequence;
class PyDict;
class PyFunction;
class PyObject;

struct KeywordArg
{
    std::string_view name;
    PyObject& arg;
};

struct Keyword
{
    std::string_view name;

    Keyword(const Keyword&) = delete;
    Keyword& operator=(const Keyword&) = delete;
    Keyword(Keyword&&) = delete;
    Keyword& operator=(Keyword&&) = delete;

    // Blatant abuse
    KeywordArg operator=(PyObject& object)
    {
        return {name, object};
    }
};

inline Keyword operator""_kw(const char* s, std::size_t n)
{
    return Keyword{std::string_view{s, n}};
}

class PyObject
{
    PyTypeObject* m_type;

    PyObject* mroLookup(int index);

    PyObject* methodLookup(int index);

public:
    explicit PyObject(PyObject& type) : m_type(reinterpret_cast<PyTypeObject*>(&type)) {}

    PyObject(const PyObject&) = delete;
    PyObject(PyObject&&) noexcept = delete;
    PyObject& operator=(const PyObject&) = delete;
    PyObject& operator=(PyObject&&) noexcept = delete;

    friend PyObject& type(PyObject& obj)
    {
        return *reinterpret_cast<PyObject*>(obj.m_type);
    }

    PyObject* getSlot(int index);

    PyObject* getSlot(std::string_view name);

    void setSlot(int index, PyObject& object);

    template <class... Args>
    PyObject& operator()(Args&&... args);

    bool operator==(PyObject& other);

    bool is(PyObject& other)
    {
        return this == &other;
    }

    template <class T>
    bool isa();

    template <class T>
    T& cast()
    {
        // TODO: static_assert(std::is_pointer_interconvertible_base_of_v<PyObject, T>);
        return *reinterpret_cast<T*>(this);
    }

    template <class T>
    T* dyn_cast()
    {
        return isa<T>() ? &cast<T>() : nullptr;
    }
};

bool isinstance(PyObject& obj, PyObject& type);

namespace Builtins
{
#define BUILTIN(name, symbol, ...) extern PyObject name asm(symbol);
#include <pylir/Interfaces/Builtins.def>
} // namespace Builtins

class PyTypeObject
{
    friend class PyObject;

    PyObject m_base;
    std::size_t m_offset;

public:
    operator PyObject&()
    {
        return m_base;
    }

    enum Slots
    {
#define TYPE_SLOT(x, ...) x,
#include <pylir/Interfaces/Slots.def>
    };
};

using PyUniversalCC = PyObject& (*)(PyFunction&, PySequence&, PyDict&);

class PyFunction
{
    friend class PyObject;

    PyObject m_base;
    PyUniversalCC m_function;

public:
    explicit PyFunction(PyUniversalCC function) : m_base(Builtins::Function), m_function(function) {}

    operator PyObject&()
    {
        return m_base;
    }

    enum Slots
    {
#define FUNCTION_SLOT(x, ...) x,
#include <pylir/Interfaces/Slots.def>
    };
};

class PySequence
{
protected:
    PyObject m_base;
    BufferComponent<PyObject*, MallocAllocator> m_buffer;

    template <class InputIter>
    PySequence(PyObject& type, InputIter begin, InputIter end) : m_base(type), m_buffer(begin, end)
    {
    }

public:
    explicit PySequence(PyObject& type) : m_base(type) {}

    operator PyObject&()
    {
        return m_base;
    }

    PyObject** begin()
    {
        return m_buffer.data();
    }

    PyObject** end()
    {
        return m_buffer.data() + m_buffer.size();
    }

    std::size_t len()
    {
        return m_buffer.size();
    }

    PyObject& getItem(std::size_t index)
    {
        return *m_buffer[index];
    }
};

class PyTuple : public PySequence
{
public:
    PyTuple() : PySequence(Builtins::Tuple) {}

    template <class... Args>
    explicit PyTuple(Args&... args) : PyTuple()
    {
        m_buffer.reserve(sizeof...(Args));
        (m_buffer.push_back(&args), ...);
    }

    template <class InputIter>
    explicit PyTuple(InputIter begin, InputIter end) : PySequence(Builtins::Tuple, begin, end)
    {
    }

    // Dangerous! Don't use it to modify tuples received from python code. Python compiler assumes tuples are
    // immutable. One may use it to construct a tuple however
    void push_back(PyObject& object)
    {
        m_buffer.push_back(&object);
    }
};

class PyList : public PySequence
{
public:
    PyList() : PySequence(Builtins::List) {}
};

class PyString
{
    PyObject m_base;
    BufferComponent<char, MallocAllocator> m_buffer;

public:
    explicit PyString(std::string_view string, PyObject& type = Builtins::Str)
        : m_base(type), m_buffer(string.begin(), string.end())
    {
    }

    operator PyObject&()
    {
        return m_base;
    }

    friend bool operator==(PyString& lhs, std::string_view sv)
    {
        return lhs.view() == sv;
    }

    friend bool operator==(const std::string_view sv, PyString& rhs)
    {
        return rhs.view() == sv;
    }

    std::string_view view() const
    {
        return std::string_view{m_buffer.data(), m_buffer.size()};
    }

    std::size_t len() const
    {
        return m_buffer.size();
    }
};

class PyDict
{
    PyObject m_base;
    HashTable<PyObject*, PyObject*, PyObjectHasher, PyObjectEqual, MallocAllocator> m_table;

public:
    PyDict(PyObject& type = Builtins::Dict) : m_base(type) {}

    operator PyObject&()
    {
        return m_base;
    }

    PyObject* tryGetItem(PyObject& key)
    {
        auto result = m_table.find(&key);
        if (result == m_table.end())
        {
            return nullptr;
        }
        return result->value;
    }

    void setItem(PyObject& key, PyObject& value)
    {
        m_table.insert_or_assign(&key, &value);
    }

    void delItem(PyObject& key)
    {
        m_table.erase(&key);
    }
};

class PyInt
{
    PyObject m_base;
    BigInt m_integer;

public:
    operator PyObject&()
    {
        return m_base;
    }

    bool boolean()
    {
        return !m_integer.isZero();
    }

    template <class T>
    T to()
    {
        return m_integer.getInteger<T>();
    }
};

class PyBaseException
{
    PyObject m_base;
    std::uintptr_t m_landingPad;
    _Unwind_Exception m_unwindHeader;
    std::uint32_t m_typeIndex;

public:
    constexpr static std::uint64_t EXCEPTION_CLASS = 0x50594C5250590000; // PYLRPY\0\0

    operator PyObject&()
    {
        return m_base;
    }

    enum Slots
    {
#define BASEEXCEPTION_SLOT(x, ...) x,
#include <pylir/Interfaces/Slots.def>
    };

    _Unwind_Exception& getUnwindHeader()
    {
        static_assert(offsetof(PyBaseException, m_unwindHeader) == alignof(_Unwind_Exception));
        return m_unwindHeader;
    }

    static PyBaseException* fromUnwindHeader(_Unwind_Exception* header)
    {
        PYLIR_ASSERT(header->exception_class == EXCEPTION_CLASS);
        return reinterpret_cast<PyBaseException*>(reinterpret_cast<char*>(header)
                                                  - offsetof(PyBaseException, m_unwindHeader));
    }

    std::uintptr_t getLandingPad() const
    {
        return m_landingPad;
    }

    void setLandingPad(std::uintptr_t landingPad)
    {
        m_landingPad = landingPad;
    }

    std::uint32_t getTypeIndex() const
    {
        return m_typeIndex;
    }

    void setTypeIndex(std::uint32_t typeIndex)
    {
        m_typeIndex = typeIndex;
    }
};

template <PyObject&>
struct PyTypeTraits;

template <>
struct PyTypeTraits<Builtins::Type>
{
    using instanceType = PyTypeObject;
    constexpr static std::size_t slotCount =
        std::initializer_list<int>{
#define TYPE_SLOT(x, ...) 0,
#include <pylir/Interfaces/Slots.def>
        }
            .size();
};

template <>
struct PyTypeTraits<Builtins::Function>
{
    using instanceType = PyFunction;
    constexpr static std::size_t slotCount =
        std::initializer_list<int>{
#define FUNCTION_SLOT(x, ...) 0,
#include <pylir/Interfaces/Slots.def>
        }
            .size();
};

#define NO_SLOT_TYPE(name, instance)                \
    template <>                                     \
    struct PyTypeTraits<Builtins::name>             \
    {                                               \
        using instanceType = instance;              \
        constexpr static std::size_t slotCount = 0; \
    }

NO_SLOT_TYPE(Int, PyInt);
NO_SLOT_TYPE(Bool, PyInt);
// TODO: NO_SLOT_TYPE(Float, PyFloat);
NO_SLOT_TYPE(Str, PyString);
NO_SLOT_TYPE(Tuple, PySequence);
NO_SLOT_TYPE(List, PySequence);
// TODO: NO_SLOT_TYPE(Set, PySet);
NO_SLOT_TYPE(Dict, PyDict);
#undef NO_SLOT_TYPE

namespace details
{
constexpr static std::size_t BaseExceptionSlotCount =
    std::initializer_list<int>{
#define BASEEXCEPTION_SLOT(x, ...) 0,
#include <pylir/Interfaces/Slots.def>
    }
        .size();
} // namespace details

#define BUILTIN_EXCEPTION(name, ...)                                              \
    template <>                                                                   \
    struct PyTypeTraits<Builtins::name>                                           \
    {                                                                             \
        using instanceType = PyBaseException;                                     \
        constexpr static std::size_t slotCount = details::BaseExceptionSlotCount; \
    };
#include <pylir/Interfaces/Builtins.def>

template <PyObject& typeObject, std::size_t slotCount = PyTypeTraits<typeObject>::slotCount>
class StaticInstance
{
    using InstanceType = typename PyTypeTraits<typeObject>::instanceType;
    static_assert(alignof(InstanceType) >= alignof(PyObject*));
    alignas(InstanceType) std::byte m_buffer[sizeof(InstanceType) + slotCount * sizeof(PyObject*)]{};

public:
    template <class... Args>
    StaticInstance(std::initializer_list<std::pair<typename InstanceType::Slots, PyObject&>> slotsInit, Args&&... args)
    {
        static_assert(std::is_standard_layout_v<std::remove_reference_t<decltype(*this)>>);
        PyObject& instance = *new (m_buffer) InstanceType(std::forward<Args>(args)...);
        for (auto& pair : slotsInit)
        {
            instance.setSlot(pair.first, pair.second);
        }
    }

    operator PyObject&()
    {
        return get();
    }

    operator InstanceType&()
    {
        return *reinterpret_cast<InstanceType*>(m_buffer);
    }

    InstanceType& get()
    {
        return *this;
    }
};

template <PyObject& typeObject>
class StaticInstance<typeObject, 0>
{
    using InstanceType = typename PyTypeTraits<typeObject>::instanceType;
    InstanceType m_object;

public:
    template <class... Args>
    StaticInstance(Args&&... args) : m_object(std::forward<Args>(args)...)
    {
    }

    operator PyObject&()
    {
        return m_object;
    }

    operator InstanceType&()
    {
        return m_object;
    }

    InstanceType& get()
    {
        return m_object;
    }
};

template <class... Args>
PyObject& PyObject::operator()(Args&&... args)
{
    PyObject* self = this;
    while (true)
    {
        auto* call = self->methodLookup(PyTypeObject::__call__);
        if (!call)
        {
            // TODO: raise Type error
        }
        if (auto* pyF = call->dyn_cast<PyFunction>())
        {
            PyTuple& tuple = alloc<PyTuple>(*self);
            PyDict& dict = alloc<PyDict>();
            (
                [&](auto&& arg)
                {
                    static_assert(
                        std::is_same_v<PyObject&, decltype(arg)> || std::is_same_v<KeywordArg&&, decltype(arg)>);
                    if constexpr (std::is_same_v<PyObject&, decltype(arg)>)
                    {
                        tuple.push_back(arg);
                    }
                    else
                    {
                        dict.setItem(alloc<PyString>(arg.name), arg.arg);
                    }
                }(std::forward<Args>(args)),
                ...);
            return pyF->m_function(*pyF, tuple, dict);
        }
        self = call;
    }
}

template <class T>
inline bool PyObject::isa()
{
    static_assert(sizeof(T) && false, "No specialization available");
    PYLIR_UNREACHABLE;
}

template <>
inline bool PyObject::isa<PyTypeObject>()
{
    return isinstance(*this, Builtins::Type);
}

template <>
inline bool PyObject::isa<PySequence>()
{
    return isinstance(*this, Builtins::Tuple) || isinstance(*this, Builtins::List);
}

template <>
inline bool PyObject::isa<PyDict>()
{
    return isinstance(*this, Builtins::Dict);
}

template <>
inline bool PyObject::isa<PyFunction>()
{
    return type(*this).is(Builtins::Function);
}

template <>
inline bool PyObject::isa<PyString>()
{
    return isinstance(*this, Builtins::Str);
}

template <>
inline bool PyObject::isa<PyInt>()
{
    return isinstance(*this, Builtins::Int);
}

template <>
inline bool PyObject::isa<PyBaseException>()
{
    return isinstance(*this, Builtins::BaseException);
}

} // namespace pylir::rt
