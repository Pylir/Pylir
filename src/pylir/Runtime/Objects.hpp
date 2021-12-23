
#pragma once

#include <pylir/Support/BigInt.hpp>
#include <pylir/Support/HashTable.hpp>

#include <array>
#include <string_view>
#include <type_traits>

namespace pylir::rt
{

class PyTypeObject;
class PySequence;
class PyDict;
class PyFunction;

class PyObject
{
    PyTypeObject* m_type;

public:
    PyObject(PyTypeObject* type) : m_type(type) {}

    PyObject(const PyObject&) = delete;
    PyObject(PyObject&&) noexcept = delete;
    PyObject& operator=(const PyObject&) = delete;
    PyObject& operator=(PyObject&&) noexcept = delete;

    PyObject* getType() const
    {
        return reinterpret_cast<PyObject*>(m_type);
    }

    template <class T>
    bool isa();

    template <class T>
    T* cast()
    {
        // TODO: static_assert(std::is_pointer_interconvertible_base_of_v<PyObject, T>);
        return reinterpret_cast<T*>(this);
    }

    template <class T>
    T* dyn_cast()
    {
        return isa<T>() ? cast<T>() : nullptr;
    }

    PyObject* getSlot(int index);

    PyObject* getSlot(std::string_view name);

    PyObject* call(PySequence* args, PyDict* keywords);

    PyObject* call(std::initializer_list<PyObject*> args);
};

static_assert(std::is_standard_layout_v<PyObject>);

namespace Builtin
{
#define BUILTIN(name, symbol, ...) extern PyObject name asm(symbol);
#include <pylir/Interfaces/Builtins.def>
} // namespace Builtin

using PyUniversalCC = PyObject* (*)(PyFunction*, PySequence*, PyDict*);

class PyTypeObject
{
    PyObject m_base;
    std::size_t m_offset;

public:
    std::size_t getOffset() const
    {
        return m_offset;
    }

    enum Slots
    {
#define TYPE_SLOT(x) x,
#include <pylir/Interfaces/Slots.def>
    };
};

static_assert(std::is_standard_layout_v<PyTypeObject>);

class PyFunction
{
    PyObject m_base;
    PyUniversalCC m_function;

public:
    enum Slots
    {
#define FUNCTION_SLOT(x) x,
#include <pylir/Interfaces/Slots.def>
    };

    PyObject* call(PySequence* args, PyDict* keywords)
    {
        return m_function(this, args, keywords);
    }
};

static_assert(std::is_standard_layout_v<PyFunction>);

template <class T>
struct BufferComponent
{
    std::size_t size{};
    std::size_t capacity{};
    T* array{};
};

class PySequence
{
    PyObject m_base;
    BufferComponent<PyObject*> m_buffer;

protected:
    PySequence(PyTypeObject* type, BufferComponent<PyObject*> data) : m_base(type), m_buffer(data) {}

public:
    PyObject** begin()
    {
        return m_buffer.array;
    }

    PyObject** end()
    {
        return m_buffer.array + m_buffer.size;
    }

    std::size_t len() const
    {
        return m_buffer.size;
    }

    PyObject* getItem(std::size_t index) const
    {
        return m_buffer.array[index];
    }
};

static_assert(std::is_standard_layout_v<PySequence>);

class PyString
{
    PyObject m_base;
    BufferComponent<char> m_buffer;

public:
    friend bool operator==(const PyString& lhs, std::string_view sv)
    {
        return lhs.view() == sv;
    }

    friend bool operator==(const std::string_view sv, PyString& rhs)
    {
        return rhs.view() == sv;
    }

    std::string_view view() const
    {
        return std::string_view{m_buffer.array, m_buffer.size};
    }

    std::size_t len() const
    {
        return m_buffer.size;
    }
};

static_assert(std::is_standard_layout_v<PySequence>);

struct PyObjectHasher
{
    std::size_t operator()(PyObject* object) const noexcept;
};

struct PyObjectEqual
{
    bool operator()(PyObject* lhs, PyObject* rhs) const noexcept;
};

class PyDict
{
    PyObject m_base;
    HashTable<PyObject*, PyObject*, PyObjectHasher, PyObjectEqual> m_table;

public:
    PyDict() : m_base(reinterpret_cast<PyTypeObject*>(&Builtin::Dict)) {}

    PyObject* tryGetItem(PyObject* key);

    void setItem(PyObject* key, PyObject* value)
    {
        m_table.insert_or_assign(key, value);
    }
};

static_assert(std::is_standard_layout_v<PyDict>);

class PyInt
{
    PyObject m_base;
    BigInt m_integer;

public:
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

// TODO: Inheritance

template <class T>
inline bool PyObject::isa()
{
    static_assert(sizeof(T) && false, "No specialization available");
    PYLIR_UNREACHABLE;
}

template <>
inline bool PyObject::isa<PyTypeObject>()
{
    return getType() == &Builtin::Type;
}

template <>
inline bool PyObject::isa<PySequence>()
{
    return getType() == &Builtin::Tuple || getType() == &Builtin::List;
}

template <>
inline bool PyObject::isa<PyDict>()
{
    return getType() == &Builtin::Dict;
}

template <>
inline bool PyObject::isa<PyFunction>()
{
    return getType() == &Builtin::Function;
}

template <>
inline bool PyObject::isa<PyString>()
{
    return getType() == &Builtin::Str;
}

template <>
inline bool PyObject::isa<PyInt>()
{
    return getType() == &Builtin::Int || getType() == &Builtin::Bool;
}

} // namespace pylir::rt
