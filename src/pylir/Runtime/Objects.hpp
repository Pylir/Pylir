
#pragma once

#include <pylir/Support/BigInt.hpp>
#include <pylir/Support/HashTable.hpp>

#include <array>
#include <string_view>
#include <type_traits>

#include <unwind.h>

#include "Builtins.hpp"
#include "GCInterface.hpp"
#include "Support.hpp"

namespace pylir::rt
{

struct KeywordArg
{
    std::string_view name;
    PyObject& arg;
};

struct Keyword
{
    std::string_view name;

    ~Keyword() = default;
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
    constexpr explicit PyObject(PyTypeObject& type) : m_type(&type) {}

    ~PyObject() = default;
    PyObject(const PyObject&) = delete;
    PyObject(PyObject&&) noexcept = delete;
    PyObject& operator=(const PyObject&) = delete;
    PyObject& operator=(PyObject&&) noexcept = delete;

    friend PyObject& type(PyObject& obj)
    {
        return *reinterpret_cast<PyObject*>(reinterpret_cast<std::uintptr_t>(obj.m_type) & ~std::uintptr_t{0b11});
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

    void clearMarking()
    {
        m_type = reinterpret_cast<PyTypeObject*>(reinterpret_cast<std::uintptr_t>(m_type) & ~std::uintptr_t(0b11));
    }

    template <class T>
    void setMark(T value)
    {
        m_type = reinterpret_cast<PyTypeObject*>((reinterpret_cast<std::uintptr_t>(m_type) & ~std::uintptr_t(0b11))
                                                 | static_cast<std::uintptr_t>(value));
    }

    template <class T>
    T getMark()
    {
        return static_cast<T>(reinterpret_cast<std::uintptr_t>(m_type) & 0b11);
    }
};

void destroyPyObject(PyObject& object);

bool isinstance(PyObject& obj, PyObject& type);

class PyTypeObject
{
    friend class PyObject;

    PyObject m_base;
    std::size_t m_offset;
    PyTypeObject* m_layoutType;
    PyTuple* m_mroTuple;

public:
    /*implicit*/ operator PyObject&() noexcept
    {
        return m_base;
    }

    constexpr static auto& layoutTypeObject = Builtins::Type;

    enum Slots
    {
#define TYPE_SLOT(x, y, ...) y,
#include <pylir/Interfaces/Slots.def>
    };

    template <class... Args>
    PyObject& operator()(Args&&... args)
    {
        return m_base(std::forward<Args>(args)...);
    }

    [[nodiscard]] std::size_t getOffset() const noexcept
    {
        return m_offset;
    }

    [[nodiscard]] PyTuple& getMROTuple() const noexcept
    {
        return *m_mroTuple;
    }
};

using PyUniversalCC = PyObject& (*)(PyFunction&, PyTuple&, PyDict&);

class PyFunction
{
    friend class PyObject;

    PyObject m_base;
    PyUniversalCC m_function;

public:
    constexpr explicit PyFunction(PyUniversalCC function) : m_base(Builtins::Function), m_function(function) {}

    /*implicit*/ operator PyObject&()
    {
        return m_base;
    }

    constexpr static auto& layoutTypeObject = Builtins::Function;

    enum Slots
    {
#define FUNCTION_SLOT(x, y, ...) y,
#include <pylir/Interfaces/Slots.def>
    };
};

class PyTuple
{
    PyObject m_base;
    std::size_t m_size;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
    PyObject* m_trailing[];
#pragma GCC diagnostic pop

public:
    explicit PyTuple(std::size_t size, PyObject& type = Builtins::Tuple)
        : m_base(type.cast<PyTypeObject>()), m_size(size)
    {
    }

    constexpr static auto& layoutTypeObject = Builtins::Tuple;

    /*implicit*/ operator PyObject&()
    {
        return m_base;
    }

    PyObject** begin()
    {
        return m_trailing;
    }

    PyObject** end()
    {
        return m_trailing + m_size;
    }

    std::size_t len()
    {
        return m_size;
    }

    PyObject& getItem(std::size_t index)
    {
        return *m_trailing[index];
    }
};

class PyList
{
    PyObject m_base;
    std::size_t m_size;
    PyTuple* m_tuple;

public:
    explicit PyList(PyObject& type = Builtins::List) : m_base(type.cast<PyTypeObject>()) {}

    /*implicit*/ operator PyObject&()
    {
        return m_base;
    }

    constexpr static auto& layoutTypeObject = Builtins::List;

    PyObject** begin()
    {
        return m_tuple->begin();
    }

    PyObject** end()
    {
        return m_tuple->begin() + m_size;
    }

    std::size_t len()
    {
        return m_size;
    }

    PyObject& getItem(std::size_t index)
    {
        return m_tuple->getItem(index);
    }
};

class PyString
{
    PyObject m_base;
    BufferComponent<char, MallocAllocator> m_buffer;

public:
    explicit PyString(std::string_view string, PyObject& type = Builtins::Str)
        : m_base(type.cast<PyTypeObject>()), m_buffer(string.begin(), string.end())
    {
    }

    constexpr static auto& layoutTypeObject = Builtins::Str;

    /*implicit*/ operator PyObject&()
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

    [[nodiscard]] std::string_view view() const
    {
        return std::string_view{m_buffer.data(), m_buffer.size()};
    }

    [[nodiscard]] std::size_t len() const
    {
        return m_buffer.size();
    }
};

class PyDict
{
    PyObject m_base;
    HashTable<PyObject*, PyObject*, PyObjectHasher, PyObjectEqual, MallocAllocator> m_table;

public:
    explicit PyDict(PyObject& type = Builtins::Dict) : m_base(type.cast<PyTypeObject>()) {}

    constexpr static auto& layoutTypeObject = Builtins::Dict;

    /*implicit*/ operator PyObject&()
    {
        return m_base;
    }

    PyObject* tryGetItem(PyObject& key)
    {
        auto* result = m_table.find(&key);
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

    auto begin()
    {
        return m_table.begin();
    }

    auto end()
    {
        return m_table.end();
    }
};

class PyInt
{
    PyObject m_base;
    BigInt m_integer;

public:
    constexpr static auto& layoutTypeObject = Builtins::Int;

    /*implicit*/ operator PyObject&()
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

    /*implicit*/ operator PyObject&()
    {
        return m_base;
    }

    constexpr static auto& layoutTypeObject = Builtins::BaseException;

    enum Slots
    {
#define BASEEXCEPTION_SLOT(x, y, ...) y,
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

    [[nodiscard]] std::uintptr_t getLandingPad() const
    {
        return m_landingPad;
    }

    void setLandingPad(std::uintptr_t landingPad)
    {
        m_landingPad = landingPad;
    }

    [[nodiscard]] std::uint32_t getTypeIndex() const
    {
        return m_typeIndex;
    }

    void setTypeIndex(std::uint32_t typeIndex)
    {
        m_typeIndex = typeIndex;
    }
};

template <PyTypeObject& typeObject, std::size_t slotCount = PyTypeTraits<typeObject>::slotCount>
class StaticInstance
{
    using InstanceType = typename PyTypeTraits<typeObject>::instanceType;
    static_assert(alignof(InstanceType) >= alignof(PyObject*));
    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    alignas(InstanceType) std::array<std::byte, sizeof(InstanceType) + slotCount * sizeof(PyObject*)> m_buffer{};

public:
    template <class... Args>
    StaticInstance(std::initializer_list<std::pair<typename InstanceType::Slots, PyObject&>> slotsInit, Args&&... args)
    {
        static_assert(std::is_standard_layout_v<std::remove_reference_t<decltype(*this)>>);
        new (m_buffer.data()) InstanceType(std::forward<Args>(args)...);
        std::array<PyObject*, slotCount> slots;
        for (auto& [index, object] : slotsInit)
        {
            slots[index] = &object;
        }
        std::memcpy(m_buffer.data() + sizeof(InstanceType), slots.data(), slots.size() * sizeof(PyObject*));
    }

    /*implicit*/ operator PyObject&()
    {
        return get();
    }

    /*implicit*/ operator InstanceType&()
    {
        return *reinterpret_cast<InstanceType*>(m_buffer);
    }

    InstanceType& get()
    {
        return *this;
    }
};

template <PyTypeObject& typeObject>
class StaticInstance<typeObject, 0>
{
    using InstanceType = typename PyTypeTraits<typeObject>::instanceType;
    InstanceType m_object;

public:
    template <class... Args>
    explicit StaticInstance(Args&&... args) : m_object(std::forward<Args>(args)...)
    {
    }

    /*implicit*/ operator PyObject&()
    {
        return m_object;
    }

    /*implicit*/ operator InstanceType&()
    {
        return m_object;
    }

    InstanceType& get()
    {
        return m_object;
    }
};

namespace details
{

template <PyTypeObject& type>
struct AllocType
{
    template <class... Args>
    decltype(auto) operator()(Args&&... args) const noexcept
    {
        using InstanceType = typename PyTypeTraits<type>::instanceType;
        void* memory = pylir_gc_alloc(sizeof(InstanceType) + sizeof(PyObject*) * PyTypeTraits<type>::slotCount);
        return *new (memory) InstanceType(std::forward<Args>(args)...);
    }
};

template <>
struct AllocType<Builtins::Tuple>
{
    template <class... Args>
    decltype(auto) operator()(std::size_t count, Args&&... args) const noexcept
    {
        using InstanceType = typename PyTypeTraits<Builtins::Tuple>::instanceType;
        std::byte* memory =
            reinterpret_cast<std::byte*>(pylir_gc_alloc(sizeof(InstanceType) + sizeof(PyObject*) * count));
        return *new (memory) InstanceType(count, std::forward<Args>(args)...);
    }
};

} // namespace details

template <PyTypeObject& type>
constexpr details::AllocType<type> alloc;

template <class... Args>
PyObject& PyObject::operator()(Args&&... args)
{
    PyObject* self = this;
    while (true)
    {
        auto* call = self->methodLookup(PyTypeObject::Call);
        if (!call)
        {
            // TODO: raise Type error
        }
        if (auto* pyF = call->dyn_cast<PyFunction>())
        {
            constexpr std::size_t tupleCount = (1 + ... + std::is_same_v<PyObject&, Args>);
            auto& tuple = alloc<Builtins::Tuple>(tupleCount);
            auto& dict = alloc<Builtins::Dict>();
            auto iter = tuple.begin();
            *iter++ = self;
            (
                [&](auto&& arg)
                {
                    static_assert(
                        std::is_same_v<PyObject&, decltype(arg)> || std::is_same_v<KeywordArg&&, decltype(arg)>);
                    if constexpr (std::is_same_v<PyObject&, decltype(arg)>)
                    {
                        *iter++ = &arg;
                    }
                    else
                    {
                        dict.setItem(alloc<Builtins::Str>(arg.name), arg.arg);
                    }
                }(std::forward<Args>(args)),
                ...);
            return pyF->m_function(*pyF, tuple, dict);
        }
        self = call;
    }
}

template <>
inline bool PyObject::isa<PyObject>()
{
    return true;
}

template <class T>
inline bool PyObject::isa()
{
    return type(*this).cast<PyTypeObject>().m_layoutType == &T::layoutTypeObject;
}

} // namespace pylir::rt
