#include "Globals.hpp"

#include <vector>

// The code in here is definitely undefined behaviour (in particular the Windows solution) and if anything deserves
// a better solution it'd be this. That said, until we have relocating GC, there are only read accesses so its
// "probably" fine, but we are still doing a bunch of aliasing violations. In particular, if we had a relocating GC
// and were in C++ code, the compiler would optimize something like:
//  global = object1;
//  functionThatTriggersRelocation();
//  return global;
//
// To simply return object1, despite its address having been relocated. This is not an issue in Python code because
// the compiler passes for the GC would see that the SSA value of object1 is alive and insert an relocation, but we
// can't do that in C++. More likely would be need some kind of Handle type that does some magic and at the very least
// escapes the underlying root. TBD

extern "C"
{
#ifdef _WIN32
    // NOLINTNEXTLINE(bugprone-reserved-identifier)
    static pylir::rt::PyObject* __start_py_root __attribute__((section("py_root$a"))) = nullptr;
    // NOLINTNEXTLINE(bugprone-reserved-identifier)
    static pylir::rt::PyObject* __stop_py_root __attribute__((section("py_root$z"))) = nullptr;
#else
    // NOLINTNEXTLINE(bugprone-reserved-identifier)
    extern pylir::rt::PyObject* __start_py_root __attribute__((weak));
    // NOLINTNEXTLINE(bugprone-reserved-identifier)
    extern pylir::rt::PyObject* __stop_py_root __attribute__((weak));
#endif

#ifdef _WIN32
    // NOLINTNEXTLINE(bugprone-reserved-identifier)
    static __attribute__((section("py_coll$a"))) pylir::rt::PyObject __start_py_coll{pylir::rt::Builtins::NoneType};
    // NOLINTNEXTLINE(bugprone-reserved-identifier)
    static __attribute__((section("py_coll$z"))) pylir::rt::PyObject __stop_py_coll{pylir::rt::Builtins::NoneType};
#else
    // NOLINTNEXTLINE(bugprone-reserved-identifier)
    extern pylir::rt::PyObject __start_py_coll __attribute__((weak));
    // NOLINTNEXTLINE(bugprone-reserved-identifier)
    extern pylir::rt::PyObject __stop_py_coll __attribute__((weak));
#endif

#ifdef _WIN32
    constexpr __attribute__((section("py_const$a")))
    // NOLINTNEXTLINE(bugprone-reserved-identifier)
    pylir::rt::PyObject __start_py_const{pylir::rt::Builtins::NoneType};
    // NOLINTNEXTLINE(bugprone-reserved-identifier)
    constexpr __attribute__((section("py_const$z"))) pylir::rt::PyObject __stop_py_const{pylir::rt::Builtins::NoneType};
#else
    // NOLINTNEXTLINE(bugprone-reserved-identifier)
    extern pylir::rt::PyObject __start_py_const __attribute__((weak));
    // NOLINTNEXTLINE(bugprone-reserved-identifier)
    extern pylir::rt::PyObject __stop_py_const __attribute__((weak));
#endif
}

bool pylir::rt::isGlobal(PyObject* object)
{
    return (object >= &__start_py_coll && object <= &__stop_py_coll)
           || (object >= &__start_py_const && object <= &__stop_py_const);
}

tcb::span<pylir::rt::PyObject*> pylir::rt::getHandles()
{
    return {&__start_py_root, &__stop_py_root};
}

tcb::span<pylir::rt::PyObject* const> pylir::rt::getCollections()
{
    static auto results = []
    {
        std::vector<pylir::rt::PyObject*> result;

        pylir::rt::PyObject* begin = &__start_py_coll;
#ifndef _WIN32
        if (!begin)
        {
            return result;
        }
#endif
        auto nextPyObject = [](pylir::rt::PyObject* object) -> pylir::rt::PyObject*
        {
            auto& typeObject = type(*object).cast<PyTypeObject>();
            std::size_t slotCount = 0;
            if (auto* slots = static_cast<PyObject&>(typeObject).getSlot(PyTypeObject::Slots))
            {
                slotCount = slots->cast<PyTuple>().len();
            }
            else if (auto* tuple = object->dyn_cast<PyTuple>())
            {
                slotCount = tuple->len();
            }
            auto* bytes = reinterpret_cast<std::byte*>(object);
            // NOLINTNEXTLINE(bugprone-sizeof-expression)
            bytes += sizeof(PyObject*) * (slotCount + typeObject.getOffset());
            if (bytes == reinterpret_cast<std::byte*>(&__stop_py_coll))
            {
                return nullptr;
            }
            // Due to section alignment there might be nulls inbetween. Skip over those till we find a non null value
            void* pointer;
            std::memcpy(&pointer, bytes, sizeof(void*));
            while (!pointer)
            {
                bytes += sizeof(void*);
                std::memcpy(&pointer, bytes, sizeof(void*));
            }
            return reinterpret_cast<PyObject*>(bytes);
        };

#ifdef _WIN32
        begin = nextPyObject(begin);
#endif

        for (; begin; begin = nextPyObject(begin))
        {
            result.push_back(begin);
        }
        return result;
    }();
    return results;
}
