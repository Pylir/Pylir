#include "Globals.hpp"

extern "C" pylir::rt::PyObject*** const pylir_roots_default = nullptr;
extern "C" pylir::rt::PyObject** const pylir_others_default = nullptr;

extern "C" pylir::rt::PyObject*** const PYLIR_WEAK_VAR(pylir_roots_start, pylir_roots_default);
extern "C" pylir::rt::PyObject*** const PYLIR_WEAK_VAR(pylir_roots_end, pylir_roots_default);
extern "C" pylir::rt::PyObject** const PYLIR_WEAK_VAR(pylir_collections_start, pylir_others_default);
extern "C" pylir::rt::PyObject** const PYLIR_WEAK_VAR(pylir_collections_end, pylir_others_default);
extern "C" pylir::rt::PyObject** const PYLIR_WEAK_VAR(pylir_constants_start, pylir_others_default);
extern "C" pylir::rt::PyObject** const PYLIR_WEAK_VAR(pylir_constants_end, pylir_others_default);

bool pylir::rt::isGlobal(PyObject* object)
{
    struct Ranges
    {
        pylir::rt::PyObject* constMin;
        pylir::rt::PyObject* constMax;
        pylir::rt::PyObject* collMin;
        pylir::rt::PyObject* collMax;
    };
    static auto ranges = []
    {
        auto [constMin, constMax] = std::minmax_element(pylir_constants_start, pylir_constants_end);
        auto [colMin, colMax] = std::minmax_element(pylir_collections_start, pylir_collections_end);
        return Ranges{*constMin, *constMax, *colMin, *colMax};
    }();
    return (object >= ranges.constMin && object <= ranges.constMax)
           || (object >= ranges.collMin && object <= ranges.collMax);
}

tcb::span<pylir::rt::PyObject**> pylir::rt::getHandles()
{
    return {pylir_roots_start, pylir_roots_end};
}

tcb::span<pylir::rt::PyObject*> pylir::rt::getCollections()
{
    return {pylir_collections_start, pylir_collections_end};
}
