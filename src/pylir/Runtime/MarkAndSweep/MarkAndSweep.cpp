#include "MarkAndSweep.hpp"

#include <pylir/Support/Util.hpp>

// Anything below 65535 would do basically
pylir::rt::MarkAndSweep pylir::rt::gc __attribute__((init_priority(200)));

pylir::rt::PyObject* pylir::rt::MarkAndSweep::alloc(std::size_t count)
{
    count = pylir::roundUpTo(count, alignof(std::max_align_t));
    switch (count / alignof(std::max_align_t))
    {
        case 1:
        case 2: return m_unit2.nextCell();
        case 3:
        case 4: return m_unit4.nextCell();
        case 5:
        case 6: return m_unit6.nextCell();
        case 7:
        case 8: return m_unit8.nextCell();
        default: return m_tree.alloc(count);
    }
}
