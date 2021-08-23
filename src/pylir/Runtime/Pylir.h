
#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct PylirTypeObject PylirTypeObject;

    typedef struct PylirObject
    {
        PylirTypeObject* m_type;
    } PylirObject;

    // In MLIR IR this has no sign. In the runtime it is signed. Basically a signed version of size_t
    typedef ptrdiff_t index;

    typedef PylirObject* (*PylirCCType)(PylirObject* self, PylirObject* tuple, PylirObject* dict);
    typedef PylirObject* (*PylirBinOp)(PylirObject* lhs, PylirObject* rhs);
    typedef PylirObject* (*PylirTernaryOp)(PylirObject*, PylirObject*, PylirObject*);
    typedef PylirObject* (*PylirUnaryOp)(PylirObject*);

    typedef struct PylirTypeObject
    {
        index m_dictPtr;
        PylirCCType m_call;
        PylirBinOp m_add;
        PylirBinOp m_sub;
        PylirBinOp m_multiply;
        PylirBinOp m_remainder;
        PylirBinOp m_divmod;
        PylirTernaryOp m_power;
        PylirUnaryOp m_negative;
        PylirUnaryOp m_positive;
        PylirUnaryOp m_absolute;
        PylirUnaryOp m_bool;
        PylirUnaryOp m_invert;
        PylirBinOp m_lshift;
        PylirBinOp m_rshift;
        PylirBinOp m_and;
        PylirBinOp m_xor;
        PylirBinOp m_or;
        PylirUnaryOp m_int;
        PylirUnaryOp m_float;
        PylirBinOp m_inplaceAdd;
        PylirBinOp m_inplaceSubtract;
        PylirBinOp m_inplaceMultiply;
        PylirBinOp m_inplaceRemainder;
        PylirTernaryOp m_inplacePower;
        PylirBinOp m_inplaceLShift;
        PylirBinOp m_inplaceRShift;
        PylirBinOp m_inplaceAnd;
        PylirBinOp m_inplaceXor;
        PylirBinOp m_inplaceOr;
        PylirBinOp m_floorDivide;
        PylirBinOp m_trueDivide;
        PylirBinOp m_inplaceFloorDivide;
        PylirBinOp m_inplaceTrueDivide;
        PylirUnaryOp m_index;
        PylirBinOp m_matrixMultiply;
        PylirBinOp m_inplaceMatrixMultiply;
        PylirUnaryOp m_length;
        PylirBinOp m_getItem;
        PylirTernaryOp m_setItem;
        PylirBinOp m_delItem;
        PylirBinOp m_missing;
        PylirUnaryOp m_iter;
        PylirBinOp m_contains;
        PylirUnaryOp m_hash;
        PylirUnaryOp m_str;
        PylirUnaryOp m_repr;
        PylirBinOp m_getAttr;
        PylirTernaryOp m_setAttr;
        PylirBinOp m_eq;
        PylirBinOp m_ne;
        PylirBinOp m_lt;
        PylirBinOp m_gt;
        PylirBinOp m_le;
        PylirBinOp m_ge;
        PylirUnaryOp m_iterNext;
        PylirObject* m_dict;
        PylirTernaryOp m_descrGet;
        PylirTernaryOp m_descrSet;
        PylirCCType m_init;
        PylirCCType m_new;
        PylirObject* m_bases;
        PylirUnaryOp m_del;
    } PylirTypeObject;

    void* pylir_gc_alloc(size_t size);

    extern PylirTypeObject pylir_integer_type_object;

    typedef struct PylirInteger
    {
        PylirObject m_base;
        size_t m_count;
        /* size_t values[]; */
    } PylirInteger;

    PylirInteger* pylir_integer_from_size_t(size_t value);

    PylirInteger* pylir_integer_mul(PylirInteger* lhs, PylirInteger* rhs);

    typedef struct PylirIntegerToIndexResult
    {
        index value;
        bool overflow;
    } PylirIntegerToIndexResult;

    PylirIntegerToIndexResult pylir_integer_to_index(PylirInteger* value);

#ifdef __cplusplus
}
#endif
