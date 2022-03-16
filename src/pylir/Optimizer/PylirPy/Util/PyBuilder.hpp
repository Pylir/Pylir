
#pragma once

#include <mlir/IR/Builders.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Support/BigInt.hpp>

namespace pylir::Py
{
class PyBuilder : public mlir::OpBuilder
{
    mlir::Location m_loc;

    void implementBlock(mlir::Block* block)
    {
        if (auto* next = getBlock()->getNextNode())
        {
            block->insertBefore(next);
        }
        else
        {
            getBlock()->getParent()->push_back(block);
        }
        setInsertionPointToStart(block);
    }

public:
    explicit PyBuilder(mlir::Operation* operationBefore)
        : mlir::OpBuilder(operationBefore), m_loc(operationBefore->getLoc())
    {
    }

    explicit PyBuilder(mlir::MLIRContext* context, llvm::Optional<mlir::Location> loc = {})
        : mlir::OpBuilder(context), m_loc(loc.getValueOr(getUnknownLoc()))
    {
    }

    [[nodiscard]] mlir::Location getCurrentLoc() const
    {
        return m_loc;
    }

    void setCurrentLoc(mlir::Location loc)
    {
        m_loc = loc;
    }

    Py::UnboundAttr getUnboundAttr()
    {
        return Py::UnboundAttr::get(getContext());
    }

    Py::ObjectAttr getObjectAttr(mlir::FlatSymbolRefAttr type, mlir::DictionaryAttr slots = {},
                                 mlir::Attribute builtinValue = {})
    {
        return Py::ObjectAttr::get(type, slots, builtinValue);
    }

    Py::TypeAttr getTypeAttr(mlir::Attribute mroTuple = {}, mlir::DictionaryAttr slots = {})
    {
        return Py::TypeAttr::get(context, mroTuple, {}, slots);
    }

    Py::IntAttr getIntAttr(BigInt bigInt)
    {
        return Py::IntAttr::get(getContext(), std::move(bigInt));
    }

    Py::BoolAttr getPyBoolAttr(bool value)
    {
        return Py::BoolAttr::get(getContext(), value);
    }

    Py::FloatAttr getFloatAttr(double value)
    {
        return Py::FloatAttr::get(getContext(), llvm::APFloat(value));
    }

    Py::StrAttr getStrAttr(llvm::StringRef value)
    {
        return Py::StrAttr::get(getContext(), value);
    }

    Py::TupleAttr getTupleAttr(llvm::ArrayRef<mlir::Attribute> value = {})
    {
        return Py::TupleAttr::get(getContext(), value);
    }

    Py::ListAttr getListAttr(llvm::ArrayRef<mlir::Attribute> value = {})
    {
        return Py::ListAttr::get(getContext(), value);
    }

    Py::DictAttr getDictAttr(llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value = {})
    {
        return Py::DictAttr::get(getContext(), value);
    }

    Py::SetAttr getSetAttr(llvm::ArrayRef<mlir::Attribute> value = {})
    {
        return Py::SetAttr::get(getContext(), value);
    }

    Py::FunctionAttr getFunctionAttr(mlir::FlatSymbolRefAttr value, mlir::Attribute qualName = {},
                                     mlir::Attribute defaults = {}, mlir::Attribute kwDefaults = {},
                                     mlir::Attribute dict = {})
    {
        return Py::FunctionAttr::get(context, value, qualName, defaults, kwDefaults, dict);
    }

    Py::UnknownType getUnknownType()
    {
        return getType<Py::UnknownType>();
    }

    using mlir::OpBuilder::create;

    template <class Op>
    Op create()
    {
        return mlir::OpBuilder::create<Op>(getCurrentLoc());
    }

    template <class Op, class First, class... Args>
    std::enable_if_t<!std::is_same_v<std::decay_t<First>, mlir::Location>, Op> create(First&& first, Args&&... args)
    {
        return mlir::OpBuilder::create<Op>(getCurrentLoc(), std::forward<First>(first), std::forward<Args>(args)...);
    }

#define BUILTIN(name, str, ...)                                   \
    mlir::FlatSymbolRefAttr get##name##Builtin()                  \
    {                                                             \
        return mlir::FlatSymbolRefAttr::get(getContext(), (str)); \
    }
#include <pylir/Interfaces/Builtins.def>

#define BUILTIN(name, _1, _2, type, ...)                                                                 \
    Py::ConstantOp create##name##Ref()                                                                   \
    {                                                                                                    \
        return create<Py::ConstantOp>(getType<::pylir::Py::ClassType>(get##type##Builtin(), llvm::None), \
                                      get##name##Builtin());                                             \
    }
#include <pylir/Interfaces/Builtins.def>

    Py::ConstantOp createConstant(mlir::Attribute constant)
    {
        if (auto ref = constant.dyn_cast<mlir::FlatSymbolRefAttr>())
        {
            return create<Py::ConstantOp>(getUnknownType(), ref);
        }
        else if (auto unbound = constant.dyn_cast<Py::UnboundAttr>())
        {
            return create<Py::ConstantOp>(unbound);
        }
        else
        {
            return create<Py::ConstantOp>(constant.cast<Py::ObjectAttrInterface>());
        }
    }

    template <std::size_t n>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
    Py::ConstantOp createConstant(const char (&c)[n])
    {
        return create<Py::ConstantOp>(getStrAttr(c));
    }

    Py::ConstantOp createConstant(llvm::StringRef string)
    {
        return create<Py::ConstantOp>(getStrAttr(string));
    }

    template <class Integer>
    std::enable_if_t<std::is_integral_v<std::decay_t<Integer>> && !std::is_same_v<bool, std::decay_t<Integer>>>
        createConstant(Integer) = delete;

    Py::ConstantOp createConstant(bool value)
    {
        return create<Py::ConstantOp>(getPyBoolAttr(value));
    }

    Py::ConstantOp createConstant(BigInt bigInt)
    {
        return create<Py::ConstantOp>(getIntAttr(std::move(bigInt)));
    }

    Py::ConstantOp createConstant(double value)
    {
        return create<Py::ConstantOp>(getFloatAttr(value));
    }

    Py::TypeOfOp createTypeOf(mlir::Value value)
    {
        return create<Py::TypeOfOp>(getUnknownType(), value);
    }

    Py::GetSlotOp createGetSlot(mlir::Value object, mlir::Value typeObject, llvm::StringRef slot)
    {
        return create<Py::GetSlotOp>(getUnknownType(), object, typeObject, slot);
    }

    Py::SetSlotOp createSetSlot(mlir::Value object, mlir::Value typeObject, llvm::StringRef slot, mlir::Value value)
    {
        return create<Py::SetSlotOp>(object, typeObject, slot, value);
    }

    Py::DictTryGetItemOp createDictTryGetItem(mlir::Value dict, mlir::Value index)
    {
        return create<Py::DictTryGetItemOp>(getUnknownType(), getI1Type(), dict, index);
    }

    Py::DictSetItemOp createDictSetItem(mlir::Value dict, mlir::Value key, mlir::Value value)
    {
        return create<Py::DictSetItemOp>(dict, key, value);
    }

    Py::DictDelItemOp createDictDelItem(mlir::Value dict, mlir::Value key)
    {
        return create<Py::DictDelItemOp>(dict, key);
    }

    Py::DictLenOp createDictLen(mlir::Value dict)
    {
        return create<Py::DictLenOp>(dict);
    }

    Py::TupleGetItemOp createTupleGetItem(mlir::Value tuple, mlir::Value index)
    {
        return create<Py::TupleGetItemOp>(getUnknownType(), tuple, index);
    }

    Py::TupleLenOp createTupleLen(mlir::Value tuple)
    {
        return create<Py::TupleLenOp>(tuple);
    }

    Py::TuplePrependOp createTuplePrepend(mlir::Value element, mlir::Value tuple)
    {
        return create<Py::TuplePrependOp>(getUnknownType(), element, tuple);
    }

    Py::TuplePopFrontOp createTuplePopFront(mlir::Value tuple)
    {
        return create<Py::TuplePopFrontOp>(getUnknownType(), tuple);
    }

    Py::ListAppendOp createListAppend(mlir::Value list, mlir::Value item)
    {
        return create<Py::ListAppendOp>(list, item);
    }

    Py::ListLenOp createListLen(mlir::Value list)
    {
        return create<Py::ListLenOp>(list);
    }

    Py::ListToTupleOp createListToTuple(mlir::Value list)
    {
        return create<Py::ListToTupleOp>(getUnknownType(), list);
    }

    Py::FunctionGetFunctionOp createFunctionGetFunction(mlir::Type functionType, mlir::Value function)
    {
        return create<Py::FunctionGetFunctionOp>(functionType, function);
    }

    Py::FunctionGetFunctionOp createFunctionGetFunction(mlir::Value function)
    {
        return create<Py::FunctionGetFunctionOp>(getUniversalCCType(context), function);
    }

    Py::ObjectHashOp createObjectHash(mlir::Value object)
    {
        return create<Py::ObjectHashOp>(object);
    }

    Py::ObjectIdOp createObjectId(mlir::Value object)
    {
        return create<Py::ObjectIdOp>(object);
    }

    Py::TypeMROOp createTypeMRO(mlir::Value typeObject)
    {
        return create<Py::TypeMROOp>(getUnknownType(), typeObject);
    }

    Py::StrCopyOp createStrCopy(mlir::Value string, mlir::Value typeObject)
    {
        return create<Py::StrCopyOp>(getUnknownType(), string, typeObject);
    }

    Py::StrHashOp createStrHash(mlir::Value string)
    {
        return create<Py::StrHashOp>(string);
    }

    Py::StrEqualOp createStrEqual(mlir::Value lhs, mlir::Value rhs)
    {
        return create<Py::StrEqualOp>(lhs, rhs);
    }

    Py::StrConcatOp createStrConcat(llvm::ArrayRef<mlir::Value> strings)
    {
        return create<Py::StrConcatOp>(strings);
    }

    Py::MROLookupOp createMROLookup(mlir::Value mroTuple, llvm::StringRef attribute)
    {
        return create<Py::MROLookupOp>(getUnknownType(), getI1Type(), mroTuple, attribute);
    }

    Py::LinearContainsOp createLinearContains(mlir::Value mroTuple, mlir::Value element)
    {
        return create<Py::LinearContainsOp>(mroTuple, element);
    }

    Py::MakeTupleOp createMakeTuple(llvm::ArrayRef<Py::IterArg> args = {})
    {
        return create<Py::MakeTupleOp>(getUnknownType(), args);
    }

    Py::MakeTupleOp createMakeTuple(llvm::ArrayRef<mlir::Value> args, mlir::ArrayAttr iterExpansion)
    {
        if (!iterExpansion)
        {
            iterExpansion = getI32ArrayAttr({});
        }
        return create<Py::MakeTupleOp>(getUnknownType(), args, iterExpansion);
    }

    Py::MakeTupleExOp createMakeTupleEx(llvm::ArrayRef<Py::IterArg> args, mlir::Block* happyPath,
                                        mlir::Block* unwindPath, llvm::ArrayRef<mlir::Value> normalOps = {},
                                        llvm::ArrayRef<mlir::Value> unwindOps = {})
    {
        return create<Py::MakeTupleExOp>(getUnknownType(), args, happyPath, normalOps, unwindPath, unwindOps);
    }

    Py::MakeTupleExOp createMakeTupleEx(llvm::ArrayRef<Py::IterArg> args, mlir::Block* unwindPath)
    {
        auto* happyPath = new mlir::Block;
        auto op = create<Py::MakeTupleExOp>(getUnknownType(), args, happyPath, mlir::ValueRange{}, unwindPath,
                                            mlir::ValueRange{});
        implementBlock(happyPath);
        return op;
    }

    Py::MakeListOp createMakeList(llvm::ArrayRef<Py::IterArg> args = {})
    {
        return create<Py::MakeListOp>(args);
    }

    Py::MakeListOp createMakeList(llvm::ArrayRef<mlir::Value> args, mlir::ArrayAttr iterExpansion)
    {
        if (!iterExpansion)
        {
            iterExpansion = getI32ArrayAttr({});
        }
        return create<Py::MakeListOp>(args, iterExpansion);
    }

    Py::MakeListExOp createMakeListEx(llvm::ArrayRef<Py::IterArg> args, mlir::Block* happyPath, mlir::Block* unwindPath,
                                      llvm::ArrayRef<mlir::Value> normalOps = {},
                                      llvm::ArrayRef<mlir::Value> unwindOps = {})
    {
        return create<Py::MakeListExOp>(args, happyPath, normalOps, unwindPath, unwindOps);
    }

    Py::MakeListExOp createMakeListEx(llvm::ArrayRef<Py::IterArg> args, mlir::Block* unwindPath)
    {
        auto* happyPath = new mlir::Block;
        auto op = create<Py::MakeListExOp>(args, happyPath, mlir::ValueRange{}, unwindPath, mlir::ValueRange{});
        implementBlock(happyPath);
        return op;
    }

    Py::MakeSetOp createMakeSet(llvm::ArrayRef<Py::IterArg> args = {})
    {
        return create<Py::MakeSetOp>(args);
    }

    Py::MakeSetOp createMakeSet(llvm::ArrayRef<mlir::Value> args, mlir::ArrayAttr iterExpansion)
    {
        if (!iterExpansion)
        {
            iterExpansion = getI32ArrayAttr({});
        }
        return create<Py::MakeSetOp>(args, iterExpansion);
    }

    Py::MakeSetExOp createMakeSetEx(llvm::ArrayRef<Py::IterArg> args, mlir::Block* happyPath, mlir::Block* unwindPath,
                                    llvm::ArrayRef<mlir::Value> normalOps = {},
                                    llvm::ArrayRef<mlir::Value> unwindOps = {})
    {
        return create<Py::MakeSetExOp>(args, happyPath, normalOps, unwindPath, unwindOps);
    }

    Py::MakeSetExOp createMakeSetEx(llvm::ArrayRef<Py::IterArg> args, mlir::Block* unwindPath)
    {
        auto* happyPath = new mlir::Block;
        auto op = create<Py::MakeSetExOp>(args, happyPath, mlir::ValueRange{}, unwindPath, mlir::ValueRange{});
        implementBlock(happyPath);
        return op;
    }

    Py::MakeDictOp createMakeDict(llvm::ArrayRef<Py::DictArg> args = {})
    {
        return create<Py::MakeDictOp>(args);
    }

    Py::MakeDictOp createMakeDict(llvm::ArrayRef<mlir::Value> keys, llvm::ArrayRef<mlir::Value> values,
                                  mlir::ArrayAttr mappingExpansion = {})
    {
        if (!mappingExpansion)
        {
            mappingExpansion = getI32ArrayAttr({});
        }
        return create<Py::MakeDictOp>(keys, values, mappingExpansion);
    }

    Py::MakeDictExOp createMakeDictEx(llvm::ArrayRef<Py::DictArg> args, mlir::Block* happyPath, mlir::Block* unwindPath,
                                      llvm::ArrayRef<mlir::Value> normalOps = {},
                                      llvm::ArrayRef<mlir::Value> unwindOps = {})
    {
        return create<Py::MakeDictExOp>(args, happyPath, normalOps, unwindPath, unwindOps);
    }

    Py::MakeDictExOp createMakeDictEx(llvm::ArrayRef<Py::DictArg> args, mlir::Block* unwindPath)
    {
        auto* happyPath = new mlir::Block;
        auto op = create<Py::MakeDictExOp>(args, happyPath, mlir::ValueRange{}, unwindPath, mlir::ValueRange{});
        implementBlock(happyPath);
        return op;
    }

    Py::MakeFuncOp createMakeFunc(mlir::FlatSymbolRefAttr function)
    {
        return create<Py::MakeFuncOp>(function);
    }

    Py::MakeObjectOp createMakeObject(mlir::Value typeObject)
    {
        return create<Py::MakeObjectOp>(getUnknownType(), typeObject);
    }

    Py::IsOp createIs(mlir::Value lhs, mlir::Value rhs)
    {
        return create<Py::IsOp>(lhs, rhs);
    }

    Py::BoolToI1Op createBoolToI1(mlir::Value input)
    {
        return create<Py::BoolToI1Op>(input);
    }

    Py::BoolFromI1Op createBoolFromI1(mlir::Value input)
    {
        return create<Py::BoolFromI1Op>(input);
    }

    Py::IntFromIntegerOp createIntFromInteger(mlir::Value integer)
    {
        return create<Py::IntFromIntegerOp>(integer);
    }

    Py::IntToIntegerOp createIntToInteger(mlir::Type integerLike, mlir::Value object)
    {
        return create<Py::IntToIntegerOp>(integerLike, getI1Type(), object);
    }

    Py::IntCmpOp createIntCmp(Py::IntCmpKind kind, mlir::Value lhs, mlir::Value rhs)
    {
        return create<Py::IntCmpOp>(kind, lhs, rhs);
    }

    Py::IntAddOp createIntAdd(mlir::Value lhs, mlir::Value rhs)
    {
        return create<Py::IntAddOp>(lhs, rhs);
    }

    Py::IntToStrOp createIntToStr(mlir::Value object)
    {
        return create<Py::IntToStrOp>(object);
    }

    Py::YieldOp createYield(mlir::ValueRange values)
    {
        return create<Py::YieldOp>(values);
    }

    Py::TypeSwitchOp createTypeSwitch(mlir::TypeRange results, mlir::Value typeObject)
    {
        return create<Py::TypeSwitchOp>(results, typeObject, mlir::ValueRange{}, 0);
    }

    Py::TypeSwitchExOp createTypeSwitchEx(mlir::TypeRange results, mlir::Value typeObject, mlir::Block* happyPath,
                                          llvm::ArrayRef<mlir::Value> normalOperands, mlir::Block* unwindPath,
                                          llvm::ArrayRef<mlir::Value> unwindOperands)
    {
        return create<Py::TypeSwitchExOp>(results, typeObject, mlir::ValueRange{}, normalOperands, unwindOperands,
                                          happyPath, unwindPath, 0);
    }

    Py::GlobalValueOp createGlobalValue(llvm::StringRef symbolName, bool constant = false,
                                        Py::ObjectAttrInterface initializer = {}, bool external = false)
    {
        return create<Py::GlobalValueOp>(symbolName, external ? mlir::StringAttr{} : getStringAttr("private"), constant,
                                         initializer);
    }

    Py::GlobalHandleOp createGlobalHandle(llvm::StringRef symbolName)
    {
        return create<Py::GlobalHandleOp>(symbolName, getStringAttr("private"));
    }

    Py::StoreOp createStore(mlir::Value value, mlir::FlatSymbolRefAttr handle)
    {
        return create<Py::StoreOp>(value, handle);
    }

    Py::LoadOp createLoad(mlir::FlatSymbolRefAttr handle)
    {
        return create<Py::LoadOp>(getUnknownType(), handle);
    }

    Py::IsUnboundValueOp createIsUnboundValue(mlir::Value value)
    {
        return create<Py::IsUnboundValueOp>(value);
    }

    Py::RaiseOp createRaise(mlir::Value exception)
    {
        return create<Py::RaiseOp>(exception);
    }

    Py::InvokeOp createInvoke(llvm::ArrayRef<mlir::Type> resultTypes, mlir::FlatSymbolRefAttr callee,
                              llvm::ArrayRef<mlir::Value> operands, mlir::Block* happyPath,
                              llvm::ArrayRef<mlir::Value> normalOperands, mlir::Block* unwindPath,
                              llvm::ArrayRef<mlir::Value> unwindOperands)
    {
        return create<Py::InvokeOp>(resultTypes, callee, operands, normalOperands, unwindOperands, happyPath,
                                    unwindPath);
    }

    Py::InvokeIndirectOp createInvokeIndirect(mlir::Value callee, llvm::ArrayRef<mlir::Value> operands,
                                              mlir::Block* happyPath, llvm::ArrayRef<mlir::Value> normalOperands,
                                              mlir::Block* unwindPath, llvm::ArrayRef<mlir::Value> unwindOperands)
    {
        return create<Py::InvokeIndirectOp>(callee, operands, normalOperands, unwindOperands, happyPath, unwindPath);
    }
};
} // namespace pylir::Py
