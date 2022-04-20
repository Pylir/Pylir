// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_TABLEGEN_UTIL
#define PYLIR_TABLEGEN_UTIL

include "mlir/IR/AttrTypeBase.td"

class ParamSpec<AttrOrTypeParameter m_type, string m_name, string m_defaultArg = ""> {
	AttrOrTypeParameter type = m_type;
	string name = m_name;
	string defaultArg = m_defaultArg;
}

class ParamSpecBuilder<list<ParamSpec> spec, code extraCode = ""> : AttrBuilder<!dag(ins,
											!foreach(x, spec, CArg<x.type.cppType, !if(x.type.isOptional, "{}", x.defaultArg)>),
															!foreach(x, spec, x.name)),
	!interleave(!foreach(x, !filter(x, spec, x.type.isOptional),
					x.name # " = " # x.name # " ? " # x.name # " : " # !subst("$_ctx", "$_ctxt", x.type.defaultValue) # ";\n"),
							"") # extraCode #
	"return Base::get($_ctxt, " # !interleave(!foreach(x, spec, x.name), ", ") # ");"
>;

class UseableAttrInterface<string name> : AttrInterface<name>,
	Attr<CPred<"$_self.isa<" # cppNamespace # "::" # name # ">()">, name # " instance">
{
	let storageType = cppNamespace # "::" # cppClassName;
    let returnType = storageType;
    let convertFromStorage = "$_self";
}

class DefaultType<string name, string description = ""> : DefaultValuedParameter<"::mlir::FlatSymbolRefAttr",
	"::mlir::FlatSymbolRefAttr::get($_ctx, ::pylir::Py::Builtins::" # name # ".name)", description>;

class CustomPrint<AttrOrTypeParameter x, string elementParse = ?, string elementPrint = ?>
	: AttrOrTypeParameter<x.cppType, x.summary, x.cppAccessorType>
{
	let allocator = x.allocator;
	let comparator = x.comparator;
	let cppStorageType = x.cppStorageType;
	let syntax = x.syntax;
	let isOptional = x.isOptional;
	let defaultValue = x.defaultValue;

	let parser = elementParse;
	let printer = elementPrint;
}

class ListPrint<string type, string delim, string elementParse = "", string elementPrint = ""> : ArrayRefParameter<type>
{
	let parser = [{
		[&]() -> mlir::FailureOr<llvm::SmallVector<}] # type # [{>> {
			llvm::SmallVector<}] # type # [{> array;
			if ($_parser.parseCommaSeparatedList(::mlir::AsmParser::Delimiter::}] # delim # [{, [&]() -> mlir::ParseResult {
					auto temp = }] # !if(!empty(elementParse), "::mlir::FieldParser<" # type # ">::parse($_parser)", elementParse) # [{;
					if (mlir::succeeded(temp))
					{
						array.push_back(std::move(*temp));
					}
					return static_cast<mlir::LogicalResult>(temp);
				}))
			{
				return ::mlir::failure();
			}
			return array;
		}()
	}];

	let printer = [{
		$_printer << '}] # !cond(!eq(delim, "Paren") : "(", !eq(delim, "Square") : "[", !eq(delim, "Braces") : "{") # [{';
		llvm::interleaveComma($_self, $_printer, [&](auto&& x){ }] # !if(!empty(elementPrint), "$_printer << x;", !subst("$_self", "x", elementPrint)) # [{; });
		$_printer << '}] # !cond(!eq(delim, "Paren") : ")", !eq(delim, "Square") : "]", !eq(delim, "Braces") : "}") # [{';
	}];
}

#endif
