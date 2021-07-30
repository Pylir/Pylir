
#ifndef PYLIR_ATTRIBUTES_TABLEGEN
#define PYLIR_ATTRIBUTES_TABLEGEN

include "pylir/Optimizer/Dialect/PylirBase.td"

class Pylir_Attr<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute"> :
    AttrDef<Pylir_Dialect, name, traits, baseCppClass>;

def Pylir_IntegerAttr : Pylir_Attr<"Integer"> {
    let mnemonic = "integer";

    let parameters = (ins "llvm::APInt":$value);

    let printer = [{
        $_printer << getMnemonic() << '<' << getValue() << '>';
    }];

    let parser = [{
        llvm::APInt integer;
        if ($_parser.parseLess() || $_parser.parseInteger(integer) || $_parser.parseGreater())
        {
            return {};
        }
        return get($_ctxt, integer);
    }];
}

def Pylir_SetAttr : Pylir_Attr<"Set", [DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let mnemonic = "set";

    let parameters = (ins ArrayRefParameter<"mlir::Attribute", "">:$value);

    let printer = [{
        $_printer << getMnemonic() << '{';
        llvm::interleaveComma(getValue(), $_printer);
        $_printer << '}';
    }];

    let parser = [{
        if ($_parser.parseLBrace())
        {
            return {};
        }
        llvm::DenseSet<mlir::Attribute> attributes;
        {
            mlir::Attribute attribute;
            if ($_parser.parseAttribute(attribute))
            {
                return {};
            }
            attributes.insert(attribute);
        }
        while (!$_parser.parseOptionalComma())
        {
            mlir::Attribute attribute;
            if ($_parser.parseAttribute(attribute))
            {
                return {};
            }
            attributes.insert(attribute);
        }
        if ($_parser.parseRBrace())
        {
            return {};
        }
        return get($_ctxt, attributes);
    }];

    let skipDefaultBuilders = 1;

    let builders = [
        AttrBuilder<(ins "llvm::DenseSet<mlir::Attribute>":$value),[{
            return Base::get($_ctxt, std::vector<mlir::Attribute>{value.begin(), value.end()});
        }]>
    ];
}


def Pylir_DictAttr : Pylir_Attr<"Dict", [DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let mnemonic = "dict";

    let parameters = (ins ArrayRefParameter<"std::pair<mlir::Attribute, mlir::Attribute>", "">:$value);

    let printer = [{
        $_printer << getMnemonic() << '{';
        llvm::interleaveComma(getValue(), $_printer, [&](const auto& pair) { $_printer << pair.first << ':' << pair.second; });
        $_printer << '}';
    }];

    let parser = [{
        if ($_parser.parseLBrace())
        {
            return {};
        }
        llvm::DenseMap<mlir::Attribute, mlir::Attribute> attributes;
        std::vector<std::pair<mlir::Attribute,mlir::Attribute>> values;
        {
            mlir::Attribute key, value;
            if ($_parser.parseAttribute(key) || $_parser.parseColon() || $_parser.parseAttribute(value))
            {
                return {};
            }
            if(attributes.insert({key,value}).second)
            {
                values.emplace_back(key,value);
            }
        }
        while (!$_parser.parseOptionalComma())
        {
            mlir::Attribute key, value;
            if ($_parser.parseAttribute(key) || $_parser.parseColon() || $_parser.parseAttribute(value))
            {
                return {};
            }
            if(attributes.insert({key,value}).second)
            {
                values.emplace_back(key,value);
            }
        }
        if ($_parser.parseRBrace())
        {
            return {};
        }
        return get($_ctxt, values);
    }];

    let skipDefaultBuilders = 1;

    let builders = [
        AttrBuilder<(ins "llvm::ArrayRef<std::pair<mlir::Attribute,mlir::Attribute>>":$value),[{
            llvm::DenseMap<mlir::Attribute, mlir::Attribute> map;
            std::vector<std::pair<mlir::Attribute, mlir::Attribute>> values;
            for (auto& iter : value)
            {
                if (map.insert(iter).second)
                {
                    values.push_back(iter);
                }
            }
            return Base::get($_ctxt, values);
        }]>
    ];

    let extraClassDeclaration = [{
        static DictAttr getAlreadySorted(mlir::MLIRContext* context,
                                         llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value);
    }];
}


#endif
