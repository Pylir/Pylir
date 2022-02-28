// RUN: pylir-opt %s --lower-type-switch --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.object = #py.type
py.globalValue @builtins.bool = #py.type

func @type_switch(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant @builtins.type
    %1 = py.constant @builtins.bool
    %2 = py.typeSwitch %arg0 {
        %3 = py.typeOf %arg0
        py.yield %3 : !py.dynamic
    } case %0 {
        py.yield %0 : !py.dynamic
    } case %1 {
        py.yield %0 : !py.dynamic
    } : !py.dynamic
    return %2 : !py.dynamic
}

// CHECK-LABEL: func @type_switch
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[TYPE:.*]] = py.constant @builtins.type
// CHECK-NEXT: %[[BOOL:.*]] = py.constant @builtins.bool
// CHECK-NEXT: %[[IS:.*]] = py.is %[[ARG0]], %[[TYPE]]
// CHECK-NEXT: cf.cond_br %[[IS]], ^[[TYPE_BLOCK:.*]], ^[[CONTINUE:[[:alnum:]]+]]
// CHECK-NEXT: ^[[TYPE_BLOCK]]:
// CHECK-NEXT: cf.br ^[[END:[[:alnum:]]+]]
// CHECK-SAME: %[[TYPE]]
// CHECK-NEXT: ^[[CONTINUE]]:
// CHECK-NEXT: %[[IS:.*]] = py.is %[[ARG0]], %[[BOOL]]
// CHECK-NEXT: cf.cond_br %[[IS]], ^[[BOOL_BLOCK:.*]], ^[[CONTINUE:[[:alnum:]]+]]
// CHECK-NEXT: ^[[BOOL_BLOCK]]:
// CHECK-NEXT: cf.br ^[[END]]
// CHECK-SAME: %[[TYPE]]
// CHECK-NEXT: ^[[CONTINUE]]:
// CHECK-NEXT: cf.br ^[[GENERIC:[[:alnum:]]+]]
// CHECK-NEXT: ^[[GENERIC]]:
// CHECK-NEXT: %[[TYPE_OF:.*]] = py.typeOf %[[ARG0]]
// CHECK-NEXT: cf.br ^[[END]]
// CHECK-SAME: %[[TYPE_OF]]
// CHECK-NEXT: ^[[END]]
// CHECK-SAME: %[[RESULT:[[:alnum:]]+]]
// CHECK-NEXT: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.object = #py.type
py.globalValue @builtins.bool = #py.type
py.globalValue @builtins.BaseException = #py.type

func @foo(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant #py.unbound
    return %0 : !py.dynamic
}

func @type_switch(%trueValue : !py.dynamic) -> !py.dynamic {
    %nothing = py.typeSwitchEx %trueValue {
        %value = call @foo(%trueValue) : (!py.dynamic) -> !py.dynamic
        py.yield %value : !py.dynamic
    } : !py.dynamic
    label ^success unwind ^failure

^success:
    return %trueValue : !py.dynamic

^failure:
    %0 = py.landingPad @builtins.BaseException
    py.landingPad.br ^handler

^handler:
    return %0 : !py.dynamic
}

// CHECK-LABEL: func @type_switch
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: cf.br ^[[GENERIC:[[:alnum:]]+]]
// CHECK-NEXT: ^[[GENERIC]]:
// CHECK-NEXT: %[[INVOKE:.*]] = py.invoke @foo(%[[ARG0]])
// CHECK-NEXT: label ^[[HAPPY:.*]] unwind ^[[FAILURE:[[:alnum:]]+]]
// CHECK-NEXT: ^[[HAPPY]]:
// CHECK-NEXT: cf.br ^[[MERGE:[[:alnum:]]+]]
// CHECK-SAME: %[[INVOKE]]
// CHECK-NEXT: ^[[MERGE]]
// CHECK-SAME: %[[RESULT:[[:alnum:]]+]]
// CHECK-NEXT: cf.br ^[[END:[[:alnum:]]+]]
// CHECK-NEXT: ^[[END]]:
// CHECK-NEXT: return %[[ARG0]]
// CHECK-NEXT: ^[[FAILURE]]:
// CHECK-NEXT: %[[EXCEPTION:.*]] = py.landingPad @builtins.BaseException
// CHECK-NEXT: py.landingPad.br ^[[HANDLER:[[:alnum:]]+]]
// CHECK-NEXT: ^[[HANDLER]]:
// CHECK-NEXT: return %[[EXCEPTION]]
