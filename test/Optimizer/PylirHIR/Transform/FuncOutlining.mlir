// RUN: pylir-opt %s --pylir-func-outlining --split-input-file | FileCheck %s

// CHECK: #[[$NONE:.*]] = #py.globalValue<builtins.None>

// CHECK-LABEL: init "__main__"
pyHIR.init "__main__" {
  // CHECK: %[[FUNC:.*]] = py.makeFunc @[[$BASIC:.*]]
  // CHECK: %[[STR:.*]] = py.constant(#py.str<"basic">)
  // CHECK: py.setSlot %[[FUNC]][%{{.*}}] to %[[STR]]
  // CHECK: %[[NONE1:.*]] = py.constant(#[[$NONE]])
  // CHECK: %[[NONE2:.*]] = py.constant(#[[$NONE]])
  // CHECK: py.setSlot %[[FUNC]][%{{.*}}] to %[[NONE1]]
  // CHECK: py.setSlot %[[FUNC]][%{{.*}}] to %[[NONE2]]
  // CHECK: test.use(%[[FUNC]])
  %0 = func "basic"(%ff0 "rest") {
    return %ff0
  }
  test.use(%0) : !py.dynamic
  init_return
}

// CHECK-LABEL: globalFunc @posDefault
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
pyHIR.globalFunc @posDefault(%arg0) {
  // CHECK: %[[FUNC:.*]] = py.makeFunc @[[$POS_DEFAULT_INNER:.*]]
  // CHECK: %[[STR:.*]] = py.constant(#py.str<"posDefaultInner">)
  // CHECK: py.setSlot %[[FUNC]][%{{.*}}] to %[[STR]]
  // CHECK: %[[KW:.*]] = py.constant(#py.str<"lol">)
  // CHECK: %[[HASH:.*]] = py.str_hash %[[KW]]
  // CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[ARG0]])
  // CHECK: %[[DICT:.*]] = py.makeDict (%[[KW]] hash(%[[HASH]]) : %[[ARG0]])
  // CHECK: py.setSlot %[[FUNC]][%{{.*}}] to %[[TUPLE]]
  // CHECK: py.setSlot %[[FUNC]][%{{.*}}] to %[[DICT]]
  // CHECK: return %[[FUNC]]
  %0 = func "posDefaultInner"(%ff0 = %arg0, %ff1 only "lol" = %arg0) {
    return %ff0
  }
  return %0
}

// CHECK: globalFunc @[[$BASIC]](%{{[[:alnum:]]+}}, %{{[[:alnum:]]+}} "rest") {
// CHECK: globalFunc @[[$POS_DEFAULT_INNER]](%{{[[:alnum:]]+}}, %{{[[:alnum:]]+}} has_default, %{{[[:alnum:]]+}} only "lol" has_default) {

// -----

// CHECK-LABEL: init "func_collision"
pyHIR.init "func_collision" {
  // CHECK: py.makeFunc @[[$BASIC1:.*]]
  // CHECK: py.makeFunc @[[$BASIC2:.*]]
  %0 = func "basic"(%ff0 "rest") {
    return %ff0
  }
  %1 = func "basic"(%ff0 "rest") {
    return %ff0
  }
  init_return
}

// CHECK: globalFunc @[[$BASIC1]](
// CHECK: globalFunc @[[$BASIC2]](

// -----

// CHECK-LABEL: init "arg_res_attrs"
pyHIR.init "arg_res_attrs" {
  // CHECK: py.makeFunc @[[$BASIC1:.*]]
  %0 = func "basic"(%ff0 "rest" { test.name = 0 : i32 }) -> {test.name = 1 : i32} {
    return %ff0
  }
  init_return
}

// CHECK: globalFunc @[[$BASIC1]](
// CHECK-SAME: "rest" {test.name = 0 : i32}
// CHECK-SAME: -> {test.name = 1 : i32}

// -----

// CHECK-LABEL: init "non_locals"
pyHIR.init "non_locals" {
  // CHECK: %[[LIST:.*]] = py.makeList ()
  %0 = py.makeList ()
  // CHECK: py.makeFunc @[[$BASIC1:[[:alnum:]]+]][%[[LIST]] : !py.dynamic]
  %1 = func "basic"() {
    return %0
  }
  init_return
}

// CHECK: globalFunc @[[$BASIC1]](
// CHECK-SAME: %[[CLOSURE:[[:alnum:]]+]]
// CHECK: %[[ARG:.*]] = py.function_closureArg %[[CLOSURE]][0] : [!py.dynamic]
// CHECK: return %[[ARG]]

// -----


// CHECK-LABEL: init "constants"
pyHIR.init "constants" {
  %0 = py.constant (#py.str<"text">)
  // CHECK: py.makeFunc @[[BASIC1:[[:alnum:]]+]]
  // CHECK-NOT: [
  // CHECK: {{$}}
  %1 = func "basic"() {
    return %0
  }
  init_return
}

// CHECK: globalFunc @[[BASIC1]](
// CHECK-NEXT: %[[ARG:.*]] = py.constant(#py.str
// CHECK-NEXT: return %[[ARG]]
