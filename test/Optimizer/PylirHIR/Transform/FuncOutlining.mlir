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
  // CHECK: init_return %[[FUNC]]
  %0 = func "basic"(%ff0 "rest") {
    return %ff0
  }
  init_return %0
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

// CHECK: globalFunc @[[$BASIC]](%{{.*}} "rest") {
// CHECK: globalFunc @[[$POS_DEFAULT_INNER]](%{{.*}} has_default, %{{.*}} only "lol" has_default) {

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
  init_return %0
}

// CHECK: globalFunc @[[$BASIC1]](
// CHECK: globalFunc @[[$BASIC2]](
