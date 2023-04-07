// RUN: pylir-opt %s -split-input-file -verify-diagnostics

pyHIR.globalFunc @ret_mismatch1() {
    %0 = func "foo"(%ff0 "rest") -> !py.dynamic {
        return %ff0
    }
    // expected-error@below {{expected no return value within function with no return type}}
    return %0
}

// -----

pyHIR.globalFunc @ret_mismatch2() -> !py.dynamic {
    // expected-error@below {{expected return value within function with return type}}
    return
}

// -----

// expected-error@below {{only one positional rest parameter allowed}}
pyHIR.globalFunc @two_pos_rest(*%arg0, *%arg1) {
    return
}

// -----

// expected-error@below {{only one keyword rest parameter allowed}}
pyHIR.globalFunc @two_keyword_rest(**%arg0, **%arg1) {
    return
}
