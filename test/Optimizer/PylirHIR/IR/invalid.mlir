// RUN: pylir-opt %s -split-input-file -verify-diagnostics

// expected-error@below {{only one positional rest parameter allowed}}
pyHIR.globalFunc @two_pos_rest(*%arg0, *%arg1) {
    return
}

// -----

// expected-error@below {{only one keyword rest parameter allowed}}
pyHIR.globalFunc @two_keyword_rest(**%arg0, **%arg1) {
    return
}
