# RUN: split-file %s %t
# RUN: pylir %t/normal.py -fsyntax-only -verify
# RUN: pylir %t/regex.py -fsyntax-only -verify
# RUN: not pylir %t/invalid-regex.py -fsyntax-only -verify 2>&1 | FileCheck %t/invalid-regex.py
# RUN: not pylir %t/not-closed-regex.py -fsyntax-only -verify 2>&1 | FileCheck %t/not-closed-regex.py
# RUN: not pylir %t/not-a-regex.py -fsyntax-only -verify 2>&1 | FileCheck %t/not-a-regex.py
# RUN: pylir %t/above.py -fsyntax-only -verify
# RUN: pylir %t/rel_plus.py -fsyntax-only -verify
# RUN: pylir %t/rel_minus.py -fsyntax-only -verify
# RUN: pylir %t/same.py -fsyntax-only -verify
# RUN: not pylir %t/not-a-warning.py -fsyntax-only -verify 2>&1 | FileCheck %t/not-a-warning.py

#--- normal.py

try:
    pass
# expected-error@below {{except clause without expression must come last}}
except:
    pass
except int:
    pass

#--- regex.py

try:
    pass
# expected-error-re@below {{except clause without {{expression|type}} must come last}}
except:
    pass
except int:
    pass

#--- invalid-regex.py

try:
    pass
# expected-error-re@below {{except clause without {{*}} must come last}}
except:
    pass
except int:
    pass

# CHECK: error: invalid regex '*'

#--- not-closed-regex.py

try:
    pass
# expected-error-re@below {{except clause without {{expression must come last}}
except:
    pass
except int:
    pass

# CHECK: error: found start of regex with no end '}}'

#--- not-a-regex.py

try:
    pass
# expected-error@below {{except clause without {{expression|type}} must come last}}
except:
    pass
except int:
    pass

# CHECK: error: Did not expect diagnostic:
# CHECK-NEXT: except clause without expression must come last
# CHECK: error: Did not encounter error at line [[# @LINE - 7]]:

#--- above.py

try:
    pass
except:
    # expected-error@above {{except clause without expression must come last}}
    pass
except int:
    pass

#--- rel_plus.py

try:
    pass
    # expected-error@+1 {{except clause without expression must come last}}
except:
    pass
except int:
    pass

#--- rel_minus.py

try:
    pass
except:
    # expected-error@-1 {{except clause without expression must come last}}
    pass
except int:
    pass

#--- same.py

try:
    pass
except: # expected-error {{except clause without expression must come last}}
    pass
except int:
    pass

#--- not-a-warning.py

try:
    pass
except: # expected-warning {{except clause without expression must come last}}
    pass
except int:
    pass

# CHECK: error: Did not expect diagnostic:
# CHECK-NEXT: except clause without expression must come last
# CHECK: error: Did not encounter warning at line [[# @LINE - 7]]:
