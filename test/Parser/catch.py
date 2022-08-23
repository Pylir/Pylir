# RUN: pylir %s -fsyntax-only -verify

try:
    pass
# expected-error@below {{except clause without expression must come last}}
except:
    pass
except int:
    pass
