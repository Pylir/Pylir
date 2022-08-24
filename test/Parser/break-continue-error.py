# RUN: pylir %s -fsyntax-only -verify

# expected-error@below {{occurrence of 'break' outside of loop}}
break

# expected-error@below {{occurrence of 'continue' outside of loop}}
continue

while True:
    def foo():
        # expected-error@below {{occurrence of 'break' outside of loop}}
        break

while True:
    def foo():
        # expected-error@below {{occurrence of 'continue' outside of loop}}
        continue

while True:
    class Foo:
        # expected-error@below {{occurrence of 'break' outside of loop}}
        break

while True:
    class Foo:
        # expected-error@below {{occurrence of 'continue' outside of loop}}
        continue

while True:
    pass
else:
    # expected-error@below {{occurrence of 'break' outside of loop}}
    break

while True:
    pass
else:
    # expected-error@below {{occurrence of 'continue' outside of loop}}
    continue
