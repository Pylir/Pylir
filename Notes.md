# Getting & Setting Attributes

## Getter:

* Lookup `__getattribute__` in type
  - Goes through MRO
  - If found -> execute and return
* Lookup attribute in type
  - Goes through MRO
  - If found && has `__get__` && `__set__` methods -> execute `__get__` and return
* Lookup in `__dict__` if available
* if type lookup result was successful && had `__get__` but not `__set__` -> execute `__get__` and return
* if type lookup result was successful -> return it
* Lookup `__getattr__` in type
  - Goes through MRO
  - If found -> execute and return Else:
    raise AttributeError

Worst case unoptimized lookup count: len(MRO) (`__getattribute__`) + 1 (`__dict__`) + len(MRO) + (`__getattr__`)

Lookup for `__getattribute__` and `__getattr__` could potentially be combined making it len(MRO) + 1.

Would improve the worst case but potentially worsen the best case

Setter:

* Lookup attribute in type
  - Goes through MRO
  - If found && has `__set__` methods -> execute `__set__` and return
* Set in `__dict__` if available Else:
  raise AttributeError

# Method lookup

* Lookup in type
  - Goes through MRO
  - if found && has `__get__` -> execute and return
  - if found return result Else:
    - raise AttributeError or TypeError

# Attempting a call

```python
def call(callable, *args, **kwd):
  while callable is not function and hasattr(callable, '__call__'):
    callable = callable.__call__
  if callable is not function:
    raise TypeError
  callable(*args, **kwd)
```
