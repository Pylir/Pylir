# Interfaces

There are two kinds of interfaces:

* Generic Interfaces
* Dialect Interfaces

A dialect interface may either directly depend on the dialect or simply
conceptually belong in the domain of the dialect.
The distinction is mostly a matter of code organization, rather than a semantic
difference.

Dialect interfaces are found in the `Interfaces` directory of the dialects'
directory, while generic interfaces are found in the top level `Interfaces`
directory of the optimizer.

```{toctree}
:maxdepth: 3
:caption: Interfaces

GenericInterfaces
PylirHIRInterfaces
PylirPyInterfaces
```
