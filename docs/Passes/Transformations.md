# Transformation Passes

Transformation passes within Pylir may either be "generic", as in, not depending
on any upstream {doc}`dialects <../Dialects/index>`, or specific to a dialect.
This is also reflected in the directory structure, with generic passes living in
the top level `Transforms` directory of the optimizer, while the specific passes
are found in the `Transforms` directory of the dialects' directory.

## Generic Transformation Passes

```{include} ../TableGen/Passes/PylirTransform.md
```

## 'py' Transformation Passes

```{include} ../TableGen/Passes/PylirPyTransform.md
```

## 'pyMem' Transformation Passes

```{include} ../TableGen/Passes/PylirMemTransforms.md
```

## 'pyHIR' Transformation Passes

```{include} ../TableGen/Passes/PylirHIRTransform.md
```
