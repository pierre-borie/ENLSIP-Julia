# ENLSIP Documentation

# Notations

The transpose of a given matrix `M` is written `M'`.

It corresponds to `Adjoint` operation in Julia, equivalent to transposition when using matrices with floats. 

## Structures

### Structures to contain informations

```@autodocs
Modules = [Enlsip]
Order   = [:type]
Filter = t -> typeof(t) === DataType && !(t <: Enlsip.EvalFunc)
```

### Evaluation functions structures

```@autodocs
Modules = [Enlsip]
Order   = [:type]
Filter = t -> typeof(t) === DataType && t <: Enlsip.EvalFunc
```


## Functions

```@autodocs
Modules = [Enlsip]
Order   = [:function]
```
