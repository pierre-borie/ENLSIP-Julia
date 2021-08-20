# Enlsip Documentation


## Functions

```@autodocs
Modules = [Enlsip]
Order   = [:function]
```

## Evaluation functions structures

```@autodocs
Modules = [Enlsip]
Order   = [:type]
Filter = t -> typeof(t) === DataType && t <: Enlsip.EvalFunc
```