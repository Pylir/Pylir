
# Rationales

This document is a random document in which I am documenting design choices for the future, so that if I ever reconsider
any choices I can go over the circumstances and arguments that lead to specific decisions.

## Alloca, Load and Store

Early on the issue of how to handle variables arose. Initially I was going to emit an already pure SSA form from the 
frontend by taking care of control flow merge points and forming variant types from the incoming values. An advantage
with this approach I realised early on as well, is that trivial constant folding could occur which would automatically
also lead to type deduction. 

The reason I am currently going with alloca, load and store instead though are:
* Merging control flow points is a pain, doesn't matter whether through block arguments or structured control flow Ops,
    but ESPECIALLY using structured control flow ops. One has to modify and insert ops in the predecessors to cast to 
    the variant type. The variant type itself is only known once all predecessors values have been generated
* "Casting" types is a lot easier through alloca. The alloca produces a handle type that can't be used in any ops but 
    store and load. Store ops allow storing ANY kind of Python type, and loading allows to load any kind of type. The 
    type erasure/casting to Pylir_UnknownType is therefore basically built in
* I am currently thinking that using a mem2reg style algorithm might be easier to implement. Type deduction should then
    work through folding as well as canonicalization patterns
