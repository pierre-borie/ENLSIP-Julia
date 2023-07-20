module Enlsip

# Packages

using LinearAlgebra, Polynomials
using Formatting, Printf

# include source files
for f in ["enlsip_functions", "structures", "solver"]
    include("./$f.jl")
end

end # module