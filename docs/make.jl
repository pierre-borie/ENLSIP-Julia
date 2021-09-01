push!(LOAD_PATH,"../src/")

using Documenter, Enlsip

makedocs(sitename="Enlsip.jl")

deploydocs(repo="github.com/pierre-borie/ENLSIP-Julia")
