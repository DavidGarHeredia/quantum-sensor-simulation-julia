using YAML
using ArgParse

include("simulate.jl")
include("save_results.jl")


function parse_commandline()
    s = ArgParseSettings()
    # TODO: add --output_file
    @add_arg_table! s begin
        "config_file"
            help = "path to YAML configuration file with simulation options"
            required = true
    end

    return parse_args(s)
end


function main()
    args = parse_commandline()
    config = YAML.load_file(args["config_file"], dicttype=Dict{Symbol,Any})
    # for nqubits, entanglement,
    config[:θ][:low] = deg2rad(config[:θ][:low])
    config[:θ][:high] = deg2rad(config[:θ][:high])
    measures = simulate(;config...)
    # save results
    save_results(measures)
end


main()