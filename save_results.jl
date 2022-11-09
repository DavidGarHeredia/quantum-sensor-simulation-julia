using HDF5
using TensorCast


function save_results(measures; filename="output.h5")
    # Convert measures into a 3-D array
    @cast measures[c, m, q] := measures[c][m][q] lazy=false
    # Create a directory to store the files with the results
    data_dir = "data"
    if !isdir(data_dir)
        mkpath(data_dir)
    end
    # Write the data
    h5open(joinpath(data_dir, filename), "w") do fid
        group = create_group(fid, "data")
        write(create_dataset(group, "measures", Int64, size(measures)), measures)
    end
end
