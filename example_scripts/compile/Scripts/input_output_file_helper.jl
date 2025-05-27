function get_file_pathsV1(inputs_dir::String, extension::String) :: Tuple{Array{String,1}, Array{String,1}}
    #execute ls -Sr command to get the list of files in the directory
    file_names = read(`ls $(inputs_dir) -Sr`, String) #file names in order ascending size
    #split the file names by new line
    file_names = split(file_names, "\n")
    #filter out empty strings
    file_names = filter(x -> length(x) > 0 && endswith(x, extension), file_names)
    #join the directory path with the file names
    input_file_paths = joinpath.(inputs_dir, file_names)
    #filter out files that don't end in .json 
    return input_file_paths, file_names
end

function get_file_pathsV1(inputs_dir::String) :: Tuple{Array{String,1}, Array{String,1}}
    return get_file_pathsV1(inputs_dir, ".json")
end

function create_output_folderV1(output_dir::String, input_file_path)
    output_dir = joinpath(output_dir, split(basename(input_file_path), ".")[1])
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    return output_dir
end

function create_output_folderV2(output_dir::String, input_file_path ::String, rank ::Int)
    output_dir = joinpath(output_dir, split(basename(input_file_path), ".")[1])
    if rank != 0 
        return output_dir
    end
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    return output_dir
end

function get_output_folders(path_to_outputs)
     #files at path 
    #all folders at path_to_outputs
    run_folders = readdir(path_to_outputs)
    #all folders full path 
    full_folder_paths = [joinpath(path_to_outputs, folder) for folder in run_folders]
    #remove if not a directory
    full_folder_paths = [folder for folder in full_folder_paths if isdir(folder)]
    return  run_folders, full_folder_paths, full_folder_paths
end