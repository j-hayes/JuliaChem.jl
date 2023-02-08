
struct JCTiming
    iteration_times :: Dict{String, Float64}
end

function create_jctiming()
    return JCTiming(Dict{String, Float64}())
end

function print_iteration_times(timings::JCTiming)
    #print integer keys first, then the rest
    timings_deep_copy = deepcopy(timings)
    sorted_pairs = sort(collect(pairs(timings_deep_copy.iteration_times)), by= x->get_sort_index_from_key(x[1]))
    for key_value_pair in sorted_pairs
        if isnothing(tryparse(Int, key_value_pair[1]))
            continue
        end
        println("$(key_value_pair[1]): $(key_value_pair[2]) seconds")
        delete!(timings_deep_copy.iteration_times, key_value_pair[1])    
    end
    sorted_pairs = sort(collect(pairs(timings_deep_copy.iteration_times)), by= x->x[1])
    for key_value_pair in sorted_pairs
        println("$(key_value_pair[1]): $(key_value_pair[2]) seconds")
    end 
end

function get_sort_index_from_key(key::String)
   parsed_key = tryparse(Int, key)
   if isnothing(parsed_key)
     return -1
   else
     return parsed_key
   end
end

export JCTiming, create_jctiming, print_iteration_times