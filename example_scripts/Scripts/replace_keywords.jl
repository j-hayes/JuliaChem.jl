function replace_keywordsV1(dict, replacement_dict)
    for (key, value) in replacement_dict
        dict[key] = value
    end
end

function replace_model(model_dict, replacement_dict)
    replace_keywordsV1(model_dict, replacement_dict)
end

function replace_scf_keywords(keywords_dict, replacement_dict)
    replace_keywordsV1(keywords_dict["scf"], replacement_dict)
end