using CSV
using DataFrames: DataFrame

dataset = "Soccer"
dirty_table = CSV.File("datasets/$(dataset)_dirty.csv") |> DataFrame
clean_table = CSV.File("datasets/$(dataset)_clean.csv") |> DataFrame

# In the dirty data, CSV.jl infers that PhoneNumber, ZipCode, and ProviderNumber
# are strings. Our PClean script also models these columns as string-valued.
# However, in the clean CSV file (without typos) it infers they are
# numbers. To facilitate comparison of PClean's results (strings) with 
# ground-truth, we preprocess the clean values to convert them into strings.
clean_table[!, :birthyear] = map(x -> "$x", clean_table[!, :birthyear])
clean_table[!, :season] = map(x -> "$x", clean_table[!, :season])

# Stores sets of unique observed values of each attribute.
possibilities = Dict(col => remove_missing(unique(collect(dirty_table[!, col])))
                     for col in propertynames(dirty_table))
