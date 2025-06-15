using CSV
using DataFrames: DataFrame

dataset = "beers"
dirty_table = CSV.File("datasets/$(dataset)_dirty.csv") |> DataFrame
clean_table = CSV.File("datasets/$(dataset)_clean.csv") |> DataFrame

# In the dirty data, CSV.jl infers that PhoneNumber, ZipCode, and ProviderNumber
# are strings. Our PClean script also models these columns as string-valued.
# However, in the clean CSV file (without typos) it infers they are
# numbers. To facilitate comparison of PClean's results (strings) with 
# ground-truth, we preprocess the clean values to convert them into strings.
clean_table[!, :id] = map(x -> "$x", clean_table[!, :id])
clean_table[!, :ounces] = map(x -> "$x", clean_table[!, :ounces])
clean_table[!, :abv] = map(x -> "$x", clean_table[!, :abv])
clean_table[!, :ibu] = map(x -> "$x", clean_table[!, :ibu])
clean_table[!, :brewery_id] = map(x -> "$x", clean_table[!, :brewery_id])

dirty_table[!, :id] = map(x -> "$x", dirty_table[!, :id])
dirty_table[!, :ounces] = map(x -> "$x", dirty_table[!, :ounces])
dirty_table[!, :abv] = map(x -> "$x", dirty_table[!, :abv])
dirty_table[!, :ibu] = map(x -> "$x", dirty_table[!, :ibu])
dirty_table[!, :brewery_id] = map(x -> "$x", dirty_table[!, :brewery_id])

# Stores sets of unique observed values of each attribute.
possibilities = Dict(col => remove_missing(unique(collect(dirty_table[!, col])))
                     for col in propertynames(dirty_table))
