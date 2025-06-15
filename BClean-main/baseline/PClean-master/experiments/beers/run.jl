using PClean

include("load_data.jl")

PClean.@model BeersModel begin 
    @class Country begin
        @learned state_proportions::ProportionsParameter
        state ~ ChooseProportionally(possibilities[:state], state_proportions)
        city ~ StringPrior(3, 30, possibilities[:city])
    end;
    @class Brewery begin
        brewery_id ~ ChooseUniformly(possibilities[:brewery_id])
        brewery_name ~ ChooseUniformly(possibilities[:brewery_name])
        country ~ Country
    end;
    @class Beer begin
        id ~ StringPrior(3, 30, possibilities[:id])
        beers_name ~ ChooseUniformly(possibilities[:beer_name])
        style ~ ChooseUniformly(possibilities[:style])
        ounces ~ StringPrior(3, 4, possibilities[:ounces])
        abv ~ StringPrior(3, 6, possibilities[:abv])
        ibu ~ StringPrior(1, 3, possibilities[:ibu])
    end;
    @class Record begin
        begin
            beers       ~ Beer;                            
            ids     ~ AddTypos(beers.id);
            beers_name  ~ AddTypos(beers.beers_name);        
            ouncess ~ AddTypos(beers.ounces);
            abv        ~ AddTypos(beers.abv);              
            ibu    ~ AddTypos(beers.ibu);
            styles      ~ AddTypos(beers.style);
        end
        begin
            brewerys ~ Brewery
            brewery_ids ~ AddTypos(brewerys.brewery_id);
            brewery_names ~ AddTypos(brewerys.brewery_name);
            states ~ AddTypos(brewerys.country.state);
            citys ~ AddTypos(brewerys.country.city);
        end
    end;
end;

query = @query BeersModel.Record [
    id               beers.id                ids
    beer_name        beers.beers_name       beers_name
    style            beers.style             styles
    ounces           beers.ounces            ouncess
    abv              beers.abv               abv
    ibu              beers.ibu               ibu
    brewery_id       brewerys.brewery_id     brewery_ids
    brewery_name     brewerys.brewery_name   brewery_names
    state            brewerys.country.state  states
    city             brewerys.country.city   citys
];

config = PClean.InferenceConfig(1, 2; use_mh_instead_of_pg=true);
observations = [ObservedDataset(query, dirty_table)];
@time begin 
    trace = initialize_trace(observations, config);
    run_inference!(trace, config);
end

results = evaluate_accuracy(dirty_table, clean_table, trace.tables[:Record], query)
# PClean.save_results("F://code//Julia-demo//PClean-master//PClean-master//results", "hospital", trace, observations)
println(results)
