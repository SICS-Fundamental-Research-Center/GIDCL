using PClean

include("load_data.jl")

PClean.@model SoccerModel begin 
    @class Team begin
        teamname ~ StringPrior(3, 50, possibilities[:team])
        city ~ StringPrior(3, 50, possibilities[:city])
        stadium ~ StringPrior(3, 50, possibilities[:stadium])
        manager ~ ChooseUniformly(possibilities[:manager])
    end;
    @class Player begin
        name ~ StringPrior(3, 50, possibilities[:name])
        surname ~ StringPrior(3, 50, possibilities[:surname])
        birthplace ~ StringPrior(3, 50, possibilities[:birthplace])
        birthyear ~ ChooseUniformly(possibilities[:birthyear])
    end;
    @class Record begin
        pos ~ ChooseUniformly(possibilities[:position])
        season ~ ChooseUniformly(possibilities[:season])
        player ~ Player
        teams ~ Team
        pos_obs     ~ AddTypos(pos);         season_obs ~ AddTypos(season)
        teamname_obs     ~ AddTypos(teams.teamname);         city_obs ~ AddTypos(teams.city)
        stadium_obs     ~ AddTypos(teams.stadium);         manager_obs ~ AddTypos(teams.manager)
        name_obs     ~ AddTypos(player.name);         surname_obs ~ AddTypos(player.surname)
        birthyear_obs     ~ AddTypos(player.birthyear);         birthplace_obs ~ AddTypos(player.birthplace)
    end;
end;

query = @query SoccerModel.Record [
    position         pos                    pos_obs
    season           season                 season_obs
    team             teams.teamname         teamname_obs
    city             teams.city             city_obs
    stadium          teams.stadium          stadium_obs
    manager          teams.manager          manager_obs
    name             player.name            name_obs
    surname          player.surname         surname_obs
    birthyear        player.birthyear       birthyear_obs
    birthplace       player.birthplace      birthplace_obs
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
