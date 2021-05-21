using Distributions

function quadrupolar_starting_value(sites::Int64)
    starting_values = zeros(5 * sites - 1)
    starting_values[1:5:end] = rand(Uniform(0, 9), sites)  # √Qcc
    starting_values[2:5:end] = rand(Uniform(0, 1), sites)  # √σQcc
    starting_values[3:5:end] = rand(Uniform(0, 1), sites)  # √η
    starting_values[4:5:end] = rand(Uniform(0, 1), sites)  # √ση
    if sites > 1
        starting_values[5:5:end] = rand(Uniform(0, 1 / sites), sites - 1)
    end
    return starting_values
end

function quadrupolar_lb(sites::Int64)
    starting_values = zeros(5 * sites - 1)
    return starting_values
end

function quadrupolar_ub(sites::Int64)
    starting_values = zeros(5 * sites - 1)
    starting_values[1:5:end] .= 9  # √Qcc
    starting_values[2:5:end] .= 1  # √σQcc
    starting_values[3:5:end] .= 1  # √η
    starting_values[4:5:end] .= 1  # √ση
    if sites > 1
        starting_values[5:5:end] .= 1
    end
    return starting_values
end

function quadrupolar_tol(sites::Int64)
    tol = zeros(5 * sites)
    tol[1:5:end] .= 0.5
    tol[2:5:end] .= 0.1
    tol[3:5:end] .= 0.1
    tol[4:5:end] .= 0.1
    tol[5:5:end] .= 0.05
    return tol
end

function CSA_starting_value(sites::Int64)
    starting_values = zeros(7 * sites - 1)
    starting_values[1:7:end] = rand(Uniform(-4000, 4000), sites)
    starting_values[2:7:end] = rand(Uniform(0, 800), sites)
    starting_values[3:7:end] = rand(Uniform(-4000, 4000), sites)
    starting_values[4:7:end] = rand(Uniform(0, 400), sites)
    starting_values[5:7:end] = rand(Uniform(0, 1), sites)
    starting_values[6:7:end] = rand(Uniform(0, 1), sites)
    starting_values[7:7:end] = map(x -> 1 / sites, 1:sites-1)
    return starting_values
end

function CSA_lb(sites::Int64)
    starting_values = zeros(7 * sites - 1)
    starting_values[1:7:end] .= -4000
    starting_values[2:7:end] .= 0.000001
    starting_values[3:7:end] .= -4000
    starting_values[4:7:end] .= 0.000001
    starting_values[5:7:end] .= 0.000001
    starting_values[6:7:end] .= 0.000001
    starting_values[7:7:end] .= 0.0
    return starting_values
end

function CSA_ub(sites::Int64)
    starting_values = zeros(7 * sites - 1)
    starting_values[1:7:end] .= 4000.0
    starting_values[2:7:end] .= 800.0
    starting_values[3:7:end] .= 4000
    starting_values[4:7:end] .= 400.0
    starting_values[5:7:end] .= 1.0
    starting_values[6:7:end] .= 1.0
    starting_values[7:7:end] .= 1.0
    return starting_values
end

function CSA_tol(sites::Int64)
    tol = zeros(7 * sites)
    tol[1:7:end] .= 100.0
    tol[2:7:end] .= 20.0
    tol[3:7:end] .= 30.0
    tol[4:7:end] .= 5.0
    tol[5:7:end] .= 0.1
    tol[6:7:end] .= 0.1
    tol[7:7:end] .= 0.05
    return tol
end
