using LinearAlgebra
using SparseArrays
using DataStructures


mutable struct SolverData
    const MAX_ITER::UInt
    const NUMBER_OF_VARIABLES::UInt
    const NUMBER_OF_INEQ_CONSTRAINTS::UInt
    const TIME_LIMIT::UInt
    const SILENT::Bool
    last_iter::UInt
    termination_status::String
    termination_time::Float64
    cost_per_iter::Vector{Float64}
    LB_per_iter::Vector{Float64}
    LR_per_iter::Vector{Float64}
    UB_per_iter::Vector{Float64}
    function SolverData(max_iter::UInt, 
                        number_of_variables::UInt, 
                        number_of_ineq_contraints::UInt,
                        time_limit::UInt,
                        silent::Bool
                        )
        MAX_ITER = max_iter
        NUMBER_OF_VARIABLES = number_of_variables
        NUMBER_OF_INEQ_CONSTRAINTS = number_of_ineq_contraints
        TIME_LIMIT = time_limit
        SILENT = silent
        cost_per_iter = Vector{Float64}(undef, max_iter)
        LB_per_iter = Vector{Float64}(undef, max_iter)
        LR_per_iter::Vector{Float64} = Vector{Float64}(undef, max_iter)
        UB_per_iter::Vector{Float64} = Vector{Float64}(undef, max_iter)
        last_iter = UInt(0)
        termination_status = "None"
        termination_time = 0
        new(MAX_ITER, NUMBER_OF_VARIABLES, NUMBER_OF_INEQ_CONSTRAINTS, TIME_LIMIT, SILENT,
        last_iter, termination_status, termination_time, cost_per_iter, LB_per_iter, LR_per_iter, UB_per_iter)
    end
end

function is_valid(solution_variable::Vector{Bool}, A::SparseMatrixCSC{Bool, UInt}, b::Vector{Float64}, number_of_constraints::UInt)
    is_valid_result = true

    for constraint_index in 1:number_of_constraints
        is_valid_result = A[constraint_index,:]'solution_variable > b[constraint_index]
        if !is_valid_result
            break
        end
    end
    return is_valid_result
end

# SUBGRAD

function find_optimal_solution_of_LR!(lagrangian_cost::Vector{Float64}, solution_variable::Vector{Bool}, number_of_variables::UInt)
    for var_index in 1:number_of_variables
        if lagrangian_cost[var_index] > 0
            solution_variable[var_index] = 0
        else
            solution_variable[var_index] = 1
        end
    end
end

function update_lagrangian_multiplier!(lagrange_multipliers::Vector{Float64}, sub_grad::Vector{Float64}, step_size::Float64, ϵ::Float64 = 0.2)
    for i in 1:size(lagrange_multipliers)[1]
        lagrange_multipliers[i] = max(0.0, lagrange_multipliers[i] + (1+ϵ)*step_size*sub_grad[i])
    end
end

function update_lagrangian_cost!(lagrangian_cost::Vector{Float64}, lagrange_multipliers::Vector{Float64}, c::Vector{Float64}, Aᵀ::SparseMatrixCSC{Bool, UInt})
    SparseArrays.mul!(lagrangian_cost, Aᵀ, lagrange_multipliers)
    for i in 1:size(lagrangian_cost)[1]
        lagrangian_cost[i] = -lagrangian_cost[i] + c[i]
    end
end

function update_subgradient!(subgradient::Vector{Float64}, x::Vector{Bool}, A::SparseMatrixCSC{Bool, UInt}, b::Vector{Float64})
    SparseArrays.mul!(subgradient, A, x)
    for i in 1:size(subgradient)[1]
        subgradient[i] = -subgradient[i] + b[i]
    end
end


# SET COVERING GREEDY 

function get_nz_row_index_from_column!(A::SparseMatrixCSC{Bool, UInt}, column::UInt, row_vec_buffer::Vector{UInt})::UInt
    buff_index::UInt = UInt(0)
    for row_index in nzrange(A, column)
        buff_index += 1
        row_vec_buffer[buff_index] = rowvals(A)[row_index]
    end
    return buff_index
end

function get_nz_column_index_from_line!(Aᵀ::SparseMatrixCSC{Bool, UInt}, row::UInt, column_vec_buffer::Vector{UInt})::UInt
    buff_index::UInt = UInt(0)
    for column_index in nzrange(Aᵀ, row) # colum of A
        buff_index += 1
        column_vec_buffer[buff_index] = rowvals(Aᵀ)[column_index]
    end
    return buff_index
end


"""
Update the number of nodes that a set covers, but hasn't beem cover by another selected set.
"""
function update_n_nodes_not_covered!(n_nodes_not_covered::Vector{UInt},
    row_vec_buffer::Vector{UInt},
    column_vec_buffer::Vector{UInt},
    last_row_buff_index::UInt,
    nodes_covered::Vector{Bool},
    changed_sets::Vector{UInt},
    set_changed::Vector{Bool},
    Aᵀ::SparseMatrixCSC{Bool, UInt})::UInt

    number_of_sets_changed::UInt = 0
    set_changed .= 0
    
    for buffer_line_index in 1:last_row_buff_index
        node_covered_index = row_vec_buffer[buffer_line_index]
        # Not covered in other iteration
        if nodes_covered[node_covered_index] == 0 
            # Get all column indexes at line index of row_vec_buffer
            last_column_buffer_idx = get_nz_column_index_from_line!(Aᵀ, node_covered_index, column_vec_buffer)
            for buffer_col_index in 1:last_column_buffer_idx
                set_index = column_vec_buffer[buffer_col_index]
                n_nodes_not_covered[set_index] -= 1
                # Not count a previous changed set
                if set_changed[set_index] == 0
                    number_of_sets_changed += 1
                    changed_sets[number_of_sets_changed] = set_index
                    set_changed[set_index] = 1
                end
            end
            nodes_covered[node_covered_index] = 1
        end
    end
    return number_of_sets_changed
end

"""
Updates the sets position in the heap based on changed_sets vector.
"""
function update_greedy_cost_heap!(greedy_cost_heap::MutableBinaryMinHeap{Float64},
    n_nodes_not_covered::Vector{UInt64},
    changed_sets::Vector{UInt64},
    number_of_sets_changed::UInt,
    c::Vector{Float64})

    for changed_set_index in 1:number_of_sets_changed
        set_index = changed_sets[changed_set_index]
        if n_nodes_not_covered[set_index] ≤ 0 
            greedy_cost_heap[Int(set_index)] = Inf
        elseif greedy_cost_heap[Int(set_index)] != Inf
            greedy_cost_heap[Int(set_index)] = c[set_index]/n_nodes_not_covered[set_index]
        end
    end
end

function greedy_algorithm_for_SCP!(solution_var::Vector{Bool}, 
                           solution_var_LR::Vector{Bool},
                           n_nodes_not_covered::Vector{UInt64},
                           changed_sets::Vector{UInt64},
                           set_changed::Vector{Bool},
                           nodes_covered::Vector{Bool},
                           greedy_cost_heap::MutableBinaryMinHeap{Float64},
                           column_vec_buffer::Vector{UInt},
                           row_vec_buffer::Vector{UInt},
                           c::Vector{Float64}, A::SparseMatrixCSC{Bool, UInt}, Aᵀ::SparseMatrixCSC{Bool, UInt},
                           number_of_variables::UInt)::Float64
    
    # Seting values

    not_covered::Bool = true
    nom_LR_variables_used = false
    cost::Float64 = 0
    number_of_sets_changed::UInt = UInt(0)
    nodes_covered .= 0

    #copy!(n_nodes_not_covered, sum(A, dims=1)[1,:])
    sum!(n_nodes_not_covered, Aᵀ)

    # Set nom LR variables = Inf
    
    for i in 1:number_of_variables
        greedy_cost_heap[Int(i)] = solution_var_LR[i] == 1 ? c[i]/n_nodes_not_covered[i] : Inf
    end
    
    while not_covered 
        
        mean_cost, selected_element = top_with_handle(greedy_cost_heap)
        last_row_buff_index = get_nz_row_index_from_column!(A, UInt(selected_element), row_vec_buffer)
        
        if mean_cost < Inf
            solution_var[selected_element] = 1
            cost += c[selected_element]
        elseif !nom_LR_variables_used
            
            # Set nom LR variables != Inf
            for i in 1:Int(number_of_variables)
                if solution_var_LR[i] == 0 && n_nodes_not_covered[i] > 0
                    greedy_cost_heap[i] = c[i]/n_nodes_not_covered[i]
                end
            end
            nom_LR_variables_used = true
        else
            not_covered = false
        end
        
        
        number_of_sets_changed = update_n_nodes_not_covered!(n_nodes_not_covered,
        row_vec_buffer,
        column_vec_buffer,
        last_row_buff_index,
        nodes_covered,
        changed_sets,
        set_changed, Aᵀ)

        
        update_greedy_cost_heap!(greedy_cost_heap,
        n_nodes_not_covered,
        changed_sets,
        number_of_sets_changed,
        c)
        
    end
    return cost
end


# MAIN


function binary_problem_lagrangian_solver(c::Vector{Float64}, A::SparseMatrixCSC{Bool, UInt}, b::Vector{Float64}, max_iter::UInt = UInt(500),
    time_limit::UInt = UInt(3600),
    ϵ::Float64 = 0.2,
    δ_init::Float64 = 2.0,
    count_trigger::UInt = UInt(500),
    silent::Bool = false)::SolverData

    initial_time::Float64 = time()
    number_of_ineq_constraints::UInt = A.m # Number of lines
    number_of_variables::UInt = A.n # Number of colums

    solver_data::SolverData = SolverData(max_iter, number_of_variables, number_of_ineq_constraints, time_limit, silent)
    
    # Lagrangian Relaxation data
    lagrange_multipliers::Vector{Float64} = zeros(Float64, number_of_ineq_constraints)
    lagrangian_cost::Vector{Float64} = Vector{Float64}(undef, number_of_variables)
    subgradient::Vector{Float64} = Vector{Float64}(undef, number_of_ineq_constraints)
    solution_variable_LR::Vector{Bool} = zeros(Bool, number_of_variables)
    
    # SCP problem data    
    solution_variable::Vector{Bool} = zeros(Bool, number_of_variables)
    upper_bound_solution::Vector{Bool} = Vector{Bool}(undef, number_of_variables)
    upper_bound_lagrange_multiplier::Vector{Float64} = Vector{Float64}(undef, number_of_ineq_constraints)
    upper_bound_cost::Float64 = Inf
    lower_bound_cost::Float64 = 0
    cost::Float64 = 0
    
    # Greedy Algorithm data
    n_nodes_not_covered::Vector{UInt} = Vector{UInt}(undef, number_of_variables)
    nodes_covered::Vector{Bool} = Vector{Bool}(undef, number_of_ineq_constraints)
    greedy_cost_heap::MutableBinaryMinHeap{Float64} = MutableBinaryMinHeap(lagrangian_cost)
    
    changed_sets::Vector{UInt} = zeros(UInt, number_of_variables)
    set_changed::Vector{Bool} = zeros(Bool, number_of_variables)

    column_vec_buffer::Vector{UInt} = Vector{UInt}(undef, number_of_variables)
    row_vec_buffer::Vector{UInt} = Vector{UInt}(undef, number_of_ineq_constraints)

    Aᵀ::SparseMatrixCSC{Bool, UInt} = SparseArrays.transpose(A)
    
    # Other definitions
    iteration_counter::UInt = 0
    not_solved::Bool = true
    δ::Float64 = δ_init
    last_print_time::Float64 = 0.0

    while not_solved && iteration_counter < max_iter
        
        iteration_counter += 1
        
        update_lagrangian_cost!(lagrangian_cost, lagrange_multipliers, c, Aᵀ)

        find_optimal_solution_of_LR!(lagrangian_cost, solution_variable_LR, number_of_variables)

        cost_of_LR = lagrangian_cost'solution_variable_LR + sum(lagrange_multipliers)

        update_subgradient!(subgradient, solution_variable_LR, A, b)

        if cost_of_LR > lower_bound_cost
            lower_bound_cost = cost_of_LR
        end

        cost = greedy_algorithm_for_SCP!(solution_variable, 
        solution_variable_LR,
        n_nodes_not_covered,
        changed_sets,
        set_changed,
        nodes_covered,
        greedy_cost_heap,
        column_vec_buffer,
        row_vec_buffer,
        c, A, Aᵀ, number_of_variables)

        if cost < upper_bound_cost
            upper_bound_cost = cost
            copy!(upper_bound_solution, solution_variable)
            copy!(upper_bound_lagrange_multiplier, lagrange_multipliers)
        end

        if (upper_bound_cost - lower_bound_cost) < 1
            not_solved = false
            solver_data.termination_status = "Optimal"
        end
        
        step_size = δ*(upper_bound_cost - cost_of_LR)/sum(abs2, subgradient)

        update_lagrangian_multiplier!(lagrange_multipliers, subgradient, step_size, ϵ)

        if iteration_counter%count_trigger == count_trigger - 1
            δ = δ/2
        end

        if time() - initial_time > last_print_time + 1 && !silent || iteration_counter == 1
            last_print_time = time() - initial_time
            print("t: $(Int(round(last_print_time)))s, it: $(iteration_counter), cost: $(cost), LB: $(lower_bound_cost), UB: $(upper_bound_cost), δ: $(δ), step_size: $(step_size), Gap:$((upper_bound_cost-lower_bound_cost)/lower_bound_cost)\n")
        end

        solver_data.LB_per_iter[iteration_counter] = lower_bound_cost
        solver_data.UB_per_iter[iteration_counter] = upper_bound_cost
        solver_data.cost_per_iter[iteration_counter] = cost
        solver_data.LR_per_iter[iteration_counter] = cost_of_LR
    end
    solver_data.last_iter = iteration_counter
    return solver_data
end