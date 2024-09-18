using SparseArrays

"""
Convert a Vector{UInt8} buffer to a UInt number, considedring the last index `last_index`
"""
function get_uint_from_char_array(char_buffer::Vector{UInt8}, last_index::Int)
    uint_value::UInt = UInt(0)
    for symbol_index in 1:last_index
        number_value::UInt8 = UInt(char_buffer[symbol_index] - UInt8('0'))
        uint_value += number_value*10^(last_index - symbol_index)
    end
    return uint_value
end

"""
Receive a string and generate a list of UInt numbers in the `uint_vector_buffer`, using a char buffer `char_buffer`
"""
function get_numbers_from_line_string!(string_var::String, uint_vector_buffer::Vector{UInt}, char_buffer::Vector{UInt8})
    char_count = 0
    string_count = 0
    for char_var in string_var
        if char_var == ' '
            if (char_count > 0)
                string_count += 1
                uint_vector_buffer[string_count] = get_uint_from_char_array(char_buffer, char_count)
            end
            char_count = 0
        else
            char_count += 1
            char_buffer[char_count] = char_var
        end
    end
    return string_count
end

"""
Read a rail**** file and return a set cover sparse matrix, set cost vector, number of nodes, and number of sets.

# Inputs
- `file_name::string` : Path to the file

# Returns
- `cover_sparse_matrix::SparseMatrixCSC{Bool}` : Matrix where eath line contais the nodes index covered by the set
- `cost_vector::Vector{UInt}` : Cost to use the set
- `num_nodes::{UInt}`
- `num_sets::{UInt}`
"""
function gen_sparse_matrix_and_cost_vector_from_file(file_name::String)::Tuple{SparseMatrixCSC{Bool}, Vector{UInt}, UInt, UInt}
    
    file = open(file_name, "r")

    set_id::Int = 1

    first_line::String = readline(file)
    splited_string::Vector{String} = split(first_line, " ", keepempty = false)
    num_nodes::UInt = parse(UInt, splited_string[1])
    num_sets::UInt = parse(UInt, splited_string[2])

    sets_index_list::Vector{UInt} = []
    nodes_index_list::Vector{UInt} = []
    cost_vector::Vector{UInt} = Vector{UInt}(undef, num_sets)

    splited_values = Vector{UInt}(undef, 50)
    char_buffer = Vector{UInt8}(undef, 50)

    # Begining on the second line
    for line in eachline(file)
        
        number_of_elements = get_numbers_from_line_string!(line, splited_values, char_buffer)
        
        cost = splited_values[1]
        num_covered_nodes = splited_values[2]

        @assert num_covered_nodes == (number_of_elements - 2)

        for list_index in 3:(num_covered_nodes + 2)
            push!(sets_index_list, set_id)
            push!(nodes_index_list, splited_values[list_index])
        end
        cost_vector[set_id] = cost

        set_id += 1
    end
    close(file)

    return (sparse(nodes_index_list, sets_index_list, true, num_nodes, num_sets), cost_vector, num_nodes, num_sets)
end

