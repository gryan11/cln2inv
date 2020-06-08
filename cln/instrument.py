import sys, os, shutil
import subprocess
import re
from collections import defaultdict
import pandas as pd
from copy import deepcopy
import numpy as np
pd.set_option('display.max_rows', None, 'display.max_columns', None)


def line_simplify(line):   # e.g., change a line "int a; {a=1;}" to four lines "int a; \\ { \\ a=1; \\ }"
    remaining_line = line.strip()
    simplified_lines = []
    match_result = re.search('[{};]', remaining_line)
    while match_result is not None:           # may find ";{}" in a raw string, currently does not consider this
        pos, char = match_result.start(), match_result.group(0)
        if char == ';':
            simplified_lines.append(remaining_line[:pos + 1])
        else:
            assert char == '{' or char == '}'
            simplified_lines += [remaining_line[:pos], char]
        remaining_line = remaining_line[pos + 1:]
        match_result = re.search('[{};]', remaining_line)
    simplified_lines.append(remaining_line)
    empty_removed = [line.strip() + '\n' for line in simplified_lines if not line == '']
    return empty_removed

    
def instrument(path, basename, max_iter=50, record_initial_state=False):
    src = path + "/c/" + basename + ".c"
    intermediate = path + "/instrumented/" + basename + ".c.tmp"
    out = path + "/instrumented/" + basename + ".c" 
    header_file =  path + "/csv/"+basename+"_header.csv"
    while_index = 0
    free_vars = []
    sampled_vars = []
    unreferenced_sampled_vars = []
    unknown_count = 0
    precondition = []
    last_line_is_while = False
    loop_reached = False

    # this first pass will find the free variables, sampled variables and the preconditions
    # most instrumenting is done in this pass, including iteration capping
    with open (intermediate, "w") as tmpfile, open (src, "r") as f:
        tmpfile.write("#include <stdlib.h>\n")
        tmpfile.write("#include <stdio.h>\n")
        tmpfile.write("#include <assert.h>\n")
        tmpfile.write("#include <time.h>\n")
        for full_line in f:
            if full_line.strip().startswith('//'):
                tmpfile.write(full_line.strip() + '\n')
                continue
            simplified_lines = line_simplify(full_line)
            for line in simplified_lines:
                if line.find('unknown()') >= 0:
                    unknown_count += 1
                    line = line.replace('unknown()', 'rand()%2 < unknown_' + str(unknown_count))
                if line.startswith('while'):
                    while_index += 1
                    loop_reached = True
                    loop_condition = line[5:].strip()
                    tmpfile.write("int while_counter_" + str(while_index) + " = 0;\n")
                    tmpfile.write('while (while_counter_' + str(while_index) + ' < ' + str(max_iter) + ')\n')
                    tmpfile.write('{\n')
                    last_line_is_while = True
                    tmpfile.write('if (!' + loop_condition + ') break;\n')
                    continue

                if last_line_is_while:
                    assert line == '{\n'
                    last_line_is_while = False
                    continue

                if line.find('main') >= 0:    # function declaration, hacking with the dataset
                    tmpfile.write('int main(int argc, char** argv)\n')
                    continue

                if line.startswith('assume'):
                    line = line.replace('assume', 'assert')

                tmpfile.write(line)
                line = line.strip()

                # find and remove unreferenced variables
                unreferenced_sampled_vars_old = unreferenced_sampled_vars.copy()
                for var in unreferenced_sampled_vars_old:
                    if re.search('(\W|^)' + var + '\W', line) is not None:
                        unreferenced_sampled_vars.remove(var)

                if not loop_reached:
                    if line.startswith('int'):
                        new_int_strs = line[4:-1].split(',')
                        for new_int_str in new_int_strs:
                            if new_int_str.find('=') >= 0:  # initialized var
                                new_int = new_int_str.split('=')[0]
                                sampled_vars.append(new_int.strip())
                                unreferenced_sampled_vars.append(new_int.strip())
                            else:
                                free_vars.append(new_int_str.strip())
                                sampled_vars.append(new_int_str.strip())
                                unreferenced_sampled_vars.append(new_int_str.strip())
                    elif line.startswith('assert'):
                        print(line)
                        precondition.append(re.search('\(.*\);$', line).group(0)[1:-2].strip('()'))
                    elif re.search('[^<>]=', line) is not None:  # single assignment initialization
                        var_name = line.strip('()').split('=')[0].strip()
                        assert var_name in free_vars
                        free_vars.remove(var_name)

    for var in unreferenced_sampled_vars:
        sampled_vars.remove(var)
        if var in free_vars:
            free_vars.remove(var)
    assert while_index == 1  # no nested loop

    # this second pass add the command line argument reading, and print statement
    with open(intermediate, "r") as tmpfile, open(out, "w") as outfile, open(header_file, "w") as header_fd:
        free_var_index = 1
        loop_reached, last_line_is_while, last_line_is_main = False, False, False
        for line in tmpfile:
            # read initial values of free variables from command line
            if not loop_reached and line.startswith('int') and line.find('main') < 0:
                line = line.strip()
                new_line_exprs = []
                new_int_strs = line[4:-1].split(',')
                for new_int_str in new_int_strs:
                    new_int_str = new_int_str.strip()
                    if new_int_str.find('=') < 0 and new_int_str in free_vars:
                        new_line_exprs.append(new_int_str + '=atoi(argv[' + str(free_var_index) + '])')
                        free_var_index += 1
                    else:
                        new_line_exprs.append(new_int_str)
                outfile.write('int ' + ', '.join(new_line_exprs) + ';\n')
                continue

            elif line.startswith('while') and record_initial_state:
                # record the initial values of all sampled variables
                initial_vars = []
                for var in sampled_vars:
                    outfile.write('int ' + var + '0 = ' + var + ';\n')
                    initial_vars.append(var + '0')
                sampled_vars += initial_vars

            outfile.write(line)

            if line.startswith('while'):
                loop_reached, last_line_is_while = True, True
            elif last_line_is_while:
                print_list = [str(while_index), "while_counter_" + str(while_index) + "++", "1"] + sampled_vars
                format_str = ["%d ", "%d ", "%d "] + ["%d " for _ in range(len(sampled_vars))]
                print_stmt = "printf(\"{} \\n\", {});\n".format(", ".join(format_str), ", ".join(print_list))
                outfile.write(print_stmt)
                # write the separate header file
                header_fd.write("init,final,1," + ",".join(print_list[3:]) + "\n")
                last_line_is_while = False
            elif line.startswith('int main'):
                last_line_is_main = True
            elif last_line_is_main and unknown_count > 0:
                last_line_is_main = False
                outfile.write('srand(time(0));\n')
                for i in range(1, unknown_count + 1):
                    outfile.write('int unknown_' + str(i) + ' = atoi(argv[' + str(len(free_vars) + i) + ']);\n')
                for i in range(1, unknown_count + 1):
                    free_vars.append('unknown_' + str(i))
                    precondition += ['unknown_' + str(i) + ' >= 0', 'unknown_' + str(i) + ' <= 2']
    os.remove(intermediate)

    print('free vars:', free_vars)
    print('precondition:', precondition)
    return free_vars, precondition


def gen_initial_points(params, precondition, width, large_sample_num):
    def str_to_numeral(str):
        try:
            return float(eval(str))
        except:
            return None
    def shuffle_return(arr):
        new_arr = deepcopy(arr)
        np.random.shuffle(new_arr)
        return new_arr
    bounds = {param: {'upper': float('inf'), 'lower': float('-inf')} for param in params}
    for equation in precondition:
        match_result = re.match('^(.+)(==|>=|<=|>|<)(.+)$', equation)
        first, op, second = match_result.group(1).strip(), match_result.group(2), match_result.group(3).strip()
        if op == '>':
            op = '>='
            second = second + ' + 1'
        elif op == '<':
            op = '<='
            second = second + ' - 1'
        if first in params:  # currently assumes that variable is on the left side
            bound_num = str_to_numeral(second)
            if bound_num is not None:
                if op == '>=':
                    bounds[first]['lower'] = np.maximum(bounds[first]['lower'], bound_num)
                elif op == '<=':
                    bounds[first]['upper'] = np.minimum(bounds[first]['upper'], bound_num)
    # now we have the (optional) upper and lower bound for each variable
    values_for_each_var = defaultdict(list)

    def random_norepeat(low, high, num):
        return np.random.choice(np.arange(low, high), np.minimum(num, int(high-low)), replace=False)
    for param in params:
        upper, lower = bounds[param]['upper'], bounds[param]['lower']
        if lower != float('-inf') and upper != float('inf'):
            if upper - lower < width:
                values_for_each_var[param] = np.concatenate((np.array([lower, upper]), shuffle_return(np.arange(lower + 1, upper, 1))))
            else:
                values_for_each_var[param] = np.concatenate((np.array([lower, upper]), random_norepeat(lower+1, upper, 4 * width)))
        elif lower != float('-inf') and upper == float('inf'):
            values_for_each_var[param] = np.concatenate((np.arange(lower, lower + width, 1), random_norepeat(lower + width, lower + width * 3, width),
                                                         random_norepeat(lower + width * 3, lower + width * 10, 2 * width), random_norepeat(lower + width * 10, lower + width * 20, 2 * width)))
        elif lower == float('-inf') and upper != float('inf'):
            values_for_each_var[param] = np.concatenate((np.arange(upper - width + 1, upper + 1, 1), random_norepeat(upper - 3 * width + 1, upper - width + 1, width),
                                                         random_norepeat(upper - 10 * width + 1, upper - 3 * width + 1, 2 * width), random_norepeat(upper - 20 * width + 1, upper - 10 * width + 1, 2 * width)))
        elif lower == float('-inf') and upper == float('inf'):
            values_for_each_var[param] = np.concatenate((np.array([0, 1, -1]), shuffle_return(np.concatenate((np.arange(2, width), np.arange(-width, -1)))),
                                                         shuffle_return(np.concatenate((random_norepeat(width, 5*width, 2 * width), random_norepeat(-5*width, -width, 2 * width))))))
        else:
            assert False
    return values_for_each_var


def sample_core(initial_points, uniq, start_run_id=0):
    run_dir = 'runtemp'
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    run_id = start_run_id
    for initial_point in initial_points:
        value_list = ['{}'.format(value) for value in initial_point]
        with open(run_dir + "/" + basename + str(run_id) + ".csv", "w") as outfile:
            subprocess.run([path + "/bin/" + basename] + value_list, stdout=outfile, stderr=subprocess.PIPE)
        run_id += 1
    if len(initial_points) == 0:  # no free var for this program
        with open(run_dir + "/" + basename + '0' + ".csv", "w") as outfile:
            subprocess.run([path + "/bin/" + basename], stdout=outfile, stderr=subprocess.PIPE)
        run_id += 1

    with open(path + "/csv/" + basename + "_header.csv", 'r') as header_file:
        line = header_file.readline()
        line_splited = line.strip().split(',')
        dfhs = line_splited + ['run_id']
    run_traces = []
    for i in range(start_run_id, run_id):
        logf = run_dir + "/" + basename + str(i) + ".csv"

        with open(logf, 'r') as run_out_file:
            lines = run_out_file.readlines()
        for line in lines:
            splited_line = line.strip().split(',')
            line_list = [float(word.strip()) for word in splited_line] + [i]
            run_traces.append(line_list)
    dfs = pd.DataFrame(run_traces, columns=dfhs)

    # cleanup
    subprocess.run(['rm', run_dir + '/' + basename + '*.csv'], stderr=subprocess.DEVNULL)
    # shutil.rmtree(run_dir)

    if uniq:
        subset = list(dfs.columns)
        if 'run_id' in subset:
            subset.remove('run_id')
        dfs = dfs.drop_duplicates(subset=subset)
    dfs = dfs.reset_index(drop=True)
    dfs = dfs.drop(columns=['run_id'])  # in ICLR we don't need run_id
    num_sample = dfs.shape[0]
    return dfs, num_sample, run_id


def expand_quadratic(dfs):
    var_names = list(dfs.columns[3:])
    data_dict = {var_name: dfs[var_name].to_numpy() for var_name in var_names}
    # for idx1 in range(len(var_names)):
    #     for idx2 in range(idx1, len(var_names)):
    #         var_name_1, var_name_2 = var_names[idx1], var_names[idx2]
    #         cross_var_name = '(* ' + var_name_1 + ' ' + var_name_2 + ')'
    #         dfs[cross_var_name] = data_dict[var_name_1] * data_dict[var_name_2]
    for var in var_names:
        cross_var_name = '(* ' + var + ' ' + var + ')'
        dfs[cross_var_name] = data_dict[var] * data_dict[var]
    return dfs


def sample(path, basename, params, precondition, width=10, large_sample_num=12, uniq=True, quadratic=False):
    subprocess.run(["gcc", "-o{}".format(path + "/bin/" + basename), path + "/instrumented/" + basename + ".c",  "-lm"])
    values_for_each_var = gen_initial_points(params, precondition, width, large_sample_num)

    def cartesian_expand_samples(values_for_each_var, width):
        def cartesian_product(*arrays):
            # https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
            la = len(arrays)
            dtype = np.result_type(*arrays)
            arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
            for i, a in enumerate(np.ix_(*arrays)):
                arr[..., i] = a
            return arr.reshape(-1, la)
        if len(values_for_each_var) == 0:
            return []
        single_var_value_list = []
        for param in params:
            single_var_value_list.append(values_for_each_var[param][:width])
        initial_points = cartesian_product(*single_var_value_list)
        return initial_points

    initial_points = cartesian_expand_samples(values_for_each_var, width)
    dfs, num_sample, run_id = sample_core(initial_points, uniq)

    if quadratic:  # add second-order terms
        dfs = expand_quadratic(dfs)

    if os.path.exists(path + "/csv/" + basename + "_header.csv"):
        os.remove(path + "/csv/" + basename + "_header.csv")
    if os.path.exists(path + "/csv/" + basename + ".csv"):
        os.remove(path + "/csv/" + basename + ".csv")
    dfs.to_csv(path + "/csv/" + basename + ".csv", index=False)


def read_config(path):
    with open(path + '/cln2inv.config', 'r') as f:
        quadratic_line = f.readline()
        initial_line = f.readline()
        assert quadratic_line.startswith('quadratic:') and initial_line.startswith('initial:')
        quadratic_files = quadratic_line[10:].strip().split(',')
        initial_files = initial_line[8:].strip().split(',')
    return quadratic_files, initial_files

    
if __name__ == "__main__":

    basenames = [str(i) for i in range(1, 134)]
    path="../benchmarks/code2inv"
    if len(sys.argv) == 2:
        print ("run on single file: " + sys.argv[1])
        basenames = [sys.argv[1]]

    quadratic_list, record_initial_state_list = read_config(path)

    for basename in basenames:
        print("Generating execution traces for", basename)
        if not os.path.exists(os.path.join(path, "instrumented")):
            os.mkdir(os.path.join(path, "instrumented"))
        if not os.path.exists(os.path.join(path, "csv")):
            os.mkdir(os.path.join(path, "csv"))
        if not os.path.exists(os.path.join(path, "bin")):    
            os.mkdir(os.path.join(path, "bin"))

        params, precondition = instrument(path, basename, record_initial_state=(basename in record_initial_state_list))
        sample(path, basename, params, precondition, uniq=True, quadratic=(basename in quadratic_list))
