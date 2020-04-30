from cln.template_gen import TemplateGen
from cln.invariant_checking import InvariantChecker
from cln.cln_training import CLNTrainer

import pandas as pd
import numpy as np
from z3 import And, Real
import sys, time


def load_consts(problem_num, const_file):
    consts = []
    with open(const_file, 'rt') as const_file:
        for line in const_file:
            number, consts_original, consts_expand = line.strip().split()
            if int(number) == problem_num:
                consts = consts_expand.split(',')
                consts = [int(const) for const in consts]
                
    return consts




def run_code2inv_problem(problem_num):
    fname = str(problem_num) + '.c'
    csvname = str(problem_num) + '.csv'
    src_path = 'benchmarks/code2inv/c/'
    check_path = 'benchmarks/code2inv/smt2/'
    trace_path = 'benchmarks/code2inv/traces/'

    if problem_num in [26, 27, 31, 32, 61, 62, 72, 75, 106]:
        print(problem_num,'theoretically unsolvable')
        return False, '', 0

    start_time = time.time()
    consts = load_consts(problem_num, 'benchmarks/code2inv/smt2/const.txt')
    

    templateGen = TemplateGen(src_path+fname, trace_path + csvname)
    trainer = CLNTrainer(trace_path + csvname)
    invariantChecker = InvariantChecker(fname, check_path)

    non_loop_invariant = None
    if problem_num in [110, 111, 112, 113]:
        # non_loop_invariant = And(Real('sn') == 0, Real('i') == 1, Real('n') < 0)
        non_loop_invariant = And(Real('sn') == 0, Real('i') == 1, Real('n') < 1)
    elif problem_num in [118, 119, 122, 123]:
        non_loop_invariant = And(Real('sn') == 0, Real('i') == 1, Real('size') < 1)

    solved, inv_str = False, ''
    for n_template in range(10_000):
        cln_template = templateGen.get_next_template()
        if cln_template is None:
            break
        rerun = True
        max_epoch = 2000
        restarts = 0
        while not solved and rerun and restarts < 10:
            invariant, rerun = trainer.build_train_cln(cln_template, consts,
                    max_epoch=max_epoch, non_loop_invariant=non_loop_invariant,
                    pname=problem_num)

            try:
                solved, inv_str = invariantChecker.check_cln(invariant)
            except:
                # rerun if z3 throws an error
                rerun = True
            max_epoch += 1000
            restarts += 1

        if solved:
            break

    runtime = time.time() - start_time
    return solved, inv_str, runtime

    
def main():
    if len(sys.argv) > 1:
        print('running problem', sys.argv[1])
        problem = int(sys.argv[1])
        solved, inv_str, runtime = run_code2inv_problem(problem)
        print(problem, 'Solved?', solved, 'Time: {:0.2f}s'.format(runtime))
        print('invariant:', inv_str)

    else:
        print('running entire benchmark, use python cln2inv.py <problem_number> to run single problem')
        total_solved = 0
        total = 0
        unsolvable = 0
        runtimes = []
        start = 1
        for i in range(start, 134):
            solved, inv_str, runtime = run_code2inv_problem(i)
            total += solved
            if i in [26, 27, 31, 32, 61, 62, 72, 75, 106]:
                unsolvable += 1
            else:
                assert(solved)
            runtimes.append(runtime)

            print(i, 'Solved?', solved, 'Total Solved: {}/{}'.format(total, 124), 'Time: {:0.2f}s'.format(runtime))
            print('Invariant:', inv_str)
        print('Avg. Runtime: {:0.1f}s, Max Runtime: {:0.1f}s'.format(np.mean(runtimes), np.max(runtimes)))


if __name__=='__main__':
    main()
