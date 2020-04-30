from cln.condition_parser import parse_file_conditions
from collections import defaultdict
import pandas as pd
import numpy as np
import re, operator
import z3




class TemplateOp():
    def __init__(self):
        self.params = []


class Or(TemplateOp):
    def __init__(self):
        self.params = []
        
    def __str__(self, depth=0):
        s = str('    '*depth) + 'Or' + '\n'
        for p in self.params:
            s += p.__str__(depth=depth+1)
        return s
    
    def to_z3(self):
        param_z3s = [p.to_z3() for p in self.params]
        return z3.Or(param_z3s)

    def is_static(self):
        return np.all([p.is_static() for p in self.params])
        
        
class And(TemplateOp):
    def __init__(self):
        self.params = []
        
    def __str__(self, depth=0):
        s = str('    '*depth) + 'Or' + '\n'
        for p in self.params:
            s += p.__str__(depth=depth+1)
        return s
    
    def to_z3(self):
        param_z3s = [p.to_z3() for p in self.params]
        return z3.And(param_z3s)

    def is_static(self):
        return np.all([p.is_static() for p in self.params])

    
class Not(TemplateOp):
    def __init__(self):
        self.params = []
        
    def __str__(self, depth=0):
        s = str('    '*depth) + 'Not' + '\n'
        for p in self.params:
            s += p.__str__(depth=depth+1)
        return s
    
    def to_z3(self):
        assert(len(self.params) == 1)
        return z3.Not(self.params[0].to_z3())

    def is_static(self):
        return np.all([p.is_static() for p in self.params])

        
class Constraint(TemplateOp):
    OPS = {
            '=': operator.eq,
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
          }
    
    def __init__(self, op):
        self.params = []
        self.coeffs = {}
        self.static = True
        self.op = op
        
    def __str__(self, depth=0):
        s = str('    '*depth) + self.op + '\n'
        for p in self.params:
            if isinstance(p, TemplateOp):
                s += p.__str__(depth=depth+1)
            else:
                s += str('    '*(depth+1)) + str(p) + '\n'
        return s
    
    def to_z3(self):
        expr = 0
        for var, coeff in self.coeffs.items():
            expr += z3.Real(var)*coeff
        op_func = Constraint.OPS[self.op]
        return op_func(expr, 0)

    def is_static(self):
        return self.static

            
        
class NumOp(TemplateOp):
    def __init__(self, op):
        self.params = []
        self.op = op
        
    def __str__(self, depth=0):
        s = str('    '*depth) + self.op + '\n'
        for p in self.params:
            if isinstance(p, TemplateOp):
                s += p.__str__(depth=depth+1)
            else:
                s += str('    '*(depth+1)) + str(p) + '\n'
        return s
    
    def to_z3(self):
        raise NotImplementedError('NumOp does not support z3 conversion')
    
    

def get_op(smt):
    smt = smt.lstrip()
    if smt.startswith('and'):
        return And(), smt[3:]
    elif smt.startswith('or'):
        return Or(), smt[2:]
    elif smt.startswith('not'):
        return Not(), smt[3:]

    # parse full constraint
    elif smt.startswith('>='):
        return Constraint('>='), smt[2:]
    elif smt.startswith('<='):
        return Constraint('<='), smt[2:]
    elif smt.startswith('>'):
        return Constraint('>'), smt[1:]
    elif smt.startswith('<'):
        return Constraint('<'), smt[1:]
    elif smt.startswith('='):
        return Constraint('='), smt[1:]
    elif smt.startswith('+'):
        return NumOp('+'), smt[1:]
    elif smt.startswith('-'):
        return NumOp('-'), smt[1:]
    elif smt.startswith('*'):
        return NumOp('*'), smt[1:]
    raise ValueError("Invalid op "+smt[:2])


def get_coeffs(smt, coeffs={}, factor=1):
    if isinstance(smt, int):
        coeffs['1'] += factor*smt
    elif isinstance(smt, str):
        coeffs[smt] += factor

    elif not isinstance(smt, NumOp):
        raise ValueError("Unexpected term in get_coeffs "+str(smt))

    elif smt.op == '*':
        for p in smt.params:
            if isinstance(p, int):
                factor *= p
        for p in smt.params:
            get_coeffs(p, coeffs, factor)

    elif smt.op == '+':
        for p in smt.params:
            get_coeffs(p, coeffs, factor)

    elif smt.op == '-':
        get_coeffs(smt.params[0], coeffs, factor)
        get_coeffs(smt.params[1], coeffs, -factor)
    else:
        raise ValueError("Unexpected op in get_coeffs "+str(smt))


def flatten_constraint(constr):
    coeffs = defaultdict(lambda: 0)
    get_coeffs(constr.params[0], coeffs, 1)
    get_coeffs(constr.params[1], coeffs, -1)
    constr.coeffs = coeffs


def load_trace(csv_name):
    df = pd.read_csv(csv_name)
    df_data = df.drop(columns=['init', 'final'])
    df_data['1'] = 1
    return df_data


def build_simple_template(smt):
    smt = smt.lstrip(' (')
    root_op, smt = get_op(smt)
    stack = [root_op]
    while smt:
        cur_op = stack[-1]
        smt = smt.lstrip()
        if smt.startswith('('):
            op, smt = get_op(smt[1:])
            stack.append(op)
            cur_op.params.append(op)
        elif smt.startswith(')'):
            if isinstance(cur_op, Constraint):
                flatten_constraint(cur_op)
            stack.pop()
            smt = smt[1:]
        else:
            m = re.search(r'[\w\d]+', smt)
            word = m.group()
            smt = smt[m.end():]
            if word.isnumeric():
                cur_op.params.append(int(word))
            else:
                cur_op.params.append(word)
    return root_op

class TemplateGen():
    
    def __init__(self, c_file, csv_name):
        
        self.c_file = c_file
        self.csv_name = csv_name
        
        # parse conditions
        self.condition = parse_file_conditions(self.c_file)
        condition = self.condition
        
        # generate simple templates
        collection = []
        collection = collection + (condition['preconds'])
        collection.append(condition['predicate'])
        collection.append(condition['postcondition']['assert'])
        collection = collection + (condition['postcondition']['ifs'])
        if condition['predicate']:
            collection.append( "(or " + condition['predicate'] + " " + condition['postcondition']['assert'] + ")")
        collection = [i for i in collection if i is not None]
        ands = [ "(and " + i + " " + j + " )" for i in collection for j in collection]
        ors = [ "(or " + i + " " + j + " )" for i in collection for j in collection]
        collection = collection + ands + ors
        
        self.simple_template_smts = collection
        self.simple_index = 0
        self.generic_index = 0


    def build_generic_template(self, generic_index, pred_str):
        df = load_trace(self.csv_name)
        
        var_names = list(df.columns)
        template = And()
        if generic_index < 1:
            eq = Constraint('=')
            eq.coeffs = {var:0 for var in var_names}
            eq.static = False
            template.params.append(eq)
        else:
            if generic_index%2:
                eq_relation = And()
            else:
                eq_relation = Or()
            n_eqs = (generic_index+3)//2
            for _ in n_eqs:
                eq = Constraint('=')
                eq.coeffs = {var:0 for var in var_names}
                eq.static = False
                eq_relation.params.append(eq)
            template.params.append(eq_relation)

        if pred_str is not None and (('<' in pred_str) or ('<=' in pred_str)):
            pred_str = pred_str.strip(' ()')
            pred_l = pred_str.split()
            try:
                v1 = float(pred_l[1])
            except ValueError:
                v1 = pred_l[1]
            try:
                v2 = float(pred_l[2])
            except ValueError:
                v2 = pred_l[2]

            if isinstance(v1, str) and isinstance(v2, str):
                pred_constr = Constraint(pred_l[0])
                pred_constr.coeffs[v1] = 1
                pred_constr.coeffs[v2] = -1
                pred_constr.static = False
                template.params.append(pred_constr)

        for var in var_names:
            if var == '1':
                continue
            lt = Constraint('<')
            lt.coeffs = {var: 0, '1': 0}
            gt = Constraint('>')
            gt.coeffs = {var: 0, '1': 0}
            template.params += [lt, gt]
        return template


    def get_next_template(self):
        if self.simple_index < len(self.simple_template_smts):
            cln_template = build_simple_template(self.simple_template_smts[self.simple_index])
            self.simple_index += 1
            return cln_template
        elif self.generic_index < 1:
            pred = self.condition['predicate']
            cln_template = self.build_generic_template(self.generic_index, pred)
            self.generic_index += 1
            return cln_template
        else:
            return None
