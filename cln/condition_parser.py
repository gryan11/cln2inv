import re
from subprocess import run
import json


OPERATORS = set(['+', '-', '*', '/', '(', ')', '@', '<', '#', '>', '!', '='])
PRIORITY = {'+':1, '-':1, '*':2, '/':2}
def infix_to_prefix(formula):
    op_stack = []
    exp_stack = []
    for ch in formula:
        if not ch in OPERATORS:
            exp_stack.append(ch)
        elif ch == '(':
            op_stack.append(ch)
        elif ch == ')':
            while op_stack[-1] != '(':
                op = op_stack.pop()
                a = exp_stack.pop()
                b = exp_stack.pop()
                exp_stack.append( " ".join(["(",op,b,a,")"]) )
            op_stack.pop() # pop '('
        else:
            while op_stack and op_stack[-1] != '(' and PRIORITY[ch] <= PRIORITY[op_stack[-1]]:
                op = op_stack.pop()
                a = exp_stack.pop()
                b = exp_stack.pop()
                exp_stack.append( " ".join([op,b,a]) )
            op_stack.append(ch)

    
    # leftover
    while op_stack:
        op = op_stack.pop()
        a = exp_stack.pop()
        b = exp_stack.pop()
        exp_stack.append(  " ".join([op,b,a] ))
    return exp_stack[-1]

def clean_up (line, paren=False):
    left = "" 
    right = ""
    if paren:
        left = "( "
        right = " )"
    
    clean = line.split('(')[1:]
    clean = left.join(clean)
    clean = clean.split(')')[:-1] 
    clean = right.join(clean)
    
    return clean.strip()
    
def full_prefix(line):
    clean = line
    if "(" in line:    
        clean = clean_up(line, paren=True)
    

    encode = re.sub(r'<=', '@', clean)
    encode = re.sub(r'>=', '#', encode)
    encode = re.sub(r'!=', '!', encode)
    encode = re.sub(r'==', '=', encode)

    prefix = infix_to_prefix(encode.split())
    decode = re.sub(r'@', '<=', prefix)
    decode = re.sub(r'#', '>=', decode)
    decode = re.sub(r'!', '!=', decode)

    
    if decode.strip()[0] == "(":
        decode = clean_up(decode,paren=True)

    return decode

def op_conversion(l):
    if '==' in l:
        l = re.sub(r'==', '=', l)
    if "!=" in l:
        l = "not (" + re.sub(r'!=', '=', l) + ")"
    return l

def to_prefix(l,line):
    if len(l.split()) != 3:
        out = full_prefix(line)
        #print("WARNING: MORE THAN 3 TOKENS",out)
    else:
        out = full_prefix(l)
    return out
    

def parse_file_conditions(fname):
    with open(fname, 'rt') as input_file:
            preconditions = []
            predicate = None
    
            post_conditions = {'ifs':[], 'assert':None}
            
            before_loop = True
            post_condition = False
            ifs = []     
            
            for line in input_file:
                if "while" in line:
                    before_loop = False
                    if ("unknown" in line):
                        continue
                    l = clean_up(line)
                    l = to_prefix(l,line)
                    
                    l = op_conversion(l)
    
                    predicate = "( " + l + " )"
                    
                if before_loop:
                    if "assume" in line:
                        l = clean_up(line)
                        
                        l = to_prefix(l,line)
                        l = op_conversion(l)
                        preconditions.append('(' + l + ')')
                    elif '=' in line:
                        
                        line = line.strip()
                        line = line.strip(';')
                        l = line.strip('()')
                        
                        if l.split()[0] == 'int':
                            
                            l = l.split()
                            l = " ".join(l[1:])
                        
                        l = to_prefix(l,line)
                        l = op_conversion(l)
                        preconditions.append('(' + l + ')')
                
                if "post-condition" in line:
                    post_condition = True
                    
        
                if '//' in line:
                    continue
                if post_condition and "if" in line and post_condition == True:
                    l = clean_up(line)
                    l = to_prefix(l,line)
                    
                    l = op_conversion(l)
                    ifs.append("( not ("+ l +"))")
                if 'assert' in line:
                    l = clean_up(line)              
                    l = to_prefix(l,line)
                    l = " ".join(l.split())
                    l = op_conversion(l)
                    post_conditions['ifs'] = ifs
                    post_conditions['assert'] = "( " + l + " )"
                    
            conditions = {'preconds': preconditions, 'predicate': predicate,
             'postcondition': post_conditions}
            return conditions
