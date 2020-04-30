import z3

class InvariantChecker():
    
    def __init__(self, c_file, smt_check_path):
        self.c_file = c_file
        
        # read top pre rec post checks in
        with open(smt_check_path+'/'+c_file+'.smt.1') as f:
            self.top = f.read()
            
        with open(smt_check_path+'/'+c_file+'.smt.2') as f:
            self.pre = f.read()
            
        with open(smt_check_path+'/'+c_file+'.smt.3') as f:
            self.loop = f.read()
            
        with open(smt_check_path+'/'+c_file+'.smt.4') as f:
            self.post = f.read()
            
        # self.solver = z3.Solver()
        # self.solver.set("timeout", 2000)
        
    def check(self, inv_str):
        for check in [self.post, self.pre, self.loop]:
            full_check = self.top + inv_str + check
            solver = z3.Solver()
            solver.set("timeout", 2000)
            solver.from_string(full_check)
            res = solver.check()

            # self.solver.push()
            # self.solver.from_string(full_check)
            # res = self.solver.check()
            # self.solver.pop()
               
            if res != z3.unsat:
                return False
            
        return True

    def check_cln(self, inv_smts):
        correct = False
        for inv_smt in inv_smts:
            inv_str = inv_smt.sexpr()
            inv_str = inv_str.replace('|', '')
            correct = self.check(inv_str)
            if correct:
                return True, inv_str
        return False, ''

            
