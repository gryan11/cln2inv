from cln.template_gen import And, Or, Not, Constraint, load_trace

from sklearn.preprocessing import normalize
from math import gcd, floor
from fractions import Fraction
from collections import defaultdict
import numpy as np
import json
import operator
import torch
import z3

def prod_tnorm(fs):
    if len(fs) == 1:
        return fs[0]
    return torch.prod(torch.stack(fs), dim=0)
    # return reduce(lambda f1, f2: f1*f2, fs)

def prod_tconorm(fs):
    if len(fs) == 1:
        return fs[0]
    return reduce(lambda f1, f2: f1 + f2 - f1*f2, fs)

def sigmoid(x, B, eps):
    # B = torch.exp(B)
    return torch.sigmoid(B*x - eps)

def gaussian (data, k):
    #assumes the form 2^-(kx^2) where k is an approximation for std dev
    #applies it uniformly across the whole matrix
    #instead of AOC = 1 we scale to have gaus(0) = 1
    data = - 0.5*((data/k) ** 2)
    data = data.exp()
    return data


def gt(x, B, eps):
    return sigmoid(x, B, eps)


def ge(x, B, eps):
    return sigmoid(x, B, -eps)


def lt(x, B, eps):
    return sigmoid(-x, B, eps)


def le(x, B, eps):
    return sigmoid(-x, B, -eps)

CLN_OPS = {
            '=': gaussian,
            '>': gt,
            '<': lt,
            '>=': ge,
            '<=': le,
          }


class Tnorm(torch.nn.Module):
    def __init__(self, tnorm=prod_tnorm):
        super(Tnorm, self).__init__()
        self.subclauses = []
        self.tnorm = tnorm

    def forward(self, x, xn):
        inputs = [subclause.forward(x, xn) for subclause in self.subclauses]
        return self.tnorm(inputs)

    def get(self, search_type):
        results = [subclause.get(search_type) for subclause in self.subclauses]
        if isinstance(self, search_type):
            results += [[self]]
        return [result for sub in results for result in sub]

class Tconorm(torch.nn.Module):
    def __init__(self, tconorm=prod_tconorm):
        super(Tconorm, self).__init__()
        self.subclauses = []
        self.tconorm = tconorm

    def forward(self, x, xn):
        inputs = [subclause.forward(x, xn) for subclause in self.subclauses]
        return self.tconorm(inputs)

    def get(self, search_type):
        results = [subclause.get(search_type) for subclause in self.subclauses]
        if isinstance(self, search_type):
            results += [[self]]
        return [result for sub in results for result in sub]


class Neg(torch.nn.Module):
    def __init__(self, tnorm=prod_tnorm):
        super(Neg, self).__init__()
        self.subclauses = []

    def forward(self, x, xn):
        subclause = self.subclauses[0].forward(x, xn)
        return 1.0 - subclause

    def get(self, search_type):
        results = [subclause.get(search_type) for subclause in self.subclauses]
        if isinstance(self, search_type):
            results += [[self]]
        return [result for sub in results for result in sub]


class EqConstraint(torch.nn.Module):
    def __init__(self, coeffs, var_names):
        super(EqConstraint, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, len(coeffs)).uniform_(-1, 1))
        self.min_std = 0.1

    def forward(self, x, xn):
        weight = self.weight
        min_std = self.min_std
        with torch.no_grad():
            norm = torch.max(torch.abs(weight))
            self.weight /= norm
            # for weight in self.weight:
                # weight /= torch.max(torch.abs(weight))
        out = torch.nn.functional.linear(xn, self.weight).squeeze()
        outputs_std = max([out.std().detach(), min_std])
        activation = gaussian(out, outputs_std)
        return activation

    def get(self, search_type):
        return [self] if isinstance(self, search_type) else []



# IneqConstraint
class IneqConstraint(torch.nn.Module):
    def __init__(self, coeffs, var_names, op_str):
        super(IneqConstraint, self).__init__()
        coeffs = defaultdict(lambda:0, coeffs)
        coeff_w = [coeffs[var] for var in var_names]
        self.weight = torch.tensor(coeff_w, dtype=torch.float).reshape(1, -1)

        self.b = torch.nn.Parameter(torch.tensor([0.0], requires_grad=True))
        self.B = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.eps = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))

        self.op_str = op_str
        self.op = CLN_OPS[op_str]

    def forward(self, x, xn):
        with torch.no_grad():
            self.B.clamp_(min=0.1)
            self.eps.clamp_(min=0.5)

        weight, b = self.weight, self.b
        out = torch.nn.functional.linear(x, weight).squeeze() - b
        activation = self.op(out, self.B, self.eps)
        return activation

    def get(self, search_type):
        return [self] if isinstance(self, search_type) else []



# build model from template
def build_cln(template, var_names):
    weights = []
    bs = []
    Bs = []
    epss = []
    def build_node(t_node):
        cln_node = None
        if isinstance(t_node, Constraint):
            if not t_node.static:
                if t_node.op == '=':
                    cln_node = EqConstraint(t_node.coeffs, var_names)
                    weights.append(cln_node.weight)
                else:
                    if len(t_node.coeffs) > 1:
                        cln_node = IneqConstraint(t_node.coeffs, var_names, t_node.op)
                        bs.append(cln_node.b)
                        Bs.append(cln_node.B)
                        epss.append(cln_node.eps)

                cln_node.template = t_node
            return cln_node


        if isinstance(t_node, And):
            cln_node = Tnorm()
        elif isinstance(t_node, Or):
            cln_node = Tconorm()
        elif isinstance(t_node, Not):
            cln_node = Neg()
        else:
            raise ValueError("Invalid template node "+str(t_node))

        subclauses = [build_node(p) for p in t_node.params]
        cln_node.subclauses = [sc for sc in subclauses if sc is not None]
        return cln_node if cln_node.subclauses else None

    return build_node(template), weights, bs, Bs, epss

def data_normalize(data):
    data = 10 * normalize(data, norm='l2', axis=1)
    return data

# train model
class CLNTrainer():
    def __init__(self, csv_name):
        self.csv_name = csv_name
        self.df_data = None

    def build_train_cln(self, template, consts, max_epoch=4000,
             non_loop_invariant=None, max_denominator = 10, pname=1):
        loss_threshold = 1e-6


        if template.is_static():
            return [template.to_z3()], False


        if self.df_data is None:
            self.df_data = load_trace(self.csv_name)
        df_data = self.df_data
        data = df_data.to_numpy(dtype=np.float) 
        data = np.unique(data, axis=0)
        data_n = data_normalize(data)
        self.data = data
        self.data_n = data_n
        var_names = list(df_data.columns)

        cln_model, weights, bs, Bs, epss = build_cln(template, var_names)

        ges, les, eqs = infer_single_var_bounds_consts(df_data, consts)

        input_size = data.shape[1]
        coeff = None

        if input_size > 1:
            converged = False

            # data preparation
            inputs_np = np.array(data_n, copy=True)
            means_input, std_input = np.zeros([input_size], dtype=np.double), np.zeros([input_size], dtype=np.double)
            for i in range(input_size):
                means_input[i] = np.mean(data_n[:, i])
                std_input[i] = np.std(data_n[:, i])
                inputs_np[:, i] = (data_n[:, i] - means_input[i])
            inputs_n = torch.tensor(inputs_np, dtype=torch.float)
            inputs = torch.tensor(data, dtype=torch.float)

            b_factor=0.010

            B_factor, eps_factor = 0.025, 0.025
            B_target = 20.0

            loss_trace = []

            optimizer = torch.optim.Adam(weights+bs+Bs+epss, lr=0.01)
            for epoch in range(max_epoch):
                optimizer.zero_grad()
                cln_out = cln_model(inputs, inputs_n).squeeze()
                primary_loss = 1 - cln_out.mean()

                if primary_loss < loss_threshold:
                    converged = True
                    break

                l_bs, l_Bs, l_eps = 0, 0, 0
                if bs:
                    l_bs = b_factor*torch.norm(torch.cat(bs), p=1)
                    # l_bs = b_factor*torch.norm((torch.cat(bs)*.5)**2, p=1)
                if Bs:
                    l_Bs  = torch.clamp(B_factor*(B_target - torch.cat(Bs).mean()), min=0)
                if epss:
                    l_eps = eps_factor*torch.norm(torch.cat(epss), p=2)

                loss = primary_loss + l_bs + l_Bs + l_eps

                # loss_trace.append(loss.item())
                # if epoch%10 == 9 and np.std(loss_trace[-9:]) < 1e-5:
                    # break

                loss.backward()
                torch.nn.utils.clip_grad_norm_(cln_model.parameters(), 0.01)

                optimizer.step()

            # calculate final coeff
            weight = torch.stack(weights)
            coeff_ = weight.detach().numpy().reshape([input_size])
            scaled_coeff = np.round(coeff_/np.abs(coeff_).min())
            coeff = []
            denominator = 1
            for i in range(input_size):
                a = Fraction.from_float(float(coeff_[i])).limit_denominator(max_denominator)
                coeff.append(a)
                denominator = denominator * a.denominator // gcd(denominator, a.denominator)
            coeff = np.asarray([[floor(a * denominator) for a in coeff]])

            # extract ineqs
            ineq_constrs = cln_model.get(IneqConstraint)

            ineqs = []
            for ineq_constr in ineq_constrs:
                coeffs = {}
                for w, var in zip(ineq_constr.weight.flatten(), var_names):
                    if w != 0:
                        coeffs[var] = int(round(w.item()))
                ineqs.append((coeffs, int(round(ineq_constr.b.item())), ineq_constr.op_str))


        Is = construct_invariant(var_names, coeff, ges, les, eqs, ineqs, '', non_loop_invariant)
        if scaled_coeff.max() < 50: # large coeffs cause z3 timeouts
            scaled_Is = construct_invariant(var_names, scaled_coeff.reshape(1,-1), ges, les, eqs, ineqs, '', non_loop_invariant)
            Is.extend(scaled_Is)

        # for I in Is:
            # print(I)
        return Is, True



# extract checks from model
def construct_invariant(var_names, eq_coeff, ges, les, eqs, learned_ineqs, pred_str='',
        non_loop_invariant=None):

    OPS = {
            '=': operator.eq,
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
          }

    pred1, pred2 = None, None
    if pred_str is not None:
        if ('<' in pred_str) or ('<=' in pred_str):
            pred = pred_str.split()
            try:
                v1 = int(pred[2])
            except ValueError:
                v1 = z3.Real(pred[2])
            try:
                v2 = int(pred[3])
            except ValueError:
                v2 = z3.Real(pred[3])
            
            if pred[1] == '<':
                pred1 = v1 < v2
                pred2 = v1 <= v2
            elif pred[1] == '<=':
                pred1 = v1 <= v2
                pred2 = v1 <= v2 + 1

    reals = []
    for var in var_names:
        if var == '1':
            reals.append(1)
        else:
            reals.append(z3.Real(var))
        
    ands = []
    ineqs = []

    if eq_coeff is not None:
        eq_constraint = 0 * 0
        if (eq_coeff[0,0] != 0):
            eq_constraint = reals[0]*eq_coeff[0,0]
        for i, real in enumerate(reals[1:]):
            if ( eq_coeff[0,i+1] != 0):
                eq_constraint += eq_coeff[0, i+1] * real

        if isinstance(eq_constraint == 0, z3.BoolRef):
            ands += [eq_constraint == 0]
        
    for ge in ges:
        ands.append(reals[var_names.index(ge[0])] >= ge[1])
        ineqs.append(reals[var_names.index(ge[0])] >= ge[1])

    for le in les:
        ands.append(reals[var_names.index(le[0])] <= le[1])
        ineqs.append(reals[var_names.index(ge[0])] >= ge[1])

    for eq_ in eqs:
        if eq_[0] != '1':
            ands.append(reals[var_names.index(eq_[0])] == eq_[1])

    for ineq in learned_ineqs:
        coeffs, b, op_str = ineq
        op = OPS[op_str]
        z3_ineq = 0
        for i, (var, coeff) in enumerate(coeffs.items()):
            if i == 0:
                z3_ineq = reals[var_names.index(var)]*coeff
            else:
                z3_ineq += reals[var_names.index(var)]*coeff

        ands.append(op(z3_ineq, b))


    I0 = z3.And(*ineqs)
    I1 = z3.And(*ands)
    I2, I3 = None, None
    if pred1 is not None and pred2 is not None:
        I2 = z3.And(*ands, pred1)
        I3 = z3.And(*ands, pred2)

    if non_loop_invariant is not None:
        I1 = z3.Or(I1, non_loop_invariant)
        if pred1 is not None and pred2 is not None:
            I2 = z3.Or(I2, non_loop_invariant)
            I3 = z3.Or(I3, non_loop_invariant)

    Is = [I1]
    if I2 is not None and I3 is not None:
        Is.extend([I2, I3])
    Is.append(I0)
    return Is


def infer_single_var_bounds_consts(df, consts):
    ges = []
    les = []
    eqs = []

    for var in df.columns:
        if var in ('init', 'final'):
            continue

        max_v = max(df[var].unique())
        min_v = min(df[var].unique())
        if max_v == min_v:
            if max_v in consts:
                eqs.append( (var, max_v ))
            continue
        if max_v in consts:
            les.append( (var, max_v) )
        if min_v in consts:
            ges.append( (var, min_v) )
            
    return ges, les, eqs


