# -*- coding: utf-8 -*-

#requires gurobipy
#!pip install gurobipy

import gurobipy as gp
import os

"""# Paths to Folders for Data Importation and Saving Results"""

Data_folder = './Data_Instances'
Results_folder = './FairIR_Results'

os.makedirs(Results_folder, exist_ok=True)

"""## FAIRIR Code"""

import numpy as np
import time
import uuid

#from models.Basic import Basic
from gurobipy import *

class Basic(object):
    """Paper matching formulated as a linear program."""
    def __init__(self, loads, coverages, weights, loads_lb=None):
        """Initialize the Basic matcher

        Args:
            loads - a list of integers specifying the maximum number of papers
                  for each reviewer.
            coverages - a list of integers specifying the number of reviews per
                 paper.
            weights - the affinity matrix (np.array) of papers to reviewers.
                   Rows correspond to reviewers and columns correspond to
                   papers.
            loads_lb - a list of integers specifying the min number of papers
                  for each reviewer (optional).

        Returns:
            initialized matcher.
        """
        self.n_rev = np.size(weights, axis=0)
        self.n_pap = np.size(weights, axis=1)
        self.loads = loads
        self.loads_lb = loads_lb
        self.coverages = coverages

        assert(np.sum(coverages) <= np.sum(loads))
        if loads_lb is not None:
            assert(np.sum(coverages) >= np.sum(loads_lb))

        self.weights = weights
        self.id = uuid.uuid4()
        self.m = Model("%s: basic matcher" % str(self.id))
        self.solution = None
        self.m.setParam('OutputFlag', 0)

        # Primal vars.
        start = time.time()
        coeff = list(self.weights.flatten())
        print('#info Basic:flatten %s' % (time.time() - start))
        self.lp_vars = self.m.addVars(self.n_rev, self.n_pap, vtype=GRB.BINARY,
                                      name='x', obj=coeff)
        self.m.update()
        print('#info Basic:Time to add vars %s' % (time.time() - start))

        # Objective.
        self.m.setObjective(self.m.getObjective(), GRB.MAXIMIZE)
        self.m.update()

        # Constraints.
        start = time.time()
        self.m.addConstrs((self.lp_vars.sum(r, '*') <= l
                           for r, l in enumerate(self.loads)))
        self.m.addConstrs((self.lp_vars.sum('*', p) == c
                           for p, c in enumerate(self.coverages)))
        if self.loads_lb is not None:
            self.m.addConstrs((self.lp_vars.sum(r, '*') >= l
                               for r, l in enumerate(self.loads_lb)))

        self.m.update()
        print('#info Basic:Time to add constraints %s' % (time.time() - start))

    @staticmethod
    def var_name(i, j):
        """The name of the variable corresponding to reviewer i and paper j."""
        return "x_" + str(i) + "," + str(j)

    @staticmethod
    def indices_of_var(v):
        """Get the indices associated with a particular var_name (above)."""
        name = v.varName
        indices = name[2:].split(',')
        i, j = int(indices[0]), int(indices[1])
        return i, j

    def solve(self):
        """Solve the ILP.

        If we were not able to solve the ILP optimally or suboptimally, then
        raise an error.  If we are able to solve the ILP, save the solution.

        Args:
            None.

        Returns:
            An np array corresponding to the solution.
        """
        self.m.optimize()
        if self.m.status == GRB.OPTIMAL:
            self.solution = self.sol_as_mat()
        return self.solution

    def sol_as_mat(self):
        if self.m.status == GRB.OPTIMAL:
            if self.solution is None:
                self.solution = np.zeros((self.n_rev, self.n_pap))
            for key, var in self.lp_vars.items():
                self.solution[key[0], key[1]] = var.x
            return self.solution
        else:
            raise Exception('Must solve the model optimally before calling!')

    def status(self):
        """Return the status code of the solver."""
        return self.m.status

    def turn_on_verbosity(self):
        """Turn on vurbosity for debugging."""
        self.m.setParam('OutputFlag', 1)

    def objective_val(self):
        """Get the objective value of a solved lp."""
        return self.m.ObjVal

# if __name__ == "__main__":
#     init_makespan = 0.7
#     ws = np.array([
#         [0.1, 0.1, 0.9, 0.9],
#         [0.1, 0.1, 0.9, 0.9],
#         [0.1, 0.1, 0.6, 0.8],
#         [0.1, 0.1, 0.4, 0.3]
#     ])
#     a = np.array([2, 2, 2, 2])
#     b = np.array([2, 2, 2, 2])
#     x = Basic(a, b, ws, loads_lb=None)
#     s = time.time()
#     x.solve()
#     print(x.sol_as_mat())
#     print(time.time() - s)
#     print('[done.]')

class FairIR(Basic):
    """Fair paper matcher via iterative relaxation.

    """

    def __init__(self, loads, loads_lb, coverages, weights, thresh=0.0):
        """Initialize.

        Args:
            loads - a list of integers specifying the maximum number of papers
                  for each reviewer.
            loads_lb - a list of ints specifying the minimum number of papers
                  for each reviewer.
            coverages - a list of integers specifying the number of reviews per
                 paper.
            weights - the affinity matrix (np.array) of papers to reviewers.
                   Rows correspond to reviewers and columns correspond to
                   papers.

            Returns:
                initialized makespan matcher.
        """
        self.n_rev = np.size(weights, axis=0)
        self.n_pap = np.size(weights, axis=1)
        self.loads = loads
        self.loads_lb = loads_lb
        self.coverages = coverages
        self.weights = weights
        self.id = uuid.uuid4()
        self.m = Model("%s: FairIR" % str(self.id))
        self.makespan = thresh
        self.solution = None

        self.m.setParam('OutputFlag', 0)

        self.load_ub_name = 'lib'
        self.load_lb_name = 'llb'
        self.cov_name = 'cov'
        self.ms_constr_prefix = 'ms'
        self.round_constr_prefix = 'round'

        # primal variables
        start = time.time()
        self.lp_vars = []
        for i in range(self.n_rev):
            self.lp_vars.append([])
            for j in range(self.n_pap):
                self.lp_vars[i].append(self.m.addVar(ub=1.0,
                                                     name=self.var_name(i, j)))
        self.m.update()
        print('#info FairIR:Time to add vars %s' % (time.time() - start))

        start = time.time()
        # set the objective
        obj = LinExpr()
        for i in range(self.n_rev):
            for j in range(self.n_pap):
                obj += self.weights[i][j] * self.lp_vars[i][j]
        self.m.setObjective(obj, GRB.MAXIMIZE)
        print('#info FairIR:Time to set obj %s' % (time.time() - start))

        start = time.time()
        # load upper bound constraints.
        for r, load in enumerate(self.loads):
            self.m.addConstr(sum(self.lp_vars[r]) <= load,
                             self.lub_constr_name(r))

        # load load bound constraints.
        if self.loads_lb is not None:
            for r, load in enumerate(self.loads_lb):
                self.m.addConstr(sum(self.lp_vars[r]) >= load,
                                 self.llb_constr_name(r))

        # coverage constraints.
        for p, cov in enumerate(self.coverages):
            self.m.addConstr(sum([self.lp_vars[i][p]
                                  for i in range(self.n_rev)]) == cov,
                             self.cov_constr_name(p))

        # makespan constraints.
        for p in range(self.n_pap):
            self.m.addConstr(sum([self.lp_vars[i][p] * self.weights[i][p]
                                  for i in range(self.n_rev)]) >= self.makespan,
                             self.ms_constr_name(p))
        self.m.update()
        print('#info FairIR:Time to add constr %s' % (time.time() - start))

    def ms_constr_name(self, p):
        """Name of the makespan constraint for paper p."""
        return '%s%s' % (self.ms_constr_prefix, p)

    def lub_constr_name(self, r):
        """Name of load upper bound constraint for reviewer r."""
        return '%s%s' % (self.load_ub_name, r)

    def llb_constr_name(self, r):
        """Name of load lower bound constraint for reviewer r."""
        return '%s%s' % (self.load_lb_name, r)

    def cov_constr_name(self, p):
        """Name of coverage constraint for paper p."""
        return '%s%s' % (self.cov_name, p)

    def change_makespan(self, new_makespan):
        """Change the current makespan to a new_makespan value.

        Args:
            new_makespan - the new makespan constraint.

        Returns:
            Nothing.
        """
        for c in self.m.getConstrs():
            if c.getAttr("ConstrName").startswith(self.ms_constr_prefix):
                self.m.remove(c)
                # self.m.update()

        for p in range(self.n_pap):
            self.m.addConstr(sum([self.lp_vars[i][p] * self.weights[i][p]
                                  for i in range(self.n_rev)]) >= new_makespan,
                             self.ms_constr_prefix + str(p))
        self.makespan = new_makespan
        self.m.update()

    def sol_as_mat(self):
        if self.m.status == GRB.OPTIMAL or self.m.status == GRB.SUBOPTIMAL:
            solution = np.zeros((self.n_rev, self.n_pap))
            for v in self.m.getVars():
                i, j = self.indices_of_var(v)
                solution[i, j] = v.x
            self.solution = solution
            return solution
        else:
            raise Exception(
                'You must have solved the model optimally or suboptimally '
                'before calling this function.')

    def integral_sol_found(self):
        """Return true if all lp variables are integral."""
        sol = self.sol_as_dict()
        return all(sol[self.var_name(i, j)] == 1.0 or
                   sol[self.var_name(i, j)] == 0.0
                   for i in range(self.n_rev) for j in range(self.n_pap))

    def fix_assignment(self, i, j, val):
        """Round the variable x_ij to val."""
        self.lp_vars[i][j].ub = val
        self.lp_vars[i][j].lb = val

    def find_ms(self):
        """Find an the highest possible makespan.

        Perform a binary search on the makespan value. Each time, solve the
        makespan LP without the integrality constraint. If we can find a
        fractional value to one of these LPs, then we can round it.

        Args:
            None

        Return:
            Highest feasible makespan value found.
        """
        mn = 0.0
        mx = np.max(self.weights) * np.max(self.coverages)
        ms = mx
        best = None
        self.change_makespan(ms)
        start = time.time()
        self.m.optimize()
        print('#info FairIR:Time to solve %s' % (time.time() - start))
        for i in range(10):
            print('#info FairIR:ITERATION %s ms %s' % (i, ms))
            if self.m.status == GRB.INFEASIBLE:
                mx = ms
                ms -= (ms - mn) / 2.0
            else:
                assert(best is None or ms > best)
                assert(self.m.status == GRB.OPTIMAL)
                best = ms
                mn = ms
                ms += (mx - ms) / 2.0
            self.change_makespan(ms)
            self.m.optimize()
        return best

    def solve(self):
        """Find a makespan and solve the ILP.

        Run a binary search to find an appropriate makespan and then solve the
        ILP. If solved optimally or suboptimally then save the solution.

        Args:
            mn - the minimum feasible makespan (optional).
            mx - the maximum possible makespan( optional).
            itr - the number of iterations of binary search for the makespan.
            log_file - the string path to the log file.

        Returns:
            The solution as a matrix.
        """
        if self.makespan <= 0:
            print('#info FairIR: searching for fairness threshold')
            ms = self.find_ms()
        else:
            print('#info FairIR: config fairness threshold: %s' % self.makespan)
            ms = self.makespan
        self.change_makespan(ms)
        self.round_fractional(np.ones((self.n_rev, self.n_pap)) * -1)

        sol = {}
        for v in self.m.getVars():
            sol[v.varName] = v.x

    def sol_as_dict(self):
        """Return the solution to the optimization as a dictionary.

        If the matching has not be solved optimally or suboptimally, then raise
        an exception.

        Args:
            None.

        Returns:
            A dictionary from var_name to value (either 0 or 1)
        """
        if self.m.status == GRB.OPTIMAL or self.m.status == GRB.SUBOPTIMAL:
            _sol = {}
            for v in self.m.getVars():
                _sol[v.varName] = v.x
            return _sol
        else:
            raise Exception(
                'You must have solved the model optimally or suboptimally '
                'before calling this function.\nSTATUS %s\tMAKESPAN %f' % (
                    self.m.status, self.makespan))

    def round_fractional(self, integral_assignments=None, count=0):
        """Round a fractional solution.

        This is the meat of the iterative relaxation approach.  First, if the
        solution to the relaxed LP is integral, then we're done--return the
        solution. Otherwise, here's what we do:
        1. if a variable is integral, lock it's value to that integer.
        2. find all papers with exactly 2 or 3 fractionally assigned revs and
           drop the makespan constraint on that reviewer.
        3. if no makespan constraints dropped, find a reviewer with exactly two
           fraction assignments and drop the load constraints on that reviewer.

        Args:
            integral_assignments - np.array of revs x paps (initially None).
            log_file - the log file if exists.
            count - (int) to keep track of the number of calls to this function.

        Returns:
            Nothing--has the side effect or storing an assignment matrix in this
            class.
        """
        if integral_assignments is None:
            integral_assignments = np.ones((self.n_rev, self.n_pap)) * -1

        self.m.optimize()

        if self.m.status != GRB.OPTIMAL and self.m.status != GRB.SUBOPTIMAL:
            assert False, '%s\t%s' % (self.m.status, self.makespan)

        if self.integral_sol_found():
            return
        else:
            frac_assign_p = {}
            frac_assign_r = {}
            sol = self.sol_as_dict()
            fractional_vars = []

            # Find fractional vars.
            for i in range(self.n_rev):
                for j in range(self.n_pap):
                    if j not in frac_assign_p:
                        frac_assign_p[j] = []
                    if i not in frac_assign_r:
                        frac_assign_r[i] = []

                    if sol[self.var_name(i, j)] == 0.0 and \
                                    integral_assignments[i][j] != 0.0:
                        self.fix_assignment(i, j, 0.0)
                        integral_assignments[i][j] = 0.0

                    elif sol[self.var_name(i, j)] == 1.0 and \
                                    integral_assignments[i][j] != 1.0:
                        self.fix_assignment(i, j, 1.0)
                        integral_assignments[i][j] = 1.0

                    elif sol[self.var_name(i, j)] != 1.0 and \
                                    sol[self.var_name(i, j)] != 0.0:
                        frac_assign_p[j].append(
                            (i, j, sol[self.var_name(i, j)]))
                        frac_assign_r[i].append(
                            (i, j, sol[self.var_name(i, j)]))
                        fractional_vars.append((i, j, sol[self.var_name(i, j)]))

                        integral_assignments[i][j] = sol[self.var_name(i, j)]

            # First try to elim a makespan constraint.
            removed = False
            for (paper, frac_vars) in frac_assign_p.items():
                if len(frac_vars) == 2 or len(frac_vars) == 3:
                    for c in self.m.getConstrs():
                        if c.ConstrName == self.ms_constr_name(paper):
                            self.m.remove(c)
                            removed = True

            # If necessary remove a load constraint.
            if not removed:
                for (rev, frac_vars) in frac_assign_r.items():
                    if len(frac_vars) == 2:
                        for c in self.m.getConstrs():
                            if c.ConstrName == self.lub_constr_name(rev) or \
                                    c.ConstrName == self.llb_constr_name(rev):
                                self.m.remove(c)
            self.m.update()
            return self.round_fractional(integral_assignments, count + 1)

def Fairir(num_inst):
    instance_file_name = f'instance_{num_inst}.npz'
    instance_file_path = os.path.join(Data_folder, instance_file_name)
    instance_data = np.load(instance_file_path)

    #affinity score matrix and t
    matrix=instance_data['affinity_scores']
    t = int(instance_data['number_t'])

    #computation of n,m, upper and lower bounds l1 and l2
    n , m = matrix.shape[0], matrix.shape[1]
    l1 = math.floor(n * t / m)
    l2 = math.ceil(n * t / m)
    
    mtr=np.transpose(matrix)
    
    loads =np.full(m, l2)
    loads_lb = np.full(m, l1)
    coverages = np.full(n, t)
    
    init_makespan = 0.7
    ws=mtr
    print(ws)

    x = FairIR(loads, loads_lb, coverages, ws)
    s = time.time()
    x.solve()
    d=x.sol_as_mat()
    print(x.sol_as_mat())
    print(x.objective_val())
    Time=time.time() - s
    print(time.time() - s)
    print("[done.]")
    d2 = np.transpose(d)
    print(np.shape(d2),np.shape(matrix))

    # Save the results to the destination folder
    result_file_name = f'instance_{num_inst}_FairIR_result.npz'
    result_file_path = os.path.join(Results_folder, result_file_name)
    np.savez(result_file_path, matrix_assign=d2,time=Time)

if __name__ == "__main__":
    #Loading the instance data
    
    # 1 = Instance C1, 2 = Instance C2, 3 = Instance C3, 4 = Instance C4, 5 = Instance C5,
    # 30 = MIDL, 40 = CVPR'17, 50 = CVPR'18, 60 = ICA2IT'19, 70 = CVPR'18Extd_2, 80 = ICLR'18, 
    # 90 = CVPR'17Extd_4, 100 = CVPR'17Extd_3, 110 = MIDL'18Extd_4, 120 = ICA2IT'19Extd_4, 130 = ICLR'18Extd_4. 
    Fairir(30)
