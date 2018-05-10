from abc import ABCMeta
import traceback
import numpy as np

# Can be a module if loaded later
pulp = None


class Solution(object):
    pass



class SolutionPulp(Solution):
    def __init__(self, status, variables):
        super(SolutionPulp, self).__init__()
        self._status = status
        self._variables = variables

    @property
    def status(self):
        return pulp.LpStatus[self._status]

    def variables_dict(self):
        """Return a dict associating the names with the values."""
        return {v.name: v.varValue for v in self._variables}

    def __str__(self):
        return str(self.variables_dict())



def solver(backend='pulp'):
    backend = backend.lower()
    if backend == 'pulp':
        return SolverPulp()
    else:
        raise ValueError("Unknown solver " + backend)



class Solver(metaclass=ABCMeta):
    def copy(self):
        raise NotImplementedError("Must be implemented by subclass")

    def add_constraint(self, c):
        """Add a constraint to be solved."""
        raise NotImplementedError("Must be implemented by subclass")

    def solve(self, variables=None):
        raise NotImplementedError("Not implemented yet")



class SolverPulp(Solver):
    def __init__(self):
        super(SolverPulp, self).__init__()
        global pulp
        import pulp
        self._prob = pulp.LpProblem()
        self._lpvars = {}
        self._conststore = {}
        self._stopped = False
        self._cache_expr = {}

    def copy(self):
        new = SolverPulp()
        new._prob = self._prob.copy()
        new._lpvars = self._lpvars
        new._stopped = self._stopped
        new._conststore = self._conststore.copy()
        for (e, _) in self._cache_expr.values():
            new._cache_copy_expr(e)
        return new

    @staticmethod
    def _expr_domain(e):
        assert isinstance(e, Expression)

        if isinstance(e, Variable):
            return e.domain

        if e.op == '=':
            return DomainRange(0, 2)
        elif e.op == '+':
            return sum((v.domain for v in e.values), DomainRange())
        else:
            raise ValueError("Can't evaluate the domain of a '%s' expression" % new.op)

    def _cache_copy_expr(self, c):
        assert isinstance(c, Expression)

        if c.name not in self._cache_expr:
            if isinstance(c, Variable):
                copy = c.copy()
                copy.parents = []
                self._cache_expr[c.name] = (c, copy)
            else:
                newvalues = [self._cache_copy_expr(v) for v in c.values]
                new = Expression(c.op, *newvalues)
                new.name = c.name
                new.domain = self._expr_domain(new)

                for v in newvalues:
                    v.parents.append(new)

                new.parents = []
                self._cache_expr[c.name] = (c, new)

        elif self._cache_expr[c.name][0] is not c and self._cache_expr[c.name][1] is not c:
            raise ValueError("Two distinct expressions with the name '%s' in the same model" % c.name)

        return self._cache_expr[c.name][1]

    def add_constraint(self, c):
        if isinstance(c, Expressions):
            for v in c.flat:
                if v is not None:
                    self.add_constraint(v)
            return

        if not c.iscomparison():
            raise ValueError("Constraint can only be a comparison expressions")

        c = self._cache_copy_expr(c)

        if self._stopped:
            self._prob += self._convert_constraint(c)
            return

        if c.name in self._conststore:
            raise ValueError("Constraint %s already in the model" % c)

        self._conststore[c.name] = c

    def _convert_constraint(self, c):
        if isinstance(c, Variable):
            if c.isinteger():
                return pulp.LpAffineExpression(int(c))
            return self._add_lpvar(c)

        lpexpr = [self._convert_constraint(v) for v in c.values]

        # TODO handle specially when comparing the result of two constraints
        if c.op == '+':
            return sum(lpexpr)
        if c.op == '=':
            return lpexpr[0] == lpexpr[1]

    def _add_lpvar(self, v):
        if v.name not in self._lpvars:
            lv = pulp.LpVariable(v.name, v.domain.min, v.domain.max - 1, pulp.LpInteger)
            self._lpvars[v.name] = lv

        return self._lpvars[v.name]

    def stoponlinesolve(self):
        for c in self._conststore.values():
            self._prob += self._convert_constraint(self._cache_copy_expr(c))
        self._conststore = {}
        self._stopped = True

    def solve(self):
        if not self._stopped:
            self.stoponlinesolve()

        status = self._prob.solve()
        variables = self._prob.variables()
        return SolutionPulp(status, variables)



class Domain(metaclass=ABCMeta):
    pass



class DomainRange(Domain):
    def __init__(self, low=None, up=None):
        self.min = low
        self.max = up

    @staticmethod
    def fromrange(r):
        assert isinstance(r, range), "DomainRange.fromrange only accepts range objects"
        assert r.step == 1, "Sparse ranges not implemented yet"
        return DomainRange(r.start, r.stop)

    def copy(self):
        return DomainRange(self.min, self.max)

    def __len__(self):
        if self.min is None or self.max is None:
            return float('inf')
        return max(self.max - self.min, 0)

    def __str__(self):
        if len(self) == 1:
            return "[%d]" % self.min
        return "[%d..%d]" % (self.min, self.max - 1)

    def __add__(self, other):
        if self.min is None:
            low = other.min
        elif other.min is None:
            low = self.min
        else:
            low = self.min + other.min

        if self.max is None:
            up = other.max
        elif other.max is None:
            up = self.max
        else:
            up = self.max + other.max - 1

        return DomainRange(low, up)



class Expression(object):
    _expridx = 0

    @classmethod
    def new_name(cls):
        name = "expr_%d" % Expression._expridx
        Expression._expridx += 1
        return name


    def __init__(self, op, *values):
        self.name = self.new_name()
        self.op = op
        self.values = []

        for v in values:
            assert isinstance(v, (Expression, int, np.integer)), \
                "Can only build expressions out of expressions or integers. Got: %s" % type(v)

            if isinstance(v, (int, np.integer)):
                v = Variable(v)
            self.values.append(v)

    def iscomparison(self):
        return self.op in '='

    def isariththmetic(self):
        return self.op in '+'

    def __add__(self, value):
        if self.op == '+':
            values = self.values
        else:
            values = [self]

        if isinstance(value, Expression) and value.op == '+':
            values += value.values
        else:
            values.append(value)

        return Expression('+', *values)

    def __radd__(self, value):
        return self + value

    def __eq__(self, value):
        return Expression('=', self, value)

    def __str__(self):
        return (" %s " % self.op).join(str(v) for v in self.values)

    def variables(self):
        if isinstance(self, Variable):
            return [self]
        return sum([v.variables() for v in self.values if isinstance(v, Expression)], [])



class Variable(Expression):
    _varidx = 0

    @classmethod
    def new_name(cls):
        name = "var_%d" % Variable._varidx
        Variable._varidx += 1
        return name

    def __init__(self, domain=None, name=None):
        super(Variable, self).__init__(None)

        if domain is None:
            domain = DomainRange()
        elif isinstance(domain, range):
            domain = DomainRange.fromrange(domain)
        elif isinstance(domain, (int, np.integer)):
            domain = DomainRange(domain, domain + 1)

        if name is not None:
            self.name = name
        self.domain = domain

    def copy(self):
        return Variable(self.domain.copy(), self.name)

    def __str__(self):
        if self.isinteger():
            return str(int(self))
        return "%s%s" % (self.name, self.domain)

    def __hash__(self):
        return hash(self.name)

    def isinteger(self):
        return len(self.domain) == 1

    def __int__(self):
        assert self.isinteger()
        return int(self.domain.min)



class Expressions(np.ndarray):
    """
    A convenient class to manipulate arrays of Expression.
    """

    # To unstandand this black magic, refer to the numpy documentation about
    # subclassing ndarray.
    def __new__(subtype, shape, domain=None, name_prefix=None):
        arr = super(Expressions, subtype).__new__(subtype, shape, dtype=np.object)
        if name_prefix is None:
            name_prefix = Variable.new_name()

        arr.flat = [Variable(domain, "%s_%d" % (name_prefix, i)) for i in range(len(arr.flat))]

        return arr

    def __array_finalize__(self, obj):
        pass

    def _call_ufunc(self, ufunc, method, *inputs, **kwargs):
        if ufunc == np.equal and method == '__call__':
            bc = np.broadcast(*inputs)
            results = kwargs.get('out', None)
            if results is None:
                results = np.empty(bc.shape, dtype=np.object)
            results.flat = [(a == b) for a, b in bc]
        else:
            results = super(Expressions, self).__array_ufunc__(ufunc, method, *inputs, **kwargs)
            if results is NotImplemented:
                return NotImplemented

        return results

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # This is needed because numpy doesn't want the comparison ufuncs to
        # return anything else than booleans.
        # https://github.com/numpy/numpy/issues/10948
        try:
            inputs = [i.view(np.ndarray) if isinstance(i, Expressions) else i for i in inputs]

            outputs = kwargs.pop('out', None)
            if outputs is not None:
                # The output to convert back to Expressions object
                convout = [isinstance(o, Expressions) for o in outputs]
                # kwargs['out'] must be a tuple
                kwargs['out'] = tuple(o.view(np.ndarray) if v else o for v, o in zip(convout, outputs))
            else:
                convout = [True] * ufunc.nout

            results = self._call_ufunc(ufunc, method, *inputs, **kwargs)

            if ufunc.nout == 1:
                results = (results,)

            results = tuple(np.asarray(r).view(Expressions) if v else r for v, r in zip(convout, results))

            return results[0] if ufunc.nout == 1 else results

        except BaseException as e:
            print("raised in __array_ufunc__:")
            traceback.print_exc()



class Variables(Expressions):
    def __new__(subtype, shape, domain=None, name_prefix=None):
        arr = super(Variables, subtype).__new__(subtype, shape)
        if name_prefix is None:
            name_prefix = Variable.new_name()

        arr.flat = [Variable(domain, "%s_%d" % (name_prefix, i)) for i in range(len(arr.flat))]

        return arr



class Zeros(Expressions):
    def __new__(subtype, shape, name_prefix=None):
        arr = super(Zeros, subtype).__new__(subtype, shape)
        arr.flat = 0
        return arr
