import z3
from lark import Lark, Transformer
from functools import lru_cache

fol_grammar = """
?start: expr

?expr: or_expr "→" expr   -> implies
     | or_expr "->" expr  -> implies
     | or_expr "↔" expr   -> iff
     | or_expr "<->" expr -> iff
     | or_expr

?or_expr: or_expr "∨" and_expr -> logical_or
        | or_expr "|" and_expr -> logical_or
        | and_expr

?and_expr: and_expr "∧" not_expr -> logical_and
         | and_expr "&" not_expr -> logical_and
         | not_expr

?not_expr: "¬" not_expr -> negate
         | "~" not_expr -> negate
         | comp_expr

?comp_expr: arith_expr "=" arith_expr  -> eq
          | arith_expr "==" arith_expr -> eq
          | arith_expr "!=" arith_expr -> neq
          | arith_expr "<" arith_expr  -> lt
          | arith_expr ">" arith_expr  -> gt
          | arith_expr "<=" arith_expr -> lte
          | arith_expr ">=" arith_expr -> gte
          | arith_expr

?arith_expr: arith_expr "+" term -> add
           | arith_expr "-" term -> sub
           | term

?term: term "*" atom -> mul
     | term "/" atom -> div
     | atom

?atom: CNAME "("(expr ("," expr)*)? ")" -> function_call
     | quant
     | CNAME -> constant
     | INT -> integer
     | FLOAT -> float
     | "(" expr ")"

quant: QUANT_OP CNAME expr -> quantifier
QUANT_OP: "forall" | "∀" | "exists" | "∃"

%import common.CNAME
%import common.INT
%import common.FLOAT
%import common.WS
%ignore WS
"""

fol_parser = Lark(fol_grammar, parser="lalr")

@lru_cache(maxsize=4096)
def sota_parse_text(premise: str):
    return fol_parser.parse(premise)

class AdvancedZ3Transformer(Transformer):
    def __init__(self, types_dict: dict, context: z3.Context):
        super().__init__()
        self.context = context
        self.types = types_dict or {}
        self.sort_object = z3.DeclareSort("Object", ctx=self.context)

        self.sort_map = {
            "Int": z3.IntSort(ctx=self.context),
            "Real": z3.RealSort(ctx=self.context),
            "Bool": z3.BoolSort(ctx=self.context),
            "Object": self.sort_object,
        }

        self.constants = {}
        self.functions = {}

    def integer(self, token):
        return z3.IntVal(int(token), ctx=self.context)

    def float(self, token):
        return z3.RealVal(float(token), ctx=self.context)

    def constant(self, token):
        name = str(token)
        if name not in self.constants:
            declared_type = self.types.get(name, "Object")
            sort = self.sort_map.get(declared_type, self.sort_object)
            self.constants[name] = z3.Const(name, sort)
        return self.constants[name]

    def function_call(self, args):
        name = str(args[0])
        params = args[1:]

        key = f"{name}_{len(params)}"

        if key not in self.functions:
            declared_type = self.types.get(name, "Bool")
            range_sort = self.sort_map.get(
                declared_type, z3.BoolSort(ctx=self.context)
            )

            domain = [p.sort() for p in params]

            self.functions[key] = z3.Function(
                name, *(domain + [range_sort])
            )

        return self.functions[key](*params)

    def add(self, args):
        return args[0] + args[1]

    def sub(self, args):
        return args[0] - args[1]

    def mul(self, args):
        return args[0] * args[1]

    def div(self, args):
        return args[0] / args[1]

    def eq(self, args):
        return args[0] == args[1]

    def neq(self, args):
        return args[0] != args[1]

    def lt(self, args):
        return args[0] < args[1]

    def gt(self, args):
        return args[0] > args[1]

    def lte(self, args):
        return args[0] <= args[1]

    def gte(self, args):
        return args[0] >= args[1]

    def implies(self, args):
        return z3.Implies(args[0], args[1])

    def logical_or(self, args):
        return z3.Or(args[0], args[1])

    def logical_and(self, args):
        return z3.And(args[0], args[1])

    def negate(self, args):
        return z3.Not(args[0])

    def quantifier(self, args):
        op = str(args[0])
        var_name = str(args[1])
        expr = args[2]

        if var_name not in self.constants:
            self.constants[var_name] = z3.Const(
                var_name, self.sort_object
            )
        var = self.constants[var_name]

        if op in ("forall", "∀"):
            return z3.ForAll([var], expr)
        if op in ("exists", "∃"):
            return z3.Exists([var], expr)
