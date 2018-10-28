import numpy as np
import tensorflow as tf

class Sentinel(object):
    """
    Sentinel object
    """
    def __init__(self,
            name):
        assert isinstance(name, str)
        self.name = '<' + name + '>'

    def __repr__(self):
        return self.name

class Struct(object):
    """
    Converts data into struct accessible with "."
    """
    def __init__(self, data):
        for name, value in data.iteritems():
            setattr(self, name, self._wrap(value))


    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)): 
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(value) if isinstance(value, dict) else value


    def __repr__(self): 
        return 'Struct({%s})' % str(', '.join("'%s': %s" % (k, repr(v)) \
                for (k, v) in self.__dict__.iteritems()))



class Filter(object):
    """
    Evaluates a logical expression where the symbols are keys to a dictionary
    of tensorflow tensors
    """
    str_to_token = {
            'and': lambda left, right: tf.logical_and(left, right),
            'or': lambda left, right: tf.logical_or(left, right),
            'not': lambda var: tf.logical_not(var),
            '(': '(',
            ')': ')',
            }
    empty_res = True


    def __init__(self, expression):
        """Either use a logical expression to initialize Filter or
        define a func and keys that takes the data dict and these filter keys
        as input: func(data, keys)"""
        if isinstance(expression, str):
            self.expression = expression
            self.token_lst = self.create_token_lst(self.expression)
        elif isinstance(expression, tuple):
            assert callable(expression[0]), expression[0]
            assert isinstance(expression[1], list), expression[1]
            self.func = expression[0]
            self.keys = expression[1]
            if len(expression) > 2:
                assert isinstance(expression[2], dict), expression[2]
                self.kwargs = expression[2]
            else:
                self.kwargs = {}
        else:
            raise Exception('Unknown initialization')


    def create_token_lst(self, expression, str_to_token=str_to_token):
        """create token list:
        'True or False' -> [True, lambda..., False]"""
        s = expression.replace('(', ' ( ')
        s = s.replace(')', ' ) ')

        token_lst = []
        self.keys = []
        for it in s.split():
            if it in str_to_token:
                token_lst.append(str_to_token[it])
            else:
                token_lst.append(it)
                self.keys.append(it)
        self.keys = np.unique(self.keys)
        return token_lst


    def find(self, lst, what, start=0):
        return [i for i, it in enumerate(lst) if it == what and i >= start]


    def parens(self, token_lst):
        """returns:
            (bool)parens_exist, left_paren_pos, right_paren_pos
        """
        left_lst = self.find(token_lst, '(')

        if not left_lst:
            return False, -1, -1

        left = left_lst[-1]

        #can not occur earlier, hence there are args and op.
        right = self.find(token_lst, ')', left + 1)[0]

        return True, left, right


    def bool_eval(self, token_lst, data):
        """token_lst has length 3 and format: [left_arg, operator, right_arg]
        operator(left_arg, right_arg) is returned"""
        try:
            if len(token_lst) == 2:
                assert callable(token_lst[0])
                operator = token_lst[0]
                if isinstance(token_lst[1], str):
                    var = tf.cast(data[token_lst[1]], tf.bool)
                else:
                    var = token_lst[1]
                return operator(var)
            else:
                assert len(token_lst) == 3
                assert callable(token_lst[1])
                operator = token_lst[1]
                if isinstance(token_lst[0], str):
                    lhs = tf.cast(data[token_lst[0]], tf.bool)
                else:
                    lhs = token_lst[0]
                if isinstance(token_lst[2], str):
                    rhs = tf.cast(data[token_lst[2]], tf.bool)
                else:
                    rhs = token_lst[2]

                return operator(lhs, rhs)
        except AssertionError:
            raise AssertionError('Every expression has to be ' + \
                    'encapsulated in brackets as a 3-tuple:\n' + \
                    '(left_arg operator right_arg)\n' + \
                    '\'not\' has to be written as:\n' + \
                    '(not arg)\n' + \
                    'Your given expression: %s' % self.expression)


    def formatted_bool_eval(self, token_lst, data, empty_res=empty_res):
        """eval a formatted (i.e. of the form 'ToFa(ToF)') string"""
        if not token_lst:
            return self.empty_res

        if len(token_lst) == 1:
            if isinstance(token_lst[0], str):
                return data[token_lst[0]]
            else:
                return token_lst[0]

        has_parens, l_paren, r_paren = self.parens(token_lst)

        if not has_parens:
            return self.bool_eval(token_lst, data)

        token_lst[l_paren:r_paren + 1] = [self.bool_eval(
            token_lst[l_paren+1:r_paren], data)]

        return self.formatted_bool_eval(token_lst, data, self.bool_eval)


    def eval(self, data):
        """The actual 'eval' routine,
        if 's' is empty, 'True' is returned,
        otherwise 's' is evaluated according to parentheses nesting."""
        self.data = data
        if hasattr(self, 'func'):
            return self.func(self.data, self.keys, **self.kwargs)
        else:
            return self.formatted_bool_eval(
                    self.token_lst, self.data)


def filter_tests(): 
    sess = tf.Session()
    data = {'a': [1, 0], 'b': [0, 1], 'c': [1, 0]}
    #data = {'a': 1, 'b': 0, 'c': 1}
    # Test and
    expr = 'a and b'
    f1 = Filter(expr)
    assert (sess.run(f1.eval(data)) == [False, False]).all(), '%s != %s' % \
            (expr, [False, False])
    # Test or
    expr = 'a or b'
    f2 = Filter(expr)
    assert (sess.run(f2.eval(data)) == [True, True]).all(), '%s != %s' % \
            (expr, [True, True])
    # Test chain
    expr = '(a and b) and c'
    f3 = Filter(expr)
    assert (sess.run(f3.eval(data)) == [False, False]).all(), '%s != %s' % \
            (expr, [False, False])
    # Test brackets
    expr = 'a and (c and b)'
    f4 = Filter(expr)
    assert (sess.run(f4.eval(data)) == [False, False]).all(), '%s != %s' % \
            (expr, [False, False])
    # Test nested brackets
    expr = '((a and b) or (a and (a or b)))'
    f5 = Filter(expr)
    assert (sess.run(f5.eval(data)) == [True, False]).all(), '%s != %s' % \
            (expr, [True, False])
    # Test not
    expr = '(not (b or (not (b or a))))'
    f6 = Filter(expr)
    assert (sess.run(f6.eval(data)) == [True, False]).all(), '%s != %s' % \
            (expr, [True, False])
    # Test func
    f = Filter((lambda data, keys: [data[key] for key in keys], ['a', 'b', 'c']))
    assert f.eval(data) == [data[key] for key in ['a', 'b', 'c']], '%s != %s' % \
            (f.eval(data), [data[key] for key in ['a', 'b', 'c']])

    print('FILTER TESTS PASSED')


if __name__ == '__main__':
    filter_tests()
