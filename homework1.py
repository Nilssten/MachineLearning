def f_recursive(x, b, res=1):
    if b < 0:
        return res
    current_expression = x**b + b
    return f_recursive(x, b-1, res * current_expression)