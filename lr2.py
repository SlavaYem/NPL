############################1############################
var_int = 10
var_float = 8.4
var_str = 'No'
big_int = var_int * 3.5
var_float -= 1
var_int / var_float
big_int / var_float
var_str = var_str + 'No' + 'Yes' * 3
print(big_int, var_float, var_str)
###########################2#############################
a = 7
b = 5

exp1 = a > b and a + b == 12 and a - b < 3 and b * 2 != 11
exp2 = a > b and a + b == 12 and a - b > -2 and b * 2 == 10
exp3 = a == b and a + b == 12 and a - b < 3 and b * 2 == 10
exp4 = a == b and a + b == 12 and a - b > 3 and b * 2 == 10
print(exp1)
print(exp2)
print(exp3)
print(exp4)

x = 5
y = 10
string1 = "hello"
string2 = "world"

exp1 = (x < 10) or (y > 20)
exp2 = (string1 == "hello") or (string2 == "goodbye")
exp3 = (x == 0) or (y == 0)
exp4 = (string1 == "hi") or (string2 == "bye")
print(exp1)
print(exp2)
print(exp3)
print(exp4)