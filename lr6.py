my_list = ['array1', 'array2', 'array3', 'array4']

for i in my_list:
    print(i)

for i in my_list:
    for j in i:
        print(j + '-', end='')

my_list_2 = [1, 2, 3, 4, 5, 6]

for i in my_list_2:
    print(float(i))


def func_1(a, b, c):
    return print(a + b + c)


func_1(1, 2, 3)


def multiply_by_two(num):
    results = num * 2
    return results


def print_result(num):
    print("Результат множення на два:", num)


num_1 = 10

result = multiply_by_two(num_1)
print_result(result)
