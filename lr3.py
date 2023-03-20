def lab31(x, y) -> None:
    if x > 0:
        print("Значення > 0")
    else:
        print("Значення < 0 або = 0")

    if y > 0:
        print("Значення > 0")
    else:
        print("Значення < 0 або = 0")


lab31(3, -3)


def lab32(x, y) -> None:
    if x > 0:
        print(1)
    else:
        print(-1)

    if y > 0:
        print(1)
    else:
        print(-1)


lab32(4, -4)


def lab33(x, y) -> None:
    if x > y:
        print("Значення змінної x більше, ніж значення змінної y")
    else:
        if x < y:
            print("Значення змінної x менше, ніж значення змінної y")
        else:
            print("Значення змінної x дорівнює значенню змінної y")


lab33(5, 2)


def lab34(x, y):
    if x > y:
        c = x - y
    elif x < y:
        c = x + y
    else:
        c = y
    return print(c)


lab34(10, 5)
# виведення ряду чисел Фібоначчі від 1 до 20
a, b = 0, 1
for i in range(20):
    print(a)
    a, b = b, a + b
# виведення ряду чисел Фібоначчі від 5 до 20
a, b = 0, 1
for i in range(4):
    a, b = b, a + b
for i in range(16):
    print(a)
    a, b = b, a + b
# виведення ряду парних чисел від 0 до 20
for i in range(0, 21, 2):
    print(i)
# виведення кожного третього числа від -1 до -21
num = -1
while num >= -21:
    print(num)
    num -= 3
# виведення чисел від 1 до 10 за допомогою циклу while
i = 1
while i <= 10:
    print(i)
    i += 1
# програма для вирішення прикладу 4*100-54 та перевірки відповіді користувача
ans = int(input("Розв'яжіть приклад 4*100-54: "))
if ans == 346:
    print("Правильно!")
else:
    print("Неправильно! Спробуйте ще раз.")
# програма для вирішення прикладу 4*100-54 та перевірки відповіді користувача до тих пір, поки він не введе правильну відповідь:
while True:
    answer = input("Розв'яжіть приклад 4*100-54: ")
    if answer == "346":
        print("Правильно!")
        break
    else:
        print("Неправильно. Спробуйте ще раз.")