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
