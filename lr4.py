str1 = "very big string"
print(str1[0])
print(str1[-1])
print(str1[2])
print(str1[-3])
print(len(str1))
print(str1[0:9])
print(str1[5:9])
print(str1[::3])

# Створення списків
list1 = ['apple', 'banana', 'orange', 'grape']
list2 = [10, 20, 30, 40]

# Вилучення другого елемента з першого списку
second_elem = list1.pop(1)
print("Вилучений елемент:", second_elem)

# Зміна останнього елемента другого списку
list2[-1] = 50
print("Змінений список 2:", list2)

# З'єднання списків
combined_list = list1 + list2
print("Об'єднаний список:", combined_list)

# Отримання зрізу
sliced_list = combined_list[2:5]
print("Список-зріз:", sliced_list)

# Додавання елементів до списку-зрізу
sliced_list.extend(['pineapple', 60])
print("Список-зріз з новими елементами:", sliced_list)