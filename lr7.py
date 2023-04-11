# //...1...//
row_1 = "abcde"
row_9 = len(row_1)
n_1 = (row_9 + 1) // 2
s1 = row_1[:n_1]
s2 = row_1[n_1:]
print(s1, s2)
new_row = s2 + s1
print(new_row)
# //...2...//
row_2 = "Hello world"
words_row = row_2.split()
reverse_words = words_row[::-1]
result = ''.join(reverse_words)
print(result)
# //...3...//
row_3 = "avg full f"
first_index = row_3.find('f')
last_index = row_3.rfind('f')
if first_index == last_index and first_index != -1:
    print(first_index)
elif first_index != -1:
    print(first_index, last_index)
# //...4...//
row_4 = "avg full"
first_index = row_4.find("f")
if first_index == -1:
    print("-2")
else:
    second_index = row_4.find("f", first_index + 1)
    if second_index == -1:
        print("-1")
    else:
        print(second_index)
# //...5...//
row_5 = "ffdsfsfhffh"
first_h_index = row_5.find("h")
last_h_index = row_5.rfind("h")
new_row = row_5[:first_h_index] + row_5[last_h_index + 1:]
print(new_row)
# //...6...//
row_6 = "sdf h abc h"
first_h = row_6.find("h")
last_h = row_6.rfind("h")
between_h = row_6[first_h + 1:last_h]
reversed_between_h = between_h[::-1]
new_row = row_6[:first_h + 1] + reversed_between_h + row_6[last_h:]
print(new_row)
# //...7...//
row_7 = "121"
new_row = row_7.replace("1", "one")
print(new_row)
# //...8...//
row_8 = "@1@kjfgkdfmgd@"
new_row = row_8.replace("@", "")
print(new_row)
# //...9...//
row_9 = "h dfsk h mkmkm h"
first_h = row_9.find('h')
last_h = row_9.rfind('h')
new_row = row_9[:first_h + 1] + row_9[first_h + 1:last_h].replace("h", "H") + row_9[last_h:]
print(new_row)
# //...10...//
row_10 = "2691"
new_row = ''.join([row_10[i] for i in range(len(row_10)) if i % 3 != 0])
print(new_row)
# //...11...//
with open('english_ukrainian_dict.txt', 'r') as file:
    lines = file.readlines()
dictionary = {}
for line in lines:
    words = line.strip().split(' - ')
    eng_word = words[0]
    ukr_words = words[1].split(', ')
    for ukr_word in ukr_words:
        if ukr_word not in dictionary:
            dictionary[ukr_word] = [eng_word]
        else:
            dictionary[ukr_word].append(eng_word)
with open('ukrainian_dictionary.txt', 'w') as file:
    for ukr_word in sorted(dictionary.keys()):
        file.write(f"{ukr_word} - {', '.join(dictionary[ukr_word])}\n")
