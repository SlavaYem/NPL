school = {
    '1a': 20,
    '1b': 23,
    '1c': 30,
    '2a': 19,
    '2b': 33,
}

print(school['1a'])
print(school['2a'])

school['2a'] = 23
school['1a'] = 19
school['1b'] = 21
print(school)

del school['1c']
print(school)
