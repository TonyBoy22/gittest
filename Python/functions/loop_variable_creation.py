'''
Create usable variables in a loop

https://python.plainenglish.io/how-to-dynamically-declare-variables-inside-a-loop-in-python-21e6880aaf8a

exec() statement
'''

dict={}

for i in range(5):
    key="x"+str(i)
    dict[key]=i
    # Se f
    exec(f'{key}={i}')
print(dict)

# for key, value in dict.items():
#     exec(f'{key}={value}')
print(x1)