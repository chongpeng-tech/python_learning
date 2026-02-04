from sys import argv

argv = ['ex15.py', 'ex15.txt']
script, filename = argv



txt = open(filename)

print(f"Here's your file {filename}:")
print(txt.read())

print("Type the filename again:")

file_again = input("> ")

txt_again = open(file_again)

print(txt_again.read())