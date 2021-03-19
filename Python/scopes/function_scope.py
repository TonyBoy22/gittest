# Vérifier si on doit passer les fonctions en argument
# main, a et b sont dans le même scope global, mais pas appelées dans les mêmes scope


def function_a(x1: int, x2: int):
    # Some stuff
    return x1+x2

def function_b(x1: int, x2: int):
    # Called in main, needs function_a
    # Passes arguments to function b
    new_sum = function_a(x1+2, x2+3)
    return new_sum

def main():
    new_sum = function_b(2, 3)
    print(new_sum)

if __name__ == "__main__":
    main()
    # Conclu, si deux fonctions sont créées dans le global scope, on peut les appeler
    # dans des scopes plus restreints.