from numpy.random import default_rng

def main():
    rng = default_rng()
    numbers = rng.choice(20, size=10, replace=False)
    print(numbers)

if __name__ == "__main__":
    main()