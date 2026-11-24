

from contextlib import contextmanager


@contextmanager
def print_with_demo():
    print("1. With statement demo")
    yield #等价于 yield None, yield 产生的值会作为 with ... as xxx 中 xxx 的值

    print("3. Exiting the with statement")


def main():
    with print_with_demo():
        print("2. Inside the with statement")

if __name__ == "__main__":
    main()