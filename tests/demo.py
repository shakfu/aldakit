"""Demo of the aldapy parser."""

from aldapy.parser import parse


def main():
    # Phase 1 example
    source1 = """
    piano:
      (tempo 120)
      o4 c8 d e f | g4 a b > c
    """

    print("=== Phase 1: Basic melody ===")
    ast = parse(source1)
    print(ast)
    print()

    # Phase 2 example with variables, markers, voices, cram, and repeats
    source2 = """
    # Define a motif
    theme = c4 d e f

    piano:
      %intro
      theme*2

      V1: c4 e g > c
      V2: e4 g > c e
      V0:

      {c d e}2

      [c d e f]*4

      @intro
    """

    print("=== Phase 2: Variables, voices, cram, repeats ===")
    ast2 = parse(source2)
    print(ast2)


if __name__ == "__main__":
    main()
