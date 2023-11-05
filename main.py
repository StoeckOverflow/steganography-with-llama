import argparse
from src.models import DynamicPOE

def main():
    parser = argparse.ArgumentParser(description="Encode or Decode secrets using ProbabilityOrderHider")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--encode", action="store_true", help="Encode the secret from stdin")
    group.add_argument("--decode", action="store_true", help="Decode the secret from stdin")

    args = parser.parse_args()

    dpoe = DynamicPOE()

    if args.encode:
        dpoe.hide_interface()
    elif args.decode:
        dpoe.recover_interface()

if __name__ == "__main__":
    main()