import sys
import json
import argparse
from src.hiders.probability_order_hider import ProbabilityOrderHider

def encode():
    try:
        input_data = json.load(sys.stdin)
        secret, newsfeed = input_data["secret"], input_data["feed"]

        poh = ProbabilityOrderHider(seed=1337)
        output_data = poh.hide_secret(newsfeed, secret)

        json.dump(output_data, sys.stdout)

    except Exception as e:
        print("Error during encoding:", str(e), file=sys.stderr)
        sys.exit(1)

def decode():
    try:
        input_data = json.load(sys.stdin)

        poh = ProbabilityOrderHider(seed=1337)
        revealed_secret = poh.reveal_secret(input_data)

        json.dump({'secret': revealed_secret}, sys.stdout)

    except Exception as e:
        print("Error during decoding:", str(e), file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Encode or Decode secrets using ProbabilityOrderHider")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--encode", action="store_true", help="Encode the secret from stdin")
    group.add_argument("--decode", action="store_true", help="Decode the secret from stdin")

    args = parser.parse_args()

    if args.encode:
        encode()
    elif args.decode:
        decode()

if __name__ == "__main__":
    main()