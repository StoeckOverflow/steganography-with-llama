import sys
import json
from src.hiders.probability_order_hider import ProbabilityOrderHider

def main():
    try:
        input_data = json.load(sys.stdin)
        secret, newsfeed = input_data["secret"], input_data["feed"]
        
        poh = ProbabilityOrderHider(seed=1337)
        poh.hide_secret(newsfeed, secret, "stdout")

    except Exception as e:
        print("Error:", str(e), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()