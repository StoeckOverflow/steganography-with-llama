from decimal import Decimal, getcontext


__all__ = [
    "Dec2BinConverter"
]

class Dec2BinConverter:
    @staticmethod
    def get_bin_from_decimal(dec: Decimal, bits_per_token: int = 3) -> tuple[str, str]:
        s = format(dec)
        s = s[s.find(".")+1:]
        for i, ch in enumerate(s):
            if ch != "0":
                break
        mantissa = bin(int(s[i:]))[2:]
        exp = bin(i)[2:]
        prec = bin(getcontext().prec)[2:]

        pad_with_0 = lambda x: "0"*(bits_per_token - (len(x) % bits_per_token)) + x if len(x) % bits_per_token else x 

        mantissa, exp, prec = pad_with_0(mantissa), pad_with_0(exp), pad_with_0(prec)
        
        return mantissa, exp, prec

    @staticmethod
    def get_decimal_from_bin(mantissa: str, exp: str, prec: str) -> Decimal:
        prec = int(prec, 2)
        getcontext().prec = prec
        s = "0." + int(exp, 2) * "0" + str(int(mantissa, 2))
        return Decimal(s)