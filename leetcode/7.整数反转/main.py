class Solution:
    def reverse(self, x: int) -> int:
        sign = 1
        if x < 0:
            sign = -1
        #变绝对值
        y = abs(x)
        result = int(str(y)[::-1])

        result = result * sign
        if result < -2**31 or result > 2**31 - 1:
            return 0

        return result