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

if __name__ == "__main__":
    sol = Solution() # 1. 造对象
    
    # 2. 获取输入
    # input() 拿到的是字符串，必须 int() 转成整数传给函数
    str_in = input("请输入一个整数: ") 
    num = int(str_in)
    
    # 3. 计算并打印
    ans = sol.reverse(num)
    print(f"反转结果: {ans}")
