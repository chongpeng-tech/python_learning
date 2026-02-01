class Solution:
    def isPalindrome(self, x:int) -> bool:
        #先将字符转换为str
        s = str(x)
        #将字符串反转
        reversed_s = s[::-1]

        return s == reversed_s

if __name__ == "__main__":
    sol = Solution()
    print("===回文数测试程序  (ctrl+c退出) ===")
    while True:
        