class Solution:
    def matchFollowing(self, s, p, ix):
        if len(s) - 1 < ix or len(p) - 1 < ix:
            return False
        if p[ix] != '*' and p[ix] != '.':
            if p[ix] == s[ix]:
                return True
            else:
                return False
        if p[ix] == '.':
            print(" . ")
        if p[ix] == '*':
            print(" * ")




    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        if p[0] == '*':
            return -1
        self.matchFollowing(s, p, 9)


s = "mississippi"
p = "mis*is*p*."
a = Solution()
a.isMatch(s, p)