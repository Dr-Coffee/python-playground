class Solution:
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 0:
            return 0
        list_n = []
        for i, v in enumerate(nums):
            if i == 0:
                list_n.append(1)
            else:
                n = 1
                for j in range(i):
                    if nums[i] > nums[j]:
                        if list_n[j] + 1 > n:
                            n = list_n[j] + 1
                list_n.append(n)
        return max(list_n)

nums = [1,3,6,7,9,4,10,5,6]
obj = Solution()
result = obj.lengthOfLIS(nums)
print(result)