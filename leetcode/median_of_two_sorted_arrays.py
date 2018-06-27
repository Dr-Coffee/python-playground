class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        two arrays, move pivot
        """
        n_total = len(nums1) + len(nums2)
        if len(nums1) > len(nums2):
            list_long = nums1
            list_short = nums2
        else:
            list_long = nums2
            list_short = nums1
        ix_mid = int(len(list_short)/2)
        print(list_short)
        len = len(list_short)-ix_mid+1



a1 = [1., 2., 3., 4., 5.]
a2 = [4., 6., 7., 9.]

obj = Solution()
obj.findMedianSortedArrays(a1, a2)




