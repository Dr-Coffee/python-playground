class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        n_large = 0
        n_small = 0
        v_smallest_in_large = None
        v_largest_in_small = None

        if len(nums1) > len(nums2):
            list_long = nums1
            list_short = nums2
        else:
            list_long = nums2
            list_short = nums1

        



a1 = [1., 2., 3., 4., 5.]
a2 = [4., 6., 7., 9.]

obj = Solution()
obj.findMedianSortedArrays(a1, a2)




