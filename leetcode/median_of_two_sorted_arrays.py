class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        n_total = len(nums1) + len(nums2)
        if len(nums1) > len(nums2):
            list_long = nums1
            list_short = nums2
        else:
            list_long = nums2
            list_short = nums1
        #-----------------------------------------
        ix1 = int(len(nums1)/2)
        ix2 = int(len(nums2)/2)
        if nums1[ix1] <= nums2[ix2]:
            pass




a1 = [1., 2., 3., 4., 5.]
a2 = [3.5, 5.1, 6., 7., 9.]

obj = Solution()
obj.findMedianSortedArrays(a1, a2)




