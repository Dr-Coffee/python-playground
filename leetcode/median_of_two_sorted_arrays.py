class Solution(object):
    def fun(self, nums1, nums2, b1, e1, b2, e2, sign):
        if e1 - b1 < e2 - b2:
            shift = int((e1-b1)/2)
        else:
            shift = int((e2-b2)/2)
        if sign:
            ix1 = b1 + shift
            ix2 = e2 - shift
        else:
            ix1 = e1 - shift
            ix2 = b2 + shift
        print(nums1[b1:e1+1])
        print(nums2[b2:e2+1])
        if nums1[ix1] > nums2[ix2]:
            if nums1[ix1] <= nums2[ix2+1]:
                print(nums1[ix1])
                print(nums2[ix2])
            else:
                self.fun(nums1, nums2, b1, ix1, ix2, e2, False)
        else:
            if nums1[ix2] <= nums2[ix1+1]:
                print(nums1[ix1])
                print(nums2[ix2])
            else:
                self.fun(nums1, nums2, ix1, e1, b2, ix2, True)

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



import numpy as np

a1 = np.sort(np.random.randint(100, size=20)).tolist()
a2 = np.sort(np.random.randint(100, size=20)).tolist()
a = np.sort(np.array(a1+a2)).tolist()

print(a1)
print(a2)
print(a)

print(np.median(a))



obj = Solution()
obj.fun(a1, a2, 0, len(a1)-1, 0, len(a2)-1, True)
#obj.findMedianSortedArrays(a1, a2)




