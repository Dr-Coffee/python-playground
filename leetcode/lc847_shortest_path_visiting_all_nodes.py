class Solution:
    parent = dict()
    rank = dict()

    def make_set(self, input):
        self.parent.update({input:input})

    def find_set(self, input):
        pass

    def union(self, input):
        pass

    def shortestPathLength(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: int
        """
        print(graph)

graph = [[1,2,3],[0],[0],[0]]
obj = Solution()
obj.shortestPathLength(graph)