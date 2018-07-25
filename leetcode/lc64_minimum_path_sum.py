class Solution:
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        n_row = len(grid)
        n_col = len(grid[0])

        result = [[0 for j in range(n_col)] for i in range(n_row)]
        for i in range(n_row):
            for j in range(n_col):
                if i == 0 and j == 0:
                    result[i][j] = grid[i][j]
                elif i == 0:
                    result[i][j] = grid[i][j] + result[i][j-1]
                elif j == 0:
                    result[i][j] = grid[i][j] + result[i-1][j]
                else:
                    result[i][j] = grid[i][j] + min(result[i-1][j], result[i][j-1])

        return result[n_row-1][n_col-1]


grid = [
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
obj = Solution()
print(obj.minPathSum(grid))