package leetcode;

import java.util.*;

public class EasySolution20200808 {

    public static void main(String args[]) {


        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.left.left = new TreeNode(4);
        root.left.right = new TreeNode(5);
        root.right.left = new TreeNode(6);
        root.right.right = new TreeNode(7);

        ListNode head = new ListNode(1);
        head.next = new ListNode(2);
        head.next.next = new ListNode(3);
        head.next.next.next = new ListNode(4);
        head.next.next.next.next = new ListNode(5);
        head.next.next.next.next.next = new ListNode(6);
        head.next.next.next.next.next.next = new ListNode(7);

        int[] nums = {1, 1, 1, 2, 2, 3};
        int k = 2;
        topKFrequent(nums, k);

        String s = "aaabbaaa";
        repeatedSubstringPattern(s);
        String[] str = s.split("b",2);
        for(String st : str){
            System.out.println(st);
        }

    }


    public int lenLongestFibSubseq(int[] A) {
        int N = A.length;
        Map<Integer, Integer> index = new HashMap();
        for (int i = 0; i < N; ++i) {
            index.put(A[i], i);
        }
        Map<Integer, Integer> longest = new HashMap();
        int ans = 0;
        for (int k = 0; k < N; ++k)
            for (int j = 0; j < k; ++j) {
                int i = index.getOrDefault(A[k] - A[j], -1);
                if (i >= 0 && i < j) {
                    // Encoding tuple (i, j) as integer (i * N + j)
                    int cand = longest.getOrDefault(i * N + j, 2) + 1;
                    longest.put(j * N + k, cand);
                    ans = Math.max(ans, cand);
                }
            }

        return ans >= 3 ? ans : 0;
    }

    public static boolean repeatedSubstringPattern(String s) {
        if (s == null) {
            return false;
        }
        // 长度为1，不能拆分出子串，所以直接返回false
        if (s.length() == 1) {
            return false;
        }
        // 长度大于1，可以拆分出子串，判断字符串中是否所有字符都相同
        if (s.split(String.valueOf(s.charAt(0))).length == 0) {
            return true;
        }
        // 找长度最长的因子，拆分出子串
        for (int i = s.length() - 1;i >= 2;i--) {
            if (s.length() % i == 0) {
                String pattern = s.substring(0,i);
                String arr[] = s.split(pattern);
                if (arr.length == 0) {
                    return true;
                }
            }
        }
        return false;
    }

    public boolean exist(char[][] board, String word) {
        if(board == null || board.length == 0 || board[0].length == 0 ) {
            return false;
        }
        for(int i = 0; i < board.length; i++) {
            for(int j = 0; j < board[0].length; j++) {
                if(dfs(board, i, j, word, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean dfs(char[][] board, int i, int j, String word, int cur) {
        if(cur == word.length()) {
            return true;
        }
        if(i < 0 || i >= board.length || j < 0 || j >= board[0].length || board[i][j] != word.charAt(cur)) {
            return false;
        }
        char c = board[i][j];
        //改变一下搜索过的地儿的状态，避免重复搜索
        board[i][j] = '.';
        boolean ret1 = dfs(board, i + 1, j, word, cur + 1);
        boolean ret2 = dfs(board, i - 1, j, word, cur + 1);
        boolean ret3 = dfs(board, i, j + 1, word, cur + 1);
        boolean ret4 = dfs(board, i, j - 1, word, cur + 1);
        //对于这个分支的DFS已经完成了，需要回溯还原现场
        board[i][j] = c;
        return ret1 || ret2 || ret3 || ret4;
    }

    public int canCompleteCircuit(int[] gas, int[] cost) {
        int n = gas.length;
        int total_tank = 0;
        int curr_tank = 0;
        int starting_station = 0;
        for (int i = 0; i < n; ++i) {
            total_tank += gas[i] - cost[i];
            curr_tank += gas[i] - cost[i];
            // If one couldn't get here,
            if (curr_tank < 0) {
                // Pick up the next station as the starting one.
                starting_station = i + 1;
                // Start with an empty tank.
                curr_tank = 0;
            }
        }
        return total_tank >= 0 ? starting_station : -1;
    }

    public boolean find132pattern(int[] nums) {
        int n = nums.length;
        // 132中的2
        int last = Integer.MIN_VALUE;
        // 用来存储132中的3
        Stack<Integer> sta = new Stack<>();
        if (nums.length < 3) {
            return false;
        }
        for (int i = n - 1; i >= 0; i--) {
            // 若出现132中的1则返回正确值
            if (nums[i] < last) {
                return true;
            }
            // 若当前值大于或等于2则更新2（2为栈中小于当前值的最大元素）
            while (!sta.isEmpty() && sta.peek() < nums[i]) {
                last = sta.pop();
            }
            // 将当前值压入栈中
            sta.push(nums[i]);
        }
        return false;
    }


    public void quickSort(ListNode begin, ListNode end) {
        //判断为空，判断是不是只有一个节点
        if (begin == null || end == null || begin == end) {
            return;
        }
        //从第一个节点和第一个节点的后面一个几点
        ListNode first = begin;
        ListNode second = begin.next;
        int nMidValue = begin.val;
        //结束条件，second到最后了
        while (second != end.next && second != null) {
            if (second.val < nMidValue) {
                first = first.next;
                //判断一下，避免后面的数比第一个数小，不用换的局面
                if (first != second) {
                    int temp = first.val;
                    first.val = second.val;
                    second.val = temp;
                }
            }
            second = second.next;
        }
        //判断，有些情况是不用换的，提升性能
        if (begin != first) {
            int temp = begin.val;
            begin.val = first.val;
            first.val = temp;
        }
        //前部分递归
        quickSort(begin, first);
        //后部分递归
        quickSort(first.next, end);
    }

    private static void quickSort(int[] arr, int leftIndex, int rightIndex) {
        if (leftIndex >= rightIndex) {
            return;
        }
        int left = leftIndex;
        int right = rightIndex;
        //待排序的第一个元素作为基准值
        int key = arr[left];
        //从左右两边交替扫描，直到left = right
        while (left < right) {
            while (right > left && arr[right] >= key) {
                //从右往左扫描，找到第一个比基准值小的元素
                right--;
            }
            //找到这种元素将arr[right]放入arr[left]中
            arr[left] = arr[right];

            while (left < right && arr[left] <= key) {
                //从左往右扫描，找到第一个比基准值大的元素
                left++;
            }
            //找到这种元素将arr[left]放入arr[right]中
            arr[right] = arr[left];
        }
        //基准值归位
        arr[left] = key;
        //对基准值左边的元素进行递归排序
        quickSort(arr, leftIndex, left - 1);
        //对基准值右边的元素进行递归排序。
        quickSort(arr, right + 1, rightIndex);
    }

    public int[] findMaxRight(int[] array) {
        if (array == null) {
            return array;
        }
        int size = array.length;
        int[] result = new int[size];
        for (int i = 0; i < size - 1; i++) {
            for (int j = i + 1; j < size; j++) {
                if (array[j] > array[i]) {
                    result[i] = array[j];
                    break;
                }
            }
        }
        //最后一个元素右边没有元素，所以肯定为-1
        result[size - 1] = -1;
        return result;
    }

    private int ret = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        /**
         对于任意一个节点, 如果最大和路径包含该节点, 那么只可能是两种情况:
         1. 其左右子树中所构成的和路径值较大的那个加上该节点的值后向父节点回溯构成最大路径
         2. 左右子树都在最大路径中, 加上该节点的值构成了最终的最大路径
         **/
        getMax(root);
        return ret;
    }

    private int getMax(TreeNode r) {
        if (r == null) {
            return 0;
        }
        // 如果子树路径和为负则应当置0表示最大路径不包含子树
        int left = Math.max(0, getMax(r.left));
        int right = Math.max(0, getMax(r.right));
        // 判断在该节点包含左右子树的路径和是否大于当前最大路径和
        ret = Math.max(ret, r.val + left + right);
        return Math.max(left, right) + r.val;
    }

    public int lengthOfLongestSubstring(String s) {
        int res = 0;
        Set<Character> set = new HashSet<>();
        for (int l = 0, r = 0; r < s.length(); r++) {
            char c = s.charAt(r);
            while (set.contains(c)) {
                set.remove(s.charAt(l++));
            }
            set.add(c);
            res = Math.max(res, r - l + 1);
        }
        return res;
    }

    /**
     * num1的第i位(高位从0开始)和num2的第j位相乘的结果在乘积中的位置是[i+j, i+j+1]
     * 例: 123 * 45,  123的第1位 2 和45的第0位 4 乘积 08 存放在结果的第[1, 2]位中
     * index:    0 1 2 3 4
     * <p>
     * 1 2 3
     * 4 5
     * ---------
     * 1 5
     * 1 0
     * 0 5
     * ---------
     * 0 6 1 5
     * 1 2
     * 0 8
     * 0 4
     * ---------
     * 0 5 5 3 5
     * 这样我们就可以单独都对每一位进行相乘计算把结果存入相应的index中
     **/
    public String multiply(String num1, String num2) {
        int n1 = num1.length() - 1;
        int n2 = num2.length() - 1;
        if (n1 < 0 || n2 < 0) {
            return "";
        }
        int[] mul = new int[n1 + n2 + 2];
        for (int i = n1; i >= 0; --i) {
            for (int j = n2; j >= 0; --j) {
                int bitmul = (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
                // 先加低位判断是否有新的进位
                bitmul += mul[i + j + 1];
                mul[i + j] += bitmul / 10;
                mul[i + j + 1] = bitmul % 10;
            }
        }
        StringBuilder sb = new StringBuilder();
        int i = 0;
        // 去掉前导0
        while (i < mul.length - 1 && mul[i] == 0) {
            i++;
        }
        for (; i < mul.length; ++i) {
            sb.append(mul[i]);
        }
        return sb.toString();
    }

    public int kthSmallest(TreeNode root, int k) {
        List<Integer> list = new ArrayList<>();
        inorder(root, list);
        return list.get(k - 1);
    }

    public void inorder(TreeNode node, List list) {
        if (node != null) {
            inorder(node.left, list);
            list.add(node.val);
            inorder(node.right, list);
        }
    }

    public TreeNode deleteNode(TreeNode root, int key) {
        if (root == null) {
            return null;
        }
        //当前节点值比key小，则需要删除当前节点的左子树中key对应的值，并保证二叉搜索树的性质不变
        if (key < root.val) {
            root.left = deleteNode(root.left, key);
            //当前节点值比key大，则需要删除当前节点的右子树中key对应的值，并保证二叉搜索树的性质不变
        } else if (key > root.val) {
            root.right = deleteNode(root.right, key);
            //当前节点等于key，则需要删除当前节点，并保证二叉搜索树的性质不变
        } else {
            //当前节点没有左子树
            if (root.left == null) {
                return root.right;
                //当前节点没有右子树
            } else if (root.right == null) {
                return root.left;
                //当前节点既有左子树又有右子树
            } else {
                TreeNode node = root.right;
                //找到当前节点右子树最左边的叶子结点
                while (node.left != null) {
                    node = node.left;
                }
                //将root的左子树放到root的右子树的最下面的左叶子节点的左子树上
                node.left = root.left;
                return root.right;
            }
        }
        return root;
    }

    public int sumNumbers(TreeNode root) {
        return helper(root, 0);
    }

    public int helper(TreeNode root, int i) {
        if (root == null) {
            return 0;
        }
        int temp = i * 10 + root.val;
        if (root.left == null && root.right == null) {
            return temp;
        }
        return helper(root.left, temp) + helper(root.right, temp);
    }

    public List<List<Integer>> dfs(TreeNode root, int sum) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        // Java 文档中 Stack 类建议使用 Deque 代替 Stack，注意：只使用栈的相关接口
        Deque<Integer> path = new ArrayDeque<>();
        dfs(root, sum, path, res);
        return res;
    }

    private void dfs(TreeNode node, int sum, Deque<Integer> path, List<List<Integer>> res) {
        if (node == null) {
            return;
        }
        if (node.val == sum && node.left == null && node.right == null) {
            path.addLast(node.val);
            res.add(new ArrayList<>(path));
            path.removeLast();
            return;
        }
        path.addLast(node.val);
        dfs(node.left, sum - node.val, path, res);
        dfs(node.right, sum - node.val, path, res);
        path.removeLast();
    }

    public static int[] topKFrequent(int[] nums, int k) {
        // key: 元素，value: 出现的次数
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            int times = map.getOrDefault(num, 0);
            map.put(num, times + 1);
        }
        // 最大堆
        Queue<Integer> pq = new PriorityQueue<>((o1, o2) -> (map.get(o2) - map.get(o1)));
        for (int key : map.keySet()) {
            pq.add(key);
        }
        int[] ans = new int[k];
        int index = 0;
        while (index < k) {
            ans[index++] = pq.poll();
        }
        return ans;
    }

    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
                //将当前层的最后一个节点放入结果列表
                if (i == size - 1) {
                    res.add(node.val);
                }
            }
        }
        return res;
    }

    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        boolean leftToRight = true;
        while (!queue.isEmpty()) {
            List<Integer> level = new ArrayList<>();
            //统计这一行有多少个节点
            int count = queue.size();
            //遍历这一行的所有节点
            for (int i = 0; i < count; i++) {
                //poll移除队列头部元素（队列在头部移除，尾部添加）
                TreeNode node = queue.poll();
                //判断是从左往右打印还是从右往左打印。
                if (leftToRight) {
                    level.add(node.val);
                } else {
                    level.add(0, node.val);
                }
                //左右子节点如果不为空会被加入到队列中
                if (node.left != null) {
                    queue.add(node.left);
                }
                if (node.right != null) {
                    queue.add(node.right);
                }
            }
            res.add(level);
            leftToRight = !leftToRight;
        }
        return res;
    }

    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        //BFS
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {//每层一个循环
            int size = q.size();
            List<Integer> list = new ArrayList<>();
            while (size > 0) {//一层中的节点
                TreeNode node = q.poll();
                list.add(node.val);
                if (node.left != null) {
                    q.offer(node.left);
                }
                if (node.right != null) {
                    q.offer(node.right);
                }
                size--;
            }
            res.add(0, list);//前插
        }
        return res;
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) {
            return new ArrayList<List<Integer>>();
        }
        LinkedList<TreeNode> queue = new LinkedList<TreeNode>();
        //将根节点放入队列中，然后不断遍历队列
        queue.add(root);
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        while (queue.size() > 0) {
            //获取当前队列的长度，这个长度相当于 当前这一层的节点个数
            int size = queue.size();
            List<Integer> tmp = new ArrayList<Integer>();
            //将队列中的元素都拿出来(也就是获取这一层的节点)，放到临时list中
            //如果节点的左/右子树不为空，也放入队列中
            for (int i = 0; i < size; ++i) {
                TreeNode t = queue.remove();
                tmp.add(t.val);
                if (t.left != null) {
                    queue.add(t.left);
                }
                if (t.right != null) {
                    queue.add(t.right);
                }
            }
            //将临时list加入最终返回结果中
            res.add(tmp);
        }
        return res;
    }

}
