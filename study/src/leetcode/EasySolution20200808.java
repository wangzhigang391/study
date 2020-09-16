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
        // ����Ϊ1�����ܲ�ֳ��Ӵ�������ֱ�ӷ���false
        if (s.length() == 1) {
            return false;
        }
        // ���ȴ���1�����Բ�ֳ��Ӵ����ж��ַ������Ƿ������ַ�����ͬ
        if (s.split(String.valueOf(s.charAt(0))).length == 0) {
            return true;
        }
        // �ҳ���������ӣ���ֳ��Ӵ�
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
        //�ı�һ���������ĵض���״̬�������ظ�����
        board[i][j] = '.';
        boolean ret1 = dfs(board, i + 1, j, word, cur + 1);
        boolean ret2 = dfs(board, i - 1, j, word, cur + 1);
        boolean ret3 = dfs(board, i, j + 1, word, cur + 1);
        boolean ret4 = dfs(board, i, j - 1, word, cur + 1);
        //���������֧��DFS�Ѿ�����ˣ���Ҫ���ݻ�ԭ�ֳ�
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
        // 132�е�2
        int last = Integer.MIN_VALUE;
        // �����洢132�е�3
        Stack<Integer> sta = new Stack<>();
        if (nums.length < 3) {
            return false;
        }
        for (int i = n - 1; i >= 0; i--) {
            // ������132�е�1�򷵻���ȷֵ
            if (nums[i] < last) {
                return true;
            }
            // ����ǰֵ���ڻ����2�����2��2Ϊջ��С�ڵ�ǰֵ�����Ԫ�أ�
            while (!sta.isEmpty() && sta.peek() < nums[i]) {
                last = sta.pop();
            }
            // ����ǰֵѹ��ջ��
            sta.push(nums[i]);
        }
        return false;
    }


    public void quickSort(ListNode begin, ListNode end) {
        //�ж�Ϊ�գ��ж��ǲ���ֻ��һ���ڵ�
        if (begin == null || end == null || begin == end) {
            return;
        }
        //�ӵ�һ���ڵ�͵�һ���ڵ�ĺ���һ������
        ListNode first = begin;
        ListNode second = begin.next;
        int nMidValue = begin.val;
        //����������second�������
        while (second != end.next && second != null) {
            if (second.val < nMidValue) {
                first = first.next;
                //�ж�һ�£������������ȵ�һ����С�����û��ľ���
                if (first != second) {
                    int temp = first.val;
                    first.val = second.val;
                    second.val = temp;
                }
            }
            second = second.next;
        }
        //�жϣ���Щ����ǲ��û��ģ���������
        if (begin != first) {
            int temp = begin.val;
            begin.val = first.val;
            first.val = temp;
        }
        //ǰ���ֵݹ�
        quickSort(begin, first);
        //�󲿷ֵݹ�
        quickSort(first.next, end);
    }

    private static void quickSort(int[] arr, int leftIndex, int rightIndex) {
        if (leftIndex >= rightIndex) {
            return;
        }
        int left = leftIndex;
        int right = rightIndex;
        //������ĵ�һ��Ԫ����Ϊ��׼ֵ
        int key = arr[left];
        //���������߽���ɨ�裬ֱ��left = right
        while (left < right) {
            while (right > left && arr[right] >= key) {
                //��������ɨ�裬�ҵ���һ���Ȼ�׼ֵС��Ԫ��
                right--;
            }
            //�ҵ�����Ԫ�ؽ�arr[right]����arr[left]��
            arr[left] = arr[right];

            while (left < right && arr[left] <= key) {
                //��������ɨ�裬�ҵ���һ���Ȼ�׼ֵ���Ԫ��
                left++;
            }
            //�ҵ�����Ԫ�ؽ�arr[left]����arr[right]��
            arr[right] = arr[left];
        }
        //��׼ֵ��λ
        arr[left] = key;
        //�Ի�׼ֵ��ߵ�Ԫ�ؽ��еݹ�����
        quickSort(arr, leftIndex, left - 1);
        //�Ի�׼ֵ�ұߵ�Ԫ�ؽ��еݹ�����
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
        //���һ��Ԫ���ұ�û��Ԫ�أ����Կ϶�Ϊ-1
        result[size - 1] = -1;
        return result;
    }

    private int ret = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        /**
         ��������һ���ڵ�, �������·�������ýڵ�, ��ôֻ�������������:
         1. �����������������ɵĺ�·��ֵ�ϴ���Ǹ����ϸýڵ��ֵ���򸸽ڵ���ݹ������·��
         2. ���������������·����, ���ϸýڵ��ֵ���������յ����·��
         **/
        getMax(root);
        return ret;
    }

    private int getMax(TreeNode r) {
        if (r == null) {
            return 0;
        }
        // �������·����Ϊ����Ӧ����0��ʾ���·������������
        int left = Math.max(0, getMax(r.left));
        int right = Math.max(0, getMax(r.right));
        // �ж��ڸýڵ��������������·�����Ƿ���ڵ�ǰ���·����
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
     * num1�ĵ�iλ(��λ��0��ʼ)��num2�ĵ�jλ��˵Ľ���ڳ˻��е�λ����[i+j, i+j+1]
     * ��: 123 * 45,  123�ĵ�1λ 2 ��45�ĵ�0λ 4 �˻� 08 ����ڽ���ĵ�[1, 2]λ��
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
     * �������ǾͿ��Ե�������ÿһλ������˼���ѽ��������Ӧ��index��
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
                // �ȼӵ�λ�ж��Ƿ����µĽ�λ
                bitmul += mul[i + j + 1];
                mul[i + j] += bitmul / 10;
                mul[i + j + 1] = bitmul % 10;
            }
        }
        StringBuilder sb = new StringBuilder();
        int i = 0;
        // ȥ��ǰ��0
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
        //��ǰ�ڵ�ֵ��keyС������Ҫɾ����ǰ�ڵ����������key��Ӧ��ֵ������֤���������������ʲ���
        if (key < root.val) {
            root.left = deleteNode(root.left, key);
            //��ǰ�ڵ�ֵ��key������Ҫɾ����ǰ�ڵ����������key��Ӧ��ֵ������֤���������������ʲ���
        } else if (key > root.val) {
            root.right = deleteNode(root.right, key);
            //��ǰ�ڵ����key������Ҫɾ����ǰ�ڵ㣬����֤���������������ʲ���
        } else {
            //��ǰ�ڵ�û��������
            if (root.left == null) {
                return root.right;
                //��ǰ�ڵ�û��������
            } else if (root.right == null) {
                return root.left;
                //��ǰ�ڵ��������������������
            } else {
                TreeNode node = root.right;
                //�ҵ���ǰ�ڵ�����������ߵ�Ҷ�ӽ��
                while (node.left != null) {
                    node = node.left;
                }
                //��root���������ŵ�root�������������������Ҷ�ӽڵ����������
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
        // Java �ĵ��� Stack �ཨ��ʹ�� Deque ���� Stack��ע�⣺ֻʹ��ջ����ؽӿ�
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
        // key: Ԫ�أ�value: ���ֵĴ���
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            int times = map.getOrDefault(num, 0);
            map.put(num, times + 1);
        }
        // ����
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
                //����ǰ������һ���ڵ�������б�
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
            //ͳ����һ���ж��ٸ��ڵ�
            int count = queue.size();
            //������һ�е����нڵ�
            for (int i = 0; i < count; i++) {
                //poll�Ƴ�����ͷ��Ԫ�أ�������ͷ���Ƴ���β����ӣ�
                TreeNode node = queue.poll();
                //�ж��Ǵ������Ҵ�ӡ���Ǵ��������ӡ��
                if (leftToRight) {
                    level.add(node.val);
                } else {
                    level.add(0, node.val);
                }
                //�����ӽڵ������Ϊ�ջᱻ���뵽������
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
        while (!q.isEmpty()) {//ÿ��һ��ѭ��
            int size = q.size();
            List<Integer> list = new ArrayList<>();
            while (size > 0) {//һ���еĽڵ�
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
            res.add(0, list);//ǰ��
        }
        return res;
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) {
            return new ArrayList<List<Integer>>();
        }
        LinkedList<TreeNode> queue = new LinkedList<TreeNode>();
        //�����ڵ��������У�Ȼ�󲻶ϱ�������
        queue.add(root);
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        while (queue.size() > 0) {
            //��ȡ��ǰ���еĳ��ȣ���������൱�� ��ǰ��һ��Ľڵ����
            int size = queue.size();
            List<Integer> tmp = new ArrayList<Integer>();
            //�������е�Ԫ�ض��ó���(Ҳ���ǻ�ȡ��һ��Ľڵ�)���ŵ���ʱlist��
            //����ڵ����/��������Ϊ�գ�Ҳ���������
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
            //����ʱlist�������շ��ؽ����
            res.add(tmp);
        }
        return res;
    }

}
