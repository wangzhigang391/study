package leetcode;

import java.util.*;

public class EasySolution20200717 {

    public static void main(String args[]) {
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.left.left = new TreeNode(4);
        root.left.right = new TreeNode(5);
        root.right.left = new TreeNode(6);
        root.right.right = new TreeNode(7);
        maxDepth(root);

        ListNode head = new ListNode(1);
        head.next = new ListNode(2);
        head.next.next = new ListNode(3);
        head.next.next.next = new ListNode(4);
        head.next.next.next.next = new ListNode(5);
        head.next.next.next.next.next = new ListNode(6);
        head.next.next.next.next.next.next = new ListNode(7);

        int[] nums = {1, 2, 3, 4, 5, 62, 1, 2, 3};
        reverseBetween01(head, 2, 4);

    }

    public List < Integer > inorderTraversal(TreeNode root) {
        List < Integer > res = new ArrayList < > ();
        helper(root, res);
        return res;
    }

    public void helper(TreeNode root, List < Integer > res) {
        if (root != null) {
            if (root.left != null) {
                helper(root.left, res);
            }
            res.add(root.val);
            if (root.right != null) {
                helper(root.right, res);
            }
        }
    }

    public List<Integer> preorderTraversal(TreeNode root) {
        LinkedList<TreeNode> stack = new LinkedList<>();
        LinkedList<Integer> output = new LinkedList<>();
        if (root == null) {
            return output;
        }
        stack.add(root);
        while (!stack.isEmpty()) {
            TreeNode node = stack.pollLast();
            output.add(node.val);
            if (node.right != null) {
                stack.add(node.right);
            }
            if (node.left != null) {
                stack.add(node.left);
            }
        }
        return output;
    }

    public String simplifyPath(String path) {
        Stack<String> stack = new Stack<>();
        // ���Ƚ��ַ����� ��/�� �ָ��洢���µ��ַ����� str ��
        String[] str = path.split("/");
        for (String s : str) {
            // �������ǿ�,�ҷ��ʵ����� ��..�� ��˵��Ҫ������һ��,Ҫ����ǰԪ�س�ջ
            if ( s.equals("..") ) {
                // �����õ���ǿ�� for ѭ������ͬʱ�жϣ���Ҫ�ٴ��п�
                // ����ͨ for ѭ�����д��( !stack.isEmpty() && s.equals("..") )
                if ( !stack.isEmpty() ) {
                    stack.pop();
                }
                // �������ǿղ��ҵ�ǰԪ�ز��� ��.�� ˵����ǰԪ����·����Ϣ��Ҫ��ջ
            } else if ( !s.equals("") && !s.equals(".") ) {
                stack.push(s);
            }
        }
        // ���ջ��û��Ԫ��˵��û��·����Ϣ������ ��/�� ����
        if ( stack.isEmpty() ) {
            return "/";
        }
        // �����õ� StringBuilder �����ַ�����Ч�ʸ�
        StringBuilder ans = new StringBuilder();
        for (int i = 0; i < stack.size(); i++) {
            ans.append( "/" + stack.get(i) );
        }
        return ans.toString();
    }

    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        for (String s : tokens) {
            if (s.equals("+")) {
                stack.push(stack.pop() + stack.pop());
            } else if (s.equals("-")) {
                stack.push(-stack.pop() + stack.pop());
            } else if (s.equals("*")) {
                stack.push(stack.pop() * stack.pop());
            } else if (s.equals("/")) {
                int num1 = stack.pop();
                stack.push(stack.pop() / num1);
            } else {
                stack.push(Integer.parseInt(s));
            }
        }
        return stack.pop();
    }

    public boolean isPalindrome(ListNode head) {
        List<Integer> vals = new ArrayList<>();

        // Convert LinkedList into ArrayList.
        ListNode currentNode = head;
        while (currentNode != null) {
            vals.add(currentNode.val);
            currentNode = currentNode.next;
        }

        // Use two-pointer technique to check for palindrome.
        int front = 0;
        int back = vals.size() - 1;
        while (front < back) {
            // Note that we must use ! .equals instead of !=
            // because we are comparing Integer, not int.
            if (!vals.get(front).equals(vals.get(back))) {
                return false;
            }
            front++;
            back--;
        }
        return true;
    }

    public ListNode rotateRight(ListNode head, int k) {
        if (head == null || k == 0) {
            return head;
        }
        ListNode cursor = head;
        ListNode tail = null;//βָ��
        int length = 1;
        while (cursor.next != null)//ѭ�� �õ��ܳ���
        {
            cursor = cursor.next;
            length++;
        }
        int loop = length - (k % length);//�õ�ѭ���Ĵ���
        tail = cursor;//ָ��β���
        cursor.next = head;//�ĳ�ѭ������
        cursor = head;//ָ��ͷ���
        for (int i = 0; i < loop; i++) {//��ʼѭ��
            cursor = cursor.next;
            tail = tail.next;
        }
        tail.next = null;//�ĳɵ�����
        return cursor;//���ص�ǰͷ
    }

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummyHead = new ListNode(0);
        ListNode tail = dummyHead;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                tail.next = l1;
                l1 = l1.next;
            } else {
                tail.next = l2;
                l2 = l2.next;
            }
            tail = tail.next;
        }

        tail.next = l1 == null ? l2 : l1;

        return dummyHead.next;
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        Stack<Integer> stack1 = new Stack<>();
        Stack<Integer> stack2 = new Stack<>();
        while (l1 != null) {
            stack1.push(l1.val);
            l1 = l1.next;
        }
        while (l2 != null) {
            stack2.push(l2.val);
            l2 = l2.next;
        }

        int carry = 0;
        ListNode head = null;
        while (!stack1.isEmpty() || !stack2.isEmpty() || carry > 0) {
            int sum = carry;
            sum += stack1.isEmpty() ? 0 : stack1.pop();
            sum += stack2.isEmpty() ? 0 : stack2.pop();
            ListNode node = new ListNode(sum % 10);
            node.next = head;
            head = node;
            carry = sum / 10;
        }
        return head;
    }

    public ListNode oddEvenList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        // head Ϊ������ͷ��㣬o Ϊ������β�ڵ�
        ListNode o = head;
        // p Ϊż����ͷ���
        ListNode p = head.next;
        // e Ϊż����β�ڵ�
        ListNode e = p;
        while (o.next != null && e.next != null) {
            o.next = e.next;
            o = o.next;
            e.next = o.next;
            e = e.next;
        }
        o.next = p;
        return head;
    }

    public ListNode partition(ListNode head, int x) {
        //����less��¼С��x��Ԫ��
        ListNode less = new ListNode(0);
        ListNode ptr1 = less;
        //����greatOrEqual��¼���ڵ���x��Ԫ��
        ListNode greatOrEqual = new ListNode(0);
        ListNode ptr2 = greatOrEqual;
        while (head != null) {
            if (head.val < x) {
                ptr1.next = new ListNode(head.val);
                ptr1 = ptr1.next;
            } else {
                ptr2.next = new ListNode(head.val);
                ptr2 = ptr2.next;
            }
            head = head.next;
        }
        //ƴ����������
        ptr1.next = greatOrEqual.next;
        return less.next;
    }

    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if (p == null || q == null) {
            return false;
        }
        if (p.val != q.val) {
            return false;
        }
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

    public static ListNode reverseBetween01(ListNode head, int m, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode pre = dummy;
        for (int i = 1; i < m; i++) {
            pre = pre.next;
        }
        head = pre.next;
        for (int i = m; i < n; i++) {
            ListNode nex = head.next;
            head.next = nex.next;
            nex.next = pre.next;
            pre.next = nex;
        }
        return dummy.next;
    }

    public static boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        TreeSet<Integer> set = new TreeSet<>();
        for (int i = 0; i < nums.length; ++i) {
            // Find the successor of current element
            Integer s = set.ceiling(nums[i]);
            if (s != null && s <= nums[i] + t) {
                return true;
            }
            // Find the predecessor of current element
            Integer g = set.floor(nums[i]);
            if (g != null && nums[i] <= g + t) {
                return true;
            }
            set.add(nums[i]);
            if (set.size() > k) {
                set.remove(nums[i - k]);
            }
        }
        return false;
    }

    public int numberOfBoomerangs(int[][] points) {
        HashMap<Double, Integer> hashMap = new HashMap<>();
        int count = 0;
        for (int i = 0; i < points.length; i++) {
            hashMap.clear();
            for (int j = 0; j < points.length; j++) {
                if (i == j) {
                    continue;
                }
                double distance = Math.pow(points[i][0] - points[j][0], 2) + Math.pow(points[i][1] - points[j][1], 2);
                count += hashMap.getOrDefault(distance, 0) * 2;
                hashMap.put(distance, hashMap.getOrDefault(distance, 0) + 1);
            }
        }
        return count;
    }

    //����
    public static List<List<String>> groupAnagrams(String[] strs) {
        if (strs.length == 0) {
            return new ArrayList();
        }
        Map<String, List<String>> ans = new HashMap<>();
        for (String s : strs) {
            char[] ca = s.toCharArray();
            Arrays.sort(ca);
            String key = String.valueOf(ca);
            if (!ans.containsKey(key)) {
                ans.put(key, new ArrayList());
            }
            ans.get(key).add(s);
        }
        return new ArrayList(ans.values());
    }

    public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
        Map<Integer, Integer> map = new HashMap<>();
        int res = 0;
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < B.length; j++) {
                int sumAB = A[i] + B[j];
                if (map.containsKey(sumAB)) {
                    map.put(sumAB, map.get(sumAB) + 1);
                } else {
                    map.put(sumAB, 1);
                }
            }
        }

        for (int i = 0; i < C.length; i++) {
            for (int j = 0; j < D.length; j++) {
                int sumCD = -(C[i] + D[j]);
                if (map.containsKey(sumCD)) {
                    res += map.get(sumCD);
                }
            }
        }
        return res;
    }

    public static int maxDepth(TreeNode root) {
        int res = 0;
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        // ��������
        queue.add(root);
        while (!queue.isEmpty()) {
            int n = queue.size();
            for (int i = 0; i < n; i++) {
                TreeNode curNode = queue.poll();
                if (curNode.left != null) {
                    queue.add(curNode.left);
                }
                if (curNode.right != null) {
                    queue.add(curNode.right);
                }
            }
            res++;
        }
        return res;
    }

    public List<List<Integer>> threeSum(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        // ö�� a
        for (int first = 1; first < n; first++) {
            // ��Ҫ����һ��ö�ٵ�������ͬ
            if (first > 0 && nums[first] == nums[first - 1]) {
                continue;
            }
            // c ��Ӧ��ָ���ʼָ����������Ҷ�
            int third = n - 1;
            int target = -nums[first];
            // ö�� b
            for (int second = first + 1; second < n; second++) {
                // ��Ҫ����һ��ö�ٵ�������ͬ
                if (second > first + 1 && nums[second] == nums[second - 1]) {
                    continue;
                }
                // ��Ҫ��֤ b ��ָ���� c ��ָ������
                while (second < third && nums[second] + nums[third] > target) {
                    --third;
                }
                // ���ָ���غϣ����� b ����������
                // �Ͳ��������� a+b+c=0 ���� b<c �� c �ˣ������˳�ѭ��
                if (second == third) {
                    break;
                }
                if (nums[second] + nums[third] == target) {
                    List<Integer> list = new ArrayList<Integer>();
                    list.add(nums[first]);
                    list.add(nums[second]);
                    list.add(nums[third]);
                    ans.add(list);
                }
            }
        }
        return ans;
    }

    public static String frequencySort(String s) {
        Map<Character, Integer> map = new HashMap<>();
        for (char chr : s.toCharArray()) {
            map.put(chr, map.getOrDefault(chr, 0) + 1);
        }

        PriorityQueue<Map.Entry<Character, Integer>> maxHeap = new PriorityQueue<>(
                (e1, e2) -> e2.getValue() - e1.getValue());

        maxHeap.addAll(map.entrySet());

        StringBuilder sortedString = new StringBuilder(s.length());
        while (!maxHeap.isEmpty()) {
            Map.Entry<Character, Integer> entry = maxHeap.poll();
            for (int i = 0; i < entry.getValue(); i++) {
                sortedString.append(entry.getKey());
            }
        }
        return sortedString.toString();
    }

    private static boolean isIsomorphicHelper(String s, String t) {
        int n = s.length();
        HashMap<Character, Character> map = new HashMap<>();
        for (int i = 0; i < n; i++) {
            char c1 = s.charAt(i);
            char c2 = t.charAt(i);
            if (map.containsKey(c1)) {
                if (map.get(c1) != c2) {
                    return false;
                }
            } else {
                if (map.containsValue(c2)) {
                    return false;
                }
                map.put(c1, c2);
            }
        }
        return true;
    }

    public boolean wordPattern(String pattern, String str) {
        //�Կո�ָ�str
        String[] s = str.split(" ");
        //���û��ȫ���ɶԵ�ӳ���򷵻�false
        if (s.length != pattern.length()) {
            return false;
        }
        //���ӳ��
        Map<Character, String> map = new HashMap<>();
        for (int i = 0; i < pattern.length(); i++) {
            //1. û��ӳ��ʱִ��
            if (!map.containsKey(pattern.charAt(i))) {
                //2. û��ӳ��������s[i]�ѱ�ʹ�ã���ƥ�䷵��false
                if (map.containsValue(s[i])) {
                    return false;
                }
                //3. ����ӳ��
                map.put(pattern.charAt(i), s[i]);
            } else {
                //��ǰ�ַ�����ӳ�䲻ƥ��,����false
                if (!map.get(pattern.charAt(i)).equals(s[i])) {
                    return false;
                }
            }
        }
        return true;
    }

    public static boolean isHappy(int n) {
        Set<Integer> set = new HashSet<>();
        int m = 0;
        while (true) {
            while (n != 0) {
                m += Math.pow(n % 10, 2);
                n /= 10;
            }
            if (m == 1) {
                return true;
            }
            if (set.contains(m)) {
                return false;
            } else {
                set.add(m);
                n = m;
                m = 0;
            }
        }
    }

    public boolean isSubsequence(String s, String t) {
        int n = s.length(), m = t.length();
        int i = 0, j = 0;
        while (i < n && j < m) {
            if (s.charAt(i) == t.charAt(j)) {
                i++;
            }
            j++;
        }
        return i == n;
    }

    public static int[] intersect(int[] nums1, int[] nums2) {
        Map<Integer, Integer> map = new TreeMap<>();
        for (int num : nums1) {
            if (!map.containsKey(num)) {
                map.put(num, 1);
            } else {
                map.put(num, map.get(num) + 1);
            }
        }
        List<Integer> list = new ArrayList<>();
        for (int num : nums2) {
            if (map.containsKey(num)) {
                list.add(num);
                map.put(num, map.get(num) - 1);
                if (map.get(num) == 0) {
                    map.remove(num);
                }
            }
        }
        int[] res = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            res[i] = list.get(i);
        }
        return res;
    }

    public static List<Integer> findAnagrams(String s, String p) {
        char[] arrS = s.toCharArray();
        char[] arrP = p.toCharArray();
        // ������󷵻صĽ��
        List<Integer> ans = new ArrayList<>();
        // ����һ�� needs �������� arrP �а���Ԫ�صĸ���
        int[] needs = new int[26];
        // ����һ�� window �������������������Ƿ��� arrP �е�Ԫ�أ�����¼���ֵĸ���
        int[] window = new int[26];
        // �Ƚ� arrP �е�Ԫ�ر��浽 needs ������
        for (int i = 0; i < arrP.length; i++) {
            needs[arrP[i] - 'a'] += 1;
        }
        // ���廬�����ڵ�����
        int left = 0;
        int right = 0;
        // �Ҵ��ڿ�ʼ���������ƶ�
        while (right < arrS.length) {
            int curR = arrS[right] - 'a';
            right++;
            // ���Ҵ��ڵ�ǰ���ʵ���Ԫ�� curR ������ 1
            window[curR] += 1;
            // �� window ������ curR �� needs �����ж�ӦԪ�صĸ���Ҫ���ʱ��͸��ƶ��󴰿�ָ��
            while (window[curR] > needs[curR]) {
                int curL = arrS[left] - 'a';
                left++;
                // ���󴰿ڵ�ǰ���ʵ���Ԫ�� curL ������ 1
                window[curL] -= 1;
            }
            // ���ｫ���з���Ҫ����󴰿��������뵽�˽��ս���� List ��
            if (right - left == arrP.length) {
                ans.add(left);
            }
        }
        return ans;
    }

    public int minSubArrayLen(int s, int[] nums) {
        int n = nums.length;
        if (n == 0) {
            return 0;
        }
        int ans = Integer.MAX_VALUE;
        int start = 0, end = 0;
        int sum = 0;
        while (end < n) {
            sum += nums[end];
            while (sum >= s) {
                ans = Math.min(ans, end - start + 1);
                sum -= nums[start];
                start++;
            }
            end++;
        }
        return ans == Integer.MAX_VALUE ? 0 : ans;
    }

    public int findKthLargest(int[] nums, int k) {
        int len = nums.length;
        // ʹ��һ������ k ��Ԫ�ص���С��
        PriorityQueue<Integer> minHeap = new PriorityQueue<>(k, (a, b) -> a - b);
        for (int i = 0; i < k; i++) {
            minHeap.add(nums[i]);
        }
        for (int i = k; i < len; i++) {
            // ��һ�ۣ����ó�����Ϊ�п���û�б�Ҫ�滻
            Integer topEle = minHeap.peek();
            // ֻҪ��ǰ������Ԫ�رȶѶ�Ԫ�ش󣬶Ѷ�������������Ԫ�ؽ�ȥ
            if (nums[i] > topEle) {
                minHeap.poll();
                minHeap.add(nums[i]);
            }
        }
        return minHeap.peek();
    }

    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int p1 = m - 1, p2 = n - 1, p3 = m + n - 1;
        while (p2 >= 0) {
            if (p1 >= 0 && nums1[p1] > nums2[p2]) {
                nums1[p3--] = nums1[p1--];
            } else {
                nums1[p3--] = nums2[p2--];
            }
        }
    }

    public void sortColors(int[] nums) {
        int len = nums.length;
        if (len < 2) {
            return;
        }
        int zero = 0;
        int two = len;
        int i = 0;
        while (i < two) {
            if (nums[i] == 0) {
                swap1(nums, i, zero);
                zero++;
                i++;
            } else if (nums[i] == 1) {
                i++;
            } else {
                two--;
                swap1(nums, i, two);
            }
        }
    }

    private void swap1(int[] nums, int index1, int index2) {
        int temp = nums[index1];
        nums[index1] = nums[index2];
        nums[index2] = temp;
    }

    public int removeDuplicates(int[] nums) {
        // 1.����[0,index)��ʾ�޸ĺ������ �����ظ���Ԫ��
        int index = 0;
        // 2.��ֹ����
        for (int i = 0; i < nums.length; i++) {
            // 3.ָ���˶�����
            if (index < 2 || nums[index - 2] != nums[i]) {
                nums[index] = nums[i];
                index++;
            }
        }
        // 4.���ݶ���ȷ������ֵ
        return index;
    }

    public int removeElement(int[] nums, int val) {
        int i = 0;
        for (int j = 0; j < nums.length; j++) {
            if (nums[j] != val) {
                nums[i] = nums[j];
                i++;
            }
        }
        return i;
    }

    public static ListNode reverseBetween(ListNode head, int m, int n) {
        ListNode dummyHead = new ListNode(0);
        dummyHead.next = head;

        ListNode g = dummyHead;
        ListNode p = dummyHead.next;

        int step = 0;
        while (step < m - 1) {
            g = g.next;
            p = p.next;
            step++;
        }

        for (int i = 0; i < n - m; i++) {
            ListNode removed = p.next;
            p.next = p.next.next;

            removed.next = g.next;
            g.next = removed;
        }

        return dummyHead.next;
    }

    public static String multiply(String num1, String num2) {
        if (num1.equals("0") || num2.equals("0")) {
            return "0";
        }
        int[] res = new int[num1.length() + num2.length()];
        for (int i = num1.length() - 1; i >= 0; i--) {
            int n1 = num1.charAt(i) - '0';
            for (int j = num2.length() - 1; j >= 0; j--) {
                int n2 = num2.charAt(j) - '0';
                int sum = (res[i + j + 1] + n1 * n2);
                res[i + j + 1] = sum % 10;
                res[i + j] += sum / 10;
            }
        }

        StringBuilder result = new StringBuilder();
        for (int i = 0; i < res.length; i++) {
            if (i == 0 && res[i] == 0) {
                continue;
            }
            result.append(res[i]);
        }
        return result.toString();
    }

    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }
        int rows = grid.length;
        int columns = grid[0].length;
        int[][] dp = new int[rows][columns];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < rows; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for (int j = 1; j < columns; j++) {
            dp[0][j] = dp[0][j - 1] + grid[0][j];
        }
        for (int i = 1; i < rows; i++) {
            for (int j = 1; j < columns; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[rows - 1][columns - 1];
    }

    public int twoCitySchedCost(int[][] costs) {
        int t = costs.length;
        int cha[] = new int[t];
        int sum = 0;
        for (int i = 0; i < t; i++) {
            //������Ա��A���뵽B�ǵ����Ĳ�
            cha[i] = costs[i][1] - costs[i][0];
            //���������˵�A�ǵķ���
            sum += costs[i][0];
        }
        Arrays.sort(cha);
        //��ȥӦ��B�ǵ���Ա�Ķ�������
        for (int i = 0; i < t / 2; i++) {
            sum += cha[i];
        }
        return sum;
    }

    static List<List<Integer>> list = new LinkedList<>();

    public static List<List<Integer>> combinationSum(int[] candidates, int target) {
        if (candidates.length == 0) {
            return list;
        }
        Arrays.sort(candidates);//����
        LinkedList<Integer> temp = new LinkedList<>();
        backtrack(candidates, target, temp, 0);
        return list;
    }

    public static void backtrack(int[] nums, int sum, LinkedList<Integer> temp, int start) {
        if (sum == 0) {
            list.add(new LinkedList(temp));
            System.out.println("temp == " + temp);
            return;
        }
        if (sum > 0) {
            for (int i = start; i < nums.length; i++) {
                if (nums[i] <= sum) {//��֦
                    temp.add(nums[i]);
                    System.out.println("nums[i] === " + nums[i]);
                    backtrack(nums, sum - nums[i], temp, i);
                    //����
                    System.out.println("����v === " + temp.getLast());
                    temp.removeLast();
                }
            }

        }
    }

    public int findMin(int[] nums) {
        int len = nums.length;
        if (len == 0) {
            throw new IllegalArgumentException("����Ϊ�գ�����СԪ��");
        }
        int left = 0;
        int right = len - 1;
        while (left < right) {
            // int mid = left + (right - left) / 2;
            int mid = (left + right) >>> 1;
            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else {
                // ��Ϊ��Ŀ��˵������Լ��������в������ظ�Ԫ�ء�
                // ��ʱһ���� nums[mid] < nums[right]
                right = mid;
            }
        }
        // һ��������СԪ�أ�������������ж�
        return nums[left];
    }

    public boolean search02(int[] nums, int target) {
        int len = nums.length;
        if (len == 0) {
            return false;
        }
        int left = 0;
        int right = len - 1;
        while (left < right) {
            int mid = (left + right) >>> 1;
            if (nums[mid] > nums[left]) {
                if (nums[left] <= target && target <= nums[mid]) {
                    // ����ǰ����������
                    right = mid;
                } else {
                    left = mid + 1;
                }
            } else if (nums[mid] < nums[left]) {
                // �÷�֧�������֧һ��
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            } else {
                // Ҫ�ų�����߽�֮ǰ���ȿ�һ����߽���Բ������ų�
                if (nums[left] == target) {
                    return true;
                } else {
                    left = left + 1;
                }
            }

        }
        // �����б��Ժ󣬻�Ҫ�ж�һ�£��ǲ��� target
        return nums[left] == target;
    }

    public boolean search01(int[] nums, int target) {
        int len = nums.length;
        if (len == 0) {
            return false;
        }
        int left = 0;
        int right = len - 1;
        while (left < right) {
            int mid = (left + right + 1) >>> 1;
            if (nums[mid] < nums[right]) {
                // 10 11 4 5 6 7 8 9
                // �ұߵ�һ����˳�����飬�����м���
                if (nums[mid] <= target && target <= nums[right]) {
                    left = mid;
                } else {
                    right = mid - 1;
                }
            } else if (nums[mid] > nums[right]) {
                // 4 5 9  2
                // �����һ����˳�����飬�����м���
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid;
                }
            } else {
                if (nums[right] == target) {
                    return true;
                }
                right = right - 1;
            }
        }
        return nums[left] == target;
    }

    public int[] searchRange(int[] nums, int target) {
        int len = nums.length;
        if (len == 0) {
            return new int[]{-1, -1};
        }
        int firstPosition = findFirstPosition(nums, target);
        if (firstPosition == -1) {
            return new int[]{-1, -1};
        }
        int lastPosition = findLastPosition(nums, target);
        return new int[]{firstPosition, lastPosition};
    }

    private int findFirstPosition(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = (left + right) >>> 1;
            // С��һ�����ǽ�
            if (nums[mid] < target) {
                // ��һ������������ [mid + 1, right]
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if (nums[left] == target) {
            return left;
        }
        return -1;
    }

    private int findLastPosition(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = (left + right + 1) >>> 1;
            // ����һ�����ǽ�
            if (nums[mid] > target) {
                // ��һ������������ [left, mid - 1]
                right = mid - 1;
            } else {
                left = mid;
            }
        }
        return left;
    }

    public int search(int[] nums, int target) {
        int lo = 0, hi = nums.length - 1, mid = 0;
        while (lo <= hi) {
            mid = lo + (hi - lo) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            // �ȸ��� nums[mid] �� nums[lo] �Ĺ�ϵ�ж� mid ������λ����Ҷ�
            if (nums[mid] >= nums[lo]) {
                // ���ж� target ���� mid ����߻����ұߣ��Ӷ��������ұ߽� lo �� hi
                if (target >= nums[lo] && target < nums[mid]) {
                    hi = mid - 1;
                } else {
                    lo = mid + 1;
                }
            } else {
                if (target > nums[mid] && target <= nums[hi]) {
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
        }
        return -1;
    }

    public static int divide(int dividend, int divisor) {
        if (dividend == 0) {
            return 0;
        }
        if (dividend == Integer.MIN_VALUE && divisor == -1) {
            return Integer.MAX_VALUE;
        }
        boolean negative;
        //������������Ƿ��������
        negative = (dividend ^ divisor) < 0;
        long t = Math.abs((long) dividend);
        long d = Math.abs((long) divisor);
        int result = 0;
        for (int i = 31; i >= 0; i--) {
            //�ҳ��㹻�����2^n*divisor
            if ((t >> i) >= d) {
                //���������2^n
                result += 1 << i;
                //����������ȥ2^n*divisor
                t -= d << i;
            }
        }
        //��������ȡ��
        return negative ? -result : result;
    }

    public static int minArray(int[] numbers) {
        int i = 0, j = numbers.length - 1;
        while (i < j) {
            int m = (i + j) / 2;
            if (numbers[m] > numbers[j]) {
                i = m + 1;
            } else if (numbers[m] < numbers[j]) {
                j = m;
            } else {
                j--;
            }
        }
        return numbers[i];
    }

    public void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while (i >= 0 && nums[i + 1] <= nums[i]) {
            i--;
        }
        if (i >= 0) {
            int j = nums.length - 1;
            while (j >= 0 && nums[j] <= nums[i]) {
                j--;
            }
            swap(nums, i, j);
        }
        reverse(nums, i + 1);
    }

    private void reverse(int[] nums, int start) {
        int i = start, j = nums.length - 1;
        while (i < j) {
            swap(nums, i, j);
            i++;
            j--;
        }
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public static int lastStoneWeight(int[] stones) {
        int len = stones.length;
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(len, (o1, o2) -> o2 - o1);
        for (int stone : stones) {
            maxHeap.add(stone);
        }

        while (maxHeap.size() >= 2) {
            Integer head1 = maxHeap.poll();
            Integer head2 = maxHeap.poll();

            maxHeap.offer(head1 - head2);
        }
        return maxHeap.poll();
    }

    public static boolean isCousins(TreeNode root, int x, int y) {
        if (root == null) {
            return false;
        }
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int cnt = 0;
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                // �ж�x��y��Ӧ�Ľڵ��Ƿ������ֵ�
                if (node.left != null && node.right != null) {
                    if (node.left.val == x && node.right.val == y) {
                        return false;
                    }
                    if (node.left.val == y && node.right.val == x) {
                        return false;
                    }
                }
                if (node.left != null) {
                    queue.add(node.left);
                }
                if (node.right != null) {
                    queue.add(node.right);
                }
                if (node.val == x || node.val == y) {
                    if (++cnt == 2) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    public List<TreeNode> generateTrees(int n) {
        if (n == 0) {
            return new LinkedList<TreeNode>();
        }
        return generateTrees(1, n);
    }

    public List<TreeNode> generateTrees(int start, int end) {
        List<TreeNode> allTrees = new LinkedList<TreeNode>();
        if (start > end) {
            allTrees.add(null);
            return allTrees;
        }
        // ö�ٿ��и��ڵ�
        for (int i = start; i <= end; i++) {
            // ������п��е�����������
            List<TreeNode> leftTrees = generateTrees(start, i - 1);
            // ������п��е�����������
            List<TreeNode> rightTrees = generateTrees(i + 1, end);
            // ��������������ѡ��һ������������������������ѡ��һ����������ƴ�ӵ����ڵ���
            for (TreeNode left : leftTrees) {
                for (TreeNode right : rightTrees) {
                    TreeNode currTree = new TreeNode(i);
                    currTree.left = left;
                    currTree.right = right;
                    allTrees.add(currTree);
                }
            }
        }
        return allTrees;
    }

    //897. ����˳�������
    public TreeNode increasingBST(TreeNode root) {
        List<Integer> vals = new ArrayList();
        inorder(root, vals);
        TreeNode ans = new TreeNode(0), cur = ans;
        for (int v : vals) {
            cur.right = new TreeNode(v);
            cur = cur.right;
        }
        return ans.right;
    }

    public void inorder(TreeNode node, List<Integer> vals) {
        if (node == null) {
            return;
        }
        inorder(node.left, vals);
        vals.add(node.val);
        inorder(node.right, vals);
    }

    TreeNode cur;

    public TreeNode increasingBST01(TreeNode root) {
        TreeNode ans = new TreeNode(0);
        cur = ans;
        inorder(root);
        return ans.right;
    }

    public void inorder(TreeNode node) {
        if (node == null) {
            return;
        }
        inorder(node.left);
        node.left = null;
        cur.right = node;
        cur = node;
        inorder(node.right);
    }

    public boolean isLongPressedName(String name, String typed) {
        int nameLen = name.length();
        int typedLen = typed.length();
        if (nameLen > typedLen) {
            return false;
        }
        int i = 0;
        int j = 0;
        while (i < nameLen && j < typedLen) {
            char nameChar = name.charAt(i);
            char typedChar = typed.charAt(j);
            if (nameChar != typedChar) {
                return false;
            }
            int nameCount = 0;
            int typedCount = 0;
            while (i < nameLen && name.charAt(i) == nameChar) {
                nameCount++;
                i++;
            }
            while (j < typedLen && typed.charAt(j) == typedChar) {
                typedCount++;
                j++;
            }
            if (typedCount < nameCount) {
                return false;
            }
        }
        return (i == nameLen) && (j == typedLen);
    }

    public static int[] sortedSquares(int[] A) {
        int N = A.length;
        int j = 0;
        while (j < N && A[j] < 0) {
            j++;
        }
        int i = j - 1;
        int[] ans = new int[N];
        int t = 0;
        while (i >= 0 && j < N) {
            if (A[i] * A[i] < A[j] * A[j]) {
                ans[t++] = A[i] * A[i];
                i--;
            } else {
                ans[t++] = A[j] * A[j];
                j++;
            }
        }
        while (i >= 0) {
            ans[t++] = A[i] * A[i];
            i--;
        }
        while (j < N) {
            ans[t++] = A[j] * A[j];
            j++;
        }
        return ans;
    }

    public int[] fairCandySwap(int[] A, int[] B) {
        int sa = 0;
        int sb = 0;
        for (int a : A) {
            sa += a;
        }
        for (int b : B) {
            sb += b;
        }
        int delta = (sb - sa) / 2;
        Set<Integer> setB = new HashSet();
        for (int x : B) {
            setB.add(x);
        }
        //y=x+delta
        for (int x : A) {
            if (setB.contains(x + delta)) {
                return new int[]{x, x + delta};
            }
        }
        throw null;
    }

    public int findJudge(int N, int[][] trust) {
        int[] counter = new int[N + 1];
        for (int[] idx : trust) {
            counter[idx[0]]--; // ����
            counter[idx[1]]++; // ���
        }
        for (int i = 1; i <= N; i++) {
            // ���ٵ� ��� - ���� ���� N - 1
            // ���г���Ϊ 0
            if (counter[i] == N - 1) {
                return i;
            }
        }
        return -1;
    }

    public int missingNumber(int[] nums) {
        int i = 0, j = nums.length - 1;
        while (i <= j) {
            int m = (i + j) / 2;
            if (nums[m] == m) i = m + 1;
            else j = m - 1;
        }
        return i;
    }

    public int findCircleNum(int[][] M) {
        int[] visited = new int[M.length];
        int count = 0;
        for (int i = 0; i < M.length; i++) {
            if (visited[i] == 0) {
                dfs(M, visited, i);
                count++;
            }
        }
        return count;
    }

    public void dfs(int[][] M, int[] visited, int i) {
        for (int j = 0; j < M.length; j++) {
            if (M[i][j] == 1 && visited[j] == 0) {
                visited[j] = 1;
                dfs(M, visited, j);
            }
        }
    }

    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }

        int nr = grid.length;
        int nc = grid[0].length;
        int num_islands = 0;
        for (int r = 0; r < nr; ++r) {
            for (int c = 0; c < nc; ++c) {
                if (grid[r][c] == '1') {
                    ++num_islands;
                    dfs(grid, r, c);
                }
            }
        }

        return num_islands;
    }

    void dfs(char[][] grid, int r, int c) {
        int nr = grid.length;
        int nc = grid[0].length;

        if (r < 0 || c < 0 || r >= nr || c >= nc || grid[r][c] == '0') {
            return;
        }

        grid[r][c] = '0';
        dfs(grid, r - 1, c);
        dfs(grid, r + 1, c);
        dfs(grid, r, c - 1);
        dfs(grid, r, c + 1);
    }

    public int[] twoSum(int[] numbers, int target) {
        for (int i = 0; i < numbers.length; ++i) {
            int low = i + 1;
            int high = numbers.length - 1;
            while (low <= high) {
                int mid = (high - low) / 2 + low;
                if (numbers[mid] == target - numbers[i]) {
                    return new int[]{i + 1, mid + 1};
                } else if (numbers[mid] > target - numbers[i]) {
                    high = mid - 1;
                } else {
                    low = mid + 1;
                }
            }
        }
        return new int[]{-1, -1};
    }

    public int[] distributeCandies(int candies, int num_people) {
        int[] nums = new int[num_people];
        int j = 1;
        while (candies > 0) {
            for (int i = 0; i < nums.length && candies > 0; i++) {
                if (candies > j) {
                    nums[i] += j;
                } else {
                    nums[i] += candies;
                }
                candies = candies - j;
                j++;
            }
        }
        return nums;
    }

    //1170
    public int[] numSmallerByFrequency(String[] queries, String[] words) {
        /*
        ˼·��
        ����� queries ���� �� words ������Ե���С��ĸ����Ƶ��
        ����洢�� q ���� �� w ������
        Ȼ����б������ҳ� w ���� �� �� q[i] ���Ԫ�ظ���

        ������Խ���һ���Ż������ȿ����룬���˼�룩
        ���ǿ��Զ� w �����������Ȼ������ w[j] > q[i] ʱֱ�� break,��Ϊ����ض����Ǳ� q[i] ���ģ���˱� q[i] ���Ԫ�ظ���Ϊ wlen - j
        */
        int qlen = queries.length;
        int wlen = words.length;

        int[] q = new int[qlen];
        int[] w = new int[wlen];

        for (int i = 0; i < qlen; i++) {
            q[i] = helper(queries[i]);
        }
        for (int i = 0; i < wlen; i++) {
            w[i] = helper(words[i]);
        }
        Arrays.sort(w);
        int[] res = new int[qlen];
        for (int i = 0; i < qlen; i++) {
            int j = 0;
            for (; j < wlen; j++) {
                if (w[j] > q[i]) {
                    break;
                }
            }
            res[i] = wlen - j;
        }
        return res;
    }

    /*
    �õ�ĳ���ַ�����С��ĸ��Ƶ��
    ͳ��ÿ���ַ����ֵĴ�����Ȼ����� chs�����Ȳ��� 0 ������С��ĸ��Ƶ�Σ�ֱ�ӷ��ؼ���
    */
    private int helper(String str) {
        int[] chs = new int[26];
        for (char ch : str.toCharArray()) {
            chs[ch - 'a']++;
        }
        for (int num : chs) {
            if (num != 0) {
                return num;
            }
        }
        return 0;
    }

    public int[][] merge(int[][] intervals) {
        int len = intervals.length;
        if (len < 2) {
            return intervals;
        }
        // �����������
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[0]));
        // Ҳ����ʹ�� Stack����Ϊ����ֻ���Ľ���������һ������
        List<int[]> res = new ArrayList<>();
        res.add(intervals[0]);
        for (int i = 1; i < len; i++) {
            int[] curInterval = intervals[i];
            // ÿ���±��������б��뵱ǰ������е����һ�������ĩβ�˵���бȽ�
            int[] peek = res.get(res.size() - 1);
            if (curInterval[0] > peek[1]) {
                res.add(curInterval);
            } else {
                // ע�⣬����Ӧ��ȡ���
                peek[1] = Math.max(curInterval[1], peek[1]);
            }
        }
        return res.toArray(new int[res.size()][]);
    }

    int maxCoins(int[] nums) {
        int n = nums.length;
        // ����������������
        int[] points = new int[n + 2];
        points[0] = points[n + 1] = 1;
        for (int i = 1; i <= n; i++) {
            points[i] = nums[i - 1];
        }
        // base case �Ѿ�������ʼ��Ϊ 0
        int[][] dp = new int[n + 2][n + 2];
        // ��ʼ״̬ת��
        // i Ӧ�ô�������
        for (int i = n; i >= 0; i--) {
            // j Ӧ�ô�������
            for (int j = i + 1; j < n + 2; j++) {
                // �����Ƶ��������ĸ���
                for (int k = i + 1; k < j; k++) {
                    // ������ѡ��
                    dp[i][j] = Math.max(
                            dp[i][j],
                            dp[i][k] + dp[k][j] + points[i] * points[j] * points[k]
                    );
                }
            }
        }
        return dp[0][n + 1];
    }

    public int minCount(int[] coins) {
        int cnt = 0;
        for (int e : coins) {
            if (e == 1) {
                cnt++;
            } else {
                cnt = cnt + e / 2 + e % 2;
            }
        }
        return cnt;
    }

    public int lengthOfLIS(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        int[] dp = new int[nums.length];
        dp[0] = 1;
        int maxans = 1;
        for (int i = 1; i < dp.length; i++) {
            int maxval = 0;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    maxval = Math.max(maxval, dp[j]);
                }
            }
            dp[i] = maxval + 1;
            maxans = Math.max(maxans, dp[i]);
        }
        return maxans;
    }

    public int maxProfit(int[] prices) {
        int minPrice1 = Integer.MAX_VALUE;
        int maxProfit1 = 0;
        int maxProfitAfterBuy = Integer.MIN_VALUE;
        int maxProfit2 = 0;
        for (int price : prices) {
            // 1.��һ����С����۸�
            minPrice1 = Math.min(minPrice1, price);
            // 2.��һ���������������
            maxProfit1 = Math.max(maxProfit1, price - minPrice1);
            // 3.�ڶ��ι�����ʣ�ྻ����
            maxProfitAfterBuy = Math.max(maxProfitAfterBuy, maxProfit1 - price);
            // 4.�ڶ����������ܹ���õ�������󣨵�3���ľ����� + ��4�������Ĺ�ƱǮ��
            maxProfit2 = Math.max(maxProfit2, price + maxProfitAfterBuy);
        }
        return maxProfit2;
    }

    public int maxProfitDp(int[] prices) {
        int len = prices.length;
        if (len < 2) {
            return 0;
        }
        // dp[i][j] ����ʾ [0, i] �����״̬Ϊ j ���������
        // j = 0��ʲô��������
        // j = 1���� 1 ������һ֧��Ʊ
        // j = 2���� 1 ������һ֧��Ʊ
        // j = 3���� 2 ������һ֧��Ʊ
        // j = 4���� 2 ������һ֧��Ʊ
        // ��ʼ��
        int[][] dp = new int[len][5];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        // 3 ״̬����û�з��������Ӧ�ø�ֵΪһ�������ܵ���
        for (int i = 0; i < len; i++) {
            dp[i][3] = Integer.MIN_VALUE;
        }
        // ״̬ת��ֻ�� 2 �������
        // ��� 1��ʲô������
        // ��� 2������һ��״̬ת�ƹ���
        for (int i = 1; i < len; i++) {
            // j = 0 ��ֵ��Զ�� 0
            dp[i][0] = 0;
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
            dp[i][2] = Math.max(dp[i - 1][2], dp[i - 1][1] + prices[i]);
            dp[i][3] = Math.max(dp[i - 1][3], dp[i - 1][2] - prices[i]);
            dp[i][4] = Math.max(dp[i - 1][4], dp[i - 1][3] + prices[i]);
        }
        // ���ֵֻ�����ڲ��ֹɵ�ʱ�������Դ�� 3 ����j = 0 ,j = 2, j = 4
        return Math.max(0, Math.max(dp[len - 1][2], dp[len - 1][4]));
    }

    public int maxProduct(int[] nums) {
        int[] maxF = new int[nums.length];
        int[] minF = new int[nums.length];
        maxF[0] = nums[0];
        minF[0] = nums[0];
        int ans = nums[0];
        for (int i = 1; i < nums.length; ++i) {
            maxF[i] = Math.max(maxF[i - 1] * nums[i], Math.max(nums[i], minF[i - 1] * nums[i]));
            minF[i] = Math.min(minF[i - 1] * nums[i], Math.min(nums[i], maxF[i - 1] * nums[i]));
            ans = Math.max(ans, maxF[i]);
        }
        return ans;
    }

    public int[] countBits(int num) {
        int[] ans = new int[num + 1];
        for (int i = 0; i <= num; ++i) {
            ans[i] = popcount(i);
        }
        return ans;
    }

    private int popcount(int x) {
        int count;
        for (count = 0; x != 0; ++count) {
            x &= x - 1;
        }
        return count;
    }

    public boolean isPowerOfTwo(int n) {
        if (n == 0) {
            return false;
        }
        long x = (long) n;
        return (x & (-x)) == x;
    }

    public int rangeSumBST(TreeNode root, int L, int R) {
        int ans = 0;
        Stack<TreeNode> stack = new Stack();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            if (node != null) {
                if (L <= node.val && node.val <= R) {
                    ans += node.val;
                }
                if (L < node.val) {
                    stack.push(node.left);
                }
                if (node.val < R) {
                    stack.push(node.right);
                }
            }
        }
        return ans;
    }

    public int translateNum(int num) {
        String s = String.valueOf(num);
        int a = 1, b = 1;
        for (int i = 2; i <= s.length(); i++) {
            String tmp = s.substring(i - 2, i);
            int c = tmp.compareTo("10") >= 0 && tmp.compareTo("25") <= 0 ? a + b : a;
            b = a;
            a = c;
        }
        return a;
    }

    public boolean isInterleave(String s1, String s2, String s3) {
        int n = s1.length(), m = s2.length(), t = s3.length();

        if (n + m != t) {
            return false;
        }

        boolean[][] f = new boolean[n + 1][m + 1];

        f[0][0] = true;
        for (int i = 0; i <= n; ++i) {
            for (int j = 0; j <= m; ++j) {
                int p = i + j - 1;
                if (i > 0) {
                    f[i][j] = f[i][j] || (f[i - 1][j] && s1.charAt(i - 1) == s3.charAt(p));
                }
                if (j > 0) {
                    f[i][j] = f[i][j] || (f[i][j - 1] && s2.charAt(j - 1) == s3.charAt(p));
                }
            }
        }
        return f[n][m];
    }

    public boolean canMakeArithmeticProgression(int[] arr) {
        Arrays.sort(arr);
        for (int i = 1; i < arr.length - 1; ++i) {
            if (arr[i] * 2 != arr[i - 1] + arr[i + 1]) {
                return false;
            }
        }
        return true;
    }

    public int balancedStringSplit(String s) {
        int num = 0;
        int res = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == 'L') {
                num++;
            } else {
                num--;
            }
            if (num == 0) {
                res++;
            }
        }
        return res;
    }

    public int tribonacci(int n) {
        if (n < 3) {
            return n == 0 ? 0 : 1;
        }

        int tmp, x = 0, y = 1, z = 1;
        for (int i = 3; i <= n; ++i) {
            tmp = x + y + z;
            x = y;
            y = z;
            z = tmp;
        }
        return z;
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left == null && right == null) {
            return null;
        }
        if (left == null) {
            return right;
        }
        if (right == null) {
            return left;
        }
        return root;
    }

    public TreeNode lowestCommonAncestor01(TreeNode root, TreeNode p, TreeNode q) {
        if (root.val < p.val && root.val < q.val) {
            return lowestCommonAncestor(root.right, p, q);
        }
        if (root.val > p.val && root.val > q.val) {
            return lowestCommonAncestor(root.left, p, q);
        }
        return root;
    }

    public int kthLargest(TreeNode root, int k) {
        List<Integer> list = new ArrayList<>();
        dfs(root, list);
        return list.get(list.size() - k);
    }

    private void dfs(TreeNode root, List<Integer> list) {
        if (root == null) {
            return;
        }
        if (root.left != null) {
            dfs(root.left, list);
        }
        list.add(root.val);
        if (root.right != null) {
            dfs(root.right, list);
        }
    }

    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return recur(root.left, root.right);
    }

    boolean recur(TreeNode L, TreeNode R) {
        if (L == null && R == null) {
            return true;
        }
        if (L == null || R == null || L.val != R.val) {
            return false;
        }
        return recur(L.left, R.right) && recur(L.right, R.left);
    }

    public static TreeNode mirrorTree01(TreeNode root) {
        if (root == null) return null;
        Stack<TreeNode> stack = new Stack<TreeNode>() {{
            add(root);
        }};
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            if (node.left != null) {
                stack.add(node.left);
            }
            if (node.right != null) {
                stack.add(node.right);
            }
            TreeNode tmp = node.left;
            node.left = node.right;
            node.right = tmp;
        }
        return root;
    }

    public static TreeNode mirrorTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode tmp = root.left;
        root.left = mirrorTree(root.right);
        root.right = mirrorTree(tmp);
        return root;
    }

    static boolean flag = true;

    public static boolean isBalanced(TreeNode root) {
        if (root == null) {
            return flag;
        }
        dfs(root);
        return flag;
    }

    public static int dfs(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = dfs(root.left);
        int right = dfs(root.right);

        if (Math.abs(left - right) > 1) {
            flag = false;
        }
        return Math.max(left, right) + 1;
    }

}
