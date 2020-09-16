package leetcode;

public class NQueens {
    int n;
    //��¼ÿ�ַ����Ļʺ��������
    int[] res;
    //�ܷ�����
    int count = 0;

    public int totalNQueens(int n) {
        this.n = n;
        this.res = new int[n];
        check(0); // ��0�п�ʼ����
        return count;
    }

    //���õ�k��
    public void check(int k) {
        if (k == n) {
            count++;
            return;
        }
        for (int i = 0; i < n; i++) {
            // ��λ��i �������������k��λ��
            res[k] = i;
            if (!judge(k)) {
                //����ͻ�Ļ������ݷ�����һ��
                check(k + 1);
            }
            //��ͻ�Ļ�����һ��λ��
        }
    }

    //�жϵ�k�еķ����Ƿ���֮ǰλ�ó�ͻ
    public boolean judge(int k) {
        for (int i = 0; i < k; i++) {
            if (res[k] == res[i] || Math.abs(k - i) == Math.abs(res[k] - res[i])) {
                return true;
            }
        }
        return false;
    }
}
