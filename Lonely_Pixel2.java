public class Solution {
    public int findBlackPixel(char[][] picture, int N) {
        int m = picture.length;
        int n = picture[0].length;
        int result = 0;
        
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                // if (picture[i][j]!='B' && picture[i][j]!='W') return 0;
                if (picture[i][j]=='B') {
                    // if (flow(picture,i,j,N)) result++;
                    if (flow(picture,i,j,N)) result++;
                }
            }
        }
        return result;
    }

    
    public boolean flow(char[][] picture, int i, int j, int N) {
        int m = picture.length;
        int n = picture[0].length;
        int countR = 0;
        int countC = 0;
        List<Integer> index = new ArrayList<>();
        
        for (int k=0; k<m; k++) {
            if (picture[k][j] == 'B') {
                countR++;
                index.add(k);
            }
        }
        for (int k=0; k<n; k++) {
            if (picture[i][k] == 'B') countC++;
        }
        
        if ( !(countR==N && countC==N) ) return false;
        
        for (Integer ind : index) {
            // if ( picture[i]!=picture[ind] ) return false;
            for (int k=0; k<n; k++) {
                if (picture[i][k] != picture[ind][k]) return false;
            }
        }
        
        return true;
    }
    
}
