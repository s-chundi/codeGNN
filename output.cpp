// This is a sample c++ input program 
#include <iostream>
#include <string>
using namespace std;

int main(){
    int n;
    cin >> n;
    bool longword;
    string w;
    for (int i = 0; i < n; i++){
        cin >> w;
        longword = w.length() > 10;
        if(longword){
            cout << w[0] << w.length() - 2 << w[w.length() - 1] << endl;
        } else {
            cout << w << endl;
        }
    }
    return 0;
}