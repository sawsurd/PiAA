#include "Board.hpp"
#include <ctime>
#include <limits>
#include <cmath>

bool isPrime(int N){
    for (int i = 2; i < sqrt(N); ++i) {
        if (N % i == 0) return false;
    }
    return true;
}

int main(){
    int N;
    while (true) {
        if (!(cin >> N) || N < 2 || N > 30) {
            cout << "incorrect input\n";
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
        } else {
            break;
        }
    }
    Board Board1 = Board(N);
    vector<Square> res;
    clock_t start = clock();

    if (N % 2 == 0 || N % 3 == 0 || N % 5 == 0 || N % 7 == 0){
        res = Board1.numberDivisor();
    }else {
        if (isPrime(N)) {
            Board1.board = Board1.fillSquare({0, 0, N / 2 + 1});
            Board1.board = Board1.fillSquare({0, N / 2 + 1, N / 2});
            Board1.board = Board1.fillSquare({N / 2 + 1, 0, N / 2});
            res = Board1.backtracking();
            res.insert(res.begin(), {0, N / 2 + 1, N / 2});
            res.insert(res.begin(), {N / 2 + 1, 0, N / 2});
            res.insert(res.begin(), {0, 0, N / 2 + 1});

        }
    }
    clock_t stop = clock();
    double duration = double(stop - start) / CLOCKS_PER_SEC;

    cout << res.size() << '\n';
    for (auto & square : res) {
        cout << square.x + 1 << ' ' << square.y + 1 << ' ' <<
             square.side << '\n';
    }
    cout << duration << " seconds" << endl;

    return 0;
}