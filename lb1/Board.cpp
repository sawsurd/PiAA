#include "Board.hpp"
#include <algorithm>

Board::Board(int N) : N(N) {
    board.resize(N, std::vector<int>(N, 0));
    result.resize(N);
}

vector<Square> Board::numberDivisor() {
    result.clear();

    if (N % 2 == 0) {
        fillEvenSquares();
    } else if (N % 3 == 0) {
        fillDivisibleByThreeSquares();
    } else if (N % 5 == 0) {
        fillDivisibleByFiveSquares();
    } else if (N % 7 == 0) {
        fillDivisibleBySevenSquares();
    }

    return result;
}

void Board::fillEvenSquares() {
    int half = N / 2;
    for (int i = 1; i <= 4; ++i) {
        int x = (i <= 2) ? 0 : half;
        int y = (i % 2 == 0) ? 0 : half;
        result.push_back({x, y, half});
        board = fillSquare({x, y, half});
    }
    cout << "Divisible by 2\n";
    printBoard();
}

void Board::fillDivisibleByThreeSquares() {
    result.push_back({0, 0, (N * 2) / 3 });                        //*   *   *
    result.push_back({0, (N * 2) / 3, N / 3 });                    //*   *   *
    result.push_back({N / 3, (N * 2) / 3, N / 3 });                //*   *   *
    result.push_back({(N * 2) / 3, (N * 2) / 3, N / 3 });
    result.push_back({(N * 2) / 3, 0, N / 3 });
    result.push_back({(N * 2) / 3, N / 3, N / 3 });

    for (const auto& square : result) {
        board = fillSquare(square);
    }
    cout << "Divisible by 3\n";
    printBoard();
}

void Board::fillDivisibleByFiveSquares() {
    result.push_back({0, 0, (N * 3) / 5 });
    result.push_back({0, (N * 3) / 5, (N * 2) / 5 });
    result.push_back({(N * 3) / 5, 0, (N * 2) / 5 });
    result.push_back({(N * 3) / 5, (N * 3) / 5, (N * 2) / 5 });
    result.push_back({(N * 2) / 5, (N * 3) / 5, N / 5 });
    result.push_back({(N * 2) / 5, (N * 4) / 5, N / 5 });
    result.push_back({(N * 3) / 5, (N * 2) / 5, N / 5 });
    result.push_back({(N * 4) / 5, (N * 2) / 5 , N / 5 });

    for (const auto& square : result) {
        board = fillSquare(square);
    }
    cout << "Divisible by 5\n";
    printBoard();
}

void Board::fillDivisibleBySevenSquares() {
    result.push_back({0, 0, (N * 4) / 7 });
    result.push_back({0, (N * 4) / 7, (N * 3) / 7 });
    result.push_back({(N * 4) / 7, 0, (N * 3) / 7 });
    result.push_back({(N * 3) / 7, (N * 4) / 7, N / 7});
    result.push_back({(N * 3) / 7, (N * 5) / 7, (N * 2) / 7});
    result.push_back({(N * 5) / 7, (N * 5) / 7, (N * 2) / 7});
    result.push_back({(N * 4) / 7, (N * 3) / 7, (N * 2) / 7});
    result.push_back({(N * 6) / 7, (N * 3) / 7, N / 7});
    result.push_back({(N * 6) / 7, (N * 4) / 7, N / 7});

    for (const auto& square : result) {
        board = fillSquare(square);
    }
    cout << "Divisible by 7\n";
    printBoard();
}

std::vector<std::vector<int>>& Board::fillSquare(Square square) {
    for (int i = square.x; i < square.x + square.side; ++i) {
        for (int j = square.y; j < square.y + square.side; ++j) {
            board[i][j] = 1;
        }
    }
    return board;
}

bool Board::isSquareEmpty(Square square) const {
    for (int i = square.x; i < square.x + square.side; ++i) {
        for (int j = square.y; j < square.y + square.side; ++j) {
            if (board[i][j] != 0) return false;
        }
    }
    return true;
}

Square Board::findEmptySquare() {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (board[i][j] == 0) {
                int side = N - std::max(i, j);
                while (!isSquareEmpty({i, j, side})) {
                    --side;
                }
                return {i, j, side};
            }
        }
    }
    return {0, 0, 0};
}

std::vector<std::vector<int>>& Board::deleteLastSquare(Square square) {
    for (int i = square.x; i < square.x + square.side; ++i) {
        for (int j = square.y; j < square.y + square.side; ++j) {
            board[i][j] = 0;
        }
    }
    return board;
}

bool Board::isTableFull() const {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (board[i][j] == 0) return false;
        }
    }
    return true;
}

std::vector<std::vector<int>>& Board::cutSquare(Square square) {
    for (int i = square.x; i < square.x + square.side; ++i) {
        for (int j = square.y; j < square.y + square.side; ++j) {
            if (i == square.x + square.side - 1 || j == square.y + square.side - 1) {
                board[i][j] = 0;
            }
        }
    }
    return board;
}

void Board::printBoard() const {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << board[i][j] << "  ";
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

vector<Square> Board::backtracking() {
    bool start = true;

    while (!temporary.empty() || start) {
        start = false;
        while (!isTableFull()) {
            if (temporary.size() >= result.size()) break;
            Square temp = findEmptySquare();
            cout << "Found square's size: " << temp.side << endl;
            board = fillSquare(temp);
            temporary.push_back(temp);
            cout << "Filling the table while !isTableFull or temporary.size() < result.size()\n";
            printBoard();
        }
        if (temporary.size() < result.size()) {
            cout << "New found number of temporary squares: " << temporary.size() << endl;
            result = temporary;
        }
        board = deleteLastSquare(temporary.back());
        temporary.pop_back();
        cout << "Delete last square from board\n";
        printBoard();

        while (!temporary.empty() && temporary.back().side == 1) {
            board = deleteLastSquare(temporary.back());
            temporary.pop_back();
            cout << "Delete squares from temporary while it's side == 1\n";
            printBoard();
        }

        if (!temporary.empty()) {
            board = cutSquare(temporary.back());
            temporary.back().side--;
            cout << "Cutting the last square\n";
            printBoard();
        }
    }
    cout << "Temporary is empty\n";

    return result;
}