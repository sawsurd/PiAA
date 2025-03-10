#ifndef BOARD_HPP
#define BOARD_HPP

#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

struct Square {
    int x, y, side;
};

class Board {
public:
    int N;
    vector<vector<int>> board;
    vector<Square> result;
    vector<Square> temporary;

    explicit Board(int N);
    vector<Square> numberDivisor();
    vector<Square> backtracking();

    std::vector<std::vector<int>>& fillSquare(Square square);
    bool isSquareEmpty(Square square) const;
    Square findEmptySquare();
    std::vector<std::vector<int>>& deleteLastSquare(Square square);
    bool isTableFull() const;
    std::vector<std::vector<int>>& cutSquare(Square square);
    void printBoard() const;

    void fillEvenSquares();
    void fillDivisibleByThreeSquares();
    void fillDivisibleByFiveSquares();
    void fillDivisibleBySevenSquares();
};

#endif
