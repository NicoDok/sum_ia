#include <iostream>
#include <assert.h>

#include "mathlib/matrix.h"
#include "mathlib/vector.h"

#include "tools.h"

// #define TEST01


#ifdef TEST01
void exercice01_Transposition() {
    std::cout << __CURRENT_FUNCTION__ << std::endl;

    MathLib::matrix<int64_t> matrix = {{0, 1} {2, 3} {4, 5}};

    assert(matrix[0][0] == 0);
    assert(matrix[0][1] == 1);
    assert(matrix[1][0] == 2);
    assert(matrix[1][1] == 3);
    assert(matrix[2][0] == 4);
    assert(matrix[2][1] == 5);

    matrix.transpose();

    assert(matrix[0][0] == 0);
    assert(matrix[0][1] == 2);
    assert(matrix[0][2] == 4);
    assert(matrix[1][0] == 1);
    assert(matrix[1][1] == 3);
    assert(matrix[1][2] == 5);
}
#endif

int main(int, char**) {
    MathLib::vector<uint64_t> vec({3, 2, 1});
    std::cout << "Hello" << std::endl;
#ifdef TEST01
    exercice01_Transposition();
#endif

    MathLib::vector<int64_t> tab_int(5);
    tab_int[3] = 2;
}
