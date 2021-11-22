#pragma once

#include "mathlib/vector.h"

#include <cstdint>
#include <stdexcept>

namespace MathLib {

template <class T>
class matrix {
    public:
        matrix(uint64_t l, uint64_t c);
        ~matrix();

    private:
        vector<T> m_data;
        uint64_t m_size_lines, m_size_columns;
};

template<class T>
matrix<T>::matrix(uint64_t l, uint64_t c)
{
    if (l*c < sizeof(uint64_t)) {
        m_data = vector<T>(l*c);
    } else {
        throw std::range_error("the matrix size is too big");
    }
}

}